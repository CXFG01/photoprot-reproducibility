"""Render the full PDB archive to a training corpus, on a many-core CPU box.

Phase 1.5 rendered CATH S40 domains: pre-chopped, single-chain, ~150 residues,
one object per file. This renders the whole experimental archive instead, which
changes three things and keeps one.

WHAT CHANGES

  * Input is mmCIF (gzipped, read natively by PyMOL), not CATH's extensionless
    per-domain PDB files. The archive moves toward extended PDB identifiers that
    legacy PDB format cannot represent, so mmCIF is the only future-proof choice.

  * Each entry yields more than one structural object. A figure may show the
    asymmetric unit (what was deposited) or the biological assembly (what the
    molecule actually is). Measured over a 1,500-entry sample, assembly 1 is
    IDENTICAL to the AU for 65.1% of entries, LARGER for 13.0% (the AU is a
    fragment of the real molecule) and SMALLER for 22.0% (the AU holds several
    copies). So the assembly is rendered only where it genuinely differs:
    always rendering both would duplicate two thirds of the work, and never
    rendering it would train on half-molecules for a third of the archive.

  * There are no CATH labels for 40.5% of entries. Labels are joined on pdb_id
    downstream and simply absent for the rest; those entries are index-only
    corpus. Nothing here depends on a label.

WHAT DOES NOT CHANGE

  The style space and the reproducibility contract are inherited verbatim from
  photoprot.render.spec, so renders are comparable with Phase 1.5 and every
  image is regenerable from (object_id, view, seed). Two details are load-bearing
  and were both learned the hard way in pymol_worker.py:

    * max_threads=1. PyMOL's ray tracer is multithreaded and its output depends
      on thread scheduling: without this, four renders of identical input differ
      in ~80/147,456 pixels. Costs nothing here because parallelism is across
      processes, one PyMOL each.

    * ray() then png() then wait for the file size to settle. cmd.png() hands
      the buffer off and returns, so the next job's delete('all') can land
      mid-write, losing the object and emitting a blank-but-valid PNG. Measured
      at ~5% of renders, intermittent on identical input.

Output is one tar per shard plus a parquet manifest. 2.8M loose small files
would make every subsequent listing and shuffle painful, and WebDataset reads
tars directly.

    python pdb_render.py index  --shard 0 --n-shards 32
    python pdb_render.py render --shard 0 --n-shards 32 --views 8
"""
from __future__ import annotations

import argparse
import colorsys
import io
import os
import sys
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Paths. The box is scratch: nothing here is canonical, everything is either
# re-downloadable from RCSB or regenerable from (object_id, view, seed).
# ----------------------------------------------------------------------------
WORK = Path(os.environ.get("PHOTOPROT_WORK", "/home/ubuntu/workspace"))
ARCHIVE = WORK / "data/pdb/mmCIF"
CCD_CACHE = WORK / "ccd_cache"
INDEX_DIR = WORK / "data/index"
RENDER_DIR = WORK / "data/renders"

# A render writes to tmpfs and is immediately read back into a tar, so the disk
# never sees 2.8M small files.
SCRATCH = Path("/dev/shm/photoprot")

# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
MIN_RESIDUES = 30          # below this a cartoon carries no topology worth learning
MIN_INK_FRACTION = 0.005   # a blank render is still a valid PNG; this catches it
MAX_ATOMS = 300_000        # capsids and filaments go to a deferred queue

# ----------------------------------------------------------------------------
# Style space - inherited verbatim from photoprot.render.spec
# ----------------------------------------------------------------------------
CANONICAL = {
    "representation": "cartoon",
    "color_scheme": "uniform",
    "background": "white",
    "ambient_occlusion": 0,
    "antialias": 1,
    "zoom_buffer": 2.0,
}
REPRESENTATIONS = ["cartoon", "ribbon"]
REPRESENTATION_P = [0.75, 0.25]
COLOR_SCHEMES = ["uniform", "rainbow", "sse", "mono", "bfactor"]
COLOR_P = [0.35, 0.25, 0.15, 0.15, 0.10]
BACKGROUNDS = ["white", "black", "grey80"]
BACKGROUND_P = [0.70, 0.15, 0.15]


def job_seed(object_id: str, view: int, global_seed: int) -> int:
    """Stable per-(object, view) seed.

    CRC32 rather than hash(): Python randomises str hashing per process, which
    would make renders irreproducible across runs.
    """
    return zlib.crc32(f"{object_id}:{view}:{global_seed}".encode()) & 0xFFFFFFFF


def sample_view(object_id: str, view: int, global_seed: int = 0) -> dict:
    """Every parameter of one render, deterministically derived."""
    rng = np.random.default_rng(job_seed(object_id, view, global_seed))

    # Orientation is random for EVERY view including the canonical one: a single
    # arbitrary viewpoint is the situation the model is actually deployed in.
    # Normalising a 4D Gaussian samples uniformly over SO(3); naive Euler angles
    # would concentrate near the poles.
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)

    if view == 0:
        style = dict(CANONICAL)
    else:
        style = {
            "representation": str(rng.choice(REPRESENTATIONS, p=REPRESENTATION_P)),
            "color_scheme": str(rng.choice(COLOR_SCHEMES, p=COLOR_P)),
            "background": str(rng.choice(BACKGROUNDS, p=BACKGROUND_P)),
            "ambient_occlusion": int(rng.integers(0, 2)),
            "antialias": 1,
            "zoom_buffer": float(rng.uniform(1.5, 3.0)),
        }
    style.update({
        "quat_w": float(q[0]), "quat_x": float(q[1]),
        "quat_y": float(q[2]), "quat_z": float(q[3]),
        "view": view,
        "role": "eval_single" if view == 0 else "train_view",
    })
    return style


# ============================================================================
# Phase 1: index. One gemmi parse per entry, recording what objects it yields.
# ============================================================================

def _polymer_stats(model) -> tuple[int, int, int]:
    """(atoms, residues-with-CA, chains) over peptide polymers only."""
    atoms = res = chains = 0
    for chain in model:
        ca = sum(1 for r in chain if r.find_atom("CA", "*") is not None)
        if ca == 0:
            continue
        chains += 1
        res += ca
        atoms += sum(len(r) for r in chain)
    return atoms, res, chains


def index_shard(shard: int, n_shards: int) -> None:
    import gemmi
    import pandas as pd

    files = sorted(ARCHIVE.glob("*/*.cif.gz"))
    mine = [f for i, f in enumerate(files) if i % n_shards == shard]
    print(f"[index {shard}] {len(mine):,} of {len(files):,} entries", flush=True)

    rows = []
    t0 = time.time()
    for i, path in enumerate(mine, 1):
        pdb_id = path.name.split(".")[0].lower()
        rec = {
            "pdb_id": pdb_id, "path": str(path),
            "au_atoms": 0, "au_res": 0, "au_chains": 0,
            "asm1_atoms": 0, "asm1_res": 0, "asm1_chains": 0,
            "has_asm1": False, "asm1_differs": False, "n_assemblies": 0,
            "keep_au": False, "keep_asm1": False, "deferred": False,
            "error": "",
        }
        try:
            st = gemmi.read_structure(str(path))
            st.setup_entities()
            st.remove_ligands_and_waters()
            if len(st) == 0:
                raise ValueError("no model")
            model = st[0]
            rec["au_atoms"], rec["au_res"], rec["au_chains"] = _polymer_stats(model)
            rec["n_assemblies"] = len(st.assemblies)

            asm = next((a for a in st.assemblies if a.name == "1"), None)
            if asm is not None:
                built = gemmi.make_assembly(
                    asm, model, gemmi.HowToNameCopiedChain.AddNumber)
                a_atoms, a_res, a_chains = _polymer_stats(built)
                rec.update(has_asm1=True, asm1_atoms=a_atoms,
                           asm1_res=a_res, asm1_chains=a_chains)
                rec["asm1_differs"] = (a_atoms != rec["au_atoms"]) and a_atoms > 0

            # Filters. A protein cartoon needs resolved backbone; nucleic-only
            # entries and short peptides carry nothing learnable.
            rec["keep_au"] = rec["au_res"] >= MIN_RESIDUES
            rec["keep_asm1"] = bool(
                rec["asm1_differs"] and rec["asm1_res"] >= MIN_RESIDUES)
            if rec["au_atoms"] > MAX_ATOMS or rec["asm1_atoms"] > MAX_ATOMS:
                rec["deferred"] = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(rec)
        if i % 2000 == 0:
            print(f"[index {shard}] {i}/{len(mine)} "
                  f"{i/(time.time()-t0):.0f} entries/s", flush=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"index_{shard:03d}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[index {shard}] wrote {out} ({len(rows):,} rows) "
          f"in {time.time()-t0:.0f}s", flush=True)


# ============================================================================
# Phase 2: render
# ============================================================================

def _init_pymol():
    import pymol
    pymol.finish_launching(["pymol", "-qck"])  # quiet, no GUI, no plugins
    from pymol import cmd

    # Determinism: PyMOL's ray tracer is multithreaded and its pixel output
    # depends on thread scheduling. Parallelism here is across processes.
    cmd.set("max_threads", 1)

    # The chemical component dictionary is on disk, so a structure containing an
    # unknown component resolves locally. Without this PyMOL reaches out to
    # files.rcsb.org mid-render - a per-render network call that would get us
    # throttled at 2.8M renders, and a silent bonding change when it fails.
    cmd.set("fetch_path", str(CCD_CACHE))
    return cmd


def _wait_for_stable_file(path: Path, timeout: float = 300.0, settle: float = 0.01) -> bool:
    """Block until `path` exists and its size stops changing.

    The only reliable signal that PyMOL finished writing: the write is
    asynchronous and cmd.sync() reports spurious lock_attempt timeouts headless.
    """
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last:
                return True
            last = size
        time.sleep(settle)
    return False


def _ink_fraction(path: Path) -> float:
    """Fraction of pixels that are not the modal (background) colour."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > 128:
            im = im.resize((128, 128), Image.NEAREST)
        colors = im.getcolors(maxcolors=128 * 128)
    if not colors:
        return 1.0
    total = sum(c for c, _ in colors)
    bg = max(colors, key=lambda c: c[0])[0]
    return 1.0 - bg / total


def _apply_color(cmd, sel, scheme, rng) -> None:
    """Colour schemes that real figures actually use."""
    if scheme == "uniform":
        h = rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.85)
        cmd.set_color("randhue", [r, g, b])
        cmd.color("randhue", sel)
    elif scheme == "rainbow":
        cmd.spectrum("count", "rainbow", sel + " and name CA")
    elif scheme == "sse":
        cmd.color("red", sel + " and ss H")
        cmd.color("yellow", sel + " and ss S")
        cmd.color("green", sel + " and not (ss H or ss S)")
    elif scheme == "bfactor":
        cmd.spectrum("b", "blue_white_red", sel)
    elif scheme == "mono":
        cmd.color("grey70", sel)
    else:
        raise ValueError(f"unknown colour scheme: {scheme}")


def _set_orientation(cmd, sel, quat) -> None:
    """Camera at an arbitrary orientation from a unit quaternion.

    Orient once for a sane camera distance, then overwrite the rotation block
    and re-zoom. Driving the camera with successive turn() calls would make the
    result depend on call order and be impossible to reproduce from the manifest.
    """
    w, x, y, z = quat
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    cmd.orient(sel)
    v = list(cmd.get_view())
    v[0:9] = rot
    cmd.set_view(v)


def render_one(cmd, path: str, obj_kind: str, style: dict, out: Path,
               width: int, rng) -> tuple[float, int, str | None]:
    """Render a single image. Returns (seconds, bytes, error-or-None)."""
    sel = "obj"
    cmd.delete("all")

    # `assembly` must be set BEFORE load: it tells the mmCIF reader whether to
    # apply the symmetry operators. "" is the asymmetric unit as deposited.
    cmd.set("assembly", "1" if obj_kind == "asm1" else "")
    cmd.load(path, sel)
    cmd.remove("solvent")
    cmd.remove("hydrogens")
    cmd.remove("not polymer")

    if cmd.count_atoms(sel) == 0:
        return 0.0, 0, "no polymer atoms after cleanup"

    cmd.hide("everything")
    if style["representation"] == "cartoon":
        cmd.set("cartoon_trace_atoms", 0)
        cmd.show_as("cartoon", sel)
        cmd.cartoon("automatic", sel)
    else:
        cmd.show_as("ribbon", sel)

    _apply_color(cmd, sel, style["color_scheme"], rng)
    cmd.bg_color(style["background"])
    cmd.set("ray_opaque_background", 1)
    if int(style["ambient_occlusion"]):
        cmd.set("ambient_occlusion_mode", 1)
    else:
        cmd.set("ambient_occlusion_mode", 0)
    cmd.set("antialias", int(style["antialias"]))
    cmd.set("ray_trace_mode", 0)

    _set_orientation(cmd, sel, (style["quat_w"], style["quat_x"],
                                style["quat_y"], style["quat_z"]))
    cmd.zoom(sel, float(style["zoom_buffer"]))

    if out.exists():
        out.unlink()
    t0 = time.time()
    # Raytrace synchronously, THEN write. cmd.png() alone hands the buffer off
    # and returns, so the next delete('all') can land mid-write and emit a
    # blank-but-valid PNG (~5% of renders, intermittent on identical input).
    cmd.ray(width, width)
    cmd.png(str(out), dpi=72)
    if not _wait_for_stable_file(out):
        return time.time() - t0, 0, "timeout waiting for png write"
    elapsed = time.time() - t0

    if not out.exists() or out.stat().st_size == 0:
        return elapsed, 0, "no output written"
    ink = _ink_fraction(out)
    if ink < MIN_INK_FRACTION:
        return elapsed, out.stat().st_size, f"blank image (ink={ink:.4f})"
    return elapsed, out.stat().st_size, None


def render_shard(shard: int, n_shards: int, views: int, width: int,
                 seed: int, limit: int | None = None) -> None:
    import pandas as pd

    idx_files = sorted(INDEX_DIR.glob("index_*.parquet"))
    if not idx_files:
        sys.exit("no index found - run `index` first")
    idx = pd.concat([pd.read_parquet(f) for f in idx_files], ignore_index=True)

    # One row per structural object, deterministically ordered so a shard's
    # membership does not depend on filesystem or concat order.
    objects = []
    for r in idx.itertuples():
        if r.deferred or r.error:
            continue
        if r.keep_au:
            objects.append((f"{r.pdb_id}_au", r.path, "au"))
        if r.keep_asm1:
            objects.append((f"{r.pdb_id}_asm1", r.path, "asm1"))
    objects.sort()
    mine = [o for i, o in enumerate(objects) if i % n_shards == shard]
    if limit:
        mine = mine[:limit]

    print(f"[render {shard}] {len(mine):,} objects x {views} views = "
          f"{len(mine)*views:,} images", flush=True)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"s{shard}.png"

    cmd = _init_pymol()
    import random
    rng = random.Random(seed)

    tar_path = RENDER_DIR / f"shard_{shard:03d}.tar"
    rows = []
    t0 = time.time()
    n_ok = n_retried = 0

    with tarfile.open(tar_path, "w") as tar:
        for j, (object_id, path, kind) in enumerate(mine, 1):
            for view in range(views):
                style = sample_view(object_id, view, seed)
                secs = nbytes = 0
                err = None
                attempts = 0
                # A failure is usually the transient write race rather than a
                # bad structure, so reset the session and retry once.
                for attempt in range(2):
                    attempts = attempt + 1
                    try:
                        secs, nbytes, err = render_one(
                            cmd, path, kind, style, tmp, width, rng)
                    except Exception as exc:
                        secs, nbytes, err = 0.0, 0, f"{type(exc).__name__}: {exc}"[:200]
                    if err is None:
                        break
                    try:
                        cmd.reinitialize()
                        cmd.set("max_threads", 1)
                        cmd.set("fetch_path", str(CCD_CACHE))
                    except Exception:
                        pass
                if attempts > 1 and err is None:
                    n_retried += 1

                name = f"{object_id}_v{view}.png"
                if err is None:
                    data = tmp.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0  # reproducible tars
                    tar.addfile(info, io.BytesIO(data))
                    n_ok += 1

                rows.append({
                    "render_id": f"{object_id}_v{view}",
                    "pdb_id": object_id.split("_")[0],
                    "object_id": object_id, "object_kind": kind,
                    "tar": tar_path.name, "member": name if err is None else "",
                    "seconds": round(secs, 4), "bytes": nbytes,
                    "ok": err is None, "error": err or "", "attempts": attempts,
                    "width": width, "height": width, "seed": seed,
                    "engine": "pymol-open-source-3.2.0a0",
                    **style,
                })
            if j % 200 == 0:
                done = len(rows)
                print(f"[render {shard}] {j}/{len(mine)} objects, {done:,} imgs, "
                      f"{done/(time.time()-t0):.2f} img/s", flush=True)

    pd.DataFrame(rows).to_parquet(
        RENDER_DIR / f"shard_{shard:03d}.parquet", index=False)
    if tmp.exists():
        tmp.unlink()
    total = time.time() - t0
    print(f"[render {shard}] done {n_ok:,}/{len(rows):,} ok in {total:.0f}s "
          f"({len(rows)/max(total,1):.2f} img/s, {n_retried} recovered by retry)",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="parse the archive, decide objects per entry")
    pi.add_argument("--shard", type=int, required=True)
    pi.add_argument("--n-shards", type=int, required=True)

    pr = sub.add_parser("render", help="render one shard of objects")
    pr.add_argument("--shard", type=int, required=True)
    pr.add_argument("--n-shards", type=int, required=True)
    pr.add_argument("--views", type=int, default=8)
    pr.add_argument("--width", type=int, default=384)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--limit", type=int, default=None,
                    help="objects per shard, for smoke tests")

    a = ap.parse_args()
    if a.cmd == "index":
        index_shard(a.shard, a.n_shards)
    else:
        render_shard(a.shard, a.n_shards, a.views, a.width, a.seed, a.limit)


if __name__ == "__main__":
    main()

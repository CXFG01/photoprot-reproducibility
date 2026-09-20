"""Render the PDB corpus on Modal, and a 3-protein pilot for eyeballing first.

Style comes from configs/render_v1.json via photoprot.render.pdb_spec, so the
axis space is editable without touching code and every image stays a pure
function of (object_id, view, seed).

    MODAL_PROFILE=tarongsilao modal run modal_render_pdb.py::pilot
    MODAL_PROFILE=tarongsilao modal run modal_render_pdb.py::pilot --pdb-ids 1ubq,4hhb,1ema
"""
from __future__ import annotations

import modal

APP_NAME = "photoprot-render-pdb"
VOL = "/vol"
PYMOL_VERSION = "3.2.0a0"
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

# Deliberately chosen to exercise different axes:
#   1ubq  76-residue beta-grasp, small and clean - geometry axes show clearly
#   4hhb  haemoglobin tetramer + 4 hemes - by-chain colour and ligand content
#   1ema  GFP, beta-barrel around a central helix - distinctive shape
DEFAULT_PILOT = "1ubq,4hhb,1ema"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("photoprot-pdb", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libxrender1", "libsm6", "libxext6",
                 "curl", "fonts-dejavu-core")
    .uv_pip_install(
        f"pymol-open-source=={PYMOL_VERSION}",
        "numpy==2.1.3", "pillow==11.0.0", "pandas==2.2.3", "pyarrow==18.1.0",
        "gemmi==0.6.7",
    )
    .add_local_python_source("photoprot")
    .add_local_dir("configs", remote_path="/root/configs")
)

CCD = f"{VOL}/ccd_cache"


def _wrap2(text: str, width: int) -> tuple[str, str]:
    """Greedy wrap into exactly two lines; the tail is truncated with an ellipsis."""
    line1: list[str] = []
    line2: list[str] = []
    for word in text.split():
        if len(" ".join(line1 + [word])) <= width:
            line1.append(word)
        else:
            line2.append(word)
    second = " ".join(line2)
    if len(second) > width:
        second = second[:width - 1] + "…"
    return " ".join(line1), second


@app.function(image=image, volumes={VOL: volume}, cpu=4.0, timeout=60 * 40)
def stage_ccd():
    """Split the chemical component dictionary into PyMOL's cache format once.

    Ligand content axes would otherwise make PyMOL fetch each unknown component
    from files.rcsb.org mid-render: a network call per render at corpus scale.
    """
    import gzip
    import os
    import time
    import urllib.request

    os.makedirs(CCD, exist_ok=True)
    have = len(os.listdir(CCD))
    if have > 40000:
        print(f"[ccd] already staged ({have:,} components)")
        return have

    t = time.time()
    gz = "/tmp/components.cif.gz"
    urllib.request.urlretrieve(CCD_URL, gz)
    n = 0
    name = None
    buf: list[str] = []

    def flush(name, buf):
        if not name:
            return 0
        with open(os.path.join(CCD, f"{name}.cif"), "w") as fh:
            fh.writelines(buf)
        return 1

    with gzip.open(gz, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("data_"):
                n += flush(name, buf)
                name = line.strip()[5:]
                buf = [line]
            else:
                buf.append(line)
    n += flush(name, buf)
    volume.commit()
    print(f"[ccd] staged {n:,} components in {time.time()-t:.0f}s")
    return n


@app.function(image=image, volumes={VOL: volume}, cpu=8.0, timeout=60 * 60)
def render_pilot(pdb_ids: list[str], views: int = 16, width: int = 384):
    """Render every view of a few entries, plus a labelled contact sheet each."""
    import json
    import os
    import sys
    import urllib.request

    sys.path.insert(0, "/root")
    from photoprot.render import pdb_spec, pdb_pymol
    from PIL import Image, ImageDraw, ImageFont

    cfg = pdb_spec.load_config("/root/configs/render_v1.json")
    out_root = f"{VOL}/pilot"
    os.makedirs(out_root, exist_ok=True)

    cmd = pdb_pymol.init_pymol(CCD if os.path.isdir(CCD) else None)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = bold = ImageFont.load_default()

    rows = []
    for pid in pdb_ids:
        pid = pid.strip().lower()
        cif = f"/tmp/{pid}.cif.gz"
        if not os.path.exists(cif):
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pid}.cif.gz", cif)

        object_id = f"{pid}_au"
        d = os.path.join(out_root, pid)
        os.makedirs(d, exist_ok=True)
        tiles = []
        for view in range(views):
            style = pdb_spec.sample_view(object_id, view, 0, cfg)
            png = os.path.join(d, f"{object_id}_v{view:02d}.png")
            secs, nbytes, err, att = pdb_pymol.render_with_retry(
                cmd, cif, "au", style, png, width, CCD)
            label = pdb_spec.describe(style)
            print(f"[{pid}] v{view:02d} {secs:6.2f}s {'OK ' if not err else 'ERR'} "
                  f"{label}" + (f"  <{err}>" if err else ""), flush=True)
            rows.append({"object_id": object_id, "view": view, "seconds": secs,
                         "bytes": nbytes, "ok": err is None, "error": err or "",
                         "attempts": att, "label": label, **style})
            tiles.append((png if not err else None, f"v{view}  {label}"))

        # ---- contact sheet -----------------------------------------------
        cols = 4
        rowsn = (len(tiles) + cols - 1) // cols
        lab_h, pad, title_h = 46, 6, 40
        W = cols * (width + pad) + pad
        H = title_h + rowsn * (width + lab_h + pad) + pad
        sheet = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(sheet)
        dr.text((pad, 10), f"{pid.upper()}  -  {views} views  -  {cfg['version']}",
                fill="black", font=bold)
        for i, (png, lab) in enumerate(tiles):
            cx = pad + (i % cols) * (width + pad)
            cy = title_h + (i // cols) * (width + lab_h + pad)
            if png and os.path.exists(png):
                with Image.open(png) as im:
                    sheet.paste(im.convert("RGB"), (cx, cy))
            else:
                dr.rectangle([cx, cy, cx + width, cy + width], outline="red")
                dr.text((cx + 8, cy + 8), "FAILED", fill="red", font=bold)
            # wrap the label over two lines so nothing is cut off
            line1, line2 = _wrap2(lab, 44)
            dr.text((cx + 2, cy + width + 3), line1, fill="black", font=font)
            dr.text((cx + 2, cy + width + 20), line2, fill="#555", font=font)
        sp = os.path.join(out_root, f"contact_{pid}.png")
        sheet.save(sp)
        print(f"[{pid}] contact sheet -> {sp}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_parquet(f"{out_root}/pilot_manifest.parquet", index=False)
    volume.commit()

    ok = sum(1 for r in rows if r["ok"])
    secs = [r["seconds"] for r in rows if r["ok"]]
    print(f"\n{ok}/{len(rows)} rendered ok; "
          f"mean {sum(secs)/max(len(secs),1):.2f}s, max {max(secs, default=0):.2f}s")
    return {"ok": ok, "total": len(rows),
            "mean_seconds": sum(secs) / max(len(secs), 1)}


@app.local_entrypoint()
def pilot(pdb_ids: str = DEFAULT_PILOT, views: int = 16, width: int = 384):
    print(stage_ccd.remote())
    ids = [p for p in pdb_ids.split(",") if p.strip()]
    print(render_pilot.remote(ids, views, width))


# ===========================================================================
# Full corpus
# ===========================================================================
# 24 source tars x SUB_PER_TAR containers. Sharding this way keeps each
# container's coordinate download to ONE tar (~800 MB) rather than making it
# pull the whole 19.3 GB corpus, while still fanning out past the 24 tars.
SUB_PER_TAR = 4
N_TARS = 24
N_RENDER_SHARDS = N_TARS * SUB_PER_TAR          # 96, under the 100-container cap
RENDER_CPU = 32.0                                # 64 is the hard per-container max
PROCS = 32                                       # one PyMOL per reserved core


def _index_one_tar(tar_idx: int) -> list[dict]:
    """gemmi pass over one tar: what objects does each entry yield?"""
    import os
    import tarfile

    import gemmi

    local = f"/tmp/cif{tar_idx}"
    os.makedirs(local, exist_ok=True)
    with tarfile.open(f"{VOL}/mmcif/shard_{tar_idx:03d}.tar") as tar:
        tar.extractall(local)

    def stats(model):
        atoms = res = chains = 0
        for chain in model:
            ca = sum(1 for r in chain if r.find_atom("CA", "*") is not None)
            if ca == 0:
                continue
            chains += 1
            res += ca
            atoms += sum(len(r) for r in chain)
        return atoms, res, chains

    rows = []
    for root, _, files in os.walk(local):
        for fn in files:
            if not fn.endswith(".cif.gz"):
                continue
            path = os.path.join(root, fn)
            pdb_id = fn.split(".")[0].lower()
            rec = {"pdb_id": pdb_id, "tar": tar_idx,
                   "member": os.path.relpath(path, local),
                   "au_atoms": 0, "au_res": 0, "asm1_atoms": 0, "asm1_res": 0,
                   "asm1_differs": False, "keep_au": False, "keep_asm1": False,
                   "deferred": False, "error": ""}
            try:
                st = gemmi.read_structure(path)
                st.setup_entities()
                st.remove_ligands_and_waters()
                if len(st) == 0:
                    raise ValueError("no model")
                rec["au_atoms"], rec["au_res"], _ = stats(st[0])
                asm = next((a for a in st.assemblies if a.name == "1"), None)
                if asm is not None:
                    built = gemmi.make_assembly(
                        asm, st[0], gemmi.HowToNameCopiedChain.AddNumber)
                    a_atoms, a_res, _ = stats(built)
                    rec["asm1_atoms"], rec["asm1_res"] = a_atoms, a_res
                    rec["asm1_differs"] = a_atoms != rec["au_atoms"] and a_atoms > 0
                rec["keep_au"] = rec["au_res"] >= 30
                rec["keep_asm1"] = bool(rec["asm1_differs"] and rec["asm1_res"] >= 30)
                if max(rec["au_atoms"], rec["asm1_atoms"]) > 300_000:
                    rec["deferred"] = True
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(rec)
    return rows


@app.function(image=image, volumes={VOL: volume}, cpu=4.0, timeout=60 * 60,
              max_containers=N_TARS, retries=2)
def index_tar(tar_idx: int):
    import pandas as pd
    rows = _index_one_tar(tar_idx)
    df = pd.DataFrame(rows)
    import os
    os.makedirs(f"{VOL}/index", exist_ok=True)
    df.to_parquet(f"{VOL}/index/index_{tar_idx:03d}.parquet", index=False)
    volume.commit()
    print(f"[index {tar_idx}] {len(df):,} entries | keep_au {int(df.keep_au.sum()):,} "
          f"| keep_asm1 {int(df.keep_asm1.sum()):,} | deferred {int(df.deferred.sum()):,} "
          f"| errors {int((df.error != '').sum()):,}", flush=True)
    return {"tar": tar_idx, "n": len(df), "au": int(df.keep_au.sum()),
            "asm1": int(df.keep_asm1.sum()), "deferred": int(df.deferred.sum()),
            "errors": int((df.error != "").sum())}


_W = {}


def _worker_init(ccd, cfg):
    import sys
    sys.path.insert(0, "/root")
    from photoprot.render import pdb_pymol
    _W["cmd"] = pdb_pymol.init_pymol(ccd)
    _W["cfg"] = cfg
    _W["ccd"] = ccd


def _worker_render(job):
    """One object, all its views. Returns manifest rows and PNG bytes."""
    import os
    import sys
    sys.path.insert(0, "/root")
    from photoprot.render import pdb_spec, pdb_pymol

    object_id, path, kind, views, width, seed = job
    cmd, cfg, ccd = _W["cmd"], _W["cfg"], _W["ccd"]
    out = f"/tmp/w{os.getpid()}.png"
    rows, blobs = [], []
    for view in range(views):
        style = pdb_spec.sample_view(object_id, view, seed, cfg)
        secs, nbytes, err, att = pdb_pymol.render_with_retry(
            cmd, path, kind, style, out, width, ccd)
        name = f"{object_id}_v{view:02d}.png"
        if err is None:
            with open(out, "rb") as fh:
                blobs.append((name, fh.read()))
        rows.append({"render_id": f"{object_id}_v{view:02d}",
                     "pdb_id": object_id.split("_")[0], "object_id": object_id,
                     "object_kind": kind, "member": name if err is None else "",
                     "seconds": round(secs, 4), "bytes": nbytes,
                     "ok": err is None, "error": err or "", "attempts": att,
                     "width": width, "seed": seed,
                     "engine": f"pymol-open-source-{PYMOL_VERSION}", **style})
    return rows, blobs


@app.function(image=image, volumes={VOL: volume}, cpu=RENDER_CPU, memory=65536,
              timeout=60 * 60 * 5, max_containers=N_RENDER_SHARDS, retries=1)
def render_shard(shard: int, views: int = 16, width: int = 384, seed: int = 0):
    import io
    import multiprocessing as mp
    import os
    import sys
    import tarfile
    import time

    import pandas as pd

    sys.path.insert(0, "/root")
    from photoprot.render import pdb_spec

    cfg = pdb_spec.load_config("/root/configs/render_v1.json")
    tar_idx, sub = divmod(shard, SUB_PER_TAR)

    idx = pd.read_parquet(f"{VOL}/index/index_{tar_idx:03d}.parquet")
    local = f"/tmp/cif{tar_idx}"
    os.makedirs(local, exist_ok=True)
    with tarfile.open(f"{VOL}/mmcif/shard_{tar_idx:03d}.tar") as tar:
        tar.extractall(local)

    # Filters are applied PER OBJECT from the raw atom counts, deliberately
    # ignoring the index's `deferred` column: that column encoded a per-ENTRY
    # rule which discarded a renderable AU whenever its assembly was a capsid.
    filt = cfg["filters"]
    min_res = filt["min_residues"]
    cap_au = filt["max_atoms_au"]
    cap_asm = filt["max_atoms_assembly"]

    objects = []
    for r in idx.itertuples():
        if r.error:
            continue
        p = os.path.join(local, r.member)
        if r.au_res >= min_res and 0 < r.au_atoms <= cap_au:
            objects.append((f"{r.pdb_id}_au", p, "au", int(r.au_atoms)))
        if (r.asm1_differs and r.asm1_res >= min_res
                and 0 < r.asm1_atoms <= cap_asm):
            objects.append((f"{r.pdb_id}_asm1", p, "asm1", int(r.asm1_atoms)))

    # Deal round-robin off a size-sorted list so no container gets a pile of
    # capsids while another gets a pile of peptides.
    objects.sort(key=lambda o: o[3])
    mine = [o for i, o in enumerate(objects) if i % SUB_PER_TAR == sub]

    # Then render LARGEST FIRST within the shard. Longest-processing-time-first
    # is the standard makespan heuristic: the first run sorted ascending, so
    # every container finished on its biggest structures with ~28 of 32 workers
    # idle but still billed. Measured utilisation was 28%.
    mine.sort(key=lambda o: -o[3])
    # Resume: skip objects already committed by an earlier attempt at this
    # shard. Cheap insurance - the first run lost 91 shards to a spend cap and
    # had no way to re-run only the missing work.
    # An object counts as done only if ALL its views are present. Keying on
    # object_id alone would permanently skip objects truncated mid-object by an
    # interrupted run - they would stay partial forever, and the manifest would
    # describe that accurately enough that nothing ever flagged it.
    import glob as _glob
    seen = []
    for pq in _glob.glob(f"{VOL}/renders/shard_{shard:03d}_p*.parquet"):
        try:
            seen.append(pd.read_parquet(pq, columns=["object_id", "view", "ok"]))
        except Exception as exc:
            print(f"[render {shard}] ignoring unreadable {os.path.basename(pq)}: "
                  f"{type(exc).__name__}", flush=True)
    done_ids: set[str] = set()
    n_partial = 0
    if seen:
        allrows = pd.concat(seen, ignore_index=True)
        # Done means every view was ATTEMPTED, not every view succeeded. Some
        # views fail deterministically - a blank render (ink < 0.005) is a pure
        # function of the same seed, so it fails identically every time.
        # Requiring 16 *successful* views would re-render those objects on every
        # resume, forever, and never converge.
        attempted = allrows.groupby("object_id").view.nunique()
        done_ids = set(attempted[attempted >= views].index)
        n_partial = int((attempted < views).sum())
        if n_partial:
            print(f"[render {shard}] {n_partial} objects with unattempted views "
                  f"will be re-rendered in full", flush=True)

    jobs = [(oid, p, k, views, width, seed) for oid, p, k, _ in mine
            if oid not in done_ids]

    print(f"[render {shard}] tar {tar_idx} sub {sub}: {len(mine):,} objects "
          f"({len(done_ids):,} already done, {len(jobs):,} to render) "
          f"x {views} views on {PROCS} procs", flush=True)
    if not jobs:
        print(f"[render {shard}] nothing to do", flush=True)
        return {"shard": shard, "objects": 0, "images": 0, "failed": 0,
                "seconds": 0.0}

    os.makedirs(f"{VOL}/renders", exist_ok=True)

    # Write in PARTS and commit each one. A modal.Volume only persists on
    # commit(), so the first run - which committed once at the end - had a
    # ~45 minute window in which nothing was recoverable. When the workspace
    # hit its spend limit, 91 of 96 shards lost everything they had rendered.
    # A tar cannot be appended to once closed, hence parts rather than one file.
    PART_OBJECTS = 100
    rows_all = []
    t0 = time.time()
    n_ok = 0
    rows_part = []
    out_tar = None
    # Continue past any parts a previous attempt committed, rather than
    # overwriting them and losing the work resume just found.
    existing = _glob.glob(f"{VOL}/renders/shard_{shard:03d}_p*.tar")
    part = (max((int(os.path.basename(f).split("_p")[1][:2]) for f in existing),
                default=-1) + 1)

    def open_part(p):
        return tarfile.open(f"{VOL}/renders/shard_{shard:03d}_p{p:02d}.tar", "w")

    def close_part(p, tar, rows):
        """Close a part so that its tar and its manifest always agree.

        The first interrupted run left 5 tars with no parquet, because this
        wrote one only when the part had rows - 1,984 images ended up with no
        manifest entry. A part is now either a tar WITH its parquet, or nothing.
        """
        tar.close()
        path = f"{VOL}/renders/shard_{shard:03d}_p{p:02d}"
        if rows:
            pd.DataFrame(rows).to_parquet(path + ".parquet", index=False)
        else:
            try:
                os.remove(path + ".tar")      # empty part: leave no orphan tar
            except OSError:
                pass
        volume.commit()

    ctx = mp.get_context("fork")
    out_tar = open_part(part)
    with ctx.Pool(PROCS, initializer=_worker_init,
                  initargs=(CCD if os.path.isdir(CCD) else None, cfg)) as pool:
        for j, (rows, blobs) in enumerate(
                pool.imap_unordered(_worker_render, jobs, chunksize=2), 1):
            rows_all.extend(rows)
            rows_part.extend(rows)
            for name, data in blobs:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0              # reproducible tars
                out_tar.addfile(info, io.BytesIO(data))
                n_ok += 1
            if j % PART_OBJECTS == 0:
                close_part(part, out_tar, rows_part)
                el = time.time() - t0
                print(f"[render {shard}] {j}/{len(jobs)} objects, {n_ok:,} imgs, "
                      f"{n_ok/max(el,1):.1f} img/s, committed part {part}", flush=True)
                part += 1
                rows_part = []
                out_tar = open_part(part)
    close_part(part, out_tar, rows_part)
    el = time.time() - t0
    print(f"[render {shard}] DONE {n_ok:,}/{len(rows_all):,} in {el:.0f}s "
          f"({n_ok/max(el,1):.1f} img/s)", flush=True)
    return {"shard": shard, "objects": len(jobs), "images": n_ok,
            "failed": len(rows_all) - n_ok, "seconds": el}


@app.local_entrypoint()
def index():
    res = list(index_tar.map(range(N_TARS)))
    tot = {k: sum(r[k] for r in res) for k in ("n", "au", "asm1", "deferred", "errors")}
    print(f"\nINDEX TOTAL entries {tot['n']:,} | AU objects {tot['au']:,} | "
          f"asm1 objects {tot['asm1']:,} | deferred {tot['deferred']:,} | "
          f"errors {tot['errors']:,}")
    print(f"total objects to render: {tot['au'] + tot['asm1']:,}")


@app.local_entrypoint()
def full(views: int = 16, width: int = 384, seed: int = 0):
    import time
    t0 = time.time()
    res = list(render_shard.map(range(N_RENDER_SHARDS),
                                kwargs={"views": views, "width": width, "seed": seed}))
    imgs = sum(r["images"] for r in res)
    bad = sum(r["failed"] for r in res)
    print(f"\nRENDER TOTAL {imgs:,} images, {bad:,} failed, "
          f"wall {time.time()-t0:.0f}s across {len(res)} shards")

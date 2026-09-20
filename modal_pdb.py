"""Acquire the PDB training corpus on Modal: a 40%-sequence-identity subset.

WHY 40%

The full archive is 260,089 entries, and rendering all of it at 8 views costs
~$127 of compute for a corpus that is mostly redundant. RCSB publishes sequence
clusters at 30/40/50/70/90/95/100% identity over polymer ENTITIES; covering every
40% cluster at least once needs only ~36k ENTRIES (13.8% of the archive), which
lands almost exactly on CATH S40's 34,653 domains. That keeps the project in the
same redundancy regime the Phase 0 baseline and all four splits were built in,
now at whole-entry granularity instead of pre-chopped domains.

This subset is the TRAINING corpus. It is deliberately NOT the retrieval index:
the deployed index must answer for any entry someone screenshots, including
recent structures with no close sequence relative, and covering all 260,089
entries at one view each is cheap (~$16). Filtering the training set and keeping
the index complete are not in tension once they are two different corpora.

REPRESENTATIVE CHOICE

A cluster's representative is its best-resolved member, not an arbitrary one.
Taking each cluster's first listed member would bias hard toward low-numbered
(older) entries, because the cluster files are alphabetically ordered. Entries
with no diffraction resolution (NMR, and anything reported as the -1.00
sentinel) sort last rather than being dropped, so an NMR-only cluster still gets
a representative.

STORAGE

Coordinates land in the volume as a few dozen tars, not ~36k loose files.
Modal network volumes are poor at many small files - the same lesson
modal_render.py records for its output side, applied here to the input side.

    MODAL_PROFILE=tarongsilao modal run modal_pdb.py::select
    MODAL_PROFILE=tarongsilao modal run modal_pdb.py::fetch
"""
from __future__ import annotations

import modal

APP_NAME = "photoprot-pdb"
VOL = "/vol"
LOCAL = "/scratch"

IDENTITY = 40          # sequence-identity threshold for the training subset
N_FETCH_SHARDS = 24    # parallel rsync streams; kind to RCSB, ~5 MB/s each

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("photoprot-pdb", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync", "curl", "tar")
    .uv_pip_install("pandas==2.2.3", "pyarrow==18.1.0", "numpy==2.1.3")
)

CLUSTER_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{p}.txt")
# Small derived-data index mapping every entry to its diffraction resolution.
# Cheaper and kinder than 260 paged Data-API calls just to rank representatives.
RESOLU_URL = "https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx"
HOLDINGS_URL = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
RSYNC_MODULE = "rsync.rcsb.org::ftp_data/structures/divided/mmCIF"


def _sh(cmd: str, check: bool = True, allow: tuple[int, ...] = (0,)) -> str:
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode not in allow:
        raise RuntimeError(f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


@app.function(image=image, volumes={VOL: volume}, cpu=4.0, timeout=60 * 45)
def select(identity: int = IDENTITY):
    """Build the subset: every `identity`% cluster covered by its best member."""
    import json
    import os
    import urllib.request

    import pandas as pd

    aux = f"{VOL}/aux"
    os.makedirs(aux, exist_ok=True)

    def grab(url: str, dest: str) -> str:
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            print(f"  fetching {url}", flush=True)
            urllib.request.urlretrieve(url, dest)
        print(f"  {os.path.basename(dest)}: {os.path.getsize(dest)/1e6:.1f} MB", flush=True)
        return dest

    cl_path = grab(CLUSTER_URL.format(p=identity), f"{aux}/clusters-{identity}.txt")
    res_path = grab(RESOLU_URL, f"{aux}/resolu.idx")
    hold_path = grab(HOLDINGS_URL, f"{aux}/entry_ids.json")

    held = {e.lower() for e in json.load(open(hold_path))}
    print(f"  current holdings: {len(held):,} entries", flush=True)

    # --- resolution table -------------------------------------------------
    # resolu.idx is a fixed-ish text table: "100D    ;       1.90", with -1.00
    # meaning "not a diffraction experiment" rather than "very bad".
    resolution: dict[str, float] = {}
    with open(res_path, errors="replace") as fh:
        for line in fh:
            if ";" not in line:
                continue
            left, _, right = line.partition(";")
            pid = left.strip().lower()
            if len(pid) != 4:
                continue
            try:
                v = float(right.strip())
            except ValueError:
                continue
            if v > 0:
                resolution[pid] = v
    print(f"  resolutions parsed: {len(resolution):,}", flush=True)

    # --- clusters ---------------------------------------------------------
    ent2cl: dict[str, set[int]] = {}
    n_entities = 0
    n_clusters = 0
    with open(cl_path) as fh:
        for ci, line in enumerate(fh):
            parts = line.split()
            if not parts:
                continue
            n_clusters += 1
            for m in parts:
                n_entities += 1
                e = m.split("_")[0].lower()
                ent2cl.setdefault(e, set()).add(ci)
    print(f"  {n_entities:,} entities -> {n_clusters:,} clusters at {identity}% "
          f"over {len(ent2cl):,} entries", flush=True)

    # Only consider entries we can actually download.
    candidates = [e for e in ent2cl if e in held]
    print(f"  candidates present in current holdings: {len(candidates):,}", flush=True)

    # --- greedy cover, best-resolved first --------------------------------
    NO_RES = 10_000.0  # sorts last, but still eligible
    candidates.sort(key=lambda e: (resolution.get(e, NO_RES), e))

    covered: set[int] = set()
    kept: list[str] = []
    for e in candidates:
        new = ent2cl[e] - covered
        if new:
            kept.append(e)
            covered |= new

    print(f"  kept {len(kept):,} entries covering {len(covered):,}/"
          f"{n_clusters:,} clusters "
          f"({'FULL' if len(covered) == n_clusters else 'PARTIAL'})", flush=True)

    with_res = sum(1 for e in kept if e in resolution)
    rs = sorted(resolution[e] for e in kept if e in resolution)
    if rs:
        print(f"  resolution of kept: median {rs[len(rs)//2]:.2f} A, "
              f"p90 {rs[int(.9*len(rs))]:.2f} A, "
              f"{with_res:,} with resolution / {len(kept)-with_res:,} without",
              flush=True)

    df = pd.DataFrame({
        "pdb_id": kept,
        "resolution": [resolution.get(e) for e in kept],
        "n_clusters_covered": [len(ent2cl[e]) for e in kept],
        "shard": [i % N_FETCH_SHARDS for i in range(len(kept))],
        "rel_path": [f"{e[1:3]}/{e}.cif.gz" for e in kept],
    })
    out = f"{VOL}/subset_{identity}.parquet"
    df.to_parquet(out, index=False)
    with open(f"{VOL}/subset_{identity}.txt", "w") as fh:
        fh.write("\n".join(kept) + "\n")
    volume.commit()
    print(f"  wrote {out}", flush=True)
    return {"identity": identity, "clusters": n_clusters,
            "entries": len(kept), "covered": len(covered)}


@app.function(image=image, volumes={VOL: volume}, cpu=2.0, timeout=60 * 90,
              max_containers=N_FETCH_SHARDS, retries=2)
def fetch_shard(shard: int, identity: int = IDENTITY):
    """rsync this shard's entries from RCSB, then store them as ONE tar."""
    import os

    import pandas as pd

    df = pd.read_parquet(f"{VOL}/subset_{identity}.parquet")
    mine = df[df.shard == shard]
    local = f"{LOCAL}/mmcif"
    os.makedirs(local, exist_ok=True)

    # --files-from lets one rsync stream pull exactly this shard's entries
    # instead of walking the whole 1,282-directory archive.
    list_path = f"{LOCAL}/files_{shard}.txt"
    with open(list_path, "w") as fh:
        fh.write("\n".join(mine.rel_path.tolist()) + "\n")

    # Exit 23/24 mean "some files were not transferred", which is the NORMAL
    # case here: the rsync tree is a weekly snapshot, so entries released since
    # it are absent and get picked up over HTTPS below. Only a genuine transport
    # failure should abort the shard.
    _sh(f"rsync -rlpt --port=33444 --contimeout=30 --timeout=600 "
        f"--files-from={list_path} {RSYNC_MODULE}/ {local}/",
        allow=(0, 23, 24))

    got = []
    for rp in mine.rel_path:
        p = os.path.join(local, rp)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            got.append(rp)
    missing = sorted(set(mine.rel_path) - set(got))

    # A handful of entries released since the mirror's weekly snapshot will not
    # be on rsync yet; HTTPS has them. Same gap-fill that the archive sync needed.
    for rp in list(missing):
        pid = os.path.basename(rp).split(".")[0]
        dest = os.path.join(local, rp)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = _sh(f"curl -sfL --max-time 120 -o {dest} "
                f"https://files.rcsb.org/download/{pid}.cif.gz "
                f"&& gzip -t {dest} && echo OK", check=False)
        if "OK" in r:
            got.append(rp)
            missing.remove(rp)

    os.makedirs(f"{VOL}/mmcif", exist_ok=True)
    tar_path = f"{VOL}/mmcif/shard_{shard:03d}.tar"
    _sh(f"tar -cf {tar_path} -C {local} " + " ".join(got))
    volume.commit()

    size = os.path.getsize(tar_path)
    print(f"[fetch {shard}] {len(got):,}/{len(mine):,} entries, "
          f"{size/1e6:.0f} MB tar, {len(missing)} missing", flush=True)
    return {"shard": shard, "requested": len(mine), "got": len(got),
            "missing": missing[:20], "bytes": size}


@app.local_entrypoint()
def fetch(identity: int = IDENTITY):
    import json
    results = list(fetch_shard.map(range(N_FETCH_SHARDS),
                                   kwargs={"identity": identity}))
    req = sum(r["requested"] for r in results)
    got = sum(r["got"] for r in results)
    gb = sum(r["bytes"] for r in results) / 1e9
    miss = [m for r in results for m in r["missing"]]
    print(f"\nTOTAL requested {req:,} | downloaded {got:,} | {gb:.1f} GB in "
          f"{len(results)} tars")
    if miss:
        print(f"missing ({len(miss)} shown up to 20/shard): {miss[:20]}")


@app.local_entrypoint()
def main(identity: int = IDENTITY):
    print(json.dumps(select.remote(identity), indent=2))


import json  # noqa: E402  (used by main)

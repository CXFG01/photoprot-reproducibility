"""Download and unpack the CATH release files Phase 0 depends on.

Pinned to an explicit CATH release rather than `latest-release` so the whole
pipeline stays reproducible when CATH ships a new version.
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests

from photoprot.paths import CATH_RAW, DOMAIN_PDB, ensure_dirs
from photoprot.util import progress

CATH_RELEASE = "v4_4_0"
BASE = f"https://download.cathdb.info/cath/releases/all-releases/{CATH_RELEASE}"


@dataclass(frozen=True)
class Resource:
    key: str
    subdir: str
    filename: str
    note: str

    @property
    def url(self) -> str:
        return f"{BASE}/{self.subdir}/{self.filename}"

    @property
    def dest(self) -> Path:
        return CATH_RAW / self.filename


R = CATH_RELEASE
RESOURCES: dict[str, Resource] = {
    r.key: r
    for r in [
        Resource(
            "domain_list",
            "cath-classification-data",
            f"cath-domain-list-{R}.txt",
            "C.A.T.H numbers for every classified domain",
        ),
        Resource(
            "names",
            "cath-classification-data",
            f"cath-names-{R}.txt",
            "human-readable names for each node in the hierarchy",
        ),
        Resource(
            "superfamily_list",
            "cath-classification-data",
            f"cath-superfamily-list-{R}.txt",
            "superfamily inventory",
        ),
        Resource(
            "boundaries",
            "cath-classification-data",
            f"cath-domain-boundaries-seqreschopping-{R}.txt",
            "residue ranges; needed later to chop domains out of full PDB entries",
        ),
        Resource(
            "s40_list",
            "non-redundant-data-sets",
            f"cath-dataset-nonredundant-S40-{R}.list",
            "S40 non-redundant domain ids (primary training set)",
        ),
        Resource(
            "s20_list",
            "non-redundant-data-sets",
            f"cath-dataset-nonredundant-S20-{R}.list",
            "S20 domain ids (fast dev subset)",
        ),
        Resource(
            "s40_pdb",
            "non-redundant-data-sets",
            f"cath-dataset-nonredundant-S40-{R}.pdb.tgz",
            "pre-chopped PDB file per S40 domain (~818 MB)",
        ),
    ]
}


def remote_size(url: str, timeout: int = 30) -> int | None:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except requests.RequestException:
        return None


def download(res: Resource, force: bool = False, chunk: int = 1 << 20) -> Path:
    """Stream a resource to disk, resuming a partial download if one exists."""
    dest = res.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(res.url)

    if dest.exists() and not force:
        have = dest.stat().st_size
        if total is not None and have == total:
            print(f"[fetch] {res.filename}: already complete ({have:,} B)", flush=True)
            return dest
        if total is None:
            print(f"[fetch] {res.filename}: present, size unverifiable - keeping", flush=True)
            return dest

    resume_from = dest.stat().st_size if (dest.exists() and not force) else 0
    if resume_from and total and resume_from > total:
        resume_from = 0  # local file is bogus, start over

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"
    print(f"[fetch] {res.filename}: {res.note}", flush=True)

    with requests.get(res.url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        if resume_from and r.status_code != 206:
            resume_from, mode = 0, "wb"  # server ignored Range
        bar = progress(
            total=total,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=res.filename[:46],
        )
        with open(dest, mode) as fh, bar:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)
                    bar.update(len(block))

    got = dest.stat().st_size
    if total is not None and got != total:
        raise IOError(f"{res.filename}: expected {total:,} B, got {got:,} B")
    print(f"[fetch] {res.filename}: done ({got:,} B)", flush=True)
    return dest


def extract_domain_pdbs(tgz: Path, out_dir: Path = DOMAIN_PDB, force: bool = False) -> int:
    """Unpack the S40 domain tarball into a flat directory of per-domain PDB files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in out_dir.iterdir())
    if existing and not force:
        print(f"[extract] {out_dir} already holds {existing:,} files - skipping", flush=True)
        return existing

    print(f"[extract] unpacking {tgz.name} -> {out_dir}", flush=True)
    n = 0
    with tarfile.open(tgz, "r:gz") as tf:
        bar = progress(unit=" files", desc="extract")
        with bar:
            for member in tf:
                if not member.isfile():
                    continue
                # flatten: the tarball nests files under dompdb/
                name = Path(member.name).name
                src = tf.extractfile(member)
                if src is None:
                    continue
                (out_dir / name).write_bytes(src.read())
                n += 1
                bar.update(1)
    print(f"[extract] wrote {n:,} domain PDB files", flush=True)
    return n


def fetch_all(keys: list[str] | None = None, force: bool = False) -> None:
    ensure_dirs()
    for key in keys or list(RESOURCES):
        download(RESOURCES[key], force=force)

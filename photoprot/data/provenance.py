"""Freeze exactly which bytes the dataset was built from.

Counts derived from CATH depend on which files you pulled and how you parsed
them, and the published summary statistics do not always agree with the shipped
files. Recording URLs, hashes, parsing rules and the reconciliation up front
means a number can be re-derived later instead of re-argued.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from photoprot.data.fetch_cath import BASE, CATH_RELEASE, RESOURCES
from photoprot.paths import DOMAIN_PDB, PROCESSED

PROVENANCE = PROCESSED / "provenance.json"

# How each raw file is interpreted. Recorded so a future reader does not have to
# reverse-engineer the parser to know what a column meant.
PARSING_RULES = {
    "cath_domain_list": (
        "CATH List File (CLF) format 2.0. Whitespace-separated columns: "
        "domain_id, C, A, T, H, S35, S60, S95, S100, S100_count, length, resolution."
    ),
    "resolution_sentinel": (
        "resolution == 999.0 marks NMR or unknown; mapped to NA so it cannot act "
        "as a numeric feature."
    ),
    "altloc": (
        "Alternate locations are resolved by highest occupancy per (residue, atom), "
        "NOT by altloc letter. A blank-or-'A' rule discards every atom of domains "
        "whose sole conformer carries another letter (e.g. 2qe7G01, labelled 'C')."
    ),
    "backbone": (
        "A residue is kept only if all of N, CA, C, O are present. Dropped residues "
        "leave a chain break, which DSSP handles as absence of hydrogen bonding."
    ),
    "superfamily_code": "cath_homsf is the dotted 'C.A.T.H' string, e.g. '3.40.50.300'.",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _count_superfamily_list(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            n += 1
    return n


def _distinct_h_in_domain_list(path: Path) -> tuple[int, int]:
    """Return (distinct superfamilies, number of domain rows)."""
    seen: set[str] = set()
    rows = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            seen.add(".".join(parts[1:5]))
            rows += 1
    return len(seen), rows


def build(include_slow_hashes: bool = True) -> dict:
    """Assemble the provenance record. Hashing the 818 MB tarball dominates."""
    files = {}
    for key, res in RESOURCES.items():
        if not res.dest.exists():
            files[key] = {"url": res.url, "status": "missing"}
            continue
        entry = {
            "url": res.url,
            "filename": res.filename,
            "bytes": res.dest.stat().st_size,
            "note": res.note,
        }
        if include_slow_hashes:
            entry["sha256"] = sha256(res.dest)
        files[key] = entry

    import pandas as pd

    from photoprot.paths import MANIFEST

    counts: dict = {}
    sf_path = RESOURCES["superfamily_list"].dest
    dl_path = RESOURCES["domain_list"].dest
    if sf_path.exists():
        counts["superfamilies_in_superfamily_list_file"] = _count_superfamily_list(sf_path)
    if dl_path.exists():
        n_h, n_rows = _distinct_h_in_domain_list(dl_path)
        counts["superfamilies_in_domain_list_file"] = n_h
        counts["domains_in_domain_list_file"] = n_rows
    if MANIFEST.exists():
        m = pd.read_parquet(MANIFEST)
        counts["domains_in_s40_manifest"] = int(len(m))
        counts["superfamilies_in_s40_manifest"] = int(m["cath_homsf"].nunique())
        counts["topologies_in_s40_manifest"] = int(m["cath_topo"].nunique())
        counts["architectures_in_s40_manifest"] = int(m["cath_arch"].nunique())
        counts["classes_in_s40_manifest"] = int(m["cath_class"].nunique())
    if DOMAIN_PDB.exists():
        counts["domain_pdb_files_on_disk"] = sum(1 for _ in DOMAIN_PDB.iterdir())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cath_release_requested": CATH_RELEASE,
        "base_url": BASE,
        "files": files,
        "counts": counts,
        "parsing_rules": PARSING_RULES,
        "reconciliation": {
            "question": (
                "cathdb.info advertises 6,573 superfamilies; the S40 manifest here "
                "yields 6,576. A subset cannot exceed its parent, so one of the two "
                "numbers refers to something other than the downloaded files."
            ),
            "finding": (
                "The shipped files give 6,631 superfamilies in cath-superfamily-list "
                "and 6,630 distinct H codes across cath-domain-list (the extra one, "
                "1.20.1690.30, is listed but has no classified domain). Every one of "
                "the 6,576 superfamilies in the S40 manifest is present in BOTH, with "
                "zero unmatched. The nesting 6,576 < 6,630 < 6,631 therefore holds and "
                "the parser is not inventing codes."
            ),
            "explanation": (
                "The CATH release-statistics table lists 6,573 superfamilies under "
                "'CATH-Plus 4.4.0' and 6,630 under 'CATH (daily snapshot)'. The files "
                "served from the v4_4_0 download directory match the daily-snapshot "
                "figure, not the frozen release figure. Counts derived here should be "
                "quoted against the recorded sha256 hashes rather than against the "
                "website summary."
            ),
            "verified_on": "2026-09-19",
        },
    }


def save(record: dict, path: Path = PROVENANCE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

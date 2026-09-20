"""Central path configuration.

Everything under DATA/ is gitignored and reproducible from scripts/.
Override the data root with the PHOTOPROT_DATA environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PHOTOPROT_DATA", ROOT / "data"))

RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# raw downloads
CATH_RAW = RAW / "cath"

# extracted per-domain PDB files (one file per CATH domain, already chopped)
DOMAIN_PDB = INTERIM / "dompdb"

# rendered images, laid out as renders/v<view>/<2-char prefix>/<domain_id>.png
RENDERS = DATA / "renders"

# derived tables
MANIFEST = PROCESSED / "manifest.parquet"
SSE_TABLE = PROCESSED / "sse.parquet"
SPLITS = PROCESSED / "splits.parquet"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, CATH_RAW, DOMAIN_PDB, RENDERS):
        p.mkdir(parents=True, exist_ok=True)

"""Build the domain manifest: one row per CATH S40 domain with its labels.

The manifest is the spine of the project. Everything downstream (renders,
secondary structure, splits, training) joins against `domain_id`.
"""
from __future__ import annotations

import pandas as pd

from photoprot.paths import DOMAIN_PDB, MANIFEST


def _resources():
    """Lazy import of the download registry.

    fetch_cath pulls in `requests`, which only the download path needs.
    Importing it at module scope made manifest.load() - a plain parquet read -
    depend on an HTTP library, and that crashed training on Modal where the
    image has no `requests`.
    """
    from photoprot.data.fetch_cath import RESOURCES

    return RESOURCES

# CATH List File (CLF) format 2.0
CLF_COLUMNS = [
    "domain_id",
    "C", "A", "T", "H",
    "S35", "S60", "S95", "S100", "S100_count",
    "length",
    "resolution",
]

# CATH marks NMR / unknown-resolution entries with this sentinel
RESOLUTION_SENTINEL = 999.0


def load_domain_list() -> pd.DataFrame:
    """Parse cath-domain-list into a typed frame (all classified domains)."""
    df = pd.read_csv(
        _resources()["domain_list"].dest,
        sep=r"\s+",
        comment="#",
        names=CLF_COLUMNS,
        header=None,
        dtype={"domain_id": "string"},
    )
    for col in CLF_COLUMNS[1:-1]:
        df[col] = df[col].astype("int32")
    df["resolution"] = df["resolution"].astype("float32")
    return df


def load_names() -> dict[str, str]:
    """Map a CATH node ('3.40.50') to its human-readable name."""
    names: dict[str, str] = {}
    with open(_resources()["names"].dest, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            node, _rep, label = line.split(None, 2)
            names[node] = label.lstrip(":").strip()
    return names


def load_nonredundant_ids(level: str = "S40") -> list[str]:
    key = {"S40": "s40_list", "S20": "s20_list"}[level.upper()]
    path = _resources()[key].dest
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build(level: str = "S40", require_pdb: bool = True) -> pd.DataFrame:
    """Join the non-redundant id list against CATH labels and on-disk PDB files."""
    full = load_domain_list()
    keep = set(load_nonredundant_ids(level))
    df = full[full["domain_id"].isin(keep)].copy().reset_index(drop=True)

    missing_labels = keep - set(df["domain_id"])
    if missing_labels:
        print(
            f"[manifest] {len(missing_labels):,} {level} ids absent from the domain "
            f"list (dropped); e.g. {sorted(missing_labels)[:5]}",
            flush=True,
        )

    # hierarchy codes at each level
    df["cath_class"] = df["C"].astype(str)
    df["cath_arch"] = df["cath_class"] + "." + df["A"].astype(str)
    df["cath_topo"] = df["cath_arch"] + "." + df["T"].astype(str)
    df["cath_homsf"] = df["cath_topo"] + "." + df["H"].astype(str)

    names = load_names()
    for col in ("cath_class", "cath_arch", "cath_topo", "cath_homsf"):
        df[f"{col}_name"] = df[col].map(names).astype("string")

    # provenance back to the source structure
    df["pdb_id"] = df["domain_id"].str[:4]
    df["chain"] = df["domain_id"].str[4]
    df["domain_idx"] = df["domain_id"].str[5:].astype("int32")

    # resolution sentinel -> NaN so it never silently acts as a feature
    df["is_nmr_or_unknown_res"] = df["resolution"] >= RESOLUTION_SENTINEL
    df.loc[df["is_nmr_or_unknown_res"], "resolution"] = pd.NA

    # locate the pre-chopped PDB file for each domain
    df["pdb_path"] = df["domain_id"].map(lambda d: str(DOMAIN_PDB / d))
    on_disk = {p.name for p in DOMAIN_PDB.iterdir()} if DOMAIN_PDB.exists() else set()
    df["has_pdb"] = df["domain_id"].isin(on_disk)

    n_missing = int((~df["has_pdb"]).sum())
    if n_missing:
        print(f"[manifest] {n_missing:,} domains have no PDB file on disk", flush=True)
    if require_pdb:
        df = df[df["has_pdb"]].reset_index(drop=True)

    df["nr_level"] = level.upper()
    return df


def save(df: pd.DataFrame, path=MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[manifest] wrote {len(df):,} rows -> {path}", flush=True)


def load(path=MANIFEST) -> pd.DataFrame:
    return pd.read_parquet(path)

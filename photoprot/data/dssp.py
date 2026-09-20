"""Secondary structure assignment for CATH domain PDB files.

Uses pydssp (a numpy reimplementation of the DSSP hydrogen-bond criterion), so
there is no external mkdssp binary to install. Output feeds the Phase 1 gate
baseline: if secondary-structure composition alone predicts CATH architecture
about as well as a CNN on renders, the rendering pipeline is not earning its keep.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

BACKBONE = ("N", "CA", "C", "O")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass
class DomainBackbone:
    coords: np.ndarray       # (L, 4, 3) float32, atom order N/CA/C/O
    seq: str                 # (L,) one-letter, 'X' for unknown residues
    res_ids: list[str]       # PDB residue identifiers, for provenance
    n_atoms_seen: int
    n_res_dropped: int       # residues lacking a full backbone


def read_backbone(path: str | Path) -> DomainBackbone:
    """Parse a CATH domain PDB file into an (L, 4, 3) backbone tensor.

    CATH domain files are ATOM-only, single chain, fixed-width, with no element
    column, so a direct column slice is both faster and more predictable than a
    general-purpose parser. Residues missing any backbone atom are dropped; that
    leaves a chain break, which DSSP handles naturally (no hydrogen bond across
    the gap) but which we count so it can be filtered on later.

    Alternate locations are resolved by keeping the highest-occupancy record for
    each (residue, atom). Filtering on the altloc letter instead would be wrong:
    some CATH domains (e.g. 2qe7G01) carry a single conformer labelled 'C', and a
    blank-or-'A' rule silently discards the entire file.
    """
    per_res: dict[str, dict[str, tuple[float, float, float]]] = {}
    best_occ: dict[str, dict[str, float]] = {}
    order: list[str] = []
    resname: dict[str, str] = {}
    n_atoms = 0

    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom not in BACKBONE:
                continue
            key = line[21] + line[22:27].strip()  # chain + resSeq + iCode
            if key not in per_res:
                per_res[key] = {}
                best_occ[key] = {}
                order.append(key)
                resname[key] = line[17:20].strip().upper()
            try:
                occ = float(line[54:60])
            except ValueError:
                occ = 1.0
            if atom in best_occ[key] and occ <= best_occ[key][atom]:
                continue
            per_res[key][atom] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            )
            best_occ[key][atom] = occ
            n_atoms += 1

    coords, seq, kept = [], [], []
    dropped = 0
    for key in order:
        atoms = per_res[key]
        if not all(a in atoms for a in BACKBONE):
            dropped += 1
            continue
        coords.append([atoms[a] for a in BACKBONE])
        seq.append(THREE_TO_ONE.get(resname[key], "X"))
        kept.append(key)

    arr = (
        np.asarray(coords, dtype=np.float32)
        if coords
        else np.zeros((0, 4, 3), dtype=np.float32)
    )
    return DomainBackbone(arr, "".join(seq), kept, n_atoms, dropped)


def assign_ss(coords: np.ndarray) -> str:
    """Per-residue 3-state secondary structure as a string of H / E / '-'."""
    import pydssp

    if coords.shape[0] < 4:
        return "-" * coords.shape[0]
    ss = pydssp.assign(coords, out_type="c3")
    return "".join(np.asarray(ss).ravel().tolist())


def segments(ss: str) -> list[tuple[str, int]]:
    """Run-length encode an SS string: 'HHHEE-' -> [('H',3), ('E',2), ('-',1)]."""
    out: list[tuple[str, int]] = []
    for ch in ss:
        if out and out[-1][0] == ch:
            out[-1] = (ch, out[-1][1] + 1)
        else:
            out.append((ch, 1))
    return out


# Short runs are usually noise rather than real elements; the threshold keeps the
# "element order" string interpretable as a topology sketch.
MIN_HELIX = 4
MIN_STRAND = 2


def element_string(ss: str, min_h: int = MIN_HELIX, min_e: int = MIN_STRAND) -> str:
    """Ordered sequence of substantial SS elements, e.g. 'HEEHE'.

    This is the closest a composition baseline gets to topology: it preserves the
    order in which helices and strands appear along the chain, which is exactly
    what distinguishes many CATH topologies within one architecture.
    """
    keep = []
    for kind, n in segments(ss):
        if kind == "H" and n >= min_h:
            keep.append("H")
        elif kind == "E" and n >= min_e:
            keep.append("E")
    return "".join(keep)


def summarize(domain_id: str, path: str | Path) -> dict:
    """Everything Phase 0 records about one domain's secondary structure."""
    bb = read_backbone(path)
    n = bb.coords.shape[0]
    if n == 0:
        return {
            "domain_id": domain_id, "n_res": 0, "n_res_dropped": bb.n_res_dropped,
            "ss_string": "", "element_string": "", "seq": "",
            "frac_H": np.nan, "frac_E": np.nan, "frac_C": np.nan,
            "n_helix": 0, "n_strand": 0,
            "mean_helix_len": np.nan, "mean_strand_len": np.nan,
            "max_helix_len": 0, "max_strand_len": 0, "ok": False,
        }

    ss = assign_ss(bb.coords)
    segs = segments(ss)
    hel = [ln for k, ln in segs if k == "H" and ln >= MIN_HELIX]
    strd = [ln for k, ln in segs if k == "E" and ln >= MIN_STRAND]

    return {
        "domain_id": domain_id,
        "n_res": n,
        "n_res_dropped": bb.n_res_dropped,
        "ss_string": ss,
        "element_string": element_string(ss),
        "seq": bb.seq,
        "frac_H": ss.count("H") / n,
        "frac_E": ss.count("E") / n,
        "frac_C": ss.count("-") / n,
        "n_helix": len(hel),
        "n_strand": len(strd),
        "mean_helix_len": float(np.mean(hel)) if hel else 0.0,
        "mean_strand_len": float(np.mean(strd)) if strd else 0.0,
        "max_helix_len": max(hel) if hel else 0,
        "max_strand_len": max(strd) if strd else 0,
        "ok": True,
    }

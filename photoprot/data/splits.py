"""Train/val/test splits.

The PDB is massively redundant, so a random split measures memorisation rather
than generalisation. We build several splits that answer different questions and
keep them side by side so results are always reported against a named regime:

  random      leaky control. Reported only as an upper bound; the gap between
              this and `homsf` is how much redundancy was doing the work.
  homsf       no superfamily appears in more than one split. Measures fold
              generalisation. NOTE: H-level classification is impossible here by
              construction (every test superfamily is unseen), so this split is
              scored at C/A/T level and by retrieval.
  topo        as above but held out at topology level. Harsher.
  member      within-superfamily split by S35 sequence cluster. Superfamilies are
              shared across splits but no S35 cluster is. This is the realistic
              deployment setting - an unknown picture of a protein from a family
              that CATH already knows - and the only split where H-level
              classification is meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACS = (0.8, 0.1, 0.1)


def _greedy_group_assign(
    sizes: pd.Series, fracs: tuple[float, ...], rng: np.random.Generator
) -> dict:
    """Assign whole groups to splits, keeping split sizes close to `fracs`.

    Largest groups first, each going to whichever split is currently furthest
    below its quota. Greedy bin-packing beats random group assignment here
    because CATH group sizes span three orders of magnitude - a random draw can
    easily put Immunoglobulins (957 domains) in a 10% test split.
    """
    total = float(sizes.sum())
    quota = np.array(fracs, dtype=float) * total
    filled = np.zeros(len(fracs), dtype=float)

    # descending size, random tie-break for reproducible but unbiased ordering
    jitter = rng.random(len(sizes))
    order = pd.DataFrame({"size": sizes.values, "j": jitter}, index=sizes.index)
    order = order.sort_values(["size", "j"], ascending=[False, True]).index

    out: dict = {}
    for g in order:
        deficit = quota - filled
        k = int(np.argmax(deficit))
        out[g] = SPLIT_NAMES[k]
        filled[k] += float(sizes.loc[g])
    return out


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    fracs: tuple[float, ...] = DEFAULT_FRACS,
    stratify_col: str | None = "cath_arch",
    seed: int = 0,
) -> pd.Series:
    """Disjoint-group split, balanced within each stratum."""
    rng = np.random.default_rng(seed)
    assign: dict = {}

    if stratify_col is None:
        sizes = df.groupby(group_col).size()
        assign.update(_greedy_group_assign(sizes, fracs, rng))
    else:
        # A group never spans strata because every CATH code is a prefix of the
        # one below it, so stratifying is just running the packer per stratum.
        # Stratify on ARCHITECTURE, not class: balancing only the 5 classes lets
        # the 43 architectures drift badly between train and test (measured TVD
        # 0.21), and a classifier trained on one architecture prior then scores
        # BELOW the majority baseline on a differently-distributed test set.
        for _, sub in df.groupby(stratify_col, observed=True):
            sizes = sub.groupby(group_col).size()
            assign.update(_greedy_group_assign(sizes, fracs, rng))

    return df[group_col].map(assign).astype("string")


def random_split(
    df: pd.DataFrame, fracs: tuple[float, ...] = DEFAULT_FRACS, seed: int = 0
) -> pd.Series:
    """Leaky control: domains assigned independently of homology."""
    rng = np.random.default_rng(seed)
    draw = rng.random(len(df))
    edges = np.cumsum(fracs)
    lab = np.full(len(df), SPLIT_NAMES[2], dtype=object)
    lab[draw < edges[1]] = SPLIT_NAMES[1]
    lab[draw < edges[0]] = SPLIT_NAMES[0]
    return pd.Series(lab, index=df.index, dtype="string")


def member_split(
    df: pd.DataFrame,
    fracs: tuple[float, ...] = DEFAULT_FRACS,
    min_clusters: int = 3,
    seed: int = 0,
) -> pd.Series:
    """Within-superfamily split by S35 cluster.

    Superfamilies with fewer than `min_clusters` distinct S35 clusters cannot
    contribute a held-out member without also losing their training signal, so
    they are placed entirely in train and marked as such.
    """
    rng = np.random.default_rng(seed)
    cluster = df["cath_homsf"] + "|" + df["S35"].astype(str)
    work = df.assign(_cluster=cluster)
    out = pd.Series(pd.NA, index=df.index, dtype="string")

    for homsf, sub in work.groupby("cath_homsf", observed=True):
        sizes = sub.groupby("_cluster").size()
        if len(sizes) < min_clusters:
            out.loc[sub.index] = "train"
            continue
        assign = _greedy_group_assign(sizes, fracs, rng)
        out.loc[sub.index] = sub["_cluster"].map(assign).values

    return out


def build_all(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Compute every split scheme and return a table keyed by domain_id."""
    out = pd.DataFrame({"domain_id": df["domain_id"].values})
    out["split_random"] = random_split(df, seed=seed).values
    out["split_homsf"] = grouped_split(df, "cath_homsf", seed=seed).values
    out["split_topo"] = grouped_split(df, "cath_topo", seed=seed).values
    out["split_member"] = member_split(df, seed=seed).values
    return out


def audit(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Leakage check: for each scheme, does any group straddle train and test?"""
    merged = df.merge(splits, on="domain_id", validate="1:1")
    rows = []
    schemes = {
        "split_random": "cath_homsf",
        "split_homsf": "cath_homsf",
        "split_topo": "cath_topo",
        "split_member": "cath_homsf|S35",
    }
    for scheme, group in schemes.items():
        if group == "cath_homsf|S35":
            key = merged["cath_homsf"] + "|" + merged["S35"].astype(str)
        else:
            key = merged[group]
        tmp = pd.DataFrame({"g": key, "s": merged[scheme]}).dropna()
        per_group = tmp.groupby("g")["s"].nunique()
        straddling = int((per_group > 1).sum())
        counts = merged[scheme].value_counts()
        rows.append({
            "scheme": scheme,
            "group_unit": group,
            "groups_in_multiple_splits": straddling,
            "n_train": int(counts.get("train", 0)),
            "n_val": int(counts.get("val", 0)),
            "n_test": int(counts.get("test", 0)),
            "n_unassigned": int(merged[scheme].isna().sum()),
            "tvd_class": _tvd(merged, scheme, "cath_class"),
            "tvd_arch": _tvd(merged, scheme, "cath_arch"),
        })
    return pd.DataFrame(rows)


def _tvd(merged: pd.DataFrame, scheme: str, label_col: str) -> float:
    """Total variation distance between the train and test label distributions.

    A grouped split can hold out whole superfamilies cleanly and still shift the
    label prior, which silently penalises any model that learned the training
    prior. Anything much above ~0.02 means the split, not the model, is driving
    the numbers.
    """
    tr = merged.loc[merged[scheme] == "train", label_col].value_counts(normalize=True)
    te = merged.loc[merged[scheme] == "test", label_col].value_counts(normalize=True)
    if tr.empty or te.empty:
        return float("nan")
    both = tr.align(te, fill_value=0.0)
    return round(float((both[0] - both[1]).abs().sum() / 2), 4)

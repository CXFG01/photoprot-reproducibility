"""Feature construction for the secondary-structure baseline.

This is deliberately the cheapest reasonable representation of a domain: what
secondary-structure elements it contains, how big they are, and in what order
they appear along the chain. If a CNN on rendered images cannot beat this, the
rendering pipeline is decoration and the project needs rethinking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

NUMERIC_COLS = [
    "n_res",
    "frac_H", "frac_E", "frac_C",
    "n_helix", "n_strand",
    "mean_helix_len", "mean_strand_len",
    "max_helix_len", "max_strand_len",
]


def numeric_block(df: pd.DataFrame) -> np.ndarray:
    """Composition and size features, plus a few length-normalised ratios."""
    X = df[NUMERIC_COLS].astype("float32").copy()
    n = X["n_res"].clip(lower=1)
    X["helix_per_100"] = 100.0 * X["n_helix"] / n
    X["strand_per_100"] = 100.0 * X["n_strand"] / n
    X["he_ratio"] = X["frac_H"] / (X["frac_E"] + 1e-3)
    X["log_n_res"] = np.log1p(X["n_res"])
    return X.to_numpy(dtype=np.float32)


def length_block(df: pd.DataFrame) -> np.ndarray:
    """Domain size alone - the ablation that catches a trivially easy target."""
    n = df["n_res"].astype("float32").to_numpy()
    return np.column_stack([n, np.log1p(n)]).astype(np.float32)


def ngram_block(
    train_strings: pd.Series,
    all_strings: pd.Series,
    max_features: int = 2000,
    ngram_range: tuple[int, int] = (1, 5),
) -> np.ndarray:
    """Character n-grams over the SS element string, e.g. 'HEEHE'.

    This is the only part of the baseline that carries topology information:
    architecture is largely composition, but topology is about the ORDER in
    which helices and strands run along the chain. Fitted on train only.
    """
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
        min_df=2,
    )
    vec.fit(train_strings.fillna(""))
    # The element alphabet is just {H, E}, so this caps out around 62 columns -
    # dense is smaller and simpler than sparse at this width.
    return vec.transform(all_strings.fillna("")).toarray().astype(np.float32)


def build(
    df: pd.DataFrame, train_mask: np.ndarray, kind: str = "both"
) -> np.ndarray:
    """Assemble the feature matrix for every row, fitting only on train rows."""
    blocks: list[np.ndarray] = []
    if kind == "length":
        blocks.append(length_block(df))
    else:
        if kind in ("numeric", "both"):
            blocks.append(numeric_block(df))
        if kind in ("ngram", "both"):
            blocks.append(
                ngram_block(df.loc[train_mask, "element_string"], df["element_string"])
            )
    if not blocks:
        raise ValueError(f"unknown feature kind: {kind}")
    return np.hstack(blocks).astype(np.float32)

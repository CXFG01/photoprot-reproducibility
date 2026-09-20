"""Retrieval evaluation over image embeddings.

Classification was the diagnostic; retrieval is the actual problem. This asks a
different question of the same trained model: are a query image's nearest
neighbours in embedding space STRUCTURALLY related to it?

Relevance is judged by shared CATH membership, which is a label proxy rather
than a structural measurement. The frozen protocol calls for TM-score based
grading (query-normalised, with coverage reported); that needs TM-align or
Foldseek and is deliberately not what this first probe does. Label agreement is
cheap, uses data already on hand, and is enough to tell whether the embedding
space has any structural organisation at all.

What makes the numbers meaningful is the level:

  architecture - the model was TRAINED on these labels, so retrieval here is
                 close to tautological and is reported only for completeness.
  topology     - never trained on. 1,471 classes.
  superfamily  - never trained on, AND under split_homsf every test superfamily
                 is absent from training entirely. This is genuinely zero-shot:
                 the model has never seen any member of these families.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("cath_arch", "cath_topo", "cath_homsf")
KS = (1, 5, 10)


def chance_rate(index_labels: pd.Series, query_labels: pd.Series) -> float:
    """Probability a uniformly random index entry shares the query's label.

    The correct floor to beat. With 6,576 superfamilies it is tiny, but it is not
    1/6576: the label distribution is extremely skewed, so a random draw is far
    more likely to hit a crowded family than a rare one.
    """
    counts = index_labels.value_counts()
    n = len(index_labels)
    p = query_labels.map(counts).fillna(0.0) / max(n, 1)
    return float(p.mean())


def topk(query_emb: np.ndarray, index_emb: np.ndarray, k: int,
         exclude_self: np.ndarray | None = None) -> np.ndarray:
    """Cosine top-k. Embeddings are L2-normalised, so a dot product suffices.

    `exclude_self[i]` is the index row that IS query i and must never be
    retrieved - otherwise every query trivially retrieves itself at rank 1.
    """
    sims = query_emb @ index_emb.T
    if exclude_self is not None:
        sims[np.arange(len(sims)), exclude_self] = -np.inf
    return np.argpartition(-sims, kth=k, axis=1)[:, :k], sims


def evaluate(
    query_meta: pd.DataFrame,
    index_meta: pd.DataFrame,
    query_emb: np.ndarray,
    index_emb: np.ndarray,
    self_rows: np.ndarray | None = None,
    ks: tuple[int, ...] = KS,
) -> pd.DataFrame:
    kmax = max(ks)
    idx, sims = topk(query_emb, index_emb, kmax, self_rows)

    # argpartition does not order within the selection; sort the k retrieved
    ordered = np.take_along_axis(
        idx, np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1), axis=1)

    rows = []
    for level in LEVELS:
        q = query_meta[level].to_numpy()
        ix = index_meta[level].to_numpy()
        hits = ix[ordered] == q[:, None]  # (n_query, kmax)

        # how many queries could possibly succeed: is there ANY other domain in
        # the index sharing this label? Without this a low recall is ambiguous
        # between "model failed" and "no correct answer existed".
        counts = index_meta[level].value_counts()
        available = query_meta[level].map(counts).fillna(0).to_numpy()
        if self_rows is not None:
            available = available - 1  # the query's own row is excluded
        answerable = available > 0

        rec = {"level": level, "n_query": len(q),
               "answerable": int(answerable.sum()),
               "chance_at_1": round(chance_rate(index_meta[level], query_meta[level]), 5)}
        for k in ks:
            rec[f"recall@{k}"] = round(float(hits[:, :k].any(axis=1).mean()), 4)
            rec[f"recall@{k}_answerable"] = round(
                float(hits[answerable, :k].any(axis=1).mean()) if answerable.any() else float("nan"), 4)
        # MRR over the retrieved window
        first = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
        rr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        rec["mrr"] = round(float(rr.mean()), 4)
        rows.append(rec)
    return pd.DataFrame(rows)

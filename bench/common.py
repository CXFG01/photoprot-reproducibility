"""Shared paths, retrieval and metrics for the PhotoProt benchmark.

Everything here is READ-ONLY with respect to the corpus. No script in bench/
mutates data/webp/*.tar or data/meta/manifest.parquet; all outputs land in
new directories (data/img, data/emb, data/ckpt, data/results).
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/photoprot")
META = f"{ROOT}/data/meta"
WEBP = f"{ROOT}/data/webp"
IMG = f"{ROOT}/data/img"
EMB = f"{ROOT}/data/emb"
CKPT = f"{ROOT}/data/ckpt"
RESULTS = f"{ROOT}/data/results"
AUX = f"{ROOT}/data/aux"

MANIFEST = f"{META}/manifest.parquet"
SPLITS = f"{META}/splits_component.parquet"


def load_manifest(columns=None):
    return pd.read_parquet(MANIFEST, columns=columns)


def load_splits():
    return pd.read_parquet(SPLITS)


def img_path(shard, member):
    """Path of an extracted image. shard is e.g. 'shard_000.tar'."""
    return f"{IMG}/{shard[:-4]}/{member}"


def atomic_write_npz(path, **arrays):
    """Write via .tmp + rename so an interrupted run never leaves a half file."""
    if 'emb' in arrays:
        validate_embeddings(arrays['emb'], path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"          # np.savez appends .npz unless it is there
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def l2norm(x, eps=1e-12):
    import torch
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, eps)
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def validate_embeddings(emb, label='embeddings'):
    """Reject corrupt or zero vectors before training, saving, or ranking."""
    if emb.ndim != 2 or not np.isfinite(emb).all():
        raise ValueError(f'{label}: expected a finite 2D embedding array')
    for start in range(0, len(emb), 8192):
        norms = np.linalg.norm(emb[start:start + 8192].astype(np.float32), axis=1)
        if (norms <= 1e-8).any():
            raise ValueError(f'{label}: zero embedding vector')


def retrieval_eval(q_emb, q_pdb, g_emb, g_pdb, device="cuda",
                   chunk=256, ks=(1, 5, 10, 20)):
    """Exact cosine retrieval, image -> nearest views -> max-pool per PDB entry.

    q_emb/g_emb are float arrays, assumed already L2-normalised so that a dot
    product IS the cosine. Scores are aggregated per PDB entry by MAX over that
    entry's gallery images, which is the 'best matching view wins' rule.

    Returns Recall@k, MRR and median rank of the correct PDB entry.
    """
    import torch

    validate_embeddings(q_emb, 'query embeddings')
    validate_embeddings(g_emb, 'gallery embeddings')
    pdb_vocab = sorted(set(g_pdb.tolist()))
    pdb_ix = {p: i for i, p in enumerate(pdb_vocab)}
    n_pdb = len(pdb_vocab)

    # queries whose true entry is absent from the gallery are unanswerable
    answerable = np.array([p in pdb_ix for p in q_pdb])
    if not answerable.all():
        q_emb, q_pdb = q_emb[answerable], q_pdb[answerable]

    g_idx = torch.tensor([pdb_ix[p] for p in g_pdb], dtype=torch.long,
                         device=device)
    G = torch.as_tensor(np.ascontiguousarray(g_emb), device=device,
                        dtype=torch.float32)
    truth = torch.tensor([pdb_ix[p] for p in q_pdb], dtype=torch.long,
                         device=device)

    ranks = torch.empty(len(q_pdb), dtype=torch.long, device=device)
    for s in range(0, len(q_pdb), chunk):
        e = min(s + chunk, len(q_pdb))
        Q = torch.as_tensor(np.ascontiguousarray(q_emb[s:e]), device=device,
                            dtype=torch.float32)
        sim = Q @ G.T                                     # [b, n_gallery]
        per = torch.full((e - s, n_pdb), -1e30, device=device)
        per.scatter_reduce_(1, g_idx.expand(e - s, -1), sim,
                            reduce="amax", include_self=True)
        true_score = per.gather(1, truth[s:e, None])
        ranks[s:e] = (per > true_score).sum(1) + 1
        del sim, per, Q

    r = ranks.float()
    out = {f"R@{k}": float((ranks <= k).float().mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median_rank"] = float(r.median())
    out["mean_rank"] = float(r.mean())
    out["n_queries"] = int(len(q_pdb))
    out["n_gallery_images"] = int(len(g_pdb))
    out["n_gallery_pdbs"] = int(n_pdb)
    out["n_unanswerable_dropped"] = int((~answerable).sum())
    return out

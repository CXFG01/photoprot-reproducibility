"""Retrieval benchmark: image -> nearest views -> group by PDB -> ranked entries.

Two protocols, and they must both be reported, because they answer different
questions:

  view0_query   query = view 0, gallery = views 1..15  (HEADLINE)
        View 0 is the corpus's designated `eval_single` render and is a FIXED
        canonical style for every object (white bg, uniform colour, cartoon
        only, no ray-trace), while views 1..15 are style-randomised. So this
        measures camera generalisation AND style shift together. That matches
        the README's frozen single-view protocol and the real use case.

  style_matched query = view 15, gallery = views 1..14  (CONTROL)
        Query drawn from the same randomised style distribution as the gallery,
        view 0 excluded entirely. Isolates camera generalisation alone.

The gap between the two IS the style-shift cost. Reporting only the first hides
it; reporting only the second flatters the model.

Everything is exact cosine (no ANN): the test gallery is ~80k x 1024 and fits in
the A6000 several times over, so an index would add a dependency and buy nothing.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PROTOCOLS = {
    "view0_query":   {"q_views": [0],  "g_views": list(range(1, 16))},
    "style_matched": {"q_views": [15], "g_views": list(range(1, 15))},
}


def load_emb(tag):
    files = sorted(glob.glob(f"{C.EMB}/{tag}/*.npz"))
    if not files:
        raise SystemExit(f"no embeddings at {C.EMB}/{tag}")
    ids, embs = [], []
    for f in files:
        z = np.load(f)
        C.validate_embeddings(z['emb'], f)
        ids.append(z["render_id"])
        embs.append(z["emb"])
    ids = np.concatenate(ids)
    embs = np.concatenate(embs).astype(np.float32)
    print(f"loaded {len(ids):,} embeddings dim={embs.shape[1]} from {len(files)} shards")
    return ids, embs


def run(tag, split="test", out_name=None):
    ids, embs = load_emb(tag)
    pos = {r: i for i, r in enumerate(ids)}

    man = C.load_manifest(columns=["render_id", "pdb_id", "object_id", "view"])
    sp = C.load_splits()[["pdb_id", "split"]]
    man["pdb_l"] = man.pdb_id.str.lower()
    man = man.merge(sp, left_on="pdb_l", right_on="pdb_id", suffixes=("", "_s"))
    sub = man[man.split == split]
    print(f"split={split}: {len(sub):,} images, "
          f"{sub.object_id.nunique():,} objects, {sub.pdb_l.nunique():,} entries")

    results = {}
    for name, cfg in PROTOCOLS.items():
        q = sub[sub.view.isin(cfg["q_views"])]
        g = sub[sub.view.isin(cfg["g_views"])]
        qi = np.array([pos[r] for r in q.render_id if r in pos])
        gi = np.array([pos[r] for r in g.render_id if r in pos])
        qp = q.pdb_l.to_numpy()[[i for i, r in enumerate(q.render_id) if r in pos]]
        gp = g.pdb_l.to_numpy()[[i for i, r in enumerate(g.render_id) if r in pos]]
        m = C.retrieval_eval(embs[qi], qp, embs[gi], gp)
        results[name] = m
        print(f"\n--- {name} ---")
        for k, v in m.items():
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else
                  f"  {k:24s} {v}")

    out = out_name or f"{C.RESULTS}/retrieval_{tag}_{split}.json"
    C.atomic_write_json(out, {"tag": tag, "split": split,
                              "embedding_dim": int(embs.shape[1]),
                              "protocols": results})
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dinov2l")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.tag, a.split, a.out)

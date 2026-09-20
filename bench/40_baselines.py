"""Nuisance baselines: HOG, silhouette and colour, on the SAME split and the
SAME retrieval code path as the learned model.

The README makes this mandatory, and it is the most important number of the
night: if a frozen ViT gets R@5 = 0.45 and a colour histogram gets 0.40, then
almost nothing about protein structure is being measured. These features are
deliberately cheap and structure-blind.

  hog        shape/edge orientation statistics, greyscale, background-insensitive-ish
  silhouette pure blob geometry: area, aspect, fill, Hu moments, edge density
  colour     8x8x8 RGB histogram - pure palette, no geometry at all
  nuisance   silhouette + colour concatenated

Background differs per view (white/black/grey80/gradient), so the foreground
mask is taken relative to the image's own border colour rather than assuming
white.

Only the evaluated split is featurised - the baselines never need the other 90%.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SIZE = 128


def _features(args):
    shard, member = args
    im = Image.open(C.img_path(shard, member)).convert("RGB").resize(
        (SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255.0
    g = a.mean(2)

    # --- background estimate from the border ring -------------------------
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    mask = dist > 0.12

    # --- silhouette / nuisance -------------------------------------------
    area = mask.mean()
    ys, xs = np.nonzero(mask)
    if len(xs) > 4:
        h = (ys.max() - ys.min() + 1) / SIZE
        w = (xs.max() - xs.min() + 1) / SIZE
        fill = len(xs) / max((ys.max()-ys.min()+1) * (xs.max()-xs.min()+1), 1)
        cy, cx = ys.mean() / SIZE, xs.mean() / SIZE
        sy, sx = ys.std() / SIZE, xs.std() / SIZE
    else:
        h = w = fill = cy = cx = sy = sx = 0.0
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    from skimage.measure import moments_hu, moments_central, moments_normalized
    try:
        mc = moments_central(mask.astype(float))
        hu = moments_hu(moments_normalized(mc))
        hu = np.sign(hu) * np.log1p(np.abs(hu))
    except Exception:
        hu = np.zeros(7)
    sil = np.concatenate([[area, h, w, h/max(w, 1e-6), fill, cy, cx, sy, sx,
                           gx, gy, g.mean(), g.std()], hu]).astype(np.float32)

    # --- colour histogram -------------------------------------------------
    q = np.clip((a * 8).astype(int), 0, 7)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    col = np.bincount(idx.ravel(), minlength=512).astype(np.float32)
    col /= col.sum()

    # --- HOG ---------------------------------------------------------------
    from skimage.feature import hog as skhog
    hg = skhog(g, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True).astype(np.float32)

    return sil, col, hg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    man = C.load_manifest(columns=["render_id", "pdb_id", "object_id", "view",
                                   "shard", "member"])
    sp = C.load_splits()[["pdb_id", "split"]]
    man["pdb_l"] = man.pdb_id.str.lower()
    man = man.merge(sp, left_on="pdb_l", right_on="pdb_id", suffixes=("", "_s"))
    sub = man[man.split == a.split].reset_index(drop=True)
    print(f"featurising {len(sub):,} images of split={a.split}", flush=True)

    cache = f"{C.EMB}/baselines_{a.split}.npz"
    if os.path.exists(cache):
        print(f"using cached {cache}")
        z = np.load(cache, allow_pickle=True)
        sil, col, hg, rid = z["sil"], z["col"], z["hog"], z["render_id"]
    else:
        rows = list(zip(sub.shard, sub.member))
        sil, col, hg = [], [], []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (s, c, h) in enumerate(ex.map(_features, rows, chunksize=64), 1):
                sil.append(s); col.append(c); hg.append(h)
                if i % 20000 == 0:
                    print(f"  {i:,}/{len(rows):,}", flush=True)
        sil = np.stack(sil); col = np.stack(col); hg = np.stack(hg)
        rid = sub.render_id.to_numpy().astype("U32")
        C.atomic_write_npz(cache, sil=sil, col=col, hog=hg, render_id=rid)
        print(f"wrote {cache}")

    # standardise silhouette dims so no single unit dominates the cosine
    sil = (sil - sil.mean(0)) / (sil.std(0) + 1e-6)
    feats = {
        "hog": hg,
        "silhouette": sil,
        "colour": col,
        "nuisance": np.concatenate([sil, col * 10.0], axis=1),
    }

    pos = {r: i for i, r in enumerate(rid)}
    PROTOCOLS = {
        "view0_query":   {"q": [0],  "g": list(range(1, 16))},
        "style_matched": {"q": [15], "g": list(range(1, 15))},
    }
    all_res = {}
    for fname, F in feats.items():
        F = C.l2norm(F.astype(np.float32))
        all_res[fname] = {}
        for pname, cfg in PROTOCOLS.items():
            q = sub[sub.view.isin(cfg["q"])]
            g = sub[sub.view.isin(cfg["g"])]
            qi = np.array([pos[r] for r in q.render_id])
            gi = np.array([pos[r] for r in g.render_id])
            m = C.retrieval_eval(F[qi], q.pdb_l.to_numpy(),
                                 F[gi], g.pdb_l.to_numpy())
            all_res[fname][pname] = m
            print(f"{fname:11s} {pname:14s} R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} "
                  f"R@10={m['R@10']:.4f} MRR={m['MRR']:.4f} "
                  f"med={m['median_rank']:.0f}", flush=True)

    C.atomic_write_json(f"{C.RESULTS}/baselines_{a.split}.json",
                        {"split": a.split, "features": all_res})
    print(f"wrote {C.RESULTS}/baselines_{a.split}.json")


if __name__ == "__main__":
    main()

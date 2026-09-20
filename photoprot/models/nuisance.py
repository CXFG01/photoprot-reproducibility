"""Image-only "nuisance" features: everything a model could exploit WITHOUT
understanding protein structure.

If a ResNet scores 0.60 on architecture and a random forest over silhouette
area, aspect ratio and HOG scores 0.57, then almost all of the apparent signal
is trivial shape and raster statistics, and the Phase 1 result is close to
meaningless. Knowing that number BEFORE training is the point.

Feature groups are kept separable so the source of any signal is attributable:

  shape  - silhouette geometry: how big, how elongated, how convex, how blobby.
           These are real but shallow cues. A long thin domain and a compact
           globular one differ here without any topology being understood.
  hog    - histogram of oriented gradients: coarse texture and edge layout.
           The classic "is a plain feature extractor enough" control.
  color  - colour statistics. On the canonical eval render the hue is sampled at
           RANDOM per image, so this group should carry NO signal. It is included
           precisely as a negative control: if colour features predict
           architecture above chance, something is wrong with the pipeline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.measure import moments_hu

IMAGE_SIZE = 128
HOG_KW = dict(orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
              block_norm="L2-Hys", feature_vector=True)

GROUPS = ("shape", "hog", "color")


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels differing from the modal (background) colour.

    Renders have a solid background, so the modal colour is the background and
    everything else is protein. More robust than thresholding on brightness,
    which would fail on black backgrounds.
    """
    flat = rgb.reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    return (np.abs(flat.astype(int) - bg.astype(int)).sum(1) > 30).reshape(rgb.shape[:2])


def shape_features(mask: np.ndarray) -> list[float]:
    h, w = mask.shape
    area = float(mask.sum())
    if area == 0:
        return [0.0] * 12

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = float(y1 - y0 + 1), float(x1 - x0 + 1)

    # perimeter via morphological gradient: foreground pixels with a background
    # 4-neighbour
    p = np.zeros_like(mask)
    p[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    p[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    p[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    p[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    perim = float(p.sum())

    # second moments -> elongation and orientation-free spread
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    cov = np.cov(np.stack([dy, dx]))
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1] if cov.shape == (2, 2) else np.array([0.0, 0.0])
    eig = np.maximum(eig, 1e-6)

    # radial mass profile: compact vs shell-like
    r = np.sqrt(dy ** 2 + dx ** 2)
    rmax = max(r.max(), 1e-6)

    return [
        area / (h * w),                 # foreground occupancy
        bw / max(bh, 1e-6),             # bounding box aspect ratio
        area / (bh * bw),               # fill ratio within bbox
        perim / max(area, 1e-6),        # perimeter-to-area (blobbiness)
        perim ** 2 / max(area, 1e-6),   # isoperimetric ratio
        float(np.sqrt(eig[0] / eig[1])),  # elongation
        float(np.sqrt(eig[0])) / max(h, 1),
        float(np.sqrt(eig[1])) / max(h, 1),
        float(r.mean() / rmax),
        float(r.std() / rmax),
        float((r < 0.5 * rmax).mean()),  # inner mass fraction
        float(bh * bw) / (h * w),
    ]


def color_features(rgb: np.ndarray, mask: np.ndarray) -> list[float]:
    """Colour statistics over the protein only. Negative control - see module docstring."""
    if mask.sum() == 0:
        return [0.0] * 10
    fg = rgb[mask].astype(float) / 255.0
    mx, mn = fg.max(1), fg.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return [
        *fg.mean(0).tolist(), *fg.std(0).tolist(),
        float(sat.mean()), float(sat.std()),
        float(mx.mean()), float(mn.mean()),
    ]


def extract(path: str, groups: tuple[str, ...] = GROUPS) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        rgb = np.asarray(im)

    mask = _foreground_mask(rgb)
    feats: list[float] = []
    # Append strictly in GROUPS order. Callers slice the combined vector by group,
    # so any divergence between this order and GROUPS silently hands each group
    # another group's columns.
    if "shape" in groups:
        feats += shape_features(mask)
        feats += [float(v) for v in moments_hu(mask.astype(float))]
    if "hog" in groups:
        gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float) / 255.0
        feats += hog(gray, **HOG_KW).tolist()
    if "color" in groups:
        feats += color_features(rgb, mask)

    return np.asarray(feats, dtype=np.float32)


def extract_one(args):
    """Top-level for multiprocessing (Windows spawn cannot pickle closures)."""
    path, groups = args
    try:
        return extract(path, groups)
    except Exception:
        return None

"""Decoded-image cache.

PNG decoding, not the GPU, is what limits training here: with the renderer using
the machine, a dataloader worker was delivering 16 img/s and the GPU sat idle at
0% utilisation for four samples out of five.

Decoding is also pure waste to repeat. The Phase 1 test matrix alone needs six
training runs (scratch, pretrained, greyscale, adversarial viewpoints, engine
holdout, split_random), and each would decode the same 34,649 images again.

So decode once into a uint8 memmap and read slices thereafter. Training then
needs almost no CPU, which removes the contention with rendering entirely.

Size is 256x256x3 bytes per image = 192 KB, about 6.6 GB per view. Cached at 256
rather than 224 so RandomResizedCrop still has headroom to crop from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from photoprot.paths import DATA

CACHE = DATA / "cache"
CACHE_SIZE = 256


def array_path(view: int, size: int = CACHE_SIZE) -> Path:
    return CACHE / f"v{view}_{size}.npy"


def index_path(view: int, size: int = CACHE_SIZE) -> Path:
    return CACHE / f"v{view}_{size}_index.parquet"


def is_complete(view: int, n_expected: int, size: int = CACHE_SIZE) -> bool:
    ap, ip = array_path(view, size), index_path(view, size)
    if not (ap.exists() and ip.exists()):
        return False
    idx = pd.read_parquet(ip)
    return len(idx) == n_expected


def decode_one(args) -> tuple[int, np.ndarray | None]:
    """Top-level for multiprocessing: Windows spawn cannot pickle closures."""
    i, path, size = args
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BICUBIC)
            return i, np.asarray(im, dtype=np.uint8)
    except Exception:
        return i, None


def load_cache(view: int, size: int = CACHE_SIZE) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-mapped image array plus its domain_id index.

    mmap_mode='r' means pages are faulted in on demand and shared between
    dataloader workers, so the cache costs no resident memory per worker.
    """
    arr = np.load(array_path(view, size), mmap_mode="r")
    idx = pd.read_parquet(index_path(view, size))
    return arr, idx

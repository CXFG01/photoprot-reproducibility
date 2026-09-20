"""Dataset and transforms for rendered protein images.

Two rules are enforced here rather than left to the training script, because
getting either wrong invalidates results silently:

1. NO MIRRORING. Alpha helices are right-handed; a horizontally flipped render
   is a physically impossible protein. Horizontal flip is the default first
   augmentation in almost every vision pipeline and it is wrong for this data.
   Only proper rotations are applied.
2. EVALUATION IS SINGLE-VIEW. Validation and test draw exactly one image per
   domain (view 0, the frozen canonical render). Aggregating views at test time
   would solve an easier problem than the deployment case.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from photoprot.data import manifest
from photoprot.paths import PROCESSED, RENDERS, SPLITS

# ImageNet statistics. Used for the pretrained backbone, and kept for the
# from-scratch run too so the two are directly comparable.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TARGETS = {"C": "cath_class", "A": "cath_arch"}
EVAL_VIEW = 0


def scan_render_dir(max_views: int | None = None) -> pd.DataFrame:
    """Recover the render table by walking the output directory.

    The per-view parquet is only written when a whole pass finishes, so during a
    long render there is no table for the pass in flight. Every render parameter
    is a deterministic function of (domain_id, view, seed) and the path encodes
    both, so the table can be rebuilt from the files themselves and training can
    start mid-pass instead of waiting for it to complete.
    """
    rows = []
    for vdir in sorted(RENDERS.glob("v*")):
        if not vdir.is_dir() or not vdir.name[1:].isdigit():
            continue
        view = int(vdir.name[1:])
        if max_views is not None and view >= max_views:
            continue
        for p in vdir.rglob("*.png"):
            rows.append({"domain_id": p.stem, "view": view,
                         "out_path": str(p), "ok": True})
    return pd.DataFrame(rows)


def load_render_table(max_views: int | None = None) -> pd.DataFrame:
    """Every successfully rendered image, from whichever passes exist so far.

    Prefers the per-view parquet, which carries the full style metadata, and
    falls back to a directory scan for passes still in flight.
    """
    frames = []
    for v in range(max_views if max_views is not None else 64):
        p = PROCESSED / f"renders_v{v}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))

    # Union with a disk scan rather than trusting the pass table to be complete.
    # A pass table only records what THAT run rendered: after an interrupted pass
    # is resumed, the table covers the resumed remainder and omits everything
    # rendered before the restart. Preferring the table would then silently train
    # on a fraction of the data - view 0 had 34,649 images on disk but only
    # 19,591 in its table.
    scanned = scan_render_dir(max_views)
    if not scanned.empty:
        known = set()
        if frames:
            tabled = pd.concat(frames, ignore_index=True)
            known = set(zip(tabled["domain_id"], tabled["view"]))
        extra = scanned[[
            (d, v) not in known for d, v in zip(scanned["domain_id"], scanned["view"])
        ]]
        if not extra.empty:
            n = extra.groupby("view").size().to_dict()
            print(f"[data] {len(extra):,} renders on disk but absent from the pass "
                  f"tables; recovered by scan: {n}", flush=True)
            frames.append(extra)

    if not frames:
        raise SystemExit(
            "No renders found. Run scripts/07_render.py first — training can "
            "begin as soon as some of view 0 exists."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ok"].astype(bool)].copy()
    df["exists"] = [Path(p).exists() for p in df["out_path"]]
    missing = int((~df["exists"]).sum())
    if missing:
        print(f"[data] {missing:,} recorded renders are missing on disk; dropped", flush=True)
    df = df[df["exists"]].drop(columns=["exists"])
    return df.drop_duplicates(subset=["domain_id", "view"]).reset_index(drop=True)


def build_frames(
    split_col: str = "split_homsf", max_views: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / val / test frames, already joined to labels and split membership."""
    r = load_render_table(max_views)
    m = manifest.load()[["domain_id", "cath_class", "cath_arch", "length"]]
    s = pd.read_parquet(SPLITS)[["domain_id", split_col]]
    df = r.merge(m, on="domain_id", validate="m:1").merge(s, on="domain_id", validate="m:1")

    train = df[df[split_col] == "train"]
    # val and test are single-view by construction
    val = df[(df[split_col] == "val") & (df["view"] == EVAL_VIEW)]
    test = df[(df[split_col] == "test") & (df["view"] == EVAL_VIEW)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_label_maps(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Label -> index, fitted on TRAIN ONLY so evaluation cannot leak categories."""
    maps = {}
    for key, col in TARGETS.items():
        vals = sorted(train[col].astype(str).unique())
        maps[key] = {v: i for i, v in enumerate(vals)}
    return maps


def save_label_maps(maps: dict, path: Path) -> None:
    path.write_text(json.dumps(maps, indent=2), encoding="utf-8")


def load_label_maps(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RandomQuarterTurn:
    """Rotate by a uniformly random multiple of 90 degrees.

    A module-level class rather than transforms.Lambda over a closure: Windows
    DataLoader workers use spawn, so the whole transform pipeline must pickle,
    and a local lambda raises "Can't get local object".

    Quarter turns only. They are exact camera rolls needing no interpolation and
    leaving no corner fill, and - unlike a mirror - they are proper rotations, so
    helix handedness is preserved.
    """

    _OPS = (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)

    def __call__(self, im: Image.Image) -> Image.Image:
        op = random.choice(self._OPS)
        return im if op is None else im.transpose(op)


def build_transforms(train: bool, image_size: int = 224, grayscale: bool = False):
    """Augmentation pipeline.

    Rotation is restricted to multiples of 90 degrees: those are exact camera
    rolls, need no interpolation and leave no fill artefacts at the corners,
    unlike arbitrary-angle rotation on a render with a solid background.
    """
    ops: list = []
    if train:
        ops += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # proper rotations only - never RandomHorizontalFlip
            RandomQuarterTurn(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
        ]
    else:
        ops += [transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC)]

    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class RenderDataset(Dataset):
    """One rendered image plus its CATH class and architecture indices."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_maps: dict[str, dict[str, int]],
        train: bool,
        image_size: int = 224,
        grayscale: bool = False,
    ):
        self.paths = frame["out_path"].tolist()
        self.domain_ids = frame["domain_id"].tolist()
        self.tf = build_transforms(train, image_size, grayscale)
        self.maps = label_maps
        self.labels = {}
        for key, col in TARGETS.items():
            m = label_maps[key]
            # -100 is ignore_index for cross entropy: a label unseen in training
            # cannot be predicted, and must not silently become class 0
            self.labels[key] = [m.get(str(v), -100) for v in frame[col]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = self.tf(im.convert("RGB"))
        y = {k: torch.tensor(v[i], dtype=torch.long) for k, v in self.labels.items()}
        return img, y, self.domain_ids[i]


class CachedRenderDataset(Dataset):
    """Same contract as RenderDataset, reading decoded pixels from a memmap.

    Avoids PNG decoding entirely, which is the actual bottleneck: training
    becomes GPU-bound and stops competing with the renderer for CPU.

    Memmaps are opened lazily, on first access inside each worker process, rather
    than in __init__. Handles created in the parent would otherwise have to
    survive pickling into every spawned worker.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        label_maps: dict[str, dict[str, int]],
        train: bool,
        image_size: int = 224,
        grayscale: bool = False,
        cache_size: int = 256,
    ):
        from photoprot.data import cache as C

        self.cache_size = cache_size
        self.domain_ids = frame["domain_id"].tolist()
        self.tf = build_transforms(train, image_size, grayscale)
        self._arrays: dict[int, object] = {}

        rows, views = [], []
        for view in sorted(frame["view"].unique()):
            idx = pd.read_parquet(C.index_path(int(view), cache_size))
            lookup = dict(zip(idx["domain_id"], idx["row"]))
            sub = frame[frame["view"] == view]
            missing = [d for d in sub["domain_id"] if d not in lookup]
            if missing:
                raise SystemExit(
                    f"view {view}: {len(missing):,} domains missing from the cache "
                    f"(e.g. {missing[:3]}). Re-run scripts/10_cache.py.")
            rows += [lookup[d] for d in sub["domain_id"]]
            views += [int(view)] * len(sub)

        order = frame.sort_values("view").index
        self.frame = frame.loc[order].reset_index(drop=True)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.views = np.asarray(views, dtype=np.int32)
        self.domain_ids = self.frame["domain_id"].tolist()

        self.labels = {}
        for key, col in TARGETS.items():
            m = label_maps[key]
            self.labels[key] = [m.get(str(v), -100) for v in self.frame[col]]

    def _array(self, view: int):
        if view not in self._arrays:
            from photoprot.data import cache as C

            self._arrays[view] = np.load(
                C.array_path(view, self.cache_size), mmap_mode="r")
        return self._arrays[view]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        arr = self._array(int(self.views[i]))
        img = self.tf(Image.fromarray(np.asarray(arr[self.rows[i]])))
        y = {k: torch.tensor(v[i], dtype=torch.long) for k, v in self.labels.items()}
        return img, y, self.domain_ids[i]


def class_counts(frame: pd.DataFrame, label_maps: dict, key: str) -> torch.Tensor:
    """Per-class training counts, for optional loss weighting."""
    col = TARGETS[key]
    m = label_maps[key]
    counts = torch.zeros(len(m), dtype=torch.float)
    for v, n in frame[col].astype(str).value_counts().items():
        if v in m:
            counts[m[v]] = float(n)
    return counts

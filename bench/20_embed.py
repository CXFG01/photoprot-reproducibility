"""Frozen backbone embeddings for the whole corpus, checkpointed per shard.

DINOv2-large is the default; --model can select an authorized local DINOv3.
Keep model parameters and buffers in FP32 and use autocast for mixed precision.

Renders are 384px square with the structure auto-zoomed to fill the frame, so
this resizes 384 -> 224 WITHOUT a centre crop. The processor default
(resize 256 + crop 224) would shave ~12% off every border and can clip the
structure, which would be a silent accuracy tax.

Resumable at shard granularity; each shard's .npz is written atomically.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ShardImages(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        shard, member = self.rows[i]
        im = Image.open(C.img_path(shard, member)).convert("RGB")
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(im).copy()          # copy: torch rejects read-only
        x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255)
        return (x - MEAN) / STD


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov2-large")
    ap.add_argument("--tag", default="dinov2l")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    from transformers import AutoModel
    dev = "cuda"
    print(f"loading {a.model}", flush=True)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()

    man = C.load_manifest(columns=["render_id", "shard", "member"])
    outdir = f"{C.EMB}/{a.tag}"
    os.makedirs(outdir, exist_ok=True)
    shards = sorted(man.shard.unique())
    print(f"{len(man):,} images across {len(shards)} shards -> {outdir}",
          flush=True)

    for si, shard in enumerate(shards, 1):
        dst = f"{outdir}/{shard[:-4]}.npz"
        if os.path.exists(dst):
            with np.load(dst) as cached:
                C.validate_embeddings(cached['emb'], dst)
            print(f"[{si:3d}/{len(shards)}] {shard} skip", flush=True)
            continue
        sub = man[man.shard == shard]
        ds = ShardImages(list(zip(sub.shard, sub.member)), a.size)
        dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                        pin_memory=True, shuffle=False)
        chunks = []
        t0 = time.time()
        for x in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(pixel_values=x).last_hidden_state[:, 0]  # CLS
            out = out.float()
            if not torch.isfinite(out).all():
                raise ValueError(f'Non-finite backbone output in {shard}')
            out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)
            chunks.append(out.cpu().numpy().astype(np.float16))
        emb = np.concatenate(chunks)
        C.atomic_write_npz(dst, emb=emb,
                           render_id=sub.render_id.to_numpy().astype("U32"))
        dt = time.time() - t0
        print(f"[{si:3d}/{len(shards)}] {shard} {len(emb):6,} imgs "
              f"{dt:5.1f}s ({len(emb)/dt:.0f}/s) dim={emb.shape[1]}", flush=True)
    print("embedding complete", flush=True)


if __name__ == "__main__":
    main()

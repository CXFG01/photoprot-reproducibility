"""Resumable training loop, shared by the local runner and the Modal app.

Must not import modal: the same code runs on a laptop and in a container.

Every Modal Function is preemptible and `nonpreemptible=True` is explicitly NOT
supported for GPU Functions, so a preempted run WILL be restarted from scratch
unless it can resume. That makes checkpoint/resume a correctness requirement
here, not a convenience: a checkpoint is written after every epoch carrying
optimiser, scheduler, epoch counter, early-stopping state and RNG state, so a
restart continues rather than silently re-running from epoch 1 with a different
random stream.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from photoprot.data import render_dataset as rd
from photoprot.models.resnet import MultiHeadResNet
from photoprot.util import progress

GATE = {"accuracy": 0.542, "macro_f1": 0.225}


@dataclass
class TrainConfig:
    split: str = "split_homsf"
    backbone: str = "resnet18"
    pretrained: bool = True
    epochs: int = 30
    batch_size: int = 128
    lr: float | None = None
    weight_decay: float = 0.05
    image_size: int = 224
    workers: int = 8
    max_views: int | None = None
    balanced_loss: bool = False
    grayscale: bool = False
    use_cache: bool = False
    patience: int = 6
    seed: int = 0
    tag: str = ""
    limit_train: int | None = None  # debug only: subsample the train split

    def resolved_lr(self) -> float:
        return self.lr if self.lr is not None else (3e-4 if not self.pretrained else 1e-4)

    def resolved_tag(self) -> str:
        if self.tag:
            return self.tag
        return (f"{self.backbone}_{'pretrained' if self.pretrained else 'scratch'}"
                f"_{self.split}" + ("_gray" if self.grayscale else ""))

    def fingerprint(self) -> str:
        """Hash of the settings that make two runs incomparable.

        Guards resume: silently continuing a checkpoint trained at a different
        learning rate or on a different split would produce a result that looks
        valid and is not.
        """
        keys = ("split", "backbone", "pretrained", "epochs", "batch_size",
                "weight_decay", "image_size", "max_views", "balanced_loss",
                "grayscale", "seed", "limit_train")
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        blob += f"|lr={self.resolved_lr()}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick_precision(device: torch.device) -> tuple[torch.dtype, bool]:
    """(autocast dtype, needs GradScaler).

    bf16 needs Ampere or newer. A T4 is Turing and has no bf16 support at all, so
    hardcoding bf16 would break there; detect instead of assume. fp16 needs loss
    scaling, bf16 does not.
    """
    if device.type != "cuda":
        return torch.float32, False
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
    except Exception:
        pass
    return torch.float16, True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> dict:
    model.eval()
    preds = {"C": [], "A": []}
    trues = {"C": [], "A": []}
    for x, y, _ in progress(loader, desc="eval", unit=" batch", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            logits = model(x)
        for k in preds:
            preds[k].append(logits[k].float().argmax(1).cpu())
            trues[k].append(y[k])
    out = {}
    for k in preds:
        p = torch.cat(preds[k]).numpy()
        t = torch.cat(trues[k]).numpy()
        keep = t >= 0  # -100 marks a label that was absent from training
        out[f"{k}_acc"] = float(accuracy_score(t[keep], p[keep]))
        out[f"{k}_macro_f1"] = float(
            f1_score(t[keep], p[keep], average="macro", zero_division=0))
        out[f"{k}_n"] = int(keep.sum())
    return out


def build_loaders(cfg: TrainConfig, device: torch.device):
    train_df, val_df, test_df = rd.build_frames(cfg.split, cfg.max_views)
    if cfg.limit_train:
        train_df = train_df.sample(min(cfg.limit_train, len(train_df)),
                                   random_state=cfg.seed).reset_index(drop=True)
        val_df = val_df.sample(min(cfg.limit_train // 4 or 1, len(val_df)),
                               random_state=cfg.seed).reset_index(drop=True)
        test_df = test_df.sample(min(cfg.limit_train // 4 or 1, len(test_df)),
                                 random_state=cfg.seed).reset_index(drop=True)
    maps = rd.fit_label_maps(train_df)
    cls = rd.CachedRenderDataset if cfg.use_cache else rd.RenderDataset

    def mk(frame, train):
        return cls(frame, maps, train, cfg.image_size, cfg.grayscale)

    common = dict(num_workers=cfg.workers, pin_memory=device.type == "cuda",
                  persistent_workers=cfg.workers > 0)
    train_dl = DataLoader(mk(train_df, True), batch_size=cfg.batch_size,
                          shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(mk(val_df, False), batch_size=cfg.batch_size * 2,
                        shuffle=False, **common)
    test_dl = DataLoader(mk(test_df, False), batch_size=cfg.batch_size * 2,
                         shuffle=False, **common)
    return (train_df, val_df, test_df), maps, (train_dl, val_dl, test_dl)


def train(cfg: TrainConfig, ckpt_dir: Path, on_epoch_end=None) -> dict:
    """Train with per-epoch checkpointing. Resumes automatically if a
    checkpoint for the same configuration is present in `ckpt_dir`.

    `on_epoch_end(epoch, metrics)` runs after each checkpoint is written - the
    Modal app uses it to commit the Volume so the checkpoint survives preemption.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.resolved_tag()
    last_path = ckpt_dir / f"{tag}_last.pt"
    best_path = ckpt_dir / f"{tag}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, needs_scaler = pick_precision(device)
    set_seed(cfg.seed)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[train] {gpu_name} | autocast {amp_dtype} "
          f"| GradScaler {needs_scaler}", flush=True)

    (train_df, _, _), maps, (train_dl, val_dl, test_dl) = build_loaders(cfg, device)
    rd.save_label_maps(maps, ckpt_dir / f"{tag}_labels.json")
    print(f"[train] train {len(train_df):,} images over "
          f"{train_df.domain_id.nunique():,} domains | "
          f"C={len(maps['C'])} A={len(maps['A'])}", flush=True)

    model = MultiHeadResNet(len(maps["C"]), len(maps["A"]),
                            backbone=cfg.backbone, pretrained=cfg.pretrained).to(device)

    weights = {}
    if cfg.balanced_loss:
        for k in ("C", "A"):
            c = rd.class_counts(train_df, maps, k).to(device)
            weights[k] = torch.where(
                c > 0, c.sum() / (len(c) * c.clamp(min=1)), torch.ones_like(c))
    crit = {k: nn.CrossEntropyLoss(weight=weights.get(k), ignore_index=-100)
            for k in ("C", "A")}

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.resolved_lr(),
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.resolved_lr(),
        total_steps=cfg.epochs * max(1, len(train_dl)), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    start_epoch, best, best_epoch, bad, history = 1, -1.0, -1, 0, []
    fp = cfg.fingerprint()
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("fingerprint") != fp:
            print(f"[train] checkpoint fingerprint {ck.get('fingerprint')} != {fp}; "
                  "configuration changed, starting fresh", flush=True)
        else:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if ck.get("scaler") is not None and needs_scaler:
                scaler.load_state_dict(ck["scaler"])
            start_epoch = ck["epoch"] + 1
            best, best_epoch, bad = ck["best"], ck["best_epoch"], ck["bad"]
            history = ck.get("history", [])
            if ck.get("rng"):
                restore_rng(ck["rng"])
            print(f"[train] RESUMED from epoch {ck['epoch']} "
                  f"(best macro-F1 {best:.4f} @ {best_epoch})", flush=True)

    if start_epoch > cfg.epochs:
        print("[train] checkpoint already at final epoch", flush=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y, _ in progress(train_dl, desc=f"epoch {epoch}/{cfg.epochs}",
                                unit=" batch"):
            x = x.to(device, non_blocking=True)
            y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                logits = model(x)
                # architecture is the target of record; class is a cheap
                # auxiliary task that regularises the shared trunk
                loss = crit["A"](logits["A"], y["A"]) + 0.3 * crit["C"](logits["C"], y["C"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        val = evaluate(model, val_dl, device, amp_dtype)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "lr": sched.get_last_lr()[0], "secs": time.time() - t0, **val}
        history.append(row)
        print(f"[train] epoch {epoch:3d} loss {row['train_loss']:.4f} | "
              f"val A acc {val['A_acc']:.4f} macroF1 {val['A_macro_f1']:.4f} | "
              f"{row['secs']:.0f}s", flush=True)

        # select on macro-F1: 21 of 43 architectures have under 100 training
        # domains, so accuracy rewards collapsing onto the crowded ones
        score = val["A_macro_f1"]
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                        "label_maps": maps, "config": asdict(cfg)}, best_path)
        else:
            bad += 1

        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if needs_scaler else None,
            "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad": bad,
            "history": history, "rng": rng_state(), "fingerprint": fp,
            "config": asdict(cfg), "label_maps": maps,
        }, last_path)

        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        if bad >= cfg.patience:
            print(f"[train] early stop at epoch {epoch} "
                  f"(best {best:.4f} @ {best_epoch})", flush=True)
            break

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, test_dl, device, amp_dtype)

    result = {"tag": tag, "best_epoch": ck["epoch"], "gpu": gpu_name,
              "amp_dtype": str(amp_dtype), **{f"test_{k}": v for k, v in test.items()},
              **{f"cfg_{k}": v for k, v in asdict(cfg).items()}}
    (ckpt_dir / f"{tag}_result.json").write_text(json.dumps(result, indent=2),
                                                 encoding="utf-8")
    pd.DataFrame(history).to_parquet(ckpt_dir / f"{tag}_history.parquet", index=False)

    print(f"\n=== TEST (single-view, split {cfg.split}) ===")
    print(f"  architecture : acc {test['A_acc']:.4f} | macro-F1 {test['A_macro_f1']:.4f}"
          f" | n={test['A_n']:,}")
    print(f"  class        : acc {test['C_acc']:.4f} | macro-F1 {test['C_macro_f1']:.4f}")
    if cfg.split == "split_homsf":
        da = test["A_acc"] - GATE["accuracy"]
        df1 = test["A_macro_f1"] - GATE["macro_f1"]
        print(f"  gate         : acc {GATE['accuracy']:.3f} | macro-F1 {GATE['macro_f1']:.3f}")
        print(f"  delta        : acc {da:+.4f} | macro-F1 {df1:+.4f}")
        print(f"  verdict      : "
              f"{'CLEARS' if (da > 0 and df1 > 0) else 'MIXED' if (da > 0 or df1 > 0) else 'BELOW'}")
    return result

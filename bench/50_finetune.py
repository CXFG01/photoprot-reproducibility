"""Multi-positive contrastive fine-tuning. Views of one PDB entry are positives.

Two stages, run in order, each checkpointed and resumable:

  stage A  frozen backbone + projection head, trained on the CACHED embeddings.
           Costs minutes, not hours, because the backbone never runs. This is
           the honest first question: is the frozen representation already
           good enough once it is re-projected for this task?

  stage B  unfreeze the last --unfreeze transformer blocks and train on pixels,
           with a much smaller LR on the backbone than on the head.

POSITIVES ARE KEYED ON pdb_id, not object_id. 1abc_au and 1abc_asm1 are the
same protein, and the evaluation groups candidates by pdb_id, so training on
object_id would optimise a different metric than the one reported.

NO HORIZONTAL FLIPS. Alpha helices are right-handed; a mirrored render is a
physically impossible protein. It is the default first augmentation in most
vision pipelines and it is wrong here.

Checkpoints are written atomically (.tmp then rename), so an interrupted run
never leaves a truncated checkpoint that would fail to load on resume.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from training_support import PublicationAugment, validation_panel, validate_checkpoint

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Head(nn.Module):
    def __init__(self, dim_in, dim_out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_in), nn.GELU(),
                                 nn.Linear(dim_in, dim_out))

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)


def supcon(z, labels, tau=0.07):
    """Multi-positive InfoNCE (SupCon, out-of-log form)."""
    sim = (z @ z.T) / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    npos = pos.sum(1)
    keep = npos > 0
    loss = -(logp * pos).sum(1)[keep] / npos[keep]
    return loss.mean()


def batch_sampler(groups, keys, P, K, rng, steps):
    """Each step: P entries x K views. Entries are sampled without replacement
    within a step, so every in-batch negative is a genuinely different protein."""
    for _ in range(steps):
        picked = rng.choice(len(keys), size=P, replace=False)
        idx, lab = [], []
        for j, gi in enumerate(picked):
            rows = groups[keys[gi]]
            take = rng.choice(len(rows), size=min(K, len(rows)),
                              replace=len(rows) < K)
            for t in take:
                idx.append(rows[t])
                lab.append(j)
        yield np.array(idx), np.array(lab)


def load_split_manifest():
    man = C.load_manifest(columns=["render_id", "pdb_id", "view", "shard",
                                   "member"])
    sp = C.load_splits()[["pdb_id", "split"]]
    man["pdb_l"] = man.pdb_id.str.lower()
    return man.merge(sp, left_on="pdb_l", right_on="pdb_id",
                     suffixes=("", "_s"))


def stage_a(a):
    dev = "cuda"
    files = sorted(glob.glob(f"{C.EMB}/{a.base_tag}/*.npz"))
    if not files:
        raise SystemExit(f"no embeddings at {C.EMB}/{a.base_tag}")
    ids, embs = [], []
    for f in files:
        z = np.load(f)
        C.validate_embeddings(z['emb'], f)
        ids.append(z["render_id"])
        embs.append(z["emb"])
    ids = np.concatenate(ids)
    embs = np.concatenate(embs).astype(np.float32)
    pos = {r: i for i, r in enumerate(ids)}

    man = load_split_manifest()
    tr = man[man.split == "train"]
    tr = tr[tr.render_id.isin(pos)]
    print(f"stage A train images {len(tr):,}  entries {tr.pdb_l.nunique():,}",
          flush=True)

    groups = {}
    for p, r in zip(tr.pdb_l.to_numpy(), tr.render_id.to_numpy()):
        groups.setdefault(p, []).append(pos[r])
    keys = sorted(groups)

    X = torch.from_numpy(embs).to(dev)
    head = Head(embs.shape[1], a.dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr_head, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps_a)
    rng = np.random.default_rng(a.seed)

    os.makedirs(C.CKPT, exist_ok=True)
    ck = f"{C.CKPT}/stage_a_last.pt"
    step0 = 0
    if os.path.exists(ck) and not a.fresh:
        s = torch.load(ck, map_location=dev)
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step0 = s["step"]
        print(f"resumed stage A from step {step0}", flush=True)

    t0 = time.time()
    for i, (idx, lab) in enumerate(
            batch_sampler(groups, keys, a.P, a.K, rng, a.steps_a - step0),
            step0 + 1):
        z = head(X[torch.from_numpy(idx).to(dev)])
        loss = supcon(z, torch.from_numpy(lab).to(dev), a.tau)
        if not torch.isfinite(loss):
            raise ValueError(f'Non-finite head training loss at step {i}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if i % 200 == 0:
            print(f"  A step {i}/{a.steps_a} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/max(i-step0,1):.3f}s/step)", flush=True)
        if i % a.ckpt_every == 0 or i == a.steps_a:
            tmp = ck + ".tmp"
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": i,
                        "dim_in": int(embs.shape[1]), "dim": a.dim}, tmp)
            os.replace(tmp, ck)

    out = f"{C.EMB}/{a.tag_a}"
    os.makedirs(out, exist_ok=True)
    head.eval()
    with torch.no_grad():
        chunks = [head(X[s:s + 8192]).cpu().numpy().astype(np.float16)
                  for s in range(0, len(X), 8192)]
    Z = np.concatenate(chunks)
    C.atomic_write_npz(f"{out}/all.npz", emb=Z, render_id=ids)
    print(f"stage A done -> {out}/all.npz dim={Z.shape[1]}", flush=True)


TRAIN_AUGMENTATION = 'publication_v1'


def backbone_blocks(model):
    """HF DINOv2 and DINOv3 expose their transformer blocks differently."""
    for owner in (getattr(model, 'encoder', None), getattr(model, 'model', None), model):
        if owner is not None:
            for attr in ('layer', 'layers'):
                blocks = getattr(owner, attr, None)
                if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
                    return blocks
    raise ValueError('Unsupported transformer block layout')


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, rows, size, train):
        import torchvision.transforms as T
        self.rows, self.size = rows, size
        if train and TRAIN_AUGMENTATION == 'legacy_ftb':
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
                T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomGrayscale(p=0.10),
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.20),
            ])
        elif train:
            self.tf = PublicationAugment(size)
        else:
            self.tf = T.Resize((size, size))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        shard, member = self.rows[i]
        im = Image.open(C.img_path(shard, member)).convert("RGB")
        im = self.tf(im)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        x = x.float().div_(255)
        return (x - MEAN) / STD


def stage_b(a):
    from transformers import AutoModel
    dev = "cuda"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = AutoModel.from_pretrained(a.model).to(dev)
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for blk in backbone_blocks(model)[-a.unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
            trainable.append(p)
    head = Head(model.config.hidden_size, a.dim).to(dev)
    print(f"stage B: {a.unfreeze} blocks unfrozen "
          f"({sum(p.numel() for p in trainable)/1e6:.0f}M params) + head",
          flush=True)

    opt = torch.optim.AdamW(
        [{"params": trainable, "lr": a.lr_backbone},
         {"params": head.parameters(), "lr": a.lr_head}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps_b)
    scaler = torch.amp.GradScaler("cuda")

    man = load_split_manifest()
    tr = man[man.split == "train"].reset_index(drop=True)
    rows = list(zip(tr.shard, tr.member))
    groups = {}
    for i, p in enumerate(tr.pdb_l.to_numpy()):
        groups.setdefault(p, []).append(i)
    keys = sorted(groups)
    print(f"stage B train images {len(tr):,} entries {len(keys):,}", flush=True)

    os.makedirs(C.CKPT, exist_ok=True)
    run_dir = f"{C.CKPT}/{a.run_name}"
    os.makedirs(run_dir, exist_ok=True)
    ck = f"{run_dir}/stage_b_last.pt"
    best_ck = f"{run_dir}/stage_b_best.pt"
    if a.fresh and (os.path.exists(ck) or os.path.exists(best_ck)):
        raise ValueError('Refusing to overwrite an existing run with --fresh; choose a new run_name')
    panel, panel_hash = validation_panel(man, a.val_entries, a.seed)
    protocol = {"panel_sha256": panel_hash, "entries": int(panel.pdb_l.nunique()),
                "query_view": 0, "gallery_views": "1..15", "selection_metric": "R@1",
                "aggregation": a.val_scoring, "split": "val"}
    best_score = -1.0
    history = []
    step0 = 0
    if os.path.exists(ck) and not a.fresh:
        s = torch.load(ck, map_location=dev)
        if s.get('validation_protocol') != protocol:
            raise ValueError('Resume validation protocol changed; use a new run_name')
        if s['config']['steps_b'] != a.steps_b:
            raise ValueError('Changing schedule length requires --init_checkpoint and a new run_name')
        for key in ('P', 'K', 'size', 'seed', 'model', 'dim', 'unfreeze', 'tau', 'lr_head', 'lr_backbone'):
            if s['config'][key] != getattr(a, key):
                raise ValueError(f'Resume config changed: {key}; use a new run_name')
        model.load_state_dict(s["model"])
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        step0 = s["step"]
        best_score = s.get('best_score', -1.0)
        history = s.get('validation_history', [])
        if 'rng_cpu' in s:
            torch.set_rng_state(s['rng_cpu'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in s['rng_cuda']])
        print(f"resumed stage B from step {step0}", flush=True)
    elif a.init_checkpoint:
        s = torch.load(a.init_checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(s['model']); head.load_state_dict(s['head'])
        print(f"Initialized weights from {a.init_checkpoint}; fresh optimizer/schedule", flush=True)
    C.atomic_write_json(f"{run_dir}/validation_panel.json", {**protocol, "render_ids": panel.render_id.tolist()})

    def save_checkpoint(path, step):
        tmp = path + '.tmp'
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "dim": a.dim,
                    "config": vars(a), "augmentation": TRAIN_AUGMENTATION,
                    "validation_protocol": protocol, "best_score": best_score,
                    "validation_history": history, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all()}, tmp)
        os.replace(tmp, path)

    def evaluate(step):
        nonlocal best_score
        # Validation must not advance training's random stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            metrics = validate_checkpoint(model, head, panel, ImgDS, a.size, a.workers, dev, a.val_scoring)
        history.append({'step': step, **metrics})
        if metrics['R@1'] > best_score:
            best_score = metrics['R@1']
            save_checkpoint(best_ck, step)
        C.atomic_write_json(f'{run_dir}/validation_history.json', {'protocol': protocol, 'history': history})
        print(f"VALIDATION step={step} R@1={metrics['R@1']:.4f} best={best_score:.4f}", flush=True)

    if not history:
        evaluate(step0)

    ds = ImgDS(rows, a.size, train=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=ListBatches(groups, keys, a, step0),
        num_workers=a.workers, pin_memory=True)

    rng_labels = loader.batch_sampler.labels
    model.train()
    head.train()
    t0 = time.time()
    step = step0
    for x in loader:
        step += 1
        lab = torch.from_numpy(rng_labels.pop(0)).to(dev)
        x = x.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feat = model(pixel_values=x).last_hidden_state[:, 0]
        z = head(feat.float())
        loss = supcon(z, lab, a.tau)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(trainable) + list(head.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 25 == 0:
            el = time.time() - t0
            per = el / max(step - step0, 1)
            print(f"  B step {step}/{a.steps_b} loss {loss.item():.4f} "
                  f"{per:.2f}s/step eta {(a.steps_b-step)*per/3600:.1f}h",
                  flush=True)
        if step % a.val_every == 0 or step >= a.steps_b:
            evaluate(step)
        if step % a.ckpt_every == 0 or step % a.val_every == 0 or step >= a.steps_b:
            save_checkpoint(ck, step)
            print(f"  checkpoint @ {step}", flush=True)
        if step >= a.steps_b:
            break

    selected = torch.load(best_ck, map_location=dev, weights_only=False)
    model.load_state_dict(selected['model']); head.load_state_dict(selected['head'])
    print(f"Selected checkpoint step={selected['step']} val R@1={selected['best_score']:.4f}", flush=True)
    if not a.skip_final_embed:
        embed_split(a, model, head, dev, ["test", "val"], man)


class ListBatches:
    """Batch sampler that also records each batch's positive labels."""

    def __init__(self, groups, keys, a, step0):
        rng = np.random.default_rng(a.seed + 1)
        self.batches, self.labels = [], []
        for step, (idx, lab) in enumerate(batch_sampler(groups, keys, a.P, a.K, rng,
                                      a.steps_b)):
            if step < step0:
                continue
            self.batches.append(idx.tolist())
            self.labels.append(lab)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@torch.no_grad()
def embed_split(a, model, head, dev, splits, man):
    model.eval()
    head.eval()
    out = f"{C.EMB}/{a.tag_b}"
    os.makedirs(out, exist_ok=True)
    sub = man[man.split.isin(splits)].reset_index(drop=True)
    ds = ImgDS(list(zip(sub.shard, sub.member)), a.size, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=256,
                                     num_workers=a.workers, pin_memory=True)
    chunks = []
    for x in dl:
        x = x.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feat = model(pixel_values=x).last_hidden_state[:, 0]
        chunks.append(head(feat.float()).cpu().numpy().astype(np.float16))
    Z = np.concatenate(chunks)
    C.atomic_write_npz(f"{out}/all.npz", emb=Z,
                       render_id=sub.render_id.to_numpy().astype("U32"))
    print(f"stage B embeddings -> {out}/all.npz {Z.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    ap.add_argument("--model", default="facebook/dinov2-large")
    ap.add_argument("--base_tag", default="dinov2l")
    ap.add_argument("--tag_a", default="dinov2l_headA")
    ap.add_argument("--tag_b", default="dinov2l_ftB")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--steps_a", type=int, default=4000)
    ap.add_argument("--steps_b", type=int, default=3000)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--run_name", default="stage_b_validated_v1")
    ap.add_argument("--init_checkpoint", help="Weights only; starts a fresh optimizer and schedule")
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_entries", type=int, default=512)
    ap.add_argument("--val_scoring", choices=['max','top3','top5'], default='top5')
    ap.add_argument("--skip_final_embed", action="store_true")
    a = ap.parse_args()
    if a.stage == 'b':
        if min(a.val_every, a.val_entries, a.ckpt_every, a.steps_b) <= 0:
            ap.error('Validation/checkpoint intervals and counts must be positive')
        if os.path.basename(a.run_name) != a.run_name or a.run_name in ('.', '..'):
            ap.error('run_name must be a directory name')
        if a.tag_b == 'dinov2l_ftB':
            a.tag_b = a.run_name
    (stage_a if a.stage == "a" else stage_b)(a)

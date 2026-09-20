"""Aggregate one query's similarities across each PDB's reference images."""
import torch


def build_pad_index(gpdb, n_pdb, dev):
    order = torch.argsort(gpdb)
    counts = torch.bincount(gpdb, minlength=n_pdb)
    starts = torch.cumsum(counts, 0) - counts
    ar = torch.arange(int(counts.max()), device=dev)
    valid = ar[None, :] < counts[:, None]
    flat = (starts[:, None] + ar[None, :]).clamp(max=len(gpdb)-1)
    return order[flat], valid


def aggregate(sim, pad_idx, pad_mask, rule):
    if rule not in ('max', 'top3', 'top5'):
        raise ValueError(f'Unknown scoring rule: {rule}')
    g = sim[:, pad_idx.reshape(-1)].reshape(len(sim), *pad_idx.shape)
    g = g.masked_fill(~pad_mask[None], -torch.inf)
    if rule == 'max':
        return g.max(-1).values
    top = g.topk(min(int(rule[3:]), g.shape[-1]), dim=-1).values
    valid = torch.isfinite(top)
    return torch.where(valid.any(-1), top.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1),
                       torch.full(top.shape[:-1], -torch.inf, device=g.device))

"""Rerun cached retrieval baselines on an identical held-out test gallery.

No training, feature extraction, or test-set hyperparameter selection. Gallery
statistics alone standardise silhouette features. Ties use alphabetical PDB ID,
matching the website rather than optimistic tied ranks. Writes a fresh report.
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as C
from retrieval_scoring import aggregate, build_pad_index


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    out = Path(C.RESULTS) / 'web_baselines_v1'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    man = C.load_manifest(columns=['render_id', 'pdb_id', 'view'])
    splits = C.load_splits()
    test_pdb = set(splits.loc[splits.split == 'test', 'pdb_id'].str.lower())
    man['pdb_id'] = man.pdb_id.str.lower()
    sub = man[man.pdb_id.isin(test_pdb)].sort_values('render_id').reset_index(drop=True)
    assert sub.render_id.is_unique
    target = {r: i for i, r in enumerate(sub.render_id)}
    sources = {}

    def load_features(paths):
        result = None
        seen = np.zeros(len(sub), dtype=bool)
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                ids, emb = z['render_id'], z['emb']
                take = [(j, target[r]) for j, r in enumerate(ids) if r in target]
                if result is None:
                    result = np.empty((len(sub), emb.shape[1]), np.float32)
                if take:
                    src, dst = np.array(take).T
                    assert not seen[dst].any(), f'Duplicate IDs in {path}'
                    result[dst] = emb[src]
                    seen[dst] = True
            sources[str(path.relative_to(Path(C.ROOT)))] = digest(path)
        assert seen.all(), 'Missing test embeddings'
        C.validate_embeddings(result)
        return result

    baseline_path = Path(C.EMB) / 'baselines_test.npz'
    with np.load(baseline_path, allow_pickle=False) as z:
        pos = {r: i for i, r in enumerate(z['render_id'])}
        assert len(pos) == len(z['render_id'])
        ix = np.array([pos[r] for r in sub.render_id])
        sil, col, hog = (z[k][ix] for k in ['sil', 'col', 'hog'])
    sources[str(baseline_path.relative_to(Path(C.ROOT)))] = digest(baseline_path)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'description': 'Exact PDB retrieval from held-out rendered images; cached feature rerun, not retraining.',
        'primary_protocol': 'view0_query', 'primary_scoring': 'top5',
        'protocols': {}, 'methods': [],
        'methodology': {
            'split': 'Existing sequence-component test split (40% sequence identity clustering).',
            'scoring': 'Cosine similarity; mean of the five strongest reference views per PDB. Also records max-view scoring for comparison.',
            'tie_policy': 'Descending score, then alphabetical PDB ID; no optimistic tied ranks.',
            'silhouette_normalisation': 'Mean and standard deviation fitted on gallery images only, separately per protocol; colour weight 10 as in original baseline.',
            'zero_vectors': 'Handcrafted zero vectors remain zero and receive deterministic tied rankings; learned zero vectors are rejected.',
            'limitation': 'Test-only gallery, smaller than the live 836,399-image index. Synthetic rendered queries; not an estimate of accuracy on paper figures or of homology.',
            'selection': 'Existing checkpoints and scoring rule fixed before this rerun; no test-set tuning.',
        },
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()},
    }
    methods = [
        ('colour', 'Colour histogram', 'handcrafted'),
        ('silhouette', 'Silhouette', 'handcrafted'),
        ('hog', 'Edges (HOG)', 'handcrafted'),
        ('nuisance', 'Shape + colour', 'handcrafted'),
        ('frozen', 'DINOv2 · frozen', 'learned'),
        ('head', 'DINOv2 · trained head', 'learned'),
        ('finetuned', 'PhotoProt · fine-tuned', 'learned'),
    ]
    for key, label, family in methods:
        print(f'Loading {label}', flush=True)
        if key == 'frozen':
            features = load_features(sorted((Path(C.EMB) / 'dinov2l').glob('*.npz')))
        elif key == 'head':
            features = load_features([Path(C.EMB) / 'dinov2l_headA/all.npz'])
        elif key == 'finetuned':
            features = load_features([Path(C.EMB) / 'dinov2l_ftB/all.npz'])
        else:
            features = {'colour': col, 'hog': hog}.get(key)
        row = {'id': key, 'label': label, 'family': family, 'protocols': {}}
        for name, qviews, gviews in [('view0_query', [0], list(range(1, 16))), ('style_matched', [15], list(range(1, 15)))]:
            qi = np.flatnonzero(sub.view.isin(qviews))
            gi = np.flatnonzero(sub.view.isin(gviews))
            vocab = sorted(set(sub.pdb_id.iloc[gi]))
            vpos = {p: i for i, p in enumerate(vocab)}
            gp = torch.tensor([vpos[p] for p in sub.pdb_id.iloc[gi]], device='cuda')
            truth = torch.tensor([vpos[p] for p in sub.pdb_id.iloc[qi]], device='cuda')
            pad, mask = build_pad_index(gp, len(vocab), 'cuda')
            report['protocols'][name] = {'queries': len(qi), 'gallery_images': len(gi), 'candidate_pdbs': len(vocab), 'query_views': qviews, 'gallery_views': gviews}
            if key in ('silhouette', 'nuisance'):
                stdsil = (sil - sil[gi].mean(0)) / (sil[gi].std(0) + 1e-6)
                features = stdsil if key == 'silhouette' else np.concatenate([stdsil, col * 10], axis=1)
            assert np.isfinite(features).all(), key
            f = torch.nn.functional.normalize(torch.from_numpy(features).to('cuda'), dim=1)
            gallery = f[gi].T.contiguous()
            ranks = {rule: [] for rule in ['top5', 'max']}
            with torch.inference_mode():
                for start in range(0, len(qi), 64):
                    sim = f[qi[start:start+64]] @ gallery
                    t = truth[start:start+64]
                    for rule in ranks:
                        scores = aggregate(sim, pad, mask, rule)
                        own = scores.gather(1, t[:, None])
                        # Alphabetical ordering resolves exact score ties.
                        rank = 1 + (scores > own).sum(1) + ((scores == own) & (torch.arange(len(vocab), device='cuda')[None] < t[:, None])).sum(1)
                        ranks[rule].extend(rank.cpu().tolist())
            row['protocols'][name] = {}
            for rule, rank in ranks.items():
                arr = np.array(rank)
                row['protocols'][name][rule] = {**{f'R@{k}': float((arr <= k).mean()) for k in [1, 5, 10, 20]}, 'MRR': float((1 / arr).mean()), 'median_rank': float(np.median(arr))}
            pd.DataFrame({'render_id': sub.render_id.iloc[qi].to_numpy(), 'pdb_id': sub.pdb_id.iloc[qi].to_numpy(), **ranks}).to_csv(out / f'{key}_{name}_ranks.csv', index=False)
            print(label, name, json.dumps(row['protocols'][name]['top5']), flush=True)
            del f, gallery, sim, scores, pad, mask
        report['methods'].append(row)
        C.atomic_write_json(str(out / 'report.json'), report)
        del features
    sources['data/meta/manifest.parquet'] = digest(C.MANIFEST)
    sources['data/meta/splits_component.parquet'] = digest(C.SPLITS)
    sources['bench/82_web_baselines.py'] = digest(__file__)
    report.update(status='complete', elapsed_seconds=time.monotonic() - started, source_sha256=sources)
    C.atomic_write_json(str(out / 'report.json'), report)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()

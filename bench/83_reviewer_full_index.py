"""Fresh held-out rankings against the deployed index, excluding query images."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from retrieval_scoring import aggregate, build_pad_index

ROOT = Path.home() / 'photoprot'
OUT = ROOT / 'data/results/reviewer_full_index_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    info = json.loads((ROOT / 'data/service/index.json').read_text())
    assert sha(ROOT / 'data/service/index.npz') == info['index_sha256']
    with np.load(ROOT / 'data/service/index.npz', allow_pickle=False) as z:
        ids, e, pdb = z['render_id'], z['emb'].astype(np.float32), z['pdb_id']
    assert len(set(ids)) == len(ids) == 836399 and np.isfinite(e).all()
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    assert (norms > 0).all()
    e /= norms
    G = torch.from_numpy(e).cuda(); del e
    man = pd.read_parquet(ROOT / 'data/meta/manifest.parquet').set_index('render_id').loc[ids]
    assert np.array_equal(man.pdb_id.str.lower().to_numpy(), np.char.lower(pdb))
    split = pd.read_parquet(ROOT / 'data/meta/splits_component.parquet').set_index('pdb_id')['split']
    is_test = man.pdb_id.str.lower().map(split).eq('test').to_numpy()
    vocab = sorted(set(np.char.lower(pdb))); pos = {p:i for i,p in enumerate(vocab)}
    group = torch.tensor([pos[p.lower()] for p in pdb], device='cuda')
    report = {'created_at':datetime.now(timezone.utc).isoformat(), 'status':'running',
              'index': info, 'protocols':{}, 'aggregation':'top5',
              'tie_policy':'Descending score, then alphabetical PDB ID',
              'note':'Cached embeddings from the actual service index; fresh rankings. All query-cohort images excluded. No encoder retraining or test tuning.',
              'script_sha256':sha(__file__), 'splits_sha256':sha(ROOT / 'data/meta/splits_component.parquet')}
    for name, view, excluded_views in [('view0_query', 0, [0]), ('style_matched', 15, [0, 15])]:
        start_time = time.monotonic()
        qi = np.flatnonzero(is_test & man.view.eq(view).to_numpy())
        excluded = is_test & man.view.isin(excluded_views).to_numpy()
        gi = np.flatnonzero(~excluded)
        assert not set(qi) & set(gi)
        gallery = G[gi].T.contiguous()
        pad, mask = build_pad_index(group[gi], len(vocab), 'cuda')
        rows = []
        for start in range(0, len(qi), 16):
            batch = qi[start:start+16]
            scores = aggregate(G[batch] @ gallery, pad, mask, 'top5')
            truth = group[batch]
            own = scores.gather(1, truth[:,None])
            vi = torch.arange(len(vocab), device='cuda')[None]
            ranks = 1 + ((scores > own) | ((scores == own) & (vi < truth[:,None]))).sum(1)
            for index, rank in zip(batch, ranks.cpu().tolist()):
                rows.append({'render_id':str(ids[index]), 'pdb_id':str(pdb[index]), 'rank':rank})
        ranks = np.array([r['rank'] for r in rows])
        report['protocols'][name] = {'queries':len(qi), 'gallery_images':len(gi),
             'candidate_pdbs':len(vocab), 'excluded_test_views':excluded_views,
             **{f'Top-{k}':float((ranks<=k).mean()) for k in [1,5,10]},
             'MRR':float((1/ranks).mean()), 'median_rank':float(np.median(ranks)),
             'elapsed_seconds':time.monotonic()-start_time}
        pd.DataFrame(rows).to_csv(OUT / f'{name}_ranks.csv', index=False)
        print(name, json.dumps(report['protocols'][name]), flush=True)
        del gallery, pad, mask, scores
    report['status'] = 'complete'
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

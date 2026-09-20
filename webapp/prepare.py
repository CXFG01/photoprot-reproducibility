"""Consolidate the verified DINOv2 index; package actual training-set examples."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('PHOTOPROT_ROOT', Path.home() / 'photoprot'))
OUT = ROOT / 'data/service'
EXAMPLE_PDBS = ['3i3w', '3bqz', '9ayl', '2lmk', '4r27', '3g7k']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'data/emb/dinov2l_ftB_full_corpus_v1'
    provenance = json.loads((cache / 'signature.json').read_text())
    assert sha(ROOT / 'data/ckpt/stage_b_last.pt') == provenance['checkpoint_sha256']
    assert sha(ROOT / 'data/meta/manifest.parquet') == provenance['manifest_sha256']
    man = pd.read_parquet(ROOT / 'data/meta/manifest.parquet')
    man['pdb_id'] = man.pdb_id.str.lower()
    files = [ROOT / 'data/emb/dinov2l_ftB/all.npz'] + sorted(cache.glob('missing_*.npz'))
    ids, emb = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            ids.append(z['render_id'].astype(str)); emb.append(z['emb'])
    ids, emb = np.concatenate(ids), np.concatenate(emb)
    assert len(ids) == len(set(ids)) == len(man)
    assert set(ids) == set(man.render_id)
    assert np.isfinite(emb).all() and (np.linalg.norm(emb.astype('float32'), axis=1) > 0).all()
    order = np.argsort(ids); ids, emb = ids[order], emb[order]
    aligned = man.set_index('render_id').loc[ids]
    tmp = OUT / 'index.tmp.npz'
    np.savez(tmp, render_id=ids, emb=emb, pdb_id=aligned.pdb_id.to_numpy(dtype='U4'))
    tmp.replace(OUT / 'index.npz')
    provenance.update(images=len(ids), pdb_entries=man.pdb_id.nunique(), objects=man.object_id.nunique(),
                      dimensions=int(emb.shape[1]), aggregation='top5', index_sha256=sha(OUT / 'index.npz'))
    (OUT / 'index.json').write_text(json.dumps(provenance, indent=2))
    splits = pd.read_parquet(ROOT / 'data/meta/splits_component.parquet')
    train = set(splits[splits.split == 'train'].pdb_id.str.lower())
    possible = man[(man.pdb_id.isin(train)) & (man.view == 0)].drop_duplicates('pdb_id')
    # Curated demonstrations verified against the live full-gallery top5 scorer.
    choices = EXAMPLE_PDBS
    examples = []; assets = OUT / 'examples'; assets.mkdir(exist_ok=True)
    for pdb in choices:
        row = possible[possible.pdb_id == pdb].iloc[0]
        source = ROOT / 'data/img' / row.shard.removesuffix('.tar') / row.member
        name = f'{pdb}{source.suffix}'
        shutil.copyfile(source, assets / name)
        examples.append({'id': pdb, 'pdb_id': pdb.upper(), 'image': f'/examples/{name}',
                         'filename': name, 'render_id': row.render_id, 'split': 'train',
                         'expected_top1': pdb.upper(), 'curated': True})
    first = man[(man.object_id == possible[possible.pdb_id == choices[0]].iloc[0].object_id)].sort_values('view')
    views = []
    for row in first.itertuples():
        source = ROOT / 'data/img' / row.shard.removesuffix('.tar') / row.member
        name = f'view-{row.view:02d}{source.suffix}'; shutil.copyfile(source, assets / name)
        views.append({'view': int(row.view), 'image': f'/examples/{name}'})
    (OUT / 'examples.json').write_text(json.dumps({'examples': examples, 'views': views}, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == '__main__':
    main()

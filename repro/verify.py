"""Recompute all reported Wild-v1 metrics without a GPU or third-party packages."""
import hashlib, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def subsets(rows):
    covered = [r for r in rows if r['best_source_rank'] is not None]
    mapped = [r for r in covered if r['mapping_status'] != 'panel_candidates' and 'rna_only' not in r['review_flags']]
    return {
        'confirmed_single_source_covered': [r for r in mapped if r['mapping_status']=='source_mapped' and len(r['pdb_ids'])==1],
        'source_mapped_or_multiple_depicted_covered': mapped,
        'all_candidate_covered_exploratory': covered,
        'all_queries_candidate_hits_including_absent': rows,
        'new_crops_candidate_covered_exploratory': [r for r in covered if not r['legacy_crop_id']],
        'previous_crops_candidate_covered': [r for r in covered if r['legacy_crop_id']],
    }

def metrics(rows):
    n=len(rows); ranks=[r['best_source_rank'] for r in rows]; valid=[r for r in ranks if r is not None]
    return {'n':n, **{f'hit_at_{k}':sum(r is not None and r<=k for r in ranks) for k in (1,5,10,20,50)},
            **{f'R@{k}':sum(r is not None and r<=k for r in ranks)/n if n else None for k in (1,5,10,20,50)},
            'MRR':sum(1/r for r in valid)/n if n else None,
            'median_rank':statistics.median(valid) if valid else None}

def main():
    base=ROOT/'benchmarks/wild-v1'
    data=json.loads((base/'predictions.json').read_text(encoding='utf-8'))
    manifest=json.loads((base/'query_manifest.json').read_text(encoding='utf-8'))
    rows=data['queries']; assert len(rows)==479
    assert [r['crop_id'] for r in rows]==[r['crop_id'] for r in manifest['crops']]
    for r in rows:
        top=r['top20']; assert len(top)==len({h['pdb_id'] for h in top})==20
        assert [h['rank'] for h in top]==list(range(1,21))
        assert top==sorted(top,key=lambda h:(-h['score'],h['pdb_id']))
        ranks=[s['rank'] for s in r['source_coverage'] if s['present']]
        assert r['best_source_rank']==(min(ranks) if ranks else None)
    result={name:metrics(rr) for name,rr in subsets(rows).items()}
    for name, calculated in result.items():
        for key,value in data['summary']['subsets'][name].items():
            if value is None: assert calculated[key] is None
            else: assert abs(calculated[key]-value)<1e-12,(name,key)
    assert hashlib.sha256((base/'query_embeddings.npz').read_bytes()).hexdigest()==json.loads((base/'validation.json').read_text())['query_embeddings_sha256']
    sums=ROOT/'SHA256SUMS.json'
    if sums.exists():
        for name, expected in json.loads(sums.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    print(json.dumps({'passed':True,'queries':479,'ranked_rows':9580,'subsets':result},indent=2))

if __name__=='__main__':main()

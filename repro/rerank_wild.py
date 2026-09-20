"""Recompute 479 query rankings against the released full production index."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from search import ROOT,rank
from verify import subsets,metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,default=ROOT/'downloads/index.npz')
    p.add_argument('--embeddings',type=Path,default=ROOT/'benchmarks/wild-v1/query_embeddings.npz')
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,default=ROOT/'reranked-wild.json')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a new path')
    torch.set_num_threads(4)
    original=json.loads((ROOT/'benchmarks/wild-v1/predictions.json').read_text(encoding='utf-8'))['queries']
    with np.load(a.embeddings,allow_pickle=False) as z:
        q=z['emb'].astype(np.float32);assert z['crop_id'].tolist()==[r['crop_id'] for r in original]
    assert q.shape==(479,256) and np.isfinite(q).all()
    calculated=rank(q,a.index,a.device,[r['pdb_ids'] for r in original])
    rank_matches=0; top20_matches=0;max_score_error=0
    for old,new in zip(original,calculated):
        rank_matches+=old['best_source_rank']==new['best_source_rank']
        top20_matches+=[r['pdb_id'] for r in old['top20']]==[r['pdb_id'] for r in new['top20']]
        if [r['pdb_id'] for r in old['top20']]==[r['pdb_id'] for r in new['top20']]:
            max_score_error=max(max_score_error,max(abs(x['score']-y['score']) for x,y in zip(old['top20'],new['top20'])))
    rows=[{**old,**new} for old,new in zip(original,calculated)]
    out={'comparison':{'matching_source_ranks':rank_matches,'matching_top20_lists':top20_matches,'max_score_error_matching_lists':max_score_error},'subsets':{k:metrics(v) for k,v in subsets(rows).items()},'queries':rows}
    a.output.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out['comparison'],indent=2))
    if rank_matches!=479 or top20_matches!=479:raise SystemExit('Rank differences detected; inspect hardware/precision and saved output.')
if __name__=='__main__':main()

"""Select max/top3/top5 on validation; apply to all wild single-image queries.

Reads the existing full-corpus and query caches only. No encoder or training.
"""
import copy
import importlib.util
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import retrieval_scoring as scoring

ROOT = Path.home() / 'photoprot'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    wild = module('wild', '60_wild.py')
    out = ROOT / 'data/results/wild_v1_cached_scoring'
    out.mkdir(exist_ok=True)
    cache = ROOT / 'data/emb/dinov2l_ftB_full_corpus_v1'
    signature = json.loads((cache/'signature.json').read_text())
    old = json.loads((ROOT/'data/results/wild_v1_full_corpus/predictions.json').read_text())
    assert signature['checkpoint_sha256'] == old['provenance']['checkpoint_sha256']
    assert wild.sha(ROOT/'data/ckpt/stage_b_last.pt') == signature['checkpoint_sha256']
    assert wild.sha(ROOT/'data/meta/manifest.parquet') == signature['manifest_sha256']
    legacy = ROOT/'data/emb/dinov2l_ftB/all.npz'
    legacy_check = json.loads((cache/'legacy_cache_check.json').read_text())
    assert wild.sha(legacy) == legacy_check['legacy_file_sha256']
    ids, values = [], []
    for path in [legacy] + sorted(cache.glob('missing_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            ids.append(z['render_id'].astype(str)); values.append(z['emb'].astype(np.float32))
    ids = np.concatenate(ids); E = np.concatenate(values)
    del values
    order = np.argsort(ids); ids = ids[order]; E = E[order]
    assert np.isfinite(E).all()
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    man = pd.read_parquet(ROOT/'data/meta/manifest.parquet').set_index('render_id',drop=False)
    assert len(set(ids)) == len(ids) == len(man) and set(ids) == set(man.index)
    man = man.loc[ids].reset_index(drop=True)
    man['pdb_l'] = man.pdb_id.str.lower()
    splits = pd.read_parquet(ROOT/'data/meta/splits_component.parquet')
    splitmap = dict(zip(splits.pdb_id.str.lower(),splits.split))
    man['split'] = man.pdb_l.map(splitmap)
    vocab = sorted(man.pdb_l.unique()); pix = {p:i for i,p in enumerate(vocab)}
    entry_idx = torch.tensor([pix[p] for p in man.pdb_l],device='cuda')
    G = torch.tensor(E,device='cuda'); del E
    rules = ['max','top3','top5']
    def evaluate(split, selected_rules):
        query_mask = ((man.split == split) & (man.view == 0)).to_numpy()
        qi = np.flatnonzero(query_mask)
        gi = torch.tensor(np.flatnonzero(~query_mask),device='cuda')
        gg = G[gi]
        pad, mask = scoring.build_pad_index(entry_idx[gi],len(vocab),'cuda')
        all_ranks = {r:[] for r in selected_rules}
        for start in range(0,len(qi),16):
            batch = qi[start:start+16]
            sim = G[batch] @ gg.T
            truth = entry_idx[batch]
            for rule in selected_rules:
                scores = scoring.aggregate(sim,pad,mask,rule)
                ts = scores.gather(1,truth[:,None])
                # Alphabetical PDB tie break, consistent with wild rankings.
                ix = torch.arange(len(vocab),device='cuda')[None,:]
                ranks = ((scores > ts) | ((scores == ts) & (ix < truth[:,None]))).sum(1)+1
                all_ranks[rule].extend(ranks.cpu().tolist())
        result = {}
        for rule, ranks in all_ranks.items():
            r = np.array(ranks)
            result[rule] = {'R@1':float(np.mean(r<=1)), 'R@5':float(np.mean(r<=5)),
                            'R@10':float(np.mean(r<=10)), 'MRR':float(np.mean(1/r)),
                            'n_queries':len(r), 'n_gallery_images':len(gi)}
        print(split, json.dumps(result),flush=True)
        return result
    val = evaluate('val',rules)
    selected = max(rules,key=lambda r:(val[r]['R@1'],val[r]['MRR'],-rules.index(r)))
    selection = {'selection_split':'val','primary_metric':'R@1','tie_break':'MRR then simpler rule',
                 'selected_rule':selected,'validation':val,
                 'protocol':'One view-0 query at a time; full corpus minus all view-0 queries of evaluated split'}
    wild.json_write(out/'selection.json',selection)
    test = evaluate('test',list(dict.fromkeys(['max',selected])))
    selection['test'] = test
    wild.json_write(out/'selection.json',selection)
    with np.load(ROOT/'data/results/wild_v1_full_corpus/query_embeddings.npz') as z:
        assert list(z['crop_id']) == [q['crop_id'] for q in old['queries']]
        Q = z['emb'].astype(np.float32)
    Q /= np.linalg.norm(Q,axis=1,keepdims=True)
    pad, mask = scoring.build_pad_index(entry_idx,len(vocab),'cuda')
    all_records = {r:[] for r in rules}
    for i,q in enumerate(old['queries']):
        sim = torch.tensor(Q[i:i+1],device='cuda') @ G.T
        for rule in rules:
            scores = scoring.aggregate(sim,pad,mask,rule)[0].cpu().numpy()
            order = np.argsort(-scores,kind='stable')
            ranks = np.empty(len(vocab),dtype=int); ranks[order]=np.arange(1,len(vocab)+1)
            rec = copy.deepcopy(q); hits = []
            for vi in order[:10]:
                idx = torch.where(entry_idx==int(vi))[0]
                best = int(idx[sim[0,idx].argmax()])
                row = man.iloc[best]
                hits.append({'rank':len(hits)+1,'pdb_id':vocab[vi].upper(),'score':float(scores[vi]),
                    'best_image_cosine':float(sim[0,best]),'render_id':row.render_id,'object_id':row.object_id,
                    'view':int(row.view),'shard':row.shard,'member':row.member,'corpus_split':row['split']})
            sources = []
            for p in q['pdb_ids']:
                vi = pix.get(p.lower())
                sources.append({'pdb_id':p,'present':vi is not None,'rank':int(ranks[vi]) if vi is not None else None,
                                'score':float(scores[vi]) if vi is not None else None,'corpus_split':splitmap.get(p.lower(),'absent')})
            present = [s['rank'] for s in sources if s['rank'] is not None]
            rec.update(top10=hits,source_coverage=sources,best_source_rank=min(present) if present else None)
            all_records[rule].append(rec)
        # Verify the old report is reproduced before trusting changed scoring.
        assert [h['pdb_id'] for h in all_records['max'][-1]['top10']] == [h['pdb_id'] for h in q['top10']]
    records = all_records[selected]
    summary = wild.summarize(records)
    comparison = {r:wild.summarize(rs) for r,rs in all_records.items()}
    wild.json_write(out/'wild_comparison.json',comparison)
    provenance = {**old['provenance'],'score':f'exact cosine, {selected} over reference images per PDB',
                  'selection':selection,'new_encoder_inference':False,'training_performed':False,
                  'code_sha256':wild.sha(__file__)}
    wild.make_report(out,records,summary,provenance,ROOT/'data/wild/publication_crops_v1')
    page=(out/'gallery.html').read_text()
    page=page.replace('each entry receives its highest reference-image score.',
          f'entry scoring: {selected}, selected on validation. The displayed reference is the best individual image.')
    (out/'gallery.html').write_text(page)
    wild.json_write(out/'predictions.json',{'provenance':provenance,'summary':summary,'queries':records})
    wild.json_write(out/'summary.json',{'provenance':provenance,**summary})
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['crop_id','source_ids','best_source_rank','rank','pdb_id','score','best_image_cosine'])
        w.writeheader()
        for q in records:
            for h in q['top10']:
                w.writerow({'crop_id':q['crop_id'],'source_ids':';'.join(q['pdb_ids']),
                            'best_source_rank':q['best_source_rank'],**{k:h[k] for k in ['rank','pdb_id','score','best_image_cosine']}})
    wild.json_write(out/'validation.json',{'status':'passed','max_rankings_reproduced':74,
                        'gallery_images':len(man),'queries':len(records),'checkpoint_unchanged':
                        wild.sha(ROOT/'data/ckpt/stage_b_last.pt')==signature['checkpoint_sha256']})
    print('COMPLETE',selected,json.dumps(comparison),flush=True)


if __name__=='__main__':
    main()

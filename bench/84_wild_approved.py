"""Fresh approved publication-image queries against the frozen deployed DINOv2 index."""
from __future__ import annotations
import argparse
import copy
import csv
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path.home() / 'photoprot'
sys.path.insert(0, str(ROOT))
from webapp.server import Engine, decode_image
from retrieval_scoring import aggregate

spec = importlib.util.spec_from_file_location('wild_report', Path(__file__).with_name('60_wild.py'))
wild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wild)


def metrics(rows):
    ranks = [r['best_source_rank'] for r in rows]
    n = len(rows)
    return {'n': n, **{f'hit_at_{k}': sum(x is not None and x <= k for x in ranks) for k in (1,5,10,20)},
            **{f'R@{k}': sum(x is not None and x <= k for x in ranks)/n if n else None for k in (1,5,10,20)},
            'MRR': sum(1/x for x in ranks if x is not None)/n if n else None,
            'median_rank': float(np.median([x for x in ranks if x is not None])) if any(x is not None for x in ranks) else None}


def summarize(rows):
    covered = [r for r in rows if r['best_source_rank'] is not None]
    mapped = [r for r in covered if r['mapping_status'] != 'panel_candidates' and 'rna_only' not in r['review_flags']]
    strict = [r for r in mapped if r['mapping_status'] == 'source_mapped' and len(r['pdb_ids']) == 1]
    return {'n_queries': len(rows), 'n_source_figures': len({r['source_id'] for r in rows}),
            'n_any_source_present': len(covered), 'n_no_source_present': len(rows)-len(covered),
            'n_accession_unresolved': sum(r['mapping_status']=='panel_candidates' for r in rows),
            'n_rna_only_diagnostic': sum('rna_only' in r['review_flags'] for r in rows),
            'subsets': {'confirmed_single_source_covered': metrics(strict),
                        'source_mapped_or_multiple_depicted_covered': metrics(mapped),
                        'all_candidate_covered_exploratory': metrics(covered),
                        'all_queries_candidate_hits_including_absent': metrics(rows),
                        'new_crops_candidate_covered_exploratory': metrics([r for r in covered if not r['legacy_crop_id']]),
                        'previous_crops_candidate_covered': metrics([r for r in covered if r['legacy_crop_id']])},
            'interpretation': 'Crop approval is not PDB mapping confirmation. Candidate-hit metrics accept any listed source accession and can be optimistic. Missing targets cannot be exact-ID successes. RNA is scored but excluded from protein mapped subsets. No structural grading or calibrated probabilities.'}


@torch.inference_mode()
def run(args):
    out, queries_dir = ROOT/args.output, ROOT/args.queries
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predictions.json').exists():
        raise RuntimeError('Completed run already exists; choose a new output directory')
    t0=time.monotonic()
    torch.set_num_threads(4)
    qpath=queries_dir/'manifest.json'
    manifest=json.loads(qpath.read_text())
    queries=manifest['crops']
    assert queries and all(c['review_status']=='approved' for c in queries)
    assert len({c['crop_id'] for c in queries})==len(queries)
    for c in queries:
        for key in ('standardized','model_input'):
            assert wild.sha(queries_dir/c[key+'_file'])==c[key+'_sha256'],c['crop_id']
        with Image.open(queries_dir/c['standardized_file']) as master, Image.open(queries_dir/c['model_input_file']) as small:
            assert master.mode==small.mode=='RGB' and master.size==(512,512) and small.size==(224,224)
            assert master.resize((224,224),Image.Resampling.BILINEAR).tobytes()==small.tobytes()
    info=json.loads((ROOT/'data/service/index.json').read_text())
    assert info['aggregation']=='top5'
    assert wild.sha(ROOT/'data/service/index.npz')==info['index_sha256']
    assert wild.sha(ROOT/'data/meta/manifest.parquet')==info['manifest_sha256']
    provenance={'model':'DINOv2-L ft_b','checkpoint_sha256':info['checkpoint_sha256'],
        'index_sha256':info['index_sha256'],'query_manifest_sha256':wild.sha(qpath),
        'query_count':len(queries),'n_gallery_images':info['images'],'n_gallery_entries':info['pdb_entries'],
        'score':'exact cosine, mean of top five reference-image similarities per PDB',
        'tie_policy':'descending score then alphabetical PDB ID','preprocessing':info['preprocessing'],
        'new_encoder_inference':True,'training_performed':False,'score_rule_tuned_on_this_set':False,
        'script_sha256':wild.sha(__file__),'server_code_sha256':wild.sha(ROOT/'webapp/server.py'),
        'aggregation_code_sha256':wild.sha(ROOT/'bench/retrieval_scoring.py'),
        'report_code_sha256':wild.sha(ROOT/'bench/60_wild.py'),
        'splits_sha256':wild.sha(ROOT/'data/meta/splits_component.parquet'),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'gpu':torch.cuda.get_device_name(),'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'All user-approved crops, including RNA-only diagnostic. Existing mapping certainty retained. Development diagnostic; embedded labels retained.'}
    wild.json_write(out/'protocol.json',provenance)
    wild.json_write(out/'query_manifest.json',manifest)
    print('FROZEN',json.dumps(provenance),flush=True)
    engine=Engine()
    with np.load(ROOT/'data/service/index.npz',allow_pickle=False) as z:
        ids=z['render_id'].copy(); pdb=z['pdb_id'].copy()
    man=pd.read_parquet(ROOT/'data/meta/manifest.parquet').set_index('render_id').loc[ids]
    assert np.array_equal(np.char.lower(pdb),man.pdb_id.str.lower().to_numpy())
    split=pd.read_parquet(ROOT/'data/meta/splits_component.parquet')
    splitmap=dict(zip(split.pdb_id.str.upper(),split['split']))
    vocab=[p.upper() for p in engine.pdbs]; lookup={p:i for i,p in enumerate(vocab)}
    group=np.array([lookup[p.upper()] for p in pdb])
    image_order=np.argsort(group,kind='stable'); counts=np.bincount(group,minlength=len(vocab)); starts=np.cumsum(counts)-counts
    records=[]; embeddings=[]; validations=[]
    for i,c in enumerate(queries):
        image=decode_image((queries_dir/c['standardized_file']).read_bytes())
        arr=np.asarray(image.resize((224,224),Image.Resampling.BILINEAR)).copy()
        x=torch.from_numpy(arr).permute(2,0,1).float()/255
        x=((x-engine.mean)/engine.std).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            f=engine.model(pixel_values=x).last_hidden_state[:,0]
        q=engine.head(f.float())
        assert torch.isfinite(q).all()
        embeddings.append(q[0].cpu().numpy())
        sim=(q@engine.gallery.T)
        scores=aggregate(sim,engine.pad,engine.mask,'top5')[0].cpu().numpy()
        assert np.isfinite(scores).all()
        order=np.argsort(-scores,kind='stable'); ranks=np.empty(len(vocab),dtype=np.int32); ranks[order]=np.arange(1,len(vocab)+1)
        similarities=sim[0].cpu().numpy()
        hits=[]
        for vi in order[:20]:
            indices=image_order[starts[vi]:starts[vi]+counts[vi]]
            best=int(indices[np.argmax(similarities[indices])]); row=man.iloc[best]
            # Independent top-five calculation validates the deployed aggregation.
            expected=float(np.sort(similarities[indices])[-5:].mean())
            assert abs(expected-float(scores[vi]))<2e-6
            hits.append({'rank':len(hits)+1,'pdb_id':vocab[vi],'score':float(scores[vi]),
                'best_image_cosine':float(similarities[best]),'render_id':str(ids[best]),
                'object_id':str(row.object_id),'view':int(row.view),'shard':str(row.shard),'member':str(row.member),
                'corpus_split':splitmap.get(vocab[vi],'unknown')})
        sources=[]
        for source in c['pdb_ids']:
            vi=lookup.get(source.upper())
            sources.append({'pdb_id':source.upper(),'present':vi is not None,'rank':int(ranks[vi]) if vi is not None else None,
                'score':float(scores[vi]) if vi is not None else None,'corpus_split':splitmap.get(source.upper(),'absent')})
        present=[s['rank'] for s in sources if s['present']]
        rec=copy.deepcopy(c); rec.update(top10=hits[:10],top20=hits,source_coverage=sources,best_source_rank=min(present) if present else None)
        records.append(rec)
        with (out/'progress.jsonl').open('a') as stream: stream.write(json.dumps(rec)+'\n')
        if i in (0,len(queries)//2,len(queries)-1):
            service=engine.search((queries_dir/c['standardized_file']).read_bytes())
            assert [h['pdb_id'] for h in service['results']]==[h['pdb_id'] for h in hits]
            err=max(abs(a['score']-b['score']) for a,b in zip(service['results'],hits))
            assert err<2e-6
            validations.append({'crop_id':c['crop_id'],'engine_top20_equal':True,'max_score_error':err})
        if (i+1)%20==0 or i+1==len(queries):
            print(f'QUERIES {i+1}/{len(queries)} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    wild.npz_write(out/'query_embeddings.npz',crop_id=np.array([c['crop_id'] for c in queries]),emb=np.stack(embeddings))
    summary=summarize(records)
    wild.make_report(out,records,summary,provenance,queries_dir)
    page=(out/'gallery.html').read_text()
    page=page.replace('each entry receives its highest reference-image score.', 'each entry receives the mean of its top-five reference-image similarities. The displayed reference is its best individual image.')
    (out/'gallery.html').write_text(page)
    wild.json_write(out/'predictions.json',{'provenance':provenance,'summary':summary,'queries':records})
    wild.json_write(out/'summary.json',summary)
    with (out/'predictions.csv').open('w',newline='') as stream:
        fields=['crop_id','source_ids','mapping_status','best_source_rank','rank','pdb_id','score','best_image_cosine']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            for h in r['top20']:
                writer.writerow({'crop_id':r['crop_id'],'source_ids':';'.join(r['pdb_ids']),
                    'mapping_status':r['mapping_status'],'best_source_rank':r['best_source_rank'],
                    **{k:h[k] for k in ['rank','pdb_id','score','best_image_cosine']}})
    assert wild.sha(ROOT/'data/ckpt/stage_b_last.pt')==info['checkpoint_sha256']
    assert wild.sha(qpath)==provenance['query_manifest_sha256']
    wild.json_write(out/'validation.json',{'passed':True,'n_queries':len(records),'engine_checks':validations,
        'top20_independent_aggregation_checks':20*len(records),'query_image_hashes_checked':True,
        'index_hash_checked':True,'checkpoint_unchanged':True,'query_manifest_unchanged':True,
        'query_embeddings_sha256':wild.sha(out/'query_embeddings.npz'),'elapsed_seconds':time.monotonic()-t0})
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',default='data/wild/dinov2_approved_20260920')
    parser.add_argument('--output',default='data/results/wild_dinov2_approved_20260920')
    run(parser.parse_args())

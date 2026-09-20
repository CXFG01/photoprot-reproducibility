"""Single publication crop -> full rendered corpus -> ranked PDB entries.

Frozen stage-B checkpoint. All reference images are retained; no query-view
aggregation. Missing reference embeddings are cached once, in resumable chunks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "photoprot"
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
PREPROCESS = {"size": [224, 224], "resize": "PIL bilinear (matches stage-B T.Resize)",
              "center_crop": False, "query_augmentation": False,
              "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def json_write(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def npz_write(path, **arrays):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


class Head(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, din), nn.GELU(), nn.Linear(din, dout))

    def forward(self, x):
        z = self.net(x)
        return torch.nn.functional.normalize(z, dim=1)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        return (x - MEAN) / STD


@torch.inference_mode()
def encode(paths, model, head, batch, workers):
    dl = DataLoader(Images(paths), batch_size=batch, num_workers=workers,
                    pin_memory=True, shuffle=False)
    chunks = []
    for x in dl:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = model(pixel_values=x.cuda(non_blocking=True)).last_hidden_state[:, 0]
        z = head(f.float())
        chunks.append(z.cpu().numpy().astype(np.float16))
    return np.concatenate(chunks)


def render_path(row):
    return ROOT / "data/img" / str(row.shard).removesuffix(".tar") / str(row.member)


def full_gallery(man, model, head, signature, args):
    cache = ROOT / "data/emb/dinov2l_ftB_full_corpus_v1"
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "signature.json"
    if marker.exists():
        assert json.loads(marker.read_text()) == signature, "Gallery cache signature mismatch"
    else:
        json_write(marker, signature)
    base_path = ROOT / "data/emb/dinov2l_ftB/all.npz"
    base = np.load(base_path, allow_pickle=False)
    ids = base["render_id"].astype(str)
    values = base["emb"]
    assert len(ids) == len(set(ids))
    pos = {r: i for i, r in enumerate(ids)}
    ix = man.set_index("render_id", drop=False)
    assert set(ids) <= set(ix.index)
    # Check that the legacy cache was produced by this model and preprocessing.
    sample_ids = ids[np.linspace(0, len(ids)-1, 16, dtype=int)]
    sample = ix.loc[sample_ids]
    fresh = encode([render_path(r) for r in sample.itertuples()], model, head, args.batch, 0).astype(np.float32)
    old = values[[pos[r] for r in sample_ids]].astype(np.float32)
    cos = np.sum(fresh * old, axis=1) / (np.linalg.norm(fresh, axis=1)*np.linalg.norm(old, axis=1))
    assert float(cos.min()) > .9999, f"Legacy embedding validation failed: {cos.min()}"
    json_write(cache / "legacy_cache_check.json", {"sample_size": 16, "min_cosine": float(cos.min()),
               "legacy_file_sha256": sha(base_path), "checkpoint_sha256": signature["checkpoint_sha256"]})
    missing = man[~man.render_id.isin(pos)].reset_index(drop=True)
    print(f"GALLERY total={len(man):,} entries={man.pdb_id.nunique():,} reused={len(ids):,} missing={len(missing):,}", flush=True)
    all_ids, all_values = [ids], [values]
    chunk_size = 8192
    t0 = time.time()
    n_new = 0
    for start in range(0, len(missing), chunk_size):
        part = missing.iloc[start:start+chunk_size]
        expected = part.render_id.to_numpy().astype("U64")
        dst = cache / f"missing_{start:07d}.npz"
        if dst.exists():
            z = np.load(dst, allow_pickle=False)
            assert np.array_equal(z["render_id"], expected)
            emb = z["emb"]
            state = "reused"
        else:
            emb = encode([render_path(r) for r in part.itertuples()], model, head, args.batch, args.workers)
            npz_write(dst, render_id=expected, emb=emb)
            n_new += len(part)
            state = "encoded"
        assert emb.shape == (len(part), values.shape[1]) and np.isfinite(emb).all()
        all_ids.append(expected)
        all_values.append(emb)
        elapsed = time.time() - t0
        rate = n_new / max(elapsed, .001)
        remaining = len(missing) - start - len(part)
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"GALLERY {start+len(part):,}/{len(missing):,} {state} rate={rate:.1f}/s eta={eta:.1f}min", flush=True)
    ids = np.concatenate(all_ids)
    emb = np.concatenate(all_values).astype(np.float32)
    assert len(ids) == len(man) and len(set(ids)) == len(man)
    assert set(ids) == set(man.render_id)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    # Sort by render ID for stable ties and straightforward provenance.
    order = np.argsort(ids)
    ids, emb = ids[order], emb[order]
    return ix.loc[ids].reset_index(drop=True), emb


def summarize(records):
    eligible = [r for r in records if r["mapping_status"] != "panel_candidates"]
    covered = [r for r in eligible if r["best_source_rank"] is not None]
    clean = [r for r in covered if r["clean_query_eligible"]]
    subsets = {}
    for name, rs in [("source_mapped_or_overlay_covered",covered),("clean_flag_and_covered",clean)]:
        subsets[name] = {"n_queries": len(rs), **{f"source_hit_at_{k}": sum(r["best_source_rank"] <= k for r in rs) for k in [1,5,10]},
                         "note": "Exact source-ID diagnostic on covered queries only; overlays accept any listed depicted source. Not structural grading."}
    return {"n_queries": len(records), "n_source_figures": len({r["source_id"] for r in records}),
            "n_any_source_present": sum(r["best_source_rank"] is not None for r in records),
            "n_no_source_present": sum(r["best_source_rank"] is None for r in records),
            "n_accession_unresolved": sum(r["mapping_status"] == "panel_candidates" for r in records),
            "diagnostics": subsets}


def make_report(out, records, summary, provenance, query_root):
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    cards=[]
    for r in records:
        qname = f"q_{r['crop_id']}.png"
        shutil.copyfile(query_root / r["standardized_file"], assets / qname)
        r["report_query_image"] = "assets/"+qname
        tiles=[]
        for hit in r["top10"]:
            name=f"g_{hit['render_id']}.jpg"
            dest=assets/name
            if not dest.exists():
                with Image.open(ROOT / "data/img" / hit["shard"].removesuffix(".tar") / hit["member"]) as im:
                    ImageOps.contain(im.convert("RGB"),(320,320)).save(dest,quality=92)
            hit["report_image"]="assets/"+name
            match = hit["pdb_id"] in r["pdb_ids"]
            tiles.append(f'<div class="hit {"match" if match else ""}"><img loading="lazy" src="assets/{name}"><b>#{hit["rank"]} <a href="https://www.rcsb.org/structure/{hit["pdb_id"]}">{hit["pdb_id"]}</a></b><p>cosine {hit["score"]:.4f}</p><small>{html.escape(hit["object_id"])} · view {hit["view"]} · {hit["corpus_split"]}</small></div>')
        rank_text = str(r["best_source_rank"]) if r["best_source_rank"] is not None else "absent from gallery"
        cards.append(f'<section data-search="{html.escape(r["crop_id"]+" "+" ".join(r["pdb_ids"])+" "+" ".join(h["pdb_id"] for h in r["top10"]))}"><h2>{html.escape(r["crop_id"])}</h2><p>Source candidate(s): <b>{", ".join(r["pdb_ids"])}</b> · best source rank: <b>{rank_text}</b> · {html.escape(r["mapping_status"])} · {html.escape(r["representation"])}</p><p>{html.escape(r["notes"])}</p><div class="results"><div class="query"><img src="assets/{qname}"><b>Single query image</b></div>{"".join(tiles)}</div></section>')
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhotoProt wild-image predictions</title><style>body{{font:15px/1.5 system-ui;margin:24px;background:#f4f5f7;color:#17222d}}header{{max-width:1100px}}h1{{font-size:28px}}h2{{font-size:19px}}input{{padding:12px;width:min(90%,650px)}}section{{background:white;padding:18px;border:1px solid #ddd;border-radius:8px;margin:22px 0}}.results{{display:flex;gap:12px;overflow-x:auto;padding:10px 0}}.hit,.query{{flex:0 0 210px;background:#fafafa;padding:8px;border:1px solid #ddd}}.query{{border:2px solid #3264a8}}.match{{border:2px solid #299350}}img{{width:210px;height:210px;object-fit:contain;background:white}}p{{margin:8px 0}}a{{color:#1b5187}}[hidden]{{display:none}}</style><header><h1>One publication image → the full reference-image corpus</h1><p>{len(records)} independent crop queries, {provenance['n_gallery_images']:,} reference images, {provenance['n_gallery_entries']:,} PDB entries. Frozen stage-B checkpoint, exact cosine search; each entry receives its highest reference-image score. No query-view pooling.</p><p>Cosine similarity is not a confidence probability. Green borders mark listed source IDs. Missing source IDs are reported separately; alternatives may still be structurally relevant. Publication labels remain in the input.</p><p>{summary['n_any_source_present']} queries have at least one listed source in the gallery; {summary['n_no_source_present']} have none. Exact-ID scores do not measure structural similarity.</p><p><a href="predictions.json">Predictions + provenance</a> · <a href="predictions.csv">CSV</a> · <a href="summary.json">Summary</a></p><input id="filter" aria-label="Filter predictions" placeholder="Filter by crop or PDB ID"></header>{''.join(cards)}<script>document.querySelector('#filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();for(const s of document.querySelectorAll('section'))s.hidden=!s.dataset.search.toLowerCase().includes(q);}});</script></html>'''
    (out/"gallery.html").write_text(content,encoding="utf-8")
    sheets=out/"contact_sheets";sheets.mkdir(exist_ok=True)
    for start in range(0,len(records),8):
        batch=records[start:start+8]
        sheet=Image.new("RGB",(1440,len(batch)*270),"#e7e9ed")
        draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            paths=[r["report_query_image"]]+[h["report_image"] for h in r["top10"][:5]]
            captions=[r["crop_id"]]+[f"#{h['rank']} {h['pdb_id']}  {h['score']:.3f}" for h in r["top10"][:5]]
            for i,(p,cap) in enumerate(zip(paths,captions)):
                with Image.open(out/p) as im:
                    thumb=ImageOps.pad(im.convert("RGB"),(232,232),color="white")
                    sheet.paste(thumb,(i*240+4,j*270+4))
                draw.text((i*240+4,j*270+239),cap,fill="black")
            draw.text((4,j*270+254),"source: "+",".join(r["pdb_ids"]),fill="black")
        sheet.save(sheets/f"sheet_{start//8+1:02d}.jpg",quality=92)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    query_root = ROOT / "data/wild/publication_crops_v1"
    out = ROOT / "data/results/wild_v1_full_corpus"
    out.mkdir(parents=True,exist_ok=True)
    qmanifest_path=query_root/"manifest.json"
    qmanifest=json.loads(qmanifest_path.read_text(encoding="utf-8"))
    queries=qmanifest["crops"]
    assert len({r["crop_id"] for r in queries}) == len(queries)
    for r in queries:
        assert sha(query_root/r["standardized_file"])==r["standardized_sha256"],r["crop_id"]
    manifest_path=ROOT/"data/meta/manifest.parquet"
    man=pd.read_parquet(manifest_path,columns=["render_id","pdb_id","object_id","view","shard","member"])
    man["pdb_id"]=man.pdb_id.str.lower()
    assert man.render_id.is_unique
    ckpath=ROOT/"data/ckpt/stage_b_last.pt"
    checkpoint_hash=sha(ckpath)
    signature={"checkpoint_sha256":checkpoint_hash,"manifest_sha256":sha(manifest_path),"preprocessing":PREPROCESS,"gallery":"all_manifest_rows_all_views", "cache_chunk_rows":8192}
    from transformers import AutoModel
    model=AutoModel.from_pretrained("facebook/dinov2-large",local_files_only=True)
    ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"],strict=True)
    head=Head(model.config.hidden_size,ck["dim"])
    head.load_state_dict(ck["head"],strict=True)
    step=int(ck["step"])
    del ck
    model=model.cuda().eval();head=head.cuda().eval()
    print(f"MODEL step={step} checkpoint_sha256={checkpoint_hash} queries={len(queries)}",flush=True)
    qemb=encode([query_root/r["standardized_file"] for r in queries],model,head,args.batch,0).astype(np.float32)
    qemb/=np.linalg.norm(qemb,axis=1,keepdims=True)
    npz_write(out/"query_embeddings.npz",crop_id=np.array([r["crop_id"] for r in queries]),emb=qemb)
    gallery,emb=full_gallery(man,model,head,signature,args)
    del model,head
    torch.cuda.empty_cache()
    vocab=sorted(gallery.pdb_id.unique())
    pix={p:i for i,p in enumerate(vocab)}
    gidx=torch.tensor([pix[p] for p in gallery.pdb_id],device="cuda")
    G=torch.tensor(emb,device="cuda")
    sp=pd.read_parquet(ROOT/"data/meta/splits_component.parquet")
    splits=dict(zip(sp.pdb_id.str.lower(),sp.split))
    records=[]
    for start in range(0,len(queries),8):
        sim=torch.tensor(qemb[start:start+8],device="cuda")@G.T
        per=torch.full((len(sim),len(vocab)),-1e30,device="cuda")
        per.scatter_reduce_(1,gidx.expand(len(sim),-1),sim,reduce="amax",include_self=True)
        for j,q in enumerate(queries[start:start+8]):
            scores=per[j].cpu().numpy()
            order=np.argsort(-scores,kind="stable")
            ranks=np.empty(len(vocab),dtype=np.int64);ranks[order]=np.arange(1,len(vocab)+1)
            hits=[]
            for vi in order[:10]:
                p=vocab[vi]
                gi=torch.where(gidx==int(vi))[0]
                best=int(gi[sim[j,gi].argmax()])
                row=gallery.iloc[best]
                hits.append({"rank":len(hits)+1,"pdb_id":p.upper(),"score":float(scores[vi]),
                    "render_id":row.render_id,"object_id":row.object_id,"view":int(row.view),
                    "shard":row.shard,"member":row.member,"corpus_split":splits.get(p,"unknown")})
            sources=[]
            for p in q["pdb_ids"]:
                vi=pix.get(p.lower())
                sources.append({"pdb_id":p,"present":vi is not None,"rank":int(ranks[vi]) if vi is not None else None,
                                "score":float(scores[vi]) if vi is not None else None,"corpus_split":splits.get(p.lower(),"absent")})
            valid=[s["rank"] for s in sources if s["rank"] is not None]
            rec={**q,"source_coverage":sources,"best_source_rank":min(valid) if valid else None,"top10":hits}
            records.append(rec)
            print(f"QUERY {q['crop_id']} top1={hits[0]['pdb_id']} score={hits[0]['score']:.4f} source_rank={rec['best_source_rank']}",flush=True)
    summary=summarize(records)
    provenance={**signature,"checkpoint_step":step,"query_manifest_sha256":sha(qmanifest_path),
                "n_gallery_images":len(gallery),"n_gallery_entries":len(vocab),"n_gallery_objects":int(gallery.object_id.nunique()),
                "gallery_view_counts":{str(k):int(v) for k,v in gallery.view.value_counts().sort_index().items()},
                "model":"facebook/dinov2-large + saved stage-B head", "score":"exact L2-normalized cosine; max over all images of each PDB entry", "single_query_image":True,
                "source_labels_used_for_ranking":False,"query_view_aggregation":False,"code_sha256":sha(__file__)}
    make_report(out,records,summary,provenance,query_root)
    json_write(out/"predictions.json",{"provenance":provenance,"summary":summary,"queries":records})
    json_write(out/"summary.json",{"provenance":provenance,**summary})
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["crop_id","source_ids","best_source_rank","rank","pdb_id","score","render_id","object_id","view","corpus_split"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            for h in r["top10"]:
                w.writerow({"crop_id":r["crop_id"],"source_ids":";".join(r["pdb_ids"]),"best_source_rank":r["best_source_rank"],**{k:h[k] for k in fields[3:]}})
    assert sha(ckpath)==checkpoint_hash,"Checkpoint changed during inference"
    assert len(records)==len(queries) and all(len(r["top10"])==10 for r in records)
    json_write(out/"validation.json",{"status":"passed","queries":len(records),"reference_images":len(gallery),
               "all_manifest_rows_indexed":len(gallery)==len(man),"query_hashes_verified":True,"checkpoint_unchanged":True,
               "one_image_per_query":True,"no_query_view_pooling":True,"source_labels_used_for_ranking":False})
    print("COMPLETE "+json.dumps(summary),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch",type=int,default=256)
    ap.add_argument("--workers",type=int,default=10)
    run(ap.parse_args())

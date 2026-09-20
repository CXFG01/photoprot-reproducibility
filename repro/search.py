"""Portable frozen inference and exact full-index ranking; no hosted API required."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
from transformers import AutoConfig, AutoModel
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
from retrieval_scoring import aggregate,build_pad_index

class Head(nn.Module):
    def __init__(self,dim,out):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,out))
    def forward(self,x):return nn.functional.normalize(self.net(x),dim=1)

def verified(path):
    records={r['name']:r for r in json.loads((ROOT/'artifacts/manifest.json').read_text())}
    with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==records[path.name]['sha256'],f'Artifact differs from frozen release: {path}'

@torch.inference_mode()
def encode(paths,checkpoint,device):
    verified(checkpoint)
    model=AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'configs/dinov2-config.json'))
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True)
    head=Head(model.config.hidden_size,state['dim']);head.load_state_dict(state['head'],strict=True)
    del state
    model.to(device).eval();head.to(device).eval()
    mean=torch.tensor([.485,.456,.406]).view(3,1,1);std=torch.tensor([.229,.224,.225]).view(3,1,1)
    result=[]
    for path in paths:
        with Image.open(path) as original:
            image=ImageOps.exif_transpose(original).convert('RGBA')
            bg=Image.new('RGBA',image.size,'white');bg.alpha_composite(image)
            arr=np.asarray(bg.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).copy()
        x=((torch.from_numpy(arr).permute(2,0,1).float()/255-mean)/std).unsqueeze(0).to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.startswith('cuda')):
            feat=model(pixel_values=x).last_hidden_state[:,0]
        result.append(head(feat.float())[0].cpu().numpy())
    return np.stack(result)

@torch.inference_mode()
def rank(queries,index,device,source_ids=None):
    verified(index)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(index,allow_pickle=False) as z:
        e=z['emb'].astype(np.float32);pdb=np.char.upper(z['pdb_id'])
    assert np.isfinite(e).all()
    e/=np.linalg.norm(e,axis=1,keepdims=True)
    gallery=torch.from_numpy(e).to(device);del e
    vocab=sorted(set(pdb.tolist()));lookup={p:i for i,p in enumerate(vocab)}
    groups=torch.tensor([lookup[p] for p in pdb],device=device)
    pad,mask=build_pad_index(groups,len(vocab),device)
    rows=[]
    for i,q in enumerate(queries):
        scores=aggregate(torch.as_tensor(q,device=device).unsqueeze(0)@gallery.T,pad,mask,'top5')[0].cpu().numpy()
        order=np.argsort(-scores,kind='stable');ranks=np.empty(len(vocab),dtype=np.int32);ranks[order]=np.arange(1,len(vocab)+1)
        row={'top20':[{'rank':j+1,'pdb_id':vocab[k],'score':float(scores[k])} for j,k in enumerate(order[:20])]}
        if source_ids is not None:
            row['source_coverage']=[{'pdb_id':s,'present':s in lookup,'rank':int(ranks[lookup[s]]) if s in lookup else None} for s in source_ids[i]]
            found=[s['rank'] for s in row['source_coverage'] if s['present']]
            row['best_source_rank']=min(found) if found else None
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path)
    p.add_argument('--artifacts',type=Path,default=ROOT/'downloads');p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4)
    q=encode([a.image],a.artifacts/'stage_b_last.pt',a.device)
    print(json.dumps(rank(q,a.artifacts/'index.npz',a.device)[0],indent=2))
if __name__=='__main__':main()

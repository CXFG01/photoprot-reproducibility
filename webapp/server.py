"""Single-worker GPU image retrieval service. Uploads are processed in memory."""
import asyncio
from collections import OrderedDict
import csv
from contextlib import asynccontextmanager
import hashlib
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import secrets
import sys
import time
import warnings

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from transformers import AutoModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.environ.get('PHOTOPROT_ROOT', Path.home() / 'photoprot'))
DATA = ROOT / 'data/service'
STATIC = Path(__file__).parent / 'static'
sys.path.insert(0, str(ROOT / 'bench'))
from retrieval_scoring import aggregate, build_pad_index

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
LOG = logging.getLogger('photoprot')
for extension, content_type in [('.webp','image/webp'),('.png','image/png'),('.jpg','image/jpeg'),('.js','text/javascript')]:
    mimetypes.add_type(content_type, extension)


class Head(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)


def decode_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            if im.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Please use a PNG, JPEG or WebP image.')
            if im.width * im.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError('Please use an image under 20 megapixels.')
            im.load(); im = ImageOps.exif_transpose(im)
            if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
                rgba = im.convert('RGBA'); bg = Image.new('RGBA', rgba.size, 'white')
                bg.alpha_composite(rgba); im = bg.convert('RGB')
            else:
                im = im.convert('RGB')
            return im
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError('This image could not be read. Use a PNG, JPEG or WebP under 20 megapixels.')


class Engine:
    def __init__(self):
        torch.set_num_threads(4)
        self.info = json.loads((DATA / 'index.json').read_text())
        h = hashlib.sha256()
        with (ROOT / 'data/ckpt/stage_b_last.pt').open('rb') as f:
            for part in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(part)
        if h.hexdigest() != self.info['checkpoint_sha256']:
            raise RuntimeError('Checkpoint/index provenance mismatch')
        with np.load(DATA / 'index.npz', allow_pickle=False) as z:
            e = z['emb'].astype(np.float32); pdb = z['pdb_id']; ids = z['render_id']
        if len(ids) != self.info['images'] or len(set(ids)) != len(ids) or not np.isfinite(e).all():
            raise RuntimeError('Invalid retrieval index')
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        if (norms <= 0).any(): raise RuntimeError('Zero embedding in index')
        e /= norms
        self.gallery = torch.from_numpy(e).cuda()
        self.pdbs = sorted(set(pdb.tolist()))
        lookup = {p:i for i,p in enumerate(self.pdbs)}
        groups = torch.tensor([lookup[p] for p in pdb], device='cuda')
        self.pad, self.mask = build_pad_index(groups, len(self.pdbs), 'cuda')
        self.model = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True)
        state = torch.load(ROOT / 'data/ckpt/stage_b_last.pt', map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)
        self.head = Head(self.model.config.hidden_size, state['dim'])
        self.head.load_state_dict(state['head'], strict=True)
        del state
        self.model.cuda().eval(); self.head.cuda().eval()
        self.mean = torch.tensor([.485, .456, .406]).view(3,1,1)
        self.std = torch.tensor([.229, .224, .225]).view(3,1,1)
        LOG.warning('Ready: %s images, %s PDB entries', len(ids), len(self.pdbs))

    @torch.inference_mode()
    def search(self, data):
        t = time.perf_counter()
        image = decode_image(data)
        arr = np.asarray(image.resize((224,224), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(arr).permute(2,0,1).float() / 255
        x = ((x-self.mean)/self.std).unsqueeze(0).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            features = self.model(pixel_values=x).last_hidden_state[:,0]
        query = self.head(features.float())
        if not torch.isfinite(query).all(): raise RuntimeError('Non-finite query embedding')
        scores = aggregate(query @ self.gallery.T, self.pad, self.mask, 'top5')[0]
        # Alphabetical PDB tie breaking follows the sorted vocabulary.
        order = torch.argsort(scores, descending=True, stable=True)[:20].cpu().tolist()
        values = scores.cpu().tolist()
        return {'results': [{'rank': i+1, 'pdb_id': self.pdbs[k].upper(), 'score': values[k],
                             'pdb_url': f'https://www.rcsb.org/structure/{self.pdbs[k].upper()}',
                             'image_url': f'https://cdn.rcsb.org/images/structures/{self.pdbs[k]}_assembly-1.jpeg'}
                            for i,k in enumerate(order)],
                'search_seconds': round(time.perf_counter()-t, 3), 'gallery_images': self.info['images'],
                'gallery_entries': self.info['pdb_entries'], 'aggregation': 'top5', 'model': 'DINOv2-L · ft_b'}


@asynccontextmanager
async def lifespan(app):
    app.state.engine = await asyncio.to_thread(Engine)
    app.state.gpu = asyncio.Lock()
    app.state.pending = 0
    app.state.metadata = {}
    app.state.exports = OrderedDict()
    app.state.remote_slots = asyncio.Semaphore(8)
    app.state.client = httpx.AsyncClient(timeout=12, follow_redirects=False)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware('http')
async def headers(request, call_next):
    if request.method == 'POST':
        size = request.headers.get('content-length')
        if size and (not size.isdigit() or int(size) > MAX_BYTES):
            return Response('Image exceeds the 10 MB limit.', status_code=413)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.url.path.startswith('/api/search'): response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/api/health')
def health():
    return {'status':'ready', 'images':app.state.engine.info['images'],
            'pdb_entries': app.state.engine.info['pdb_entries'], 'pending':app.state.pending}


@app.get('/api/catalog')
def catalog():
    examples = json.loads((DATA / 'examples.json').read_text())
    return {**examples, 'stats': {k:app.state.engine.info[k] for k in ['images','pdb_entries','objects','dimensions']}}


async def metadata(pdb):
    cache = app.state.metadata
    if pdb in cache: return cache[pdb]
    result = {'pdb_id':pdb, 'title':f'PDB {pdb}', 'method':'', 'resolution':None, 'organisms':[]}
    try:
        async with app.state.remote_slots:
            query = '{entry(entry_id:"' + pdb + '"){struct{title} exptl{method} rcsb_entry_info{resolution_combined} polymer_entities{rcsb_entity_source_organism{ncbi_scientific_name}}}}'
            res = await app.state.client.get('https://data.rcsb.org/graphql', params={'query':query})
            res.raise_for_status(); entry=res.json()['data']['entry']
            result.update(title=entry['struct']['title'], method=', '.join(x['method'] for x in entry.get('exptl') or []),
                          resolution=(entry.get('rcsb_entry_info',{}).get('resolution_combined') or [None])[0],
                          organisms=sorted({s['ncbi_scientific_name'] for e in entry.get('polymer_entities') or []
                                            for s in e.get('rcsb_entity_source_organism') or [] if s.get('ncbi_scientific_name')}))
            cache[pdb] = result
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        result['metadata_unavailable'] = True
    return result


@app.get('/api/entries')
async def entries(ids: str):
    pdbs = ids.upper().split(',')
    if len(pdbs) > 20 or any(not re.fullmatch(r'[A-Z0-9]{4}', p) for p in pdbs):
        raise HTTPException(400, 'Supply up to 20 valid PDB IDs.')
    return await asyncio.gather(*(metadata(p) for p in pdbs))


@app.get('/api/structure/{pdb}')
async def structure(pdb: str):
    pdb=pdb.upper()
    if not re.fullmatch(r'[A-Z0-9]{4}',pdb): raise HTTPException(400,'Invalid PDB ID.')
    folder=DATA/'structures'; folder.mkdir(exist_ok=True); path=folder/f'{pdb}.cif'
    if path.exists(): return FileResponse(path, media_type='chemical/x-cif')
    try:
        async with app.state.remote_slots, app.state.client.stream('GET',f'https://files.rcsb.org/download/{pdb}.cif') as response:
            response.raise_for_status(); chunks=[]; size=0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 12*1024*1024: raise HTTPException(413,'This structure is too large for the embedded viewer. Open it on RCSB.')
                chunks.append(chunk)
        data=b''.join(chunks)
        # Do not cache arbitrary unlimited public structures.
        if pdb.lower() in app.state.engine.pdbs or pdb=='1UBQ':
            path.write_bytes(data)
        return Response(data,media_type='chemical/x-cif',headers={'Cache-Control':'public, max-age=86400'})
    except httpx.HTTPError:
        raise HTTPException(502,'RCSB is unavailable. Please use the RCSB link.')


@app.post('/api/search')
async def search(request: Request):
    if app.state.pending >= 6: raise HTTPException(429,'Search is busy. Please try again in a moment.')
    app.state.pending += 1
    try:
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>MAX_BYTES: raise HTTPException(413,'Please upload an image smaller than 10 MB.')
        if not body: raise HTTPException(400,'Please select an image first.')
        async with app.state.gpu:
            try:
                result = await asyncio.to_thread(app.state.engine.search, bytes(body))
                token = secrets.token_urlsafe(24)
                app.state.exports[token] = (time.monotonic(), result['results'])
                while len(app.state.exports) > 128: app.state.exports.popitem(last=False)
                result['download_url'] = f'/api/export/{token}'
                return result
            except ValueError as exc: raise HTTPException(400,str(exc))
    finally:
        app.state.pending -= 1


@app.get('/api/export/{token}')
def export(token: str):
    item = app.state.exports.get(token)
    if not item or time.monotonic()-item[0]>3600:
        raise HTTPException(404,'This export has expired. Run the search again.')
    out=io.StringIO(); writer=csv.writer(out)
    writer.writerow(['rank','pdb_id','similarity','title','organisms','rcsb_url'])
    def safe(value):
        text=str(value)
        return "'"+text if text.startswith(('=','+','-','@')) else text
    for row in item[1]:
        meta=app.state.metadata.get(row['pdb_id'],{})
        writer.writerow([row['rank'],row['pdb_id'],row['score'],safe(meta.get('title','')),
                         safe('; '.join(meta.get('organisms',[]))),row['pdb_url']])
    return Response(out.getvalue(),media_type='text/csv',headers={
        'Content-Disposition':'attachment; filename="photoprot-results.csv"', 'Cache-Control':'no-store'})


app.mount('/examples', StaticFiles(directory=DATA/'examples'), name='examples')
app.mount('/', StaticFiles(directory=STATIC, html=True), name='website')

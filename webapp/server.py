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
from webapp.protection import Protection

ROOT = Path(os.environ.get('PHOTOPROT_ROOT', Path.home() / 'photoprot'))
DATA = ROOT / 'data/service'
STATIC = Path(__file__).parent / 'static'
sys.path.insert(0, str(ROOT / 'bench'))
from retrieval_scoring import aggregate, build_pad_index

MAX_BYTES = 10 * 1024 * 1024
UPLOAD_TIMEOUT = 15
QUEUE_TIMEOUT = 10
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
        with (DATA / 'index.npz').open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=self.info['index_sha256']:
                raise RuntimeError('Retrieval index checksum mismatch')
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
        state = torch.load(ROOT / 'data/ckpt/stage_b_last.pt', map_location='cpu', weights_only=True)
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
    app.state.metadata = OrderedDict()
    app.state.metadata_tasks = {}
    app.state.structures = OrderedDict()
    app.state.structure_bytes = 0
    app.state.structure_tasks = {}
    app.state.allowed_pdbs = {p.upper() for p in app.state.engine.pdbs} | {'1UBQ'}
    app.state.exports = OrderedDict()
    app.state.remote_slots = asyncio.Semaphore(8)
    app.state.client = httpx.AsyncClient(timeout=12, follow_redirects=False,
                                       limits=httpx.Limits(max_connections=8,max_keepalive_connections=8))
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(Protection)


@app.get('/api/health')
def health():
    return {'status':'ready', 'images':app.state.engine.info['images'],
            'pdb_entries': app.state.engine.info['pdb_entries'], 'pending':app.state.pending}


@app.get('/api/catalog')
def catalog():
    examples = json.loads((DATA / 'examples.json').read_text())
    return {**examples, 'stats': {k:app.state.engine.info[k] for k in ['images','pdb_entries','objects','dimensions']}}


async def fetch_metadata(pdb):
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
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        result['metadata_unavailable'] = True
    return result


async def metadata(pdb):
    cache=app.state.metadata
    if pdb in cache:
        expires,result=cache.pop(pdb)
        if expires>time.monotonic():
            cache[pdb]=(expires,result);return result
    tasks=app.state.metadata_tasks
    if pdb not in tasks:
        if len(tasks)>=32:
            return {'pdb_id':pdb,'title':f'PDB {pdb}','organisms':[],'metadata_unavailable':True}
        async def fetch():
            try:
                try:
                    async with asyncio.timeout(20):result=await fetch_metadata(pdb)
                except TimeoutError:
                    result={'pdb_id':pdb,'title':f'PDB {pdb}','organisms':[],'metadata_unavailable':True}
                cache[pdb]=(time.monotonic()+(60 if result.get('metadata_unavailable') else 86400),result)
                while len(cache)>2048:cache.popitem(last=False)
                return result
            finally:tasks.pop(pdb,None)
        tasks[pdb]=asyncio.create_task(fetch())
    return await asyncio.shield(tasks[pdb])


@app.get('/api/entries')
async def entries(ids: str):
    pdbs = ids.upper().split(',')
    if len(pdbs) > 20 or any(not re.fullmatch(r'[A-Z0-9]{4}', p) for p in pdbs):
        raise HTTPException(400, 'Supply up to 20 valid PDB IDs.')
    if any(p not in app.state.allowed_pdbs for p in pdbs):
        raise HTTPException(404,'PDB is not in this collection.')
    return await asyncio.gather(*(metadata(p) for p in pdbs))


@app.get('/api/structure/{pdb}')
async def structure(pdb: str):
    pdb=pdb.upper()
    if not re.fullmatch(r'[A-Z0-9]{4}',pdb): raise HTTPException(400,'Invalid PDB ID.')
    if pdb not in app.state.allowed_pdbs:raise HTTPException(404,'PDB is not in this collection.')
    cache=app.state.structures
    if pdb in cache:
        data=cache.pop(pdb);cache[pdb]=data
    else:
        tasks=app.state.structure_tasks
        if pdb not in tasks:
            if len(tasks)>=4:raise HTTPException(503,'Structure viewer is busy. Please try again shortly.')
            async def fetch():
                try:
                    try:
                        async with asyncio.timeout(20):data=await fetch_structure(pdb)
                    except TimeoutError:raise HTTPException(504,'Structure download timed out.')
                    while cache and app.state.structure_bytes+len(data)>64*1024*1024:
                        _,old=cache.popitem(last=False);app.state.structure_bytes-=len(old)
                    cache[pdb]=data;app.state.structure_bytes+=len(data)
                    return data
                finally:tasks.pop(pdb,None)
            tasks[pdb]=asyncio.create_task(fetch())
        data=await asyncio.shield(tasks[pdb])
    return Response(data,media_type='chemical/x-cif',headers={'Cache-Control':'public, max-age=86400'})


async def fetch_structure(pdb):
    try:
        async with app.state.remote_slots, app.state.client.stream('GET',f'https://files.rcsb.org/download/{pdb}.cif') as response:
            response.raise_for_status(); chunks=[]; size=0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 12*1024*1024: raise HTTPException(413,'This structure is too large for the embedded viewer. Open it on RCSB.')
                chunks.append(chunk)
        data=b''.join(chunks)
        return data
    except httpx.HTTPError:
        raise HTTPException(502,'RCSB is unavailable. Please use the RCSB link.')


@app.post('/api/search')
async def search(request: Request):
    if os.environ.get('PHOTOPROT_SEARCH_ENABLED','1')!='1':
        raise HTTPException(503,'Search is temporarily paused.')
    if app.state.pending >= 3: raise HTTPException(429,'Search is busy. Please try again in a moment.',headers={'Retry-After':'5'})
    app.state.pending += 1
    try:
        body=bytearray()
        try:
            async with asyncio.timeout(UPLOAD_TIMEOUT):
                async for chunk in request.stream():
                    if len(body)+len(chunk)>MAX_BYTES: raise HTTPException(413,'Please upload an image smaller than 10 MB.')
                    body.extend(chunk)
        except TimeoutError:raise HTTPException(408,'Upload timed out. Please try again.')
        if not body: raise HTTPException(400,'Please select an image first.')
        try:await asyncio.wait_for(app.state.gpu.acquire(),timeout=QUEUE_TIMEOUT)
        except TimeoutError:raise HTTPException(503,'Search queue is busy. Please try again shortly.',headers={'Retry-After':'5'})
        try:
            try:
                # Cancellation must not release the GPU lock while its thread is
                # still running. Keep ownership until real work has finished.
                work=asyncio.create_task(asyncio.to_thread(app.state.engine.search,bytes(body)))
                try:result=await asyncio.shield(work)
                except asyncio.CancelledError:
                    while not work.done():
                        try:await asyncio.shield(work)
                        except asyncio.CancelledError:continue
                        except Exception:break
                    if work.done() and not work.cancelled():work.exception()
                    raise
                token = secrets.token_urlsafe(24)
                now=time.monotonic()
                while app.state.exports and next(iter(app.state.exports.values()))[0]<now-3600:
                    app.state.exports.popitem(last=False)
                app.state.exports[token] = (time.monotonic(), result['results'])
                while len(app.state.exports) > 128: app.state.exports.popitem(last=False)
                result['download_url'] = f'/api/export/{token}'
                return result
            except ValueError as exc: raise HTTPException(400,str(exc))
        finally:app.state.gpu.release()
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
        meta=app.state.metadata.get(row['pdb_id'],(0,{}))[1]
        writer.writerow([row['rank'],row['pdb_id'],row['score'],safe(meta.get('title','')),
                         safe('; '.join(meta.get('organisms',[]))),row['pdb_url']])
    return Response(out.getvalue(),media_type='text/csv',headers={
        'Content-Disposition':'attachment; filename="photoprot-results.csv"', 'Cache-Control':'no-store'})


app.mount('/examples', StaticFiles(directory=DATA/'examples'), name='examples')
app.mount('/', StaticFiles(directory=STATIC, html=True), name='website')

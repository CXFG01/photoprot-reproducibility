"""Build pixel-preserving DINOv2 inputs and serve a local persistent review UI."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tarfile
import threading
import time
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'wild_test/protein_ribbon_paper_collection/protein_ribbon_paper_collection'
OUT = ROOT/'wild_test/dinov2_review_v1'
MANIFEST = OUT/'manifest.json'
LOCK = threading.Lock()
MEAN, STD = [.485,.456,.406], [.229,.224,.225]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def white_rgb(im):
    rgba = im.convert('RGBA')
    background = Image.new('RGBA', im.size, (255,255,255,255))
    return Image.alpha_composite(background, rgba).convert('RGB')


def transform(im):
    rgb = white_rgb(im)
    scale = 464/max(rgb.size)
    size = tuple(max(1, round(d*scale)) for d in rgb.size)
    fitted = rgb.resize(size, Image.Resampling.LANCZOS)
    offset = ((512-size[0])//2, (512-size[1])//2)
    canvas = Image.new('RGB', (512,512), 'white')
    canvas.paste(fitted, offset)
    model = canvas.resize((224,224), Image.Resampling.BILINEAR)
    return canvas, model, {'canvas_size':[512,512], 'resized_size':list(size),
        'padding_left_top':list(offset), 'long_side':464, 'resampler':'Pillow LANCZOS',
        'background_rgb':[255,255,255], 'upscaled':scale>1,
        'rotation_degrees':0, 'horizontal_flip':False}


def render(crop):
    path = SOURCE/crop['source_image_file']
    if sha(path) != crop['source_image_sha256']:
        raise ValueError('Source changed: '+crop['source_id'])
    with Image.open(path) as source:
        source.load()
        w,h = source.size
        box = crop['box_source_xyxy']
        if len(box)!=4 or any(type(x) is not int for x in box) or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
            raise ValueError('Crop rectangle is outside source image')
        native = source.crop(box)
        if native.mode not in ('1','L','LA','P','RGB','RGBA','I','I;16'):
            native = native.convert('RGB')
        native.save(OUT/crop['native_file'])
        canvas, model, spec = transform(native)
        canvas.save(OUT/crop['standardized_file'])
        model.save(OUT/crop['model_input_file'])
        crop['native_dimensions'] = list(native.size)
        crop['standardization'] = spec
        crop['source_dimensions'] = [w,h]
    for key in ('native','standardized','model_input'):
        crop[key+'_sha256'] = sha(OUT/crop[key+'_file'])
    crop['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def save_manifest(m):
    m['n_crops'] = len(m['crops'])
    m['review_counts'] = dict(Counter(c['review_status'] for c in m['crops']))
    write_json(MANIFEST,m)
    fields = ['crop_id','source_id','review_status','mapping_status','pdb_ids','review_flags','box_source_xyxy','native_file','standardized_file','model_input_file','source_image_sha256','standardized_sha256','model_input_sha256','notes']
    with (OUT/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in c.items()} for c in m['crops'])


def contact_sheets(m):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    for start in range(0,len(m['crops']),40):
        cs=m['crops'][start:start+40]
        sheet=Image.new('RGB',(1600,math.ceil(len(cs)/8)*230),'#e6e9ec')
        d=ImageDraw.Draw(sheet)
        for j,c in enumerate(cs):
            x,y=j%8*200,j//8*230
            with Image.open(OUT/c['standardized_file']) as im:
                sheet.paste(im.resize((196,196)),(x+2,y+2))
            d.text((x+4,y+201),f"{start+j+1:03d} {c['crop_id'][:23]}",font=font,fill='black')
            d.text((x+4,y+216),(', '.join(c['review_flags']) or 'pending review')[:25],font=font,fill='#884422')
        sheet.save(OUT/'contact_sheets'/f'sheet_{start//40+1:02d}.jpg',quality=90)


def build():
    if MANIFEST.exists():
        raise SystemExit('Dataset already exists; use --verify, --serve or --export. Refusing to overwrite review decisions.')
    for folder in ('native','rgb512','dinov2_224','sources','contact_sheets'):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((SOURCE/'index.csv').open(encoding='utf-8-sig')))
    audit=json.loads((ROOT/'results/wild_collection_audit_20260920/audit.json').read_text())
    hash_by_id={r['id']:r['sha256'] for r in audit['images']}
    reuse={r['crop_id']:r['source_ids'][0] for r in audit['reusable_crops']}
    by_id={r['id']:r for r in rows}
    with tarfile.open(ROOT/'results/wild_transfer_v1.tar') as archive:
        old=json.load(archive.extractfile('data/wild/publication_crops_v1/manifest.json'))
    specs=[]
    for old_crop in old['crops']:
        if old_crop['crop_id'] in reuse:
            specs.append((by_id[reuse[old_crop['crop_id']]],old_crop['box_source_xyxy'],old_crop,[]))
    order=json.loads((ROOT/'results/wild_crop_inspection/order.json').read_text(encoding='utf-8'))
    annotations=json.loads((ROOT/'scripts/wild_crop_boxes.json').read_text())
    assert sorted(b[0] for b in annotations['boxes'])==list(range(len(order)))
    for b in annotations['boxes']:
        row=order[b[0]]
        with Image.open(SOURCE/row['image_file']) as im:
            w,h=im.size
        box=[math.floor(b[1]*w/1000),math.floor(b[2]*h/1000),math.ceil(b[3]*w/1000),math.ceil(b[4]*h/1000)]
        specs.append((row,box,None,b[5:]))
    crops=[]
    sources=[]
    missing_pdb={r['pdb_id'] for r in audit['summary']['structure_issues']}
    duplicates={s:ids for ids in audit['summary']['duplicate_image_groups'] for s in ids}
    for row in rows:
        sources.append({**row,'source_sha256':hash_by_id.get(row['id']),
                        'source_preview_file':'sources/'+row['id']+'.jpg' if row['image_file'] else None})
        if row['image_file']:
            with Image.open(SOURCE/row['image_file']) as im:
                thumb=ImageOps.contain(white_rgb(im),(1500,1500))
                thumb.save(OUT/'sources'/(row['id']+'.jpg'),quality=90)
    for row,box,old_crop,flags in specs:
        old_id=old_crop['crop_id'] if old_crop else None
        crop_id=row['id']+'__'+(old_id.split('__',1)[1] if old_id else 'primary')
        pdbs=old_crop['pdb_ids'] if old_crop else [s.upper() for s in re.split(r'[;\s,]+',row['pdb_ids'].strip()) if s]
        flags=list(flags or (old_crop.get('review_flags',[]) if old_crop else []))
        if any(p in missing_pdb for p in pdbs): flags.append('coordinates_unavailable')
        if row['pdb_status']!='verified': flags.append('source_mapping_'+row['pdb_status'])
        if row['structure_type']!='experimental': flags.append('structure_'+row['structure_type'])
        if row['id'] in duplicates: flags.append('duplicate_source_image')
        if len(pdbs)>1 and not old_crop: flags.append('panel_pdb_assignment_needed')
        crop={
            'crop_id':crop_id,'source_id':row['id'],'legacy_crop_id':old_id,
            'origin':'reused_verified_rectangle' if old_crop else 'new_visual_selection',
            'source_image_file':row['image_file'],'source_image_sha256':hash_by_id[row['id']],
            'source_preview_file':'sources/'+row['id']+'.jpg',
            'box_source_xyxy':box,'initial_box_source_xyxy':box.copy(),
            'native_file':'native/'+crop_id+'.png','standardized_file':'rgb512/'+crop_id+'.png',
            'model_input_file':'dinov2_224/'+crop_id+'.png',
            'pdb_ids':pdbs,'source_pdb_ids':row['pdb_ids'],
            'mapping_status':old_crop['mapping_status'] if old_crop else 'panel_candidates',
            'review_status':'excluded' if 'rna_only' in flags else 'pending',
            'review_flags':sorted(set(flags)), 'annotations_retained':True,
            'clean_query_eligible':False,
            'representation':old_crop['representation'] if old_crop else (next((f for f in flags if f in ['surface','density','backbone','overlay','rna_only','local_detail']),'cartoon')),
            'granularity':old_crop['granularity'] if old_crop else 'needs_review',
            'protein_name':row['protein_name'],'paper_title':row['paper_title'],
            'source_url':row['source_url'],'figure_url':row['figure_url'],'doi':row['doi'],
            'grouping_key':row['doi'] or row['source_url'],'duplicate_source_ids':duplicates.get(row['id'],[]),
            'source_panel_mapping':row['figure_label']+' '+row['panel'],
            'pdb_status':row['pdb_status'],'structure_type':row['structure_type'],
            'license':row['license'],'caption':row['caption'],'pdb_evidence':row['pdb_evidence'],
            'source_notes':row['notes'],'caveats':row['caveats'],
            'notes':old_crop['notes'] if old_crop else ('RNA-only example; excluded from protein set.' if 'rna_only' in flags else ''),
        }
        render(crop)
        if old_crop:
            with tarfile.open(ROOT/'results/wild_transfer_v1.tar') as archive:
                with Image.open(archive.extractfile('data/wild/publication_crops_v1/'+old_crop['native_file'])) as archived:
                    with Image.open(OUT/crop['native_file']) as generated:
                        assert archived.convert('RGBA').tobytes()==generated.convert('RGBA').tobytes(),crop_id
        crops.append(crop)
        if len(crops)%40==0: print('Rendered',len(crops),'/',len(specs),flush=True)
    m={'schema_version':2,'dataset':'photoprot_dinov2_wild_review_v1',
        'status':'review_draft','source_index_sha256':sha(SOURCE/'index.csv'),
        'crop_annotations_sha256':sha(ROOT/'scripts/wild_crop_boxes.json'),
        'build_script_sha256':sha(__file__),'n_source_figures':len(rows),
        'n_available_source_images':sum(bool(r['image_file']) for r in rows),
        'model_preprocessing':{'model':'DINOv2','input_size':[224,224],'input_mode':'RGB',
            'resize':'512 square to 224 square PIL BILINEAR; no center crop',
            'tensor':'float32 RGB CHW divided by 255 then (x-mean)/std','mean':MEAN,'std':STD},
        'selection_policy':'74 prior crop rectangles plus one visually selected primary view for each of 405 new image records. Crops are a review draft; no predictions used. Text inside boxes retained.',
        'crops':crops,'sources':sources}
    save_manifest(m)
    shutil.copyfile(ROOT/'scripts/wild_review.html',OUT/'review.html')
    contact_sheets(m)
    verify()


def verify():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len({c['crop_id'] for c in m['crops']})==len(m['crops'])
    seen=set()
    for c in m['crops']:
        path=SOURCE/c['source_image_file']
        if path not in seen:
            assert sha(path)==c['source_image_sha256']
            seen.add(path)
        with Image.open(path) as src, Image.open(OUT/c['native_file']) as native:
            expected=src.crop(c['box_source_xyxy'])
            assert expected.size==native.size
            assert expected.convert('RGBA').tobytes()==native.convert('RGBA').tobytes(),c['crop_id']
            rgb, model, _=transform(native)
        for key,expected in [('standardized',rgb),('model_input',model)]:
            with Image.open(OUT/c[key+'_file']) as saved:
                assert saved.mode=='RGB' and saved.size==expected.size
                assert saved.tobytes()==expected.tobytes(),c['crop_id']
        for key in ('native','standardized','model_input'):
            assert sha(OUT/c[key+'_file'])==c[key+'_sha256']
    result={'passed':True,'n_crops':len(m['crops']),'n_source_images':len(seen),
            'checks':['source hash','native pixel equality','512 transform pixel equality','224 bilinear pixel equality','output hashes','unique crop IDs'],
            'manifest_sha256':sha(MANIFEST),'verified_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(OUT/'validation.json',result)
    print(json.dumps(result),flush=True)


def export(approved_only=False):
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    selected=[c for c in m['crops'] if c['review_status']=='approved'] if approved_only else m['crops']
    if not selected: raise ValueError('There are no approved crops yet')
    package=ROOT/'wild_test'/('dinov2_approved_v1.tar.gz' if approved_only else 'dinov2_review_v1_brev.tar.gz')
    export_m={k:v for k,v in m.items() if k!='sources'}
    export_m['crops']=selected
    export_m['n_crops']=len(selected)
    export_m['package_contents']='manifest.json, rgb512 and dinov2_224 only; native/source assets stay in local review dataset'
    export_m['review_counts']=dict(Counter(c['review_status'] for c in selected))
    export_m['local_manifest_sha256']=sha(MANIFEST)
    with tarfile.open(package,'w:gz') as tf:
        payload=json.dumps(export_m,ensure_ascii=False,indent=2).encode('utf-8')
        info=tarfile.TarInfo('manifest.json');info.size=len(payload)
        tf.addfile(info,io.BytesIO(payload))
        for c in selected:
            for key in ('standardized_file','model_input_file'):
                tf.add(OUT/c[key],arcname=c[key],recursive=False)
    write_json(OUT/'last_export.json',{'path':str(package),'sha256':sha(package),'n_crops':len(selected),'approved_only':approved_only,'manifest_sha256':sha(MANIFEST)})
    print(package,flush=True)
    return package


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(OUT),**kwargs)
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def do_GET(self):
        if self.path=='/': self.path='/review.html'
        if self.path.startswith('/api/manifest'):
            self.send_json(json.loads(MANIFEST.read_text(encoding='utf-8')));return
        if self.path.startswith('/original/'):
            source_id=unquote(urlparse(self.path).path[len('/original/'):])
            m=json.loads(MANIFEST.read_text(encoding='utf-8'))
            row=next((r for r in m['sources'] if r['id']==source_id),None)
            if not row or not row['image_file']: self.send_error(404);return
            data=(SOURCE/row['image_file']).read_bytes()
            self.send_response(200);self.send_header('Content-Type',self.guess_type(row['image_file']));self.end_headers();self.wfile.write(data);return
        super().do_GET()
    def send_json(self,obj,status=200):
        body=json.dumps(obj,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        try:
            # Only the same-origin review page may mutate this loopback service.
            origin=self.headers.get('Origin','')
            if origin and origin not in ('http://127.0.0.1:'+str(self.server.server_port),'http://localhost:'+str(self.server.server_port)):
                self.send_json({'error':'Origin not permitted'},403);return
            n=int(self.headers.get('Content-Length','0'))
            if n>65536: raise ValueError('Request too large')
            data=json.loads(self.rfile.read(n))
            with LOCK:
                m=json.loads(MANIFEST.read_text(encoding='utf-8'))
                if self.path=='/api/save':
                    c=next(c for c in m['crops'] if c['crop_id']==data['crop_id'])
                    before=json.loads(json.dumps(c))
                    if data['review_status'] not in ('pending','approved','excluded'): raise ValueError('Invalid review status')
                    if data['mapping_status'] not in ('panel_candidates','source_mapped','multiple_depicted'): raise ValueError('Invalid mapping status')
                    pdbs=[p.upper() for p in data['pdb_ids']]
                    if any(not re.fullmatch('[0-9][A-Z0-9]{3}',p) for p in pdbs): raise ValueError('PDB IDs must be four-character accessions')
                    c.update({k:data[k] for k in ['box_source_xyxy','review_status','mapping_status','notes']})
                    c['pdb_ids']=pdbs
                    c['clean_query_eligible']=False  # Approval does not certify text-free scientific eligibility.
                    render(c)
                    save_manifest(m)
                    with (OUT/'review_log.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'time':c['updated_at'],'before':before,'after':c},ensure_ascii=False)+'\n')
                    self.send_json({'crop':c,'review_counts':m['review_counts']});return
                if self.path=='/api/export':
                    verify()
                    package=export(bool(data.get('approved_only')))
                    self.send_json({'path':str(package)});return
            self.send_error(404)
        except Exception as exc:
            self.send_json({'error':str(exc)},400)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--build',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--serve',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--approved-only',action='store_true');p.add_argument('--port',type=int,default=8766)
    a=p.parse_args()
    if a.build: build()
    if a.verify: verify()
    if a.export: export(a.approved_only)
    if a.serve:
        print(f'Review at http://127.0.0.1:{a.port}',flush=True)
        ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

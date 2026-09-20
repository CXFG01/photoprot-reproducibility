"""Reconstruct frozen query pixels from lawfully obtained local source files."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True,help='Directory containing the manifest images/ source paths')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads((ROOT/'benchmarks/wild-v1/query_manifest.json').read_text(encoding='utf-8'))['crops']
    done=[];missing=[]
    for c in rows:
        source=(a.sources/c['source_image_file']).resolve()
        assert source.is_relative_to(a.sources.resolve())
        if not source.is_file():missing.append(c['crop_id']);continue
        assert sha(source)==c['source_image_sha256'],c['source_id']
        with Image.open(source) as im:
            crop=im.crop(c['box_source_xyxy']).convert('RGBA');bg=Image.new('RGBA',crop.size,'white');bg.alpha_composite(crop);rgb=bg.convert('RGB')
        scale=464/max(rgb.size);size=tuple(max(1,round(d*scale)) for d in rgb.size)
        canvas=Image.new('RGB',(512,512),'white');canvas.paste(rgb.resize(size,Image.Resampling.LANCZOS),((512-size[0])//2,(512-size[1])//2))
        for key,im,expected in [('standardized_file',canvas,c['standardized_sha256']),('model_input_file',canvas.resize((224,224),Image.Resampling.BILINEAR),c['model_input_sha256'])]:
            dest=a.output/c[key];dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==expected
            else:
                im.save(dest);assert sha(dest)==expected,'PNG differs; use the recorded Pillow version and inspect source/transform'
        done.append(c['crop_id'])
    print(json.dumps({'reconstructed':len(done),'missing':len(missing),'missing_crop_ids':missing},indent=2))
    if missing:raise SystemExit(2)
if __name__=='__main__':main()

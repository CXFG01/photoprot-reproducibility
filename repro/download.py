"""Download named release artifacts with SHA-256 verification and atomic completion."""
import argparse, hashlib, json, shutil, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('names',nargs='+',help='Asset filenames or all')
    p.add_argument('--output',type=Path,default=ROOT/'downloads')
    a=p.parse_args(); records=json.loads((ROOT/'artifacts/manifest.json').read_text())
    selected=records if a.names==['all'] else [r for r in records if r['name'] in a.names]
    if a.names!=['all'] and {r['name'] for r in selected}!=set(a.names):p.error('Unknown asset name')
    a.output.mkdir(parents=True,exist_ok=True)
    for r in selected:
        dest=a.output/r['name']
        if dest.exists():
            with dest.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256'],f'Existing file differs: {dest}'
            print('Verified existing',r['name']);continue
        temp=dest.with_suffix(dest.suffix+'.partial')
        req=urllib.request.Request(r['url'],headers={'User-Agent':'PhotoProt-reproducibility/0.1.0'})
        with urllib.request.urlopen(req,timeout=120) as response,temp.open('wb') as f:shutil.copyfileobj(response,f,8*1024*1024)
        with temp.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==r['sha256'] and temp.stat().st_size==r['bytes'],f'Artifact failed verification: {r["name"]}'
        temp.replace(dest);print('Downloaded and verified',r['name'])
if __name__=='__main__':main()

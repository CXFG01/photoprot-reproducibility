"""Extract the webp corpus to individual files for random access.

The tars are perfect for sequential embedding but useless for contrastive
training, which needs to pull an arbitrary (object, view) pair on demand. So
extract once, ~18 GB, leaving the tars untouched as the canonical copy.

Resumable at SHARD granularity via a .done marker written only after the whole
shard is out, so an interrupted run never leaves a shard half-extracted and
silently skips it next time.
"""
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def do_shard(tar_name):
    stem = tar_name[:-4]
    out = f"{C.IMG}/{stem}"
    done = f"{out}/.done"
    if os.path.exists(done):
        return tar_name, 0, "skip"
    os.makedirs(out, exist_ok=True)
    n = 0
    with tarfile.open(f"{C.WEBP}/{tar_name}") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".webp"):
                continue
            dst = f"{out}/{m.name}"
            if os.path.exists(dst):
                n += 1
                continue
            data = tf.extractfile(m).read()
            tmp = dst + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
            n += 1
    with open(done, "w") as fh:
        fh.write(str(n))
    return tar_name, n, "done"


def main():
    os.makedirs(C.IMG, exist_ok=True)
    tars = sorted(t for t in os.listdir(C.WEBP) if t.endswith(".tar"))
    print(f"{len(tars)} shards to extract -> {C.IMG}", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (name, n, how) in enumerate(ex.map(do_shard, tars), 1):
            total += n
            print(f"[{i:3d}/{len(tars)}] {name} {n:6,} {how}", flush=True)
    print(f"\nextracted/verified {total:,} images", flush=True)


if __name__ == "__main__":
    main()

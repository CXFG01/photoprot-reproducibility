"""Locating and driving the PyMOL render engine.

The renderer runs in its own interpreter, so everything about invoking it lives
here rather than being scattered through the scripts.
"""
from __future__ import annotations

import os
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "photoprot.render.pymol_worker"

# Candidate interpreters, best first. The open-source build is required: the
# Schrodinger evaluation build burns a "For Evaluation Only" watermark into a
# fraction of renders, invisible on white backgrounds and obvious on grey.
CANDIDATES = [
    Path(r"C:/Users/Lenovo/.local/share/mamba/envs/photoprot-render/python.exe"),
    Path(r"C:/Users/Lenovo/miniforge3/envs/photoprot-render/python.exe"),
]
WATERMARKED = Path(r"C:/Users/Lenovo/AppData/Local/Schrodinger/PyMOL2/python.exe")


def find_python(allow_watermarked: bool = False) -> Path:
    env = os.environ.get("PHOTOPROT_PYMOL")
    if env:
        return Path(env)
    for c in CANDIDATES:
        if c.exists():
            return c
    if allow_watermarked and WATERMARKED.exists():
        return WATERMARKED
    raise SystemExit(
        "No open-source PyMOL found. Create it with:\n"
        "  mamba create -n photoprot-render -c conda-forge pymol-open-source -y"
    )


def run_shard(python: Path, jobs: Path, results: Path, seed: int = 0,
              progress_every: int = 0) -> subprocess.CompletedProcess:
    """Render one shard in a fresh PyMOL process."""
    return subprocess.run(
        [str(python), "-m", WORKER_MODULE, "--jobs", str(jobs),
         "--results", str(results), "--seed", str(seed),
         "--progress-every", str(progress_every)],
        capture_output=True, text=True, cwd=str(ROOT),
    )


def job_seed(domain_id: str, view: int, global_seed: int) -> int:
    """Stable per-(domain, view) seed.

    CRC32 rather than hash(): Python randomises str hashing per process, which
    would make renders irreproducible across runs.
    """
    key = f"{domain_id}:{view}:{global_seed}".encode()
    return zlib.crc32(key) & 0xFFFFFFFF

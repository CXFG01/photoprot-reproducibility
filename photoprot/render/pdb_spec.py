"""Deterministic sampling of one render's parameters from a JSON axis config.

Pure: numpy and json only, no PyMOL, so the sampler can be unit-tested and the
manifest can be built without a renderer present.

The contract is that a render is a pure function of (object_id, view, seed).
CRC32 rather than hash(): Python randomises str hashing per process, which would
make renders irreproducible across runs.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "render_v1.json"


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def job_seed(object_id: str, view: int, global_seed: int) -> int:
    return zlib.crc32(f"{object_id}:{view}:{global_seed}".encode()) & 0xFFFFFFFF


def _draw(rng: np.random.Generator, spec: dict):
    kind = spec["type"]
    if kind == "categorical":
        i = rng.choice(len(spec["values"]), p=spec["p"])
        v = spec["values"][int(i)]
        return int(v) if isinstance(v, bool) else v
    if kind == "bernoulli":
        return int(rng.random() < spec["p"])
    if kind == "uniform":
        return float(rng.uniform(spec["lo"], spec["hi"]))
    if kind == "mixture_zero":
        if rng.random() < spec["p_zero"]:
            return 0.0
        return float(rng.uniform(spec["lo"], spec["hi"]))
    raise ValueError(f"unknown axis type: {kind}")


def _apply_degenerate(style: dict, degenerate: dict) -> dict:
    """Repair combinations that render to an unreadable image.

    Deterministic repair rather than resampling, for the same reason as the
    heldout repair: a rejection loop would make the result depend on how many
    draws were rejected.
    """
    for rule in degenerate.get("rules", []):
        if all(style.get(k) == v for k, v in rule["when"].items()):
            style.update(rule["set"])
    return style


def _apply_heldout(style: dict, heldout: dict) -> dict:
    """Keep training samples out of the reserved style region.

    Repaired deterministically rather than resampled, so the sample stays a pure
    function of (object_id, view, seed) - a resample loop would make the result
    depend on how many draws were rejected.
    """
    banned_rtm = set(heldout.get("ray_trace_mode", []))
    if style.get("ray_trace_mode") in banned_rtm:
        style["ray_trace_mode"] = 0
    for combo in heldout.get("combos", []):
        if all(style.get(k) == v for k, v in combo.items()):
            style["flat_sheets"] = 1
    return style


def sample_view(object_id: str, view: int, global_seed: int = 0,
                config: dict | None = None, heldout_ok: bool = False) -> dict:
    """Every parameter of one render, deterministically derived.

    `heldout_ok=True` skips the held-out repair, for generating the style-holdout
    TEST set on purpose.
    """
    cfg = config or load_config()
    rng = np.random.default_rng(job_seed(object_id, view, global_seed))

    # Orientation is random for EVERY view including the canonical one.
    # Normalising a 4D Gaussian is uniform over SO(3); naive Euler angles would
    # concentrate near the poles.
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)

    if view == 0:
        style = {k: v for k, v in cfg["canonical"].items() if k != "note"}
    else:
        style = {name: _draw(rng, spec) for name, spec in cfg["axes"].items()}
        style = _apply_degenerate(style, cfg.get("degenerate", {}))
        if not heldout_ok:
            style = _apply_heldout(style, cfg.get("heldout", {}))

    style.update({
        "quat_w": float(q[0]), "quat_x": float(q[1]),
        "quat_y": float(q[2]), "quat_z": float(q[3]),
        "view": int(view),
        "role": "eval_single" if view == 0 else "train_view",
        "config_version": cfg["version"],
    })
    return style


def describe(style: dict) -> str:
    """Compact one-line summary, for contact-sheet tile labels."""
    bits = [
        str(style.get("cartoon_style", "?")),
        str(style.get("color_scheme", "?")),
        f"bg={style.get('background')}",
        f"rtm={style.get('ray_trace_mode')}",
        f"aa={style.get('antialias')}",
        f"ao={style.get('ambient_occlusion')}",
    ]
    if style.get("content") != "cartoon_only":
        bits.append(str(style.get("content")))
    if float(style.get("cartoon_transparency") or 0) > 0:
        bits.append(f"transp={float(style['cartoon_transparency']):.2f}")
    if int(style.get("orthoscopic") or 0):
        bits.append("ortho")
    if int(style.get("cylindrical_helices") or 0):
        bits.append("cylhel")
    if int(style.get("fancy_helices") or 0):
        bits.append("fancyhel")
    if not int(style.get("flat_sheets", 1)):
        bits.append("nonflat")
    if int(style.get("depth_cue") or 0):
        bits.append("fog")
    return " ".join(bits)

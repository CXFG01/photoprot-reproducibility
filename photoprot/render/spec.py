"""What a single render looks like: the style space, and how a view is sampled.

Sampling is a pure function of (domain_id, view index, global seed), so the whole
dataset is reproducible and individual images can be regenerated on demand.
"""
from __future__ import annotations

import numpy as np

from photoprot.render.engine import job_seed

# View 0 is the frozen single-view evaluation render: one fixed, plain style so
# the headline benchmark is not confounded by style variation. Views 1+ carry the
# augmentation diversity.
CANONICAL = {
    "representation": "cartoon",
    "color_scheme": "uniform",
    "background": "white",
    "ambient_occlusion": 0,
    "antialias": 1,
    "zoom_buffer": 2.0,
}

REPRESENTATIONS = ["cartoon", "ribbon"]
REPRESENTATION_P = [0.75, 0.25]

# Weighted towards the schemes that dominate real figures. `tube` is excluded
# entirely: it renders ~34x slower than cartoon for no added realism.
COLOR_SCHEMES = ["uniform", "rainbow", "sse", "mono", "bfactor"]
COLOR_P = [0.35, 0.25, 0.15, 0.15, 0.10]

BACKGROUNDS = ["white", "black", "grey80"]
BACKGROUND_P = [0.70, 0.15, 0.15]


def sample_view(domain_id: str, view: int, global_seed: int = 0) -> dict:
    """Every parameter of one render, deterministically derived."""
    rng = np.random.default_rng(job_seed(domain_id, view, global_seed))

    # Orientation is random for EVERY view including the canonical one: a single
    # arbitrary viewpoint is the situation the model is actually deployed in.
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)

    if view == 0:
        style = dict(CANONICAL)
    else:
        style = {
            "representation": str(rng.choice(REPRESENTATIONS, p=REPRESENTATION_P)),
            "color_scheme": str(rng.choice(COLOR_SCHEMES, p=COLOR_P)),
            "background": str(rng.choice(BACKGROUNDS, p=BACKGROUND_P)),
            "ambient_occlusion": int(rng.integers(0, 2)),
            "antialias": 1,
            "zoom_buffer": float(rng.uniform(1.5, 3.0)),
        }

    style.update({
        "quat_w": float(q[0]), "quat_x": float(q[1]),
        "quat_y": float(q[2]), "quat_z": float(q[3]),
        "ray": 1, "ray_trace_mode": 0,
        "view": view,
        "role": "eval_single" if view == 0 else "train_view",
    })
    return style


def out_path(renders_root, view: int, domain_id: str) -> str:
    """Two-character fan-out keeps directories at a few hundred files each.

    A flat directory of 34,649 entries per view is legal but makes every listing
    and every resume scan painfully slow on Windows.
    """
    return str(renders_root / f"v{view}" / domain_id[:2] / f"{domain_id}.png")

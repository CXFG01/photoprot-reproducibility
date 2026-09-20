"""PyMOL-side render worker. Runs under PyMOL's OWN Python, not the project env.

PyMOL ships its own interpreter (3.10, with numpy/pandas/PIL/scipy but no
pyarrow), so this file must not import anything from `photoprot`. The interface
with the rest of the project is deliberately a CSV job file in and a CSV result
file out; the orchestrator keeps parquet and the manifest on its side.

Invoked as:
    <PyMOL>/python.exe -m photoprot.render.pymol_worker --jobs jobs.csv --results out.csv

Each job row carries every parameter needed to reproduce one image, including the
orientation quaternion, so a render can be regenerated exactly from the manifest.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Job columns the worker understands. Unknown columns are ignored and passed back.
REQUIRED = ("render_id", "pdb_path", "out_path")

def _cartoon(cmd, sel):
    cmd.set("cartoon_trace_atoms", 0)
    cmd.show_as("cartoon", sel)
    cmd.cartoon("automatic", sel)


def _ribbon(cmd, sel):
    cmd.show_as("ribbon", sel)


def _tube(cmd, sel):
    cmd.set("cartoon_trace_atoms", 1)
    cmd.show_as("cartoon", sel)
    cmd.cartoon("tube", sel)


REPRESENTATIONS = {"cartoon": _cartoon, "ribbon": _ribbon, "tube": _tube}


# A 384x384 render of an empty scene is a flat background that still writes a
# valid PNG. Fraction of pixels differing from the modal (background) colour is
# the cheapest reliable way to tell "rendered nothing" from "rendered something".
MIN_INK_FRACTION = 0.005


def _wait_for_stable_file(path, timeout=120.0, settle=0.01):
    """Block until `path` exists and its size stops changing.

    The only reliable signal that PyMOL has finished writing an image, since the
    write is asynchronous and cmd.sync() reports spurious lock_attempt timeouts
    in headless mode.
    """
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 0 and size == last:
                return True
            last = size
        time.sleep(settle)
    return False


def _ink_fraction(path):
    """Fraction of pixels that are not the modal background colour."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > 128:
            im = im.resize((128, 128), Image.NEAREST)
        colors = im.getcolors(maxcolors=128 * 128)
    if not colors:
        return 1.0
    total = sum(c for c, _ in colors)
    bg = max(colors, key=lambda c: c[0])[0]
    return 1.0 - bg / total


def _apply_color(cmd, sel, scheme, rng):
    """Colour schemes that real figures actually use."""
    if scheme == "uniform":
        # random hue at fixed saturation/value, the common single-colour figure
        import colorsys

        h = rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.85)
        cmd.set_color("randhue", [r, g, b])
        cmd.color("randhue", sel)
    elif scheme == "rainbow":
        cmd.spectrum("count", "rainbow", sel + " and name CA")
    elif scheme == "sse":
        cmd.color("red", sel + " and ss H")
        cmd.color("yellow", sel + " and ss S")
        cmd.color("green", sel + " and not (ss H or ss S)")
    elif scheme == "bfactor":
        cmd.spectrum("b", "blue_white_red", sel)
    elif scheme == "mono":
        cmd.color("grey70", sel)
    else:
        raise ValueError("unknown colour scheme: " + str(scheme))


def _set_orientation(cmd, sel, quat):
    """Place the camera at an arbitrary orientation given by a unit quaternion.

    PyMOL's view is an 18-float tuple whose first 9 entries are the row-major
    world->camera rotation. We orient once to get a sane camera distance, then
    overwrite the rotation block and re-zoom to fit. Driving the camera with
    successive turn() calls instead would make the orientation dependent on call
    order and impossible to reproduce from the manifest.
    """
    w, x, y, z = quat
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    cmd.orient(sel)
    view = list(cmd.get_view())
    view[0:9] = rot
    cmd.set_view(view)


def render_one(cmd, job, rng):
    """Render a single image. Returns (seconds, bytes, error-or-None)."""
    sel = "dom"
    cmd.delete("all")
    cmd.load(job["pdb_path"], sel, format="pdb")  # CATH files carry no extension

    cmd.hide("everything")
    rep = job.get("representation", "cartoon") or "cartoon"
    REPRESENTATIONS[rep](cmd, sel)

    _apply_color(cmd, sel, job.get("color_scheme", "uniform") or "uniform", rng)

    bg = job.get("background", "white") or "white"
    cmd.bg_color(bg)
    cmd.set("ray_opaque_background", 1)

    ao = int(job.get("ambient_occlusion", 0) or 0)
    if ao:
        cmd.set("ambient_occlusion_mode", 1)
        cmd.set("ray_trace_mode", 0)
    else:
        cmd.set("ambient_occlusion_mode", 0)

    cmd.set("antialias", int(job.get("antialias", 1) or 0))
    cmd.set("ray_trace_mode", int(job.get("ray_trace_mode", 0) or 0))

    quat = (
        float(job.get("quat_w", 1)), float(job.get("quat_x", 0)),
        float(job.get("quat_y", 0)), float(job.get("quat_z", 0)),
    )
    _set_orientation(cmd, sel, quat)
    cmd.zoom(sel, float(job.get("zoom_buffer", 2.0) or 2.0))

    out = job["out_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    w = int(job.get("width", 384) or 384)
    h = int(job.get("height", 384) or 384)

    t0 = time.time()
    # Image writing in headless PyMOL is asynchronous even when the raytrace is
    # not. cmd.ray() blocks, but cmd.png() hands the buffer off and returns, so
    # the next job's delete('all') can land mid-write. That corrupts the session:
    # PyMOL loses the object ("Selector-Error: Invalid selection name") and emits
    # a blank background that is still a structurally valid PNG on disk.
    # Measured at ~5% of renders, intermittent on identical input - the same
    # domain at the same orientation renders correctly three times and blanks the
    # fourth. So: raytrace synchronously, then wait for the file to finish being
    # written before touching the session again.
    cmd.ray(w, h)
    cmd.png(out, dpi=72)
    if not _wait_for_stable_file(out):
        return time.time() - t0, 0, "timeout waiting for png write"
    elapsed = time.time() - t0

    if not os.path.exists(out) or os.path.getsize(out) == 0:
        return elapsed, 0, "no output written"

    ink = _ink_fraction(out)
    if ink < MIN_INK_FRACTION:
        return elapsed, os.path.getsize(out), f"blank image (ink={ink:.4f})"

    return elapsed, os.path.getsize(out), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True, help="CSV of render jobs")
    ap.add_argument("--results", required=True, help="CSV to write timings to")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    import random

    rng = random.Random(args.seed)

    with open(args.jobs, newline="", encoding="utf-8") as fh:
        jobs = list(csv.DictReader(fh))
    for col in REQUIRED:
        if jobs and col not in jobs[0]:
            raise SystemExit(f"job file missing required column: {col}")

    import pymol

    pymol.finish_launching(["pymol", "-qck"])  # quiet, no GUI, no plugins
    from pymol import cmd

    cmd.set("max_threads", 1)  # one PyMOL per process; parallelism is external

    rows = []
    t_start = time.time()
    n_retried = 0
    for i, job in enumerate(jobs, 1):
        secs, nbytes, err, attempts = 0.0, 0, None, 0
        # A failure is usually the transient write race rather than a bad domain,
        # so reset the session and try again before writing the render off.
        for attempt in range(2):
            attempts = attempt + 1
            try:
                secs, nbytes, err = render_one(cmd, job, rng)
            except Exception as exc:  # one bad domain must not kill the shard
                secs, nbytes, err = 0.0, 0, f"{type(exc).__name__}: {exc}"
            if err is None:
                break
            try:
                cmd.reinitialize()  # only reliable recovery; ~10ms, failure-only
            except Exception:
                pass
        if attempts > 1 and err is None:
            n_retried += 1
        rows.append({
            "render_id": job["render_id"],
            "seconds": round(secs, 4),
            "bytes": nbytes,
            "ok": err is None,
            "error": err or "",
            "attempts": attempts,
        })
        if args.progress_every and i % args.progress_every == 0:
            rate = i / (time.time() - t_start)
            print(f"[worker] {i}/{len(jobs)}  {rate:.2f} img/s", flush=True)

    with open(args.results, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(
            fh, fieldnames=["render_id", "seconds", "bytes", "ok", "error", "attempts"])
        wr.writeheader()
        wr.writerows(rows)

    ok = sum(1 for r in rows if r["ok"])
    total = time.time() - t_start
    print(
        f"[worker] done {ok}/{len(rows)} ok in {total:.1f}s "
        f"({len(rows)/total:.2f} img/s, {n_retried} recovered by retry)",
        flush=True,
    )


if __name__ == "__main__":
    main()

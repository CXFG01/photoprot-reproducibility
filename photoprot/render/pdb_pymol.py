"""Apply a sampled style to PyMOL and write one PNG.

Runs under an interpreter that has PyMOL importable (the pip wheel, not the
Windows subprocess arrangement Phase 1.5 needed). Imports nothing from photoprot
except pdb_spec, so it stays usable inside a bare Modal image.

Two details are load-bearing and were both learned the hard way:

  * max_threads=1. PyMOL's ray tracer is multithreaded and its pixel output
    depends on thread scheduling - four renders of identical input otherwise
    differ in ~80/147,456 pixels. It is also FASTER in aggregate, because 32
    processes x 1 thread beats 1 process x 32 threads by 4.3x: intra-render
    threading scales sublinearly (7.51x on 32 threads) while process
    parallelism is near-perfect.

  * ray() then png() then wait for the file size to settle. cmd.png() alone
    hands the buffer off and returns, so the next delete('all') can land
    mid-write, losing the object and emitting a blank-but-valid PNG. Measured
    at ~5% of renders, intermittent on identical input.
"""
from __future__ import annotations

import colorsys
import math
import os
import time

MIN_INK_FRACTION = 0.005   # a blank render is still a valid PNG; this catches it

LIGANDS = "organic and not solvent and not polymer"

# Selected by ELEMENT rather than by PyMOL's `inorganic`, because the metal
# centre that figures actually draw as a sphere is usually inside an organic
# residue - the Fe of a haem is part of HEM, so `inorganic` misses it and the
# view silently collapses to a plain cartoon.
ION_ELEMENTS = ("Zn+Fe+Mg+Ca+Mn+Cu+Na+K+Ni+Co+Cd+Hg+Mo+W+V+Sr+Ba+Cs+Rb+Li+"
                "Al+Au+Ag+Pt+Pd+Se+Cl+Br+I")
IONS = f"(not polymer) and (not solvent) and (elem {ION_ELEMENTS})"


def init_pymol(ccd_cache: str | None = None):
    import pymol
    pymol.finish_launching(["pymol", "-qck"])   # quiet, no GUI, no plugins
    from pymol import cmd

    cmd.set("max_threads", 1)
    if ccd_cache:
        # The chemical component dictionary on disk. Without it PyMOL reaches out
        # to files.rcsb.org mid-render for unknown components - a network call per
        # render at corpus scale, and a silent bonding change when it fails.
        cmd.set("fetch_path", ccd_cache)
    return cmd


def _wait_for_stable_file(path: str, timeout: float = 600.0,
                          settle: float = 0.005) -> bool:
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


def _ink_fraction(path: str) -> float:
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


def _apply_colour(cmd, style: dict) -> None:
    scheme = style["color_scheme"]
    sel = "polymer"
    if scheme == "uniform":
        r, g, b = colorsys.hsv_to_rgb(float(style["uniform_hue"]), 0.55, 0.85)
        cmd.set_color("randhue", [r, g, b])
        cmd.color("randhue", sel)
    elif scheme == "mono":
        cmd.color("grey70", sel)
    elif scheme == "rainbow":
        cmd.spectrum("count", "rainbow", sel + " and name CA")
    elif scheme == "sse":
        cmd.color("red", sel + " and ss H")
        cmd.color("yellow", sel + " and ss S")
        cmd.color("green", sel + " and not (ss H or ss S)")
    elif scheme == "bfactor":
        cmd.spectrum("b", "blue_white_red", sel)
    elif scheme == "bychain":
        from pymol import util
        util.cbc(selection=sel)
    elif scheme == "chainbow":
        from pymol import util
        util.chainbow(selection=sel)
    else:
        raise ValueError(f"unknown colour scheme: {scheme}")


def _apply_background(cmd, style: dict) -> None:
    bg = style["background"]
    if bg == "gradient":
        # A white-to-grey vertical wash, the common "poster" background. Fixed
        # rather than sampled, so it adds no axis.
        cmd.set("bg_gradient", 1)
        cmd.set("bg_rgb_top", "white")
        cmd.set("bg_rgb_bottom", "grey60")
    else:
        cmd.set("bg_gradient", 0)
        cmd.bg_color(bg)
    cmd.set("ray_opaque_background", 1)


def _set_camera(cmd, style: dict) -> None:
    """Orientation, projection, zoom, then off-centre - in that order.

    Projection MUST be set before zoom. Setting field_of_view or orthoscopic
    after zooming changes how much of the frame the structure fills, which both
    shifts the framing and makes render cost look artificially cheap.
    """
    cmd.set("field_of_view", float(style["field_of_view"]))
    cmd.set("orthoscopic", int(style["orthoscopic"]))

    w, x, y, z = (float(style["quat_w"]), float(style["quat_x"]),
                  float(style["quat_y"]), float(style["quat_z"]))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    # Orient once for a sane camera distance, then overwrite the rotation block.
    # Driving the camera with successive turn() calls would make the result
    # depend on call order and be impossible to reproduce from the manifest.
    cmd.orient("polymer")
    v = list(cmd.get_view())
    v[0:9] = rot
    cmd.set_view(v)
    cmd.zoom("polymer", float(style["zoom_buffer"]))

    fx = float(style.get("offcentre_x") or 0.0)
    fy = float(style.get("offcentre_y") or 0.0)
    if fx or fy:
        v = list(cmd.get_view())
        # v[11] is the camera distance; half the visible height at the subject is
        # dist * tan(fov/2), so a full frame is twice that.
        half = abs(v[11]) * math.tan(math.radians(float(style["field_of_view"])) / 2)
        v[9] += fx * 2 * half
        v[10] += fy * 2 * half
        cmd.set_view(v)


def render(cmd, path: str, obj_kind: str, style: dict, out: str,
           width: int, height: int | None = None) -> tuple[float, int, str | None]:
    """Render one image. Returns (seconds, bytes, error-or-None)."""
    height = height or width
    content = style.get("content", "cartoon_only")
    want_het = content != "cartoon_only"

    cmd.delete("all")
    # `assembly` must be set BEFORE load: it tells the mmCIF reader whether to
    # apply the symmetry operators. "" is the asymmetric unit as deposited.
    cmd.set("assembly", "1" if obj_kind == "asm1" else "")
    cmd.load(path, "obj")
    cmd.remove("solvent")
    cmd.remove("hydrogens")
    if not want_het:
        cmd.remove("not polymer")

    if cmd.count_atoms("polymer") == 0:
        return 0.0, 0, "no polymer atoms after cleanup"

    cmd.hide("everything")

    # --- cartoon geometry -------------------------------------------------
    cmd.set("cartoon_trace_atoms", 0)
    cmd.set("cartoon_cylindrical_helices", int(style["cylindrical_helices"]))
    cmd.set("cartoon_fancy_helices", int(style["fancy_helices"]))
    cmd.set("cartoon_flat_sheets", int(style["flat_sheets"]))
    cmd.set("cartoon_smooth_loops", int(style["smooth_loops"]))
    cmd.set("cartoon_loop_radius", float(style["loop_radius"]))
    cmd.set("cartoon_transparency", float(style["cartoon_transparency"]))
    cmd.show_as("cartoon", "polymer")
    cmd.cartoon(style["cartoon_style"], "polymer")

    # --- extra content ----------------------------------------------------
    # `content_effective` records what was actually drawable: an entry with no
    # ligand or no metal renders identically to cartoon_only, and the manifest
    # should say so rather than claim a variation that is not in the pixels.
    content_effective = "cartoon_only"
    if content in ("ligand_sticks", "ligand_spheres"):
        if cmd.count_atoms(LIGANDS) > 0:
            cmd.show("sticks" if content == "ligand_sticks" else "spheres", LIGANDS)
            # CPK heteroatoms, which is how figures almost always draw them.
            try:
                from pymol import util
                util.cnc(LIGANDS)
            except Exception:
                pass
            content_effective = content
    elif content == "ions_spheres":
        if cmd.count_atoms(IONS) > 0:
            cmd.show("spheres", IONS)
            content_effective = content
    style["content_effective"] = content_effective

    _apply_colour(cmd, style)
    _apply_background(cmd, style)

    # --- optics -----------------------------------------------------------
    cmd.set("ray_trace_mode", int(style["ray_trace_mode"]))
    cmd.set("antialias", int(style["antialias"]))
    cmd.set("ambient_occlusion_mode", 1 if int(style["ambient_occlusion"]) else 0)
    cmd.set("depth_cue", int(style["depth_cue"]))
    if int(style["depth_cue"]):
        cmd.set("fog_start", float(style["fog_start"]))
    cmd.set("light_count", int(style["light_count"]))
    cmd.set("ambient", float(style["ambient"]))
    cmd.set("direct", float(style["direct"]))
    cmd.set("reflect", float(style["reflect"]))
    cmd.set("shininess", float(style["shininess"]))

    _set_camera(cmd, style)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    t0 = time.time()
    cmd.ray(width, height)
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


def render_with_retry(cmd, path: str, obj_kind: str, style: dict, out: str,
                      width: int, ccd_cache: str | None = None,
                      attempts: int = 2):
    """A failure is usually the transient write race, not a bad structure."""
    secs = nbytes = 0
    err = None
    used = 0
    for i in range(attempts):
        used = i + 1
        try:
            secs, nbytes, err = render(cmd, path, obj_kind, style, out, width)
        except Exception as exc:
            secs, nbytes, err = 0.0, 0, f"{type(exc).__name__}: {exc}"[:200]
        if err is None:
            break
        try:
            cmd.reinitialize()
            cmd.set("max_threads", 1)
            if ccd_cache:
                cmd.set("fetch_path", ccd_cache)
        except Exception:
            pass
    return secs, nbytes, err, used

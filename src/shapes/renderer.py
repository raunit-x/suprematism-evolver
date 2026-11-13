"""Render ShapeGenome compositions to PIL Images.

Each shape is drawn on a temporary RGBA layer, rotated around its center,
then alpha-composited onto the canvas in z-order (list order = back-to-front).

A fibrous canvas texture (random fine scratches/lines) is overlaid to give
a physical, matte-paint-on-canvas quality rather than a flat vector look.
"""

from __future__ import annotations

import math
import random as _py_random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src.shapes.genome import ShapeGenome, ShapeGroup, Shape
from src.art.palettes import SUPREMATIST_PALETTE

DEFAULT_PALETTE = list(SUPREMATIST_PALETTE.values())

_SUPERSAMPLE = 2

# Supersampling quadruples memory and time.  Up to this many output pixels
# (2048x3072, i.e. a 2048 landscape export) we render at 2x; beyond it the
# render is already so large that edge aliasing is invisible, so we render 1x.
_SUPERSAMPLE_PIXEL_BUDGET = 2048 * 3072


def _supersample_for(width: int, height: int) -> int:
    return _SUPERSAMPLE if width * height <= _SUPERSAMPLE_PIXEL_BUDGET else 1

# Cached fiber tile (generated once, reused)
_FIBER_TILE: Image.Image | None = None
_FIBER_TILE_SIZE = 512


def _palette_rgb(palette: list[tuple], idx: int) -> tuple[int, int, int]:
    r, g, b = palette[idx % len(palette)]
    return (int(r * 255), int(g * 255), int(b * 255))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------------------------------------------
# Shape drawing primitives
# ------------------------------------------------------------------

def _draw_square(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                 pw: int, ph: int, color: tuple[int, ...]) -> None:
    side = min(pw, ph)
    x0 = cx - side // 2
    y0 = cy - side // 2
    draw.rectangle([x0, y0, x0 + side, y0 + side], fill=color)


def _draw_rect(draw: ImageDraw.ImageDraw, cx: int, cy: int,
               pw: int, ph: int, color: tuple[int, ...]) -> None:
    x0 = cx - pw // 2
    y0 = cy - ph // 2
    draw.rectangle([x0, y0, x0 + pw, y0 + ph], fill=color)


def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                 pw: int, ph: int, color: tuple[int, ...]) -> None:
    r = min(pw, ph) // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_cross(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                pw: int, ph: int, color: tuple[int, ...]) -> None:
    arm = max(3, min(pw, ph) // 5)
    draw.rectangle([cx - pw // 2, cy - arm // 2,
                    cx + pw // 2, cy + arm // 2], fill=color)
    draw.rectangle([cx - arm // 2, cy - ph // 2,
                    cx + arm // 2, cy + ph // 2], fill=color)


def _draw_trapezoid(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                    pw: int, ph: int, color: tuple[int, ...]) -> None:
    half_w = pw // 2
    half_h = ph // 2
    narrow = max(2, int(half_w * 0.45))
    tl = (cx - narrow, cy - half_h)
    tr = (cx + narrow, cy - half_h)
    br = (cx + half_w, cy + half_h)
    bl = (cx - half_w, cy + half_h)
    draw.polygon([tl, tr, br, bl], fill=color)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                   pw: int, ph: int, color: tuple[int, ...]) -> None:
    top = (cx, cy - ph // 2)
    bl = (cx - pw // 2, cy + ph // 2)
    br = (cx + pw // 2, cy + ph // 2)
    draw.polygon([top, bl, br], fill=color)


def _draw_line(draw: ImageDraw.ImageDraw, cx: int, cy: int,
               pw: int, ph: int, color: tuple[int, ...]) -> None:
    thickness = max(2, min(pw, ph) // 8)
    x0 = cx - pw // 2
    y0 = cy - thickness // 2
    draw.rectangle([x0, y0, x0 + pw, y0 + thickness], fill=color)


def _draw_ellipse(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                  pw: int, ph: int, color: tuple[int, ...]) -> None:
    draw.ellipse([cx - pw // 2, cy - ph // 2,
                  cx + pw // 2, cy + ph // 2], fill=color)


def _draw_semicircle(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                     pw: int, ph: int, color: tuple[int, ...]) -> None:
    rx = pw // 2
    ry = ph // 2
    draw.pieslice([cx - rx, cy - ry, cx + rx, cy + ry],
                  start=0, end=180, fill=color)


def _draw_hexagon(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                  pw: int, ph: int, color: tuple[int, ...]) -> None:
    hw, hh = pw / 2.0, ph / 2.0
    pts = [(cx + hw, cy), (cx + hw * 0.5, cy - hh),
           (cx - hw * 0.5, cy - hh), (cx - hw, cy),
           (cx - hw * 0.5, cy + hh), (cx + hw * 0.5, cy + hh)]
    draw.polygon(pts, fill=color)


def _draw_parallelogram(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                        pw: int, ph: int, color: tuple[int, ...]) -> None:
    hw, hh = pw / 2.0, ph / 2.0
    skew = hw * 0.35
    pts = [(cx - hw + skew, cy - hh), (cx + hw + skew, cy - hh),
           (cx + hw - skew, cy + hh), (cx - hw - skew, cy + hh)]
    draw.polygon(pts, fill=color)


def _draw_lshape(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                 pw: int, ph: int, color: tuple[int, ...]) -> None:
    arm = max(3, min(pw, ph) // 4)
    draw.rectangle([cx - pw // 2, cy - ph // 2,
                    cx - pw // 2 + arm, cy + ph // 2], fill=color)
    draw.rectangle([cx - pw // 2, cy + ph // 2 - arm,
                    cx + pw // 2, cy + ph // 2], fill=color)


def _draw_tshape(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                 pw: int, ph: int, color: tuple[int, ...]) -> None:
    arm = max(3, min(pw, ph) // 4)
    draw.rectangle([cx - pw // 2, cy - ph // 2,
                    cx + pw // 2, cy - ph // 2 + arm], fill=color)
    draw.rectangle([cx - arm // 2, cy - ph // 2,
                    cx + arm // 2, cy + ph // 2], fill=color)


def _draw_arrow(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                pw: int, ph: int, color: tuple[int, ...]) -> None:
    hw, hh = pw / 2.0, ph / 2.0
    shaft = max(2, int(hh * 0.35))
    pts = [(cx, cy - hh), (cx + hw, cy), (cx + hw * 0.35, cy),
           (cx + hw * 0.35, cy + hh), (cx - hw * 0.35, cy + hh),
           (cx - hw * 0.35, cy), (cx - hw, cy)]
    draw.polygon(pts, fill=color)


def _draw_chevron(draw: ImageDraw.ImageDraw, cx: int, cy: int,
                  pw: int, ph: int, color: tuple[int, ...]) -> None:
    hw, hh = pw / 2.0, ph / 2.0
    notch = hh * 0.45
    pts = [(cx - hw, cy - hh), (cx, cy - hh + notch), (cx + hw, cy - hh),
           (cx + hw, cy + hh), (cx, cy + hh - notch), (cx - hw, cy + hh)]
    draw.polygon(pts, fill=color)


_SHAPE_DRAWERS = {
    "square": _draw_square,
    "rect": _draw_rect,
    "circle": _draw_circle,
    "cross": _draw_cross,
    "trapezoid": _draw_trapezoid,
    "triangle": _draw_triangle,
    "line": _draw_line,
    "ellipse": _draw_ellipse,
    "semicircle": _draw_semicircle,
    "hexagon": _draw_hexagon,
    "parallelogram": _draw_parallelogram,
    "lshape": _draw_lshape,
    "tshape": _draw_tshape,
    "arrow": _draw_arrow,
    "chevron": _draw_chevron,
}


# ------------------------------------------------------------------
# 3D projection helpers (Hadid / Architecton mode)
# ------------------------------------------------------------------

def _lighten(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Lighten a color by mixing toward white."""
    return (
        int(rgb[0] + (255 - rgb[0]) * factor),
        int(rgb[1] + (255 - rgb[1]) * factor),
        int(rgb[2] + (255 - rgb[2]) * factor),
    )


def _darken(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Darken a color by mixing toward black."""
    return (
        int(rgb[0] * (1 - factor)),
        int(rgb[1] * (1 - factor)),
        int(rgb[2] * (1 - factor)),
    )


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    inv = 1.0 / max(1, len(pts))
    return (sum(p[0] for p in pts) * inv, sum(p[1] for p in pts) * inv)


def _inset_polygon(pts: list[tuple[float, float]], scale: float) -> list[tuple[float, float]]:
    cx, cy = _centroid(pts)
    return [((x - cx) * scale + cx, (y - cy) * scale + cy) for x, y in pts]


def _get_shape_corners(kind: str, pw: float, ph: float) -> list[tuple[float, float]]:
    hw, hh = pw / 2.0, ph / 2.0
    if kind in ("circle", "ellipse"):
        n = 40
        return [(hw * math.cos(2 * math.pi * k / n), hh * math.sin(2 * math.pi * k / n))
                for k in range(n)]
    if kind == "triangle":
        return [(0, -hh), (hw, hh), (-hw, hh)]
    elif kind == "trapezoid":
        narrow = max(1.0, hw * 0.45)
        return [(-narrow, -hh), (narrow, -hh), (hw, hh), (-hw, hh)]
    elif kind == "line":
        thickness = max(2.0, min(pw, ph) / 8.0)
        ht = thickness / 2.0
        return [(-hw, -ht), (hw, -ht), (hw, ht), (-hw, ht)]
    elif kind == "cross":
        arm = max(1.5, min(pw, ph) / 10.0)
        return [
            (-arm, -hh), (arm, -hh), (arm, -arm),
            (hw, -arm), (hw, arm), (arm, arm),
            (arm, hh), (-arm, hh), (-arm, arm),
            (-hw, arm), (-hw, -arm), (-arm, -arm),
        ]
    elif kind == "hexagon":
        return [
            (hw, 0), (hw * 0.5, -hh), (-hw * 0.5, -hh),
            (-hw, 0), (-hw * 0.5, hh), (hw * 0.5, hh),
        ]
    elif kind == "parallelogram":
        skew = hw * 0.35
        return [
            (-hw + skew, -hh), (hw + skew, -hh),
            (hw - skew, hh), (-hw - skew, hh),
        ]
    elif kind == "lshape":
        arm = max(1.5, min(pw, ph) / 4.0)
        return [
            (-hw, -hh), (-hw + arm, -hh), (-hw + arm, hh - arm),
            (hw, hh - arm), (hw, hh), (-hw, hh),
        ]
    elif kind == "tshape":
        arm = max(1.5, min(pw, ph) / 4.0)
        return [
            (-hw, -hh), (hw, -hh), (hw, -hh + arm),
            (arm / 2, -hh + arm), (arm / 2, hh),
            (-arm / 2, hh), (-arm / 2, -hh + arm), (-hw, -hh + arm),
        ]
    elif kind == "arrow":
        shaft = hw * 0.35
        return [
            (0, -hh), (hw, 0), (shaft, 0),
            (shaft, hh), (-shaft, hh), (-shaft, 0), (-hw, 0),
        ]
    elif kind == "chevron":
        notch = hh * 0.45
        return [
            (-hw, -hh), (0, -hh + notch), (hw, -hh),
            (hw, hh), (0, hh - notch), (-hw, hh),
        ]
    else:
        return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]


def _projection_vector(shape: Shape, width: int, height: int,
                       genome: ShapeGenome) -> tuple[float, float]:
    """Return per-shape projection vector in normalized canvas units.

    Mode 0: single oblique vector.
    Mode 1: blend between two projection vectors across screen space.
    Mode 2: fan perspective from vanishing point.
    """
    mode = int(genome.camera_mode) % 3

    if mode == 0:
        ang = math.radians(genome.projection_angle)
        return (
            math.cos(ang) * genome.projection_strength,
            -math.sin(ang) * genome.projection_strength,
        )

    if mode == 1:
        a1 = math.radians(genome.projection_angle)
        a2 = math.radians(genome.projection_secondary_angle)
        s1 = max(0.0, genome.projection_strength)
        s2 = max(0.0, genome.projection_secondary_strength)
        span = 1.15 + 0.65 * abs(genome.perspective_bias)
        t = 0.5 + (shape.x - genome.projection_vanishing_x) * span
        t = _clamp(t, 0.0, 1.0)
        vx1, vy1 = math.cos(a1) * s1, -math.sin(a1) * s1
        vx2, vy2 = math.cos(a2) * s2, -math.sin(a2) * s2
        return (vx1 * (1.0 - t) + vx2 * t, vy1 * (1.0 - t) + vy2 * t)

    # mode == 2, fan from vanishing point
    px = shape.x * width
    py = shape.y * height
    vp_x = genome.projection_vanishing_x * width
    vp_y = genome.projection_vanishing_y * height
    dx = px - vp_x
    dy = py - vp_y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        ang = math.radians(genome.projection_angle)
        return (
            math.cos(ang) * genome.projection_strength,
            -math.sin(ang) * genome.projection_strength,
        )

    ux = dx / norm
    uy = dy / norm
    dist_factor = _clamp(norm / max(1.0, math.hypot(width, height)), 0.0, 1.0)
    mag = genome.projection_strength * (0.55 + 0.95 * dist_factor)
    mag *= (1.0 + 0.45 * genome.perspective_bias)
    return (ux * mag, uy * mag)


def _draw_face_details(draw: ImageDraw.ImageDraw,
                       front: list[tuple[float, float]],
                       shape: Shape,
                       detail_color: tuple[int, ...],
                       detail_light: tuple[int, ...],
                       width: int, height: int) -> None:
    """Draw bold architectural detail patterns on the front face of a cuboid.

    Each window/cell is sized as a visible fraction of the face so
    patterns read clearly even at thumbnail resolution.
    """
    if shape.detail == "none" or len(front) < 3:
        return

    cfx, cfy = _centroid(front)
    rad = math.radians(shape.rotation)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    pw = max(8, int(shape.w * width))
    ph = max(8, int(shape.h * height))

    if pw < 16 or ph < 16:
        return

    def _lts(lx, ly):
        return (int(lx * cos_r - ly * sin_r + cfx),
                int(lx * sin_r + ly * cos_r + cfy))

    def _rpoly(lx, ly, hw, hh):
        return [_lts(lx - hw, ly - hh), _lts(lx + hw, ly - hh),
                _lts(lx + hw, ly + hh), _lts(lx - hw, ly + hh)]

    span_w = pw * 0.40
    span_h = ph * 0.40
    lw = max(1, int(min(pw, ph) * 0.015))

    if shape.detail == "windows":
        win_w = max(3, int(pw * 0.11))
        win_h = max(3, int(ph * 0.14))
        gap_w = max(2, int(pw * 0.045))
        gap_h = max(2, int(ph * 0.055))
        step_x = win_w + gap_w
        step_y = win_h + gap_h
        cols = max(2, min(6, int(pw * 0.78 / max(1, step_x))))
        rows = max(1, min(5, int(ph * 0.78 / max(1, step_y))))
        for r in range(rows):
            for c in range(cols):
                lx = (c - (cols - 1) / 2.0) * step_x
                ly = (r - (rows - 1) / 2.0) * step_y
                pts = _rpoly(lx, ly, win_w / 2, win_h / 2)
                draw.polygon(pts, fill=detail_color)
                draw.line([pts[0], pts[1]], fill=detail_light, width=1)

    elif shape.detail == "checker":
        cell = max(4, int(min(pw, ph) / 5))
        cols = max(2, min(5, int(pw * 0.75 / cell)))
        rows = max(2, min(5, int(ph * 0.75 / cell)))
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    lx = (c - (cols - 1) / 2.0) * cell
                    ly = (r - (rows - 1) / 2.0) * cell
                    pts = _rpoly(lx, ly, cell / 2.2, cell / 2.2)
                    draw.polygon(pts, fill=detail_color)

    elif shape.detail == "ladder":
        n_rungs = max(3, min(8, ph // 12))
        rung_hw = span_w * 0.85
        rlw = max(2, int(min(pw, ph) * 0.02))
        for i in range(n_rungs):
            ly = (i - (n_rungs - 1) / 2.0) * span_h * 1.7 / max(1, n_rungs - 1)
            draw.line([_lts(-rung_hw, ly), _lts(rung_hw, ly)],
                      fill=detail_color, width=rlw)
        rail_hh = span_h * 0.90
        for side in [-1, 1]:
            draw.line([_lts(side * rung_hw, -rail_hh),
                       _lts(side * rung_hw, rail_hh)],
                      fill=detail_color, width=rlw)

    elif shape.detail == "lines_h":
        n_lines = max(3, min(8, ph // 10))
        line_hw = span_w * 0.92
        for i in range(n_lines):
            ly = (i - (n_lines - 1) / 2.0) * span_h * 1.7 / max(1, n_lines - 1)
            draw.line([_lts(-line_hw, ly), _lts(line_hw, ly)],
                      fill=detail_color, width=lw)

    elif shape.detail == "lines_v":
        n_lines = max(3, min(8, pw // 10))
        line_hh = span_h * 0.92
        for i in range(n_lines):
            lx = (i - (n_lines - 1) / 2.0) * span_w * 1.7 / max(1, n_lines - 1)
            draw.line([_lts(lx, -line_hh), _lts(lx, line_hh)],
                      fill=detail_color, width=lw)

    elif shape.detail == "dots":
        cols = max(2, min(5, pw // 16))
        rows = max(2, min(5, ph // 16))
        dot_r = max(2, int(min(pw, ph) * 0.028))
        for r in range(rows):
            for c in range(cols):
                lx = (c - (cols - 1) / 2.0) * span_w * 1.5 / max(1, cols - 1)
                ly = (r - (rows - 1) / 2.0) * span_h * 1.5 / max(1, rows - 1)
                sx, sy = _lts(lx, ly)
                draw.ellipse([sx - dot_r, sy - dot_r, sx + dot_r, sy + dot_r],
                             fill=detail_color)

    elif shape.detail == "grid_frame":
        gh = max(2, min(5, ph // 14))
        gw = max(2, min(5, pw // 14))
        glw = max(1, int(min(pw, ph) * 0.012))
        frame_hw = span_w * 0.90
        frame_hh = span_h * 0.90
        for i in range(gh + 1):
            ly = -frame_hh + i * (2 * frame_hh / max(1, gh))
            draw.line([_lts(-frame_hw, ly), _lts(frame_hw, ly)],
                      fill=detail_color, width=glw)
        for i in range(gw + 1):
            lx = -frame_hw + i * (2 * frame_hw / max(1, gw))
            draw.line([_lts(lx, -frame_hh), _lts(lx, frame_hh)],
                      fill=detail_color, width=glw)

    elif shape.detail == "cross_brace":
        brace_hw = span_w * 0.88
        brace_hh = span_h * 0.88
        blw = max(2, int(min(pw, ph) * 0.018))
        draw.line([_lts(-brace_hw, -brace_hh), _lts(brace_hw, brace_hh)],
                  fill=detail_color, width=blw)
        draw.line([_lts(brace_hw, -brace_hh), _lts(-brace_hw, brace_hh)],
                  fill=detail_color, width=blw)
        draw.line([_lts(-brace_hw, -brace_hh), _lts(brace_hw, -brace_hh)],
                  fill=detail_color, width=blw)
        draw.line([_lts(brace_hw, -brace_hh), _lts(brace_hw, brace_hh)],
                  fill=detail_color, width=blw)
        draw.line([_lts(brace_hw, brace_hh), _lts(-brace_hw, brace_hh)],
                  fill=detail_color, width=blw)
        draw.line([_lts(-brace_hw, brace_hh), _lts(-brace_hw, -brace_hh)],
                  fill=detail_color, width=blw)


# ------------------------------------------------------------------
# 3D lighting
# ------------------------------------------------------------------

# Single key light from the upper left (screen space, y down, unit length).
_LIGHT_DIR = (-0.55, -0.835)
_PLASTER_WHITE = (238, 233, 222)
_UMBER = (58, 47, 40)           # shade pigment for faces turned from the light
_PAPER = (222, 208, 180)        # aged paper / primed canvas
_GRAPHITE = (72, 68, 64)


def _period_pigment(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Map a palette colour to a 1920s pigment: ~12% less saturated, warmer,
    blacks lifted to warm lamp-black and whites settled to ivory."""
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r, g, b = (x + (lum - x) * 0.12 for x in (r, g, b))
    r, g, b = _mix((int(r), int(g), int(b)), (214, 196, 160), 0.05)
    lo, hi = (30, 27, 25), (242, 237, 224)
    return tuple(int(l + (h - l) * x / 255.0) for x, l, h in zip((r, g, b), lo, hi))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _face_shade(base: tuple[int, int, int], nx: float, ny: float) -> tuple[int, int, int]:
    """Lambert-style shade for a side face with 2D outward normal (nx, ny).

    The front face (facing the viewer) keeps the base colour; side faces
    turned toward the light get lighter, those turned away fall into a
    cool, desaturated shade.
    """
    n = math.hypot(nx, ny) or 1.0
    d = (nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1]) / n
    # Painted, not rendered: three or so flat values per form.
    if d >= 0.35:
        return _lighten(base, 0.16)
    if d >= -0.2:
        return _darken(base, 0.10)
    return _mix(_darken(base, 0.30), _UMBER, 0.22)


def _noise_image(size: tuple[int, int], cells: int, octaves: int = 3,
                 amp: float = 1.0, rng: np.random.Generator | None = None) -> Image.Image:
    """Smooth multi-octave value noise as an 'L' image centred on 128.

    Built from tiny random grids upscaled by PIL, so it's cheap at any size.
    """
    rng = rng or np.random.default_rng()
    w, h = size
    acc = np.zeros((256, 256), dtype=np.float32)
    total = 0.0
    for o in range(octaves):
        n = max(2, cells * (2 ** o))
        small = rng.standard_normal((n, n)).astype(np.float32)
        img = Image.fromarray(((small + 4) / 8 * 255).clip(0, 255).astype(np.uint8), "L")
        up = np.asarray(img.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0 - 0.5
        wgt = 0.5 ** o
        acc += up * wgt
        total += wgt
    acc = acc / total * 2.0 * amp
    out = Image.fromarray((128 + acc * 127).clip(0, 255).astype(np.uint8), "L")
    return out.resize((w, h), Image.BICUBIC)


# ------------------------------------------------------------------
# Oil-paint impasto
#
# A tileable height map of directional brush strokes (with fine bristle
# streaks) is lit from the key light to get raised-paint relief, then
# soft-light blended onto surfaces.  One tile per 15-degree bucket, cached.
# ------------------------------------------------------------------

_STROKE_TILE = 512
_STROKE_CACHE: dict[tuple[int, int], Image.Image] = {}


def _stroke_relief_tile(bucket: int) -> Image.Image:
    rng = _py_random.Random(9173 + bucket)
    size = _STROKE_TILE
    height = Image.new("L", (size, size), 128)
    draw = ImageDraw.Draw(height)
    base_ang = math.radians(bucket * 15.0)

    def _stroke(cx, cy, length, width, value, ang):
        dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
        for ox in (-size, 0, size):          # wrap so the tile is seamless
            for oy in (-size, 0, size):
                draw.line([(cx - dx + ox, cy - dy + oy), (cx + dx + ox, cy + dy + oy)],
                          fill=value, width=width)

    for _ in range(1400):                    # loaded brush strokes
        _stroke(rng.uniform(0, size), rng.uniform(0, size), rng.uniform(22, 85),
                rng.randint(5, 13), rng.randint(96, 164),
                base_ang + math.radians(rng.gauss(0, 11)))
    for _ in range(2200):                    # bristle drag inside the strokes
        _stroke(rng.uniform(0, size), rng.uniform(0, size), rng.uniform(12, 50),
                1, rng.randint(112, 146), base_ang + math.radians(rng.gauss(0, 6)))

    h = np.asarray(height.filter(ImageFilter.GaussianBlur(1.1)), dtype=np.float32)
    gy, gx = np.gradient(h)
    relief = -(gx * _LIGHT_DIR[0] + gy * _LIGHT_DIR[1]) * 2.6 + (h - 128.0) * 0.35
    return Image.fromarray(np.clip(128 + relief, 0, 255).astype(np.uint8), "L")


def _impasto_field(size: tuple[int, int], angle_deg: float, canvas_scale: int,
                   strength: float) -> Image.Image:
    """Relief texture covering ``size``; stroke size tracks the canvas so the
    brushwork reads the same at thumbnail and export resolution."""
    bucket = int(round((angle_deg % 180.0) / 15.0)) % 12
    tile_px = max(80, int(_STROKE_TILE * canvas_scale / 1024 * 0.6))
    key = (bucket, tile_px)
    tile = _STROKE_CACHE.get(key)
    if tile is None:
        tile = _stroke_relief_tile(bucket)
        if tile_px != _STROKE_TILE:
            tile = tile.resize((tile_px, tile_px), Image.BICUBIC)
        if len(_STROKE_CACHE) > 24:
            _STROKE_CACHE.clear()
        _STROKE_CACHE[key] = tile
    w, h = size
    field = Image.new("L", (w, h))
    ox, oy = _py_random.randrange(tile_px), _py_random.randrange(tile_px)
    for y in range(-oy, h, tile_px):
        for x in range(-ox, w, tile_px):
            field.paste(tile, (x, y))
    lut = [int(_clamp(128 + (v - 128) * strength, 0, 255)) for v in range(256)]
    return field.point(lut)


def _apply_impasto(img: Image.Image, angle_deg: float, strength: float,
                   box: tuple[int, int, int, int] | None = None) -> None:
    """Soft-light oil-paint relief onto ``img`` (in place), within ``box``."""
    W, H = img.size
    box = box or (0, 0, W, H)
    region = img.crop(box)
    field = _impasto_field(region.size, angle_deg, min(W, H), strength)
    rgb = ImageChops.soft_light(region.convert("RGB"), Image.merge("RGB", (field,) * 3))
    if region.mode == "RGBA":
        rgb.putalpha(region.getchannel("A"))
    img.paste(rgb, box[:2])


# Per-material surface: (brush relief strength, tonal mottle amplitude).
_MATERIAL_TEXTURE = {
    "concrete": (0.22, 0.05),
    "marble": (0.12, 0.07),
}


def _texture_element(layer: Image.Image, shape: Shape) -> None:
    """Give one element its own faint surface, in place: a light brush relief
    following the element's orientation plus a soft tonal mottle with its own
    random phase, so neighbouring blocks never share one continuous texture."""
    bbox = layer.getbbox()
    if bbox is None:
        return
    relief, mottle_amp = _MATERIAL_TEXTURE.get(shape.material, _MATERIAL_TEXTURE["concrete"])
    angle = shape.rotation + _py_random.gauss(0, 4)
    _apply_impasto(layer, angle, relief, bbox)
    region = layer.crop(bbox)
    mottle = _noise_image(region.size, cells=3, octaves=3, amp=mottle_amp)
    rgb = ImageChops.soft_light(region.convert("RGB"), Image.merge("RGB", (mottle,) * 3))
    rgb.putalpha(region.getchannel("A"))
    layer.paste(rgb, bbox[:2])


def _hand_edge(layer: Image.Image) -> None:
    """Slightly uneven, hand-painted silhouette for one element (in place):
    the alpha edge wobbles by ~a brush hair, never a clean vector line."""
    bbox = layer.getbbox()
    if bbox is None:
        return
    W, H = layer.size
    r = max(0.8, min(W, H) / 1400.0)
    pad = int(r * 3) + 2
    box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
           min(W, bbox[2] + pad), min(H, bbox[3] + pad))
    region = layer.crop(box)
    a = region.getchannel("A").filter(ImageFilter.GaussianBlur(r))
    w, h = region.size
    cells = int(_clamp(max(w, h) / (r * 10), 4, 64))
    wob = _noise_image((w, h), cells=cells, octaves=2, amp=0.35)
    a = ImageChops.add(a, wob, scale=1.0, offset=-128)
    a = a.point(lambda v: int(_clamp((v - 128) * 5 + 128, 0, 255)))
    # Only ever eat inward: growing the alpha would expose the transparent
    # (black) pixels around the form as a dark rim.
    region.putalpha(ImageChops.multiply(a, region.getchannel("A")))
    layer.paste(region, box[:2])


def _glassify(layer: Image.Image) -> Image.Image:
    """Turn an opaque cuboid layer into a translucent plane: faces drop to
    ~40% opacity (a thin glaze), edges stay as faint graphite lines."""
    bbox = layer.getbbox()
    if bbox is None:
        return layer
    region = layer.crop(bbox)
    a = region.getchannel("A")
    edges = ImageChops.lighter(region.convert("L").filter(ImageFilter.FIND_EDGES),
                               a.filter(ImageFilter.FIND_EDGES))
    edges = edges.point(lambda v: 255 if v > 18 else 0)
    if min(layer.size) > 1400:
        edges = edges.filter(ImageFilter.MaxFilter(3))
    edges = ImageChops.multiply(edges, a)
    rgb = region.convert("RGB")
    rgb = Image.composite(Image.new("RGB", rgb.size, _GRAPHITE), rgb, edges)
    rgb.putalpha(ImageChops.lighter(a.point(lambda v: int(v * 0.40)),
                                    edges.point(lambda v: int(v * 0.55))))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.paste(rgb, bbox[:2])
    return out


def _build_stack_3d(genome: ShapeGenome, iw: int, ih: int,
                    palette: list[tuple]) -> Image.Image:
    """Render all cuboids back-to-front with inter-object shadows."""
    shapes = genome.flatten()
    shapes.sort(key=lambda s: s.elevation)
    stack = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    for shape in shapes:
        layer = _render_cuboid_layer(shape, iw, ih, palette, genome)
        _hand_edge(layer)
        if shape.material == "glass":
            layer = _glassify(layer)
        else:
            _texture_element(layer, shape)
        _composite_layer(stack, layer)
    return stack


def _tile_l(tile: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = size
    out = Image.new("L", (w, h))
    tw, th = tile.size
    for y in range(0, h, th):
        for x in range(0, w, tw):
            out.paste(tile, (x, y))
    return out


def _paint_ground(canvas: Image.Image, genome: ShapeGenome, palette: list[tuple],
                  grain: float) -> None:
    """The ground is an element too: a flat, evenly painted sheet (paper tone
    or one muted pigment), optionally divided by one or two large flat
    planes, carrying its own faint paper tooth.  No gradients."""
    w, h = canvas.size
    base = _period_pigment(_palette_rgb(palette, genome.bg_color_idx))
    lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    ground = _mix(base, _PAPER, 0.80 if lum > 150 else 0.12)
    sheet = Image.new("RGBA", (w, h), ground + (255,))

    plane_rgb = _mix(_period_pigment(_palette_rgb(palette, genome.bg_sec_idx)), ground, 0.55)
    draw = ImageDraw.Draw(sheet)
    ext = float(max(w, h)) * 3.0
    cx, cy = genome.bg_cx * w, genome.bg_cy * h
    ang = math.radians(genome.bg_angle)
    if genome.bg_style in (0, 3, 5, 7):
        ux, uy = math.cos(ang), -math.sin(ang)
        px, py = -uy, ux
        draw.polygon([(cx + ux * ext, cy + uy * ext), (cx - ux * ext, cy - uy * ext),
                      (cx - ux * ext + px * ext, cy - uy * ext + py * ext),
                      (cx + ux * ext + px * ext, cy + uy * ext + py * ext)],
                     fill=plane_rgb + (255,))
    if genome.bg_style in (3, 7):
        # A second, narrower plane crossing the first.
        a2 = ang + math.radians(90 if genome.bg_style == 7 else 34)
        ux, uy = math.cos(a2), -math.sin(a2)
        px, py = -uy, ux
        band = max(w, h) * 0.16
        sec = _mix(_period_pigment(_palette_rgb(palette, (genome.bg_sec_idx + 2) % len(palette))),
                   ground, 0.60)
        draw.polygon([(cx + ux * ext, cy + uy * ext), (cx - ux * ext, cy - uy * ext),
                      (cx - ux * ext + px * band, cy - uy * ext + py * band),
                      (cx + ux * ext + px * band, cy + uy * ext + py * band)],
                     fill=sec + (255,))

    # Paper tooth (fibres + a little grain) and very soft uneven wash.
    rgb = _apply_finish(sheet.convert("RGB"), 5.0, grain * 0.5, 0.0)
    wash = _noise_image((w, h), cells=4, octaves=3, amp=0.06)
    rgb = ImageChops.soft_light(rgb, Image.merge("RGB", (wash,) * 3))
    canvas.paste(rgb.convert("RGBA"), (0, 0))


def _draw_construction_lines(canvas: Image.Image, genome: ShapeGenome,
                             box: tuple[int, int, int, int]) -> None:
    """Faint pencil construction lines carried through the main forms'
    axes, as in Lissitzky / Chashnik drawings.  ``box`` is where the
    composition square sits on the canvas."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    scale = min(W, H)
    shapes = sorted(genome.flatten(), key=lambda s: s.w * s.h, reverse=True)[:3]
    if not shapes:
        return
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    lw = max(1, int(round(scale / 1100)))
    ext = float(max(canvas.size)) * 2.0
    rng = _py_random.Random(int(genome.bg_angle * 1000) ^ len(genome.flatten()))
    for s in shapes:
        pdx, pdy = _projection_vector(s, W, H, genome)
        cx = x0 + s.x * W + s.elevation * scale * pdx
        cy = y0 + s.y * H + s.elevation * scale * pdy
        a = math.radians(s.rotation)
        ux, uy = math.cos(a), math.sin(a)
        for side in (-1, 1):
            if rng.random() < 0.45:
                continue
            off = side * s.h * H / 2
            ox, oy = cx - uy * off, cy + ux * off
            draw.line([(ox - ux * ext, oy - uy * ext), (ox + ux * ext, oy + uy * ext)],
                      fill=_GRAPHITE + (int(rng.uniform(38, 70)),), width=lw)
    # Uneven pencil pressure along the lines.
    pressure = _noise_image(canvas.size, cells=10, octaves=2, amp=0.9)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), pressure.point(lambda v: min(255, v + 60))))
    canvas.alpha_composite(layer)


def _render_cuboid_layer(shape: Shape, width: int, height: int,
                          palette: list[tuple],
                          genome: ShapeGenome) -> Image.Image:
    """Render a shape as an oblique-projected 3D extruded polygon with face shading.
    
    Generates local vertices based on shape kind, extrudes them, 
    and dynamically draws visible side faces using edge normals.
    """
    rgb = _period_pigment(_palette_rgb(palette, shape.color_idx))
    # ImageDraw writes RGBA pixels verbatim (no blending), so any alpha < 255
    # fill punches a hole through the face and reads as a hollow wire frame.
    # Architectons are painted solids: always fully opaque.
    alpha = 255
    material = shape.material if shape.material in ("concrete", "marble", "glass") else "concrete"

    # Plaster / stone: palette colour pulled slightly toward warm white so
    # volumes read as painted plaster rather than glossy plastic.
    if material == "glass":
        base = _mix(rgb, _PAPER, 0.25)
    elif material == "marble":
        base = _mix(rgb, _PLASTER_WHITE, 0.14)
    else:
        base = _mix(rgb, _PLASTER_WHITE, 0.06)
    front_color = base + (alpha,)

    # Projection vectors (pixels per unit)
    scale = min(width, height)
    proj_dx, proj_dy = _projection_vector(shape, width, height, genome)

    depth_dx = shape.depth * scale * proj_dx
    depth_dy = shape.depth * scale * proj_dy

    elev_dx = shape.elevation * scale * proj_dx
    elev_dy = shape.elevation * scale * proj_dy

    # Front face dimensions and center (with elevation offset)
    min_px = max(6, int(min(width, height) * 0.014))
    pw = max(min_px, int(shape.w * width))
    ph = max(min_px, int(shape.h * height))
    cx = shape.x * width + elev_dx
    cy = shape.y * height + elev_dy

    # Apply rotation around center
    rad = math.radians(shape.rotation)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    local_corners = _get_shape_corners(shape.kind, float(pw), float(ph))

    front = []
    for lx, ly in local_corners:
        front.append((lx * cos_r - ly * sin_r + cx,
                       lx * sin_r + ly * cos_r + cy))

    # Back face = front displaced by depth projection.
    back = [(x + depth_dx, y + depth_dy) for x, y in front]

    # Final safety fit: keep projected volumes fully inside render bounds.
    pad = 2.0
    all_pts = front + back
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    avail_w = max(1.0, width - 2.0 * pad)
    avail_h = max(1.0, height - 2.0 * pad)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_x > avail_w or span_y > avail_h:
        fit = min(avail_w / max(span_x, 1e-6), avail_h / max(span_y, 1e-6), 1.0)
        cx_all, cy_all = _centroid(all_pts)
        front = [((x - cx_all) * fit + cx_all, (y - cy_all) * fit + cy_all) for x, y in front]
        back = [((x - cx_all) * fit + cx_all, (y - cy_all) * fit + cy_all) for x, y in back]
        all_pts = front + back
        min_x = min(p[0] for p in all_pts)
        max_x = max(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        max_y = max(p[1] for p in all_pts)

    shift_x = 0.0
    shift_y = 0.0
    if min_x < pad:
        shift_x += pad - min_x
    if max_x > width - pad:
        shift_x -= max_x - (width - pad)
    if min_y < pad:
        shift_y += pad - min_y
    if max_y > height - pad:
        shift_y -= max_y - (height - pad)
    if abs(shift_x) > 1e-6 or abs(shift_y) > 1e-6:
        front = [(x + shift_x, y + shift_y) for x, y in front]
        back = [(x + shift_x, y + shift_y) for x, y in back]

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    def _ipoly(pts):
        return [tuple(map(int, p)) for p in pts]

    if shape.depth > 0.002:
        n_pts = len(front)
        faces_to_draw = []
        for i in range(n_pts):
            p1 = front[i]
            p2 = front[(i + 1) % n_pts]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            
            # Outward normal (assuming clockwise vertices and Y-down)
            nx = ey
            ny = -ex
            
            # Dot product with extrusion vector
            dot = nx * depth_dx + ny * depth_dy
            
            if dot > 1e-4:
                face = [p1, p2, back[(i + 1) % n_pts], back[i]]
                
                fc = _face_shade(base, nx, ny) + (alpha,)
                faces_to_draw.append((face, fc))
        
        # Sort faces by depth (furthest along extrusion vector drawn first)
        faces_to_draw.sort(key=lambda item: (
            sum(p[0] for p in item[0]) * depth_dx + 
            sum(p[1] for p in item[0]) * depth_dy
        ), reverse=True)

        for face, fc in faces_to_draw:
            # Outline in the face's own colour: closes AA seams between
            # adjacent faces without drawing cartoon contour lines.
            draw.polygon(_ipoly(face), fill=fc, outline=fc)

    # Front face (always visible, drawn last)
    front_pts = _ipoly(front)
    draw.polygon(front_pts, fill=front_color)

    # Architectural detail patterns on front face
    if shape.detail != "none":
        # Incised, low-contrast panelling rather than black "vents".
        det_dark = _darken(front_color[:3], 0.16) + (alpha,)
        det_light = _lighten(front_color[:3], 0.10) + (alpha,)
        _draw_face_details(draw, front, shape, det_dark, det_light, width, height)

    return layer


# ------------------------------------------------------------------
# Canvas fiber texture
#
# Generates a tileable greyscale texture of fine scratches / fibers
# that simulates canvas or matte-painted surface.  Drawn once,
# cached, and tiled across the image.
# ------------------------------------------------------------------

def _generate_fiber_tile(size: int = 512, num_fibers: int = 3000) -> Image.Image:
    """Create a greyscale tile of fine random fiber lines.

    Returns an 'L' mode image centered around 128 — values above 128
    are lighter scratches, below are darker.  When blended onto the
    painting it creates a natural canvas/paper feel.
    """
    tile = Image.new("L", (size, size), 128)
    draw = ImageDraw.Draw(tile)
    rng = _py_random.Random(42)

    for _ in range(num_fibers):
        cx = rng.randint(0, size - 1)
        cy = rng.randint(0, size - 1)
        angle = rng.uniform(0, math.pi)
        length = rng.randint(4, 35)
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        brightness = rng.randint(108, 148)
        draw.line(
            [(int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))],
            fill=brightness, width=1,
        )

    tile = tile.filter(ImageFilter.GaussianBlur(radius=0.4))
    return tile


def _get_fiber_tile() -> Image.Image:
    global _FIBER_TILE
    if _FIBER_TILE is None:
        _FIBER_TILE = _generate_fiber_tile(_FIBER_TILE_SIZE)
    return _FIBER_TILE


_FIBER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _tile_fiber(width: int, height: int) -> np.ndarray:
    """Tile the fiber texture to cover (width, height).

    Returns a float32 array in [-1, 1] range representing the deviation
    from neutral.  The result is cached (the tile is fixed-seed, so the
    output is deterministic) -- callers must not mutate it in place.
    """
    # Small requests (per-shape scratch layers) are served as views of one
    # shared 1024x1024 tiling instead of re-tiling for every shape.
    if width <= 1024 and height <= 1024:
        key = (1024, 1024)
    else:
        key = (width, height)
    arr = _FIBER_CACHE.get(key)
    if arr is None:
        tile_arr = np.array(_get_fiber_tile()).astype(np.float32)
        ts = _FIBER_TILE_SIZE
        rows = (key[1] + ts - 1) // ts
        cols = (key[0] + ts - 1) // ts
        tiled = np.tile(tile_arr, (rows, cols))[:key[1], :key[0]]
        arr = (tiled - 128.0) / 128.0
        if len(_FIBER_CACHE) > 4:
            _FIBER_CACHE.clear()
        _FIBER_CACHE[key] = arr
    return arr[:height, :width]


def _apply_canvas_texture(canvas: Image.Image, strength: float = 18.0) -> Image.Image:
    """Overlay the fiber texture onto the canvas.

    Works by adding the fiber deviation (scaled by strength) to the
    luminance of each pixel.  Bright fibers lighten, dark fibers darken.
    """
    arr = np.array(canvas).astype(np.float32)
    h, w = arr.shape[:2]
    fibers = _tile_fiber(w, h) * strength

    for c in range(min(arr.shape[2], 3)):
        arr[:, :, c] = np.clip(arr[:, :, c] + fibers, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), canvas.mode)


def _apply_shape_texture(layer: Image.Image, strength: float = 12.0) -> Image.Image:
    """Apply fiber texture to a shape layer, only where alpha > 0.

    Also adds very subtle smooth tonal variation (low-frequency)
    to simulate matte paint unevenness.
    """
    arr = np.array(layer)
    alpha = arr[:, :, 3].astype(np.float32)
    if alpha.max() == 0:
        return layer

    h, w = alpha.shape
    mask = alpha > 10

    # Fiber texture
    fibers = _tile_fiber(w, h) * strength
    # Randomize the crop offset so each shape gets a different patch
    ox = _py_random.randint(0, _FIBER_TILE_SIZE - 1)
    oy = _py_random.randint(0, _FIBER_TILE_SIZE - 1)
    fibers = np.roll(fibers, (oy, ox), axis=(0, 1))

    # Low-frequency tonal variation (smooth matte paint)
    rng = np.random.default_rng()
    lo_h, lo_w = max(2, h // 32), max(2, w // 32)
    lo = rng.standard_normal((lo_h, lo_w), dtype=np.float32)
    lo_up = np.array(
        Image.fromarray(((lo + 3) / 6 * 255).clip(0, 255).astype(np.uint8), "L")
        .resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    matte = (lo_up - 0.5) * strength * 0.6

    # Directional brush drag along the shape's long axis (the region is
    # rotated with the shape afterwards, so strokes follow its orientation).
    if w >= h:
        sh, sw = max(2, h // 3), max(2, w // 48)
    else:
        sh, sw = max(2, h // 48), max(2, w // 3)
    streak = rng.standard_normal((sh, sw), dtype=np.float32)
    streak_up = np.array(
        Image.fromarray(((streak + 3) / 6 * 255).clip(0, 255).astype(np.uint8), "L")
        .resize((w, h), Image.BICUBIC)
    ).astype(np.float32) / 255.0
    brush = (streak_up - 0.5) * strength * 0.55

    combined = fibers + matte + brush

    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        ch[mask] = np.clip(ch[mask] + combined[mask], 0, 255)
        arr[:, :, c] = ch.astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


# ------------------------------------------------------------------
# Painterly edge roughness
# ------------------------------------------------------------------

def _apply_edge_roughness(layer: Image.Image) -> Image.Image:
    """Add hand-painted edge irregularities to a shape layer.

    Detects edges via the alpha-channel gradient and applies:
      - Alpha erosion with random noise (uneven paint coverage at borders)
      - Slight color darkening at edges (paint pooling where strokes end)
      - Tiny speckle overspill just outside the shape boundary
    """
    arr = np.array(layer)
    alpha = arr[:, :, 3].astype(np.float32)
    if alpha.max() == 0:
        return layer

    h, w = alpha.shape
    rng = np.random.default_rng()

    # Edge detection via alpha gradient magnitude
    gy = np.abs(np.diff(alpha, axis=0, prepend=0))
    gx = np.abs(np.diff(alpha, axis=1, prepend=0))
    edge_raw = np.clip((gx + gy) / max(alpha.max(), 1.0), 0, 1)

    # Widen edge band so the effect is visible
    edge_img = Image.fromarray((edge_raw * 255).astype(np.uint8), "L")
    edge_img = edge_img.filter(ImageFilter.MaxFilter(3))
    edge = np.array(edge_img).astype(np.float32) / 255.0

    # 1) Alpha irregularity — random erosion at edges
    noise = rng.random((h, w), dtype=np.float32)
    alpha_jitter = edge * noise * 55
    new_alpha = np.clip(alpha - alpha_jitter, 0, 255)

    # 2) Tiny overspill speckles just outside the boundary
    outside = (alpha < 5) & (edge_raw > 0.1)
    speckle_chance = rng.random((h, w), dtype=np.float32) < 0.12
    speckle_mask = outside & speckle_chance
    new_alpha[speckle_mask] = rng.uniform(20, 70, size=int(speckle_mask.sum()))

    # Copy dominant color into speckle pixels so they aren't black
    if speckle_mask.any():
        for c in range(3):
            ch = arr[:, :, c].copy()
            # Propagate nearby color via a small blur of the original
            filled = np.array(
                Image.fromarray(arr[:, :, c], "L").filter(ImageFilter.BoxBlur(2))
            )
            ch[speckle_mask] = filled[speckle_mask]
            arr[:, :, c] = ch

    arr[:, :, 3] = new_alpha.astype(np.uint8)

    # 3) Color darkening at edges (paint pooling)
    darken = edge * (0.3 + 0.7 * rng.random((h, w), dtype=np.float32)) * 18
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        arr[:, :, c] = np.clip(ch - darken, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


# ------------------------------------------------------------------
# Shape layer rendering
# ------------------------------------------------------------------

def _composite_shape(canvas: Image.Image, shape: Shape,
                     palette: list[tuple]) -> None:
    """Draw one painterly shape and alpha-composite it into ``canvas`` in place.

    Only the shape's own bounding box is touched: the scratch layer is sized
    to the shape (not the whole canvas) and blended into the matching canvas
    region.  This is what keeps large renders from being O(shapes x canvas).
    """
    width, height = canvas.size
    rgb = _palette_rgb(palette, shape.color_idx)
    alpha = int(shape.opacity * 255)
    color = rgb + (alpha,)

    pw = max(1, int(shape.w * width))
    ph = max(1, int(shape.h * height))
    cx = int(shape.x * width)
    cy = int(shape.y * height)

    # Every drawer stays within ~0.68 * hypot(pw, ph) of its centre
    # (the parallelogram skew is the widest case).
    half = int(math.ceil(0.7 * math.hypot(pw, ph))) + 6
    tmp_size = 2 * half
    tmp = Image.new("RGBA", (tmp_size, tmp_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)

    drawer = _SHAPE_DRAWERS.get(shape.kind, _draw_rect)
    drawer(draw, half, half, pw, ph, color)

    # Texture + edge roughness only need to see the painted pixels.
    bbox = tmp.getbbox()
    if bbox is None:
        return
    x0 = max(0, bbox[0] - 3)
    y0 = max(0, bbox[1] - 3)
    x1 = min(tmp_size, bbox[2] + 3)
    y1 = min(tmp_size, bbox[3] + 3)
    region = tmp.crop((x0, y0, x1, y1))
    region = _apply_shape_texture(region, strength=14.0)
    region = _apply_edge_roughness(region)

    # Offset of the painted region's centre from the shape centre.
    rcx = (x0 + x1) / 2.0 - half
    rcy = (y0 + y1) / 2.0 - half

    if shape.rotation % 360 != 0:
        # Rotate just the painted region (expanded to fit) rather than the
        # whole scratch square, then move its centre to where the rotation
        # about the shape centre would have put it.
        region = region.rotate(-shape.rotation, resample=Image.BICUBIC,
                               expand=True)
        th = math.radians(shape.rotation)
        c, sn = math.cos(th), math.sin(th)
        rcx, rcy = rcx * c - rcy * sn, rcx * sn + rcy * c

    # Mask-paste onto a transparent layer (kept from the original pipeline:
    # it gives translucent shapes their characteristic soft, deeper look).
    soft = Image.new("RGBA", region.size, (0, 0, 0, 0))
    soft.paste(region, (0, 0), region)

    rw, rh = soft.size
    dx = int(round(cx + rcx - rw / 2.0))
    dy = int(round(cy + rcy - rh / 2.0))
    cx0 = max(0, dx)
    cy0 = max(0, dy)
    cx1 = min(width, dx + rw)
    cy1 = min(height, dy + rh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas.alpha_composite(
        soft.crop((cx0 - dx, cy0 - dy, cx1 - dx, cy1 - dy)), (cx0, cy0))


def _composite_layer(canvas: Image.Image, layer: Image.Image) -> None:
    """alpha_composite a full-size sparse layer, touching only its bbox."""
    bbox = layer.getbbox()
    if bbox is None:
        return
    canvas.alpha_composite(layer.crop(bbox), (bbox[0], bbox[1]))


# ------------------------------------------------------------------
# Full genome rendering
# ------------------------------------------------------------------

def _apply_grain(image: Image.Image, strength: float = 8.0) -> Image.Image:
    """Add subtle film-grain noise for a painterly, analog feel."""
    if strength <= 0:
        return image
    arr = np.array(image).astype(np.float32)
    h, w = arr.shape[:2]
    rng = np.random.default_rng()
    noise = rng.normal(0, strength, (h, w)).astype(np.float32)
    for c in range(min(arr.shape[2], 3)):
        arr[:, :, c] = np.clip(arr[:, :, c] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), image.mode)


def _apply_color_gradient(image: Image.Image, strength: float = 12.0) -> Image.Image:
    """Apply a subtle warm-to-cool diagonal color tint.

    Top-left gets a warm (slightly amber) shift, bottom-right gets a
    cool (slightly blue) shift.  The effect is very gentle — it adds
    depth and atmosphere without overpowering the palette.
    """
    if strength <= 0:
        return image
    arr = np.array(image).astype(np.float32)
    h, w = arr.shape[:2]

    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    diag = (xx + yy) / 2.0

    warm = (1.0 - diag) * strength
    cool = diag * strength

    arr[:, :, 0] = np.clip(arr[:, :, 0] + warm * 0.7, 0, 255)   # red: warm
    arr[:, :, 1] = np.clip(arr[:, :, 1] + warm * 0.3, 0, 255)   # green: slight warm
    arr[:, :, 2] = np.clip(arr[:, :, 2] + cool * 0.5, 0, 255)   # blue: cool

    return Image.fromarray(arr.astype(np.uint8), image.mode)


_FIBER_NORM: np.ndarray | None = None


def _fiber_band(y0: int, y1: int, width: int) -> np.ndarray:
    """Rows ``y0:y1`` of the fiber texture tiled to ``width`` (float32, [-1, 1]).

    Built per band so full-size renders never hold a canvas-sized texture.
    """
    global _FIBER_NORM
    if _FIBER_NORM is None:
        _FIBER_NORM = (np.array(_get_fiber_tile()).astype(np.float32) - 128.0) / 128.0
    ts = _FIBER_TILE_SIZE
    rows = _FIBER_NORM.take(np.arange(y0, y1) % ts, axis=0)
    reps = (width + ts - 1) // ts
    return np.tile(rows, (1, reps))[:, :width]


_FINISH_BAND = 256


def _apply_finish(image: Image.Image, texture: float, grain: float,
                  gradient: float) -> Image.Image:
    """Canvas fibers + film grain + warm/cool gradient in a single pass.

    Equivalent to ``_apply_canvas_texture`` -> ``_apply_grain`` ->
    ``_apply_color_gradient`` (all three are additive per channel), applied
    in horizontal bands so that memory stays bounded even for 8k canvases.
    """
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    out = np.empty_like(arr)
    rng = np.random.default_rng() if grain > 0 else None
    xs = np.linspace(0, 0.5, w, dtype=np.float32)[None, :]
    ys_all = np.linspace(0, 0.5, h, dtype=np.float32)

    for y0 in range(0, h, _FINISH_BAND):
        y1 = min(h, y0 + _FINISH_BAND)
        base = _fiber_band(y0, y1, w) * np.float32(texture)
        if rng is not None:
            base += rng.standard_normal((y1 - y0, w), dtype=np.float32) * np.float32(grain)
        if gradient > 0:
            diag = ys_all[y0:y1, None] + xs
            warm = (1.0 - diag) * np.float32(gradient)
            cool = diag * np.float32(gradient)
            deltas = (warm * np.float32(0.7), warm * np.float32(0.3),
                      cool * np.float32(0.5))
        else:
            deltas = (None, None, None)
        for c in range(3):
            ch = arr[y0:y1, :, c].astype(np.float32)
            ch += base
            if deltas[c] is not None:
                ch += deltas[c]
            np.clip(ch, 0, 255, out=ch)
            out[y0:y1, :, c] = ch.astype(np.uint8)
    return Image.fromarray(out, image.mode)


_PAPER_WARM = (232, 214, 182)


def _apply_paper_age(image: Image.Image) -> Image.Image:
    """Make a 2D composition read as gouache on an old sheet.

    Applied at output resolution (cheap): pigments eased ~8% toward a warm,
    slightly desaturated period tone, soft multi-scale mottling of the
    ground, and gentle warm toning that creeps in from the sheet's edges.
    """
    w, h = image.size
    rgb = image.convert("RGB")

    # Period pigments: slight desaturation + warm cast.
    grey = rgb.convert("L").convert("RGB")
    rgb = Image.blend(rgb, grey, 0.08)
    warm = Image.new("RGB", (w, h), _PAPER_WARM)
    rgb = ImageChops.multiply(rgb, Image.blend(Image.new("RGB", (w, h), (255, 255, 255)), warm, 0.10))

    # Mottled ground (large soft blotches + finer tooth).
    mottle = _noise_image((w, h), cells=5, octaves=5, amp=0.16)
    rgb = ImageChops.soft_light(rgb, Image.merge("RGB", (mottle, mottle, mottle)))

    # Edge toning: warm darkening toward the border, irregular via noise.
    xs = np.linspace(-1, 1, 128, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    edge = np.clip((np.maximum(np.abs(gx), np.abs(gy)) - 0.84) / 0.16, 0, 1) ** 2.0
    edge_img = Image.fromarray((edge * 255).astype(np.uint8), "L").resize((w, h), Image.BILINEAR)
    edge_noise = _noise_image((w, h), cells=7, octaves=3, amp=0.9)
    edge_img = ImageChops.multiply(edge_img, edge_noise)
    toned = ImageChops.multiply(rgb, Image.new("RGB", (w, h), (214, 190, 150)))
    rgb = Image.composite(toned, rgb, edge_img.point(lambda v: int(v * 0.40)))
    return rgb


# Render settings (module-level defaults, updated by server)
GRAIN_STRENGTH: float = 8.0
GRADIENT_STRENGTH: float = 10.0


def render_genome(genome: ShapeGenome, width: int, height: int,
                  palette: list[tuple] | None = None,
                  grain: float | None = None,
                  gradient: float | None = None) -> Image.Image:
    """Render a genome.  ``grain``/``gradient`` default to the module-level
    settings; pass them explicitly when rendering in worker processes."""
    if palette is None:
        palette = DEFAULT_PALETTE
    if grain is None:
        grain = GRAIN_STRENGTH
    if gradient is None:
        gradient = GRADIENT_STRENGTH

    if genome.projection_strength > 0.01:
        return _render_genome_3d(genome, width, height, palette, grain, gradient)

    ss = _supersample_for(width, height)
    iw = width * ss
    ih = height * ss

    bg_rgb = _palette_rgb(palette, genome.bg_color_idx)
    canvas = Image.new("RGBA", (iw, ih), bg_rgb + (255,))

    for shape in genome.flatten():
        _composite_shape(canvas, shape, palette)

    result = _apply_finish(canvas.convert("RGB"), 14.0, grain, gradient)

    if ss > 1:
        result = result.resize((width, height), Image.LANCZOS)

    return _apply_paper_age(result)


def _render_genome_3d(genome: ShapeGenome, width: int, height: int,
                       palette: list[tuple], grain: float,
                       gradient: float) -> Image.Image:
    """Render an architecton as a 1920s painting of one (Malevich /
    Lissitzky / Chashnik): flat painted ground, faint pencil construction
    lines, then forms back-to-front (painter's algorithm) in flat tonal
    values, each with its own hand-painted edge and surface.
    """
    ss = _supersample_for(width, height)
    iw = width * ss
    ih = height * ss

    canvas = Image.new("RGBA", (iw, ih))
    _paint_ground(canvas, genome, palette, grain)
    _draw_construction_lines(canvas, genome, (0, 0, iw, ih))
    canvas.alpha_composite(_build_stack_3d(genome, iw, ih, palette))

    # Painted on a sheet: no grading, shadows or overlays after this point.
    result = canvas.convert("RGB")

    if ss > 1:
        result = result.resize((width, height), Image.LANCZOS)

    return result


def render_genome_extended(genome: ShapeGenome, square_size: int,
                           orientation: str = "square",
                           palette: list[tuple] | None = None,
                           grain: float | None = None,
                           gradient: float | None = None) -> Image.Image:
    """Render with extended canvas for landscape or portrait output.

    Background is rendered at the full extended dimensions so patterns
    extend seamlessly.  Shapes are rendered at square dimensions and
    composited into the centre.  Post-processing is applied once to the
    unified canvas.
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    if grain is None:
        grain = GRAIN_STRENGTH
    if gradient is None:
        gradient = GRADIENT_STRENGTH

    if orientation == "square":
        return render_genome(genome, square_size, square_size, palette,
                             grain, gradient)

    if orientation == "landscape":
        out_w = int(square_size * 1.5)
        out_h = square_size
    else:
        out_w = square_size
        out_h = int(square_size * 1.5)

    ss = _supersample_for(out_w, out_h)
    iw = out_w * ss
    ih = out_h * ss
    sq_iw = square_size * ss
    sq_ih = square_size * ss
    off_x = (iw - sq_iw) // 2
    off_y = (ih - sq_ih) // 2

    is_3d = genome.projection_strength > 0.01

    bg_rgb = _palette_rgb(palette, genome.bg_color_idx)
    canvas = Image.new("RGBA", (iw, ih), bg_rgb + (255,))

    if is_3d:
        _paint_ground(canvas, genome, palette, grain)
        _draw_construction_lines(canvas, genome, (off_x, off_y, off_x + sq_iw, off_y + sq_ih))
        shapes_layer = _build_stack_3d(genome, sq_iw, sq_ih, palette)
    else:
        shapes_layer = Image.new("RGBA", (sq_iw, sq_ih), (0, 0, 0, 0))
        for shape in genome.flatten():
            _composite_shape(shapes_layer, shape, palette)

    canvas.paste(shapes_layer, (off_x, off_y), shapes_layer)

    result = canvas.convert("RGB")
    if not is_3d:
        result = _apply_finish(result, 14.0, grain, gradient)

    if ss > 1:
        result = result.resize((out_w, out_h), Image.LANCZOS)

    return result if is_3d else _apply_paper_age(result)


def render_population_grid(genomes: list[ShapeGenome], thumb_size: int = 128,
                           cols: int = 5,
                           palette: list[tuple] | None = None) -> Image.Image:
    n = len(genomes)
    rows = (n + cols - 1) // cols
    padding = 4
    label_h = 18
    cell = thumb_size + padding * 2 + label_h
    grid_w = cols * cell + padding
    grid_h = rows * cell + padding

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

    for i, g in enumerate(genomes):
        row, col_idx = divmod(i, cols)
        img = render_genome(g, thumb_size, thumb_size, palette)
        x = col_idx * cell + padding
        y = row * cell + padding + label_h
        grid.paste(img, (x, y))

        draw = ImageDraw.Draw(grid)
        draw.text((x + 2, y - label_h + 2), f"#{i}", fill=(180, 180, 180))

    return grid

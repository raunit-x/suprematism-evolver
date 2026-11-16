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
    if kind in ("circle", "ellipse", "cylinder"):
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
_LIGHT_IVORY = (247, 240, 222)  # the one light every lit face shares
_SHADE_TAUPE = (96, 80, 70)     # the one shade every shaded face shares

# How far the 3D ground sits behind the structure (0 = on the same plane).
# Painted, not optical: ground fields are mixed toward the ground tone
# (lower value and chroma than the forms), laid flat (no relief plates),
# and shadows that reach the ground land detached and lighter.
GROUND_RECESSION = 0.85
GROUND_DROP = 0.075              # extra height above the ground, in canvas units


def _recede(rgb: tuple[int, int, int], ground: tuple[int, int, int],
            ease: float) -> tuple[int, int, int]:
    """A ground-field colour pushed back: its own ease toward the ground
    plus the recession share of what is left."""
    return _mix(rgb, ground, ease + (1.0 - ease) * GROUND_RECESSION * 0.6)


def _period_pigment(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Map a palette colour to a period pigment: barely desaturated (primaries
    stay strong, as in Hadid's Tektonik), a touch warmer, blacks a deep
    lamp-black and whites a clean ivory."""
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r, g, b = (x + (lum - x) * 0.03 for x in (r, g, b))
    r, g, b = _mix((int(r), int(g), int(b)), (214, 196, 160), 0.03)
    lo, hi = (17, 16, 18), (247, 244, 236)
    return tuple(int(l + (h - l) * x / 255.0) for x, l, h in zip((r, g, b), lo, hi))


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _face_shade(base: tuple[int, int, int], nx: float, ny: float) -> tuple[int, int, int]:
    """Painted value for a side face with 2D outward normal (nx, ny).

    One light for the whole picture, as in Lissitzky's Prouns: every face
    turned to the light moves toward the same warm ivory, every face turned
    away toward the same warm umber.  Shared light and shade colours are
    what make differently coloured blocks sit together in one space.
    The roof (front face) keeps the block's own colour.
    """
    n = math.hypot(nx, ny) or 1.0
    d = (nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1]) / n
    # Walls keep their own hue (a darker red is still red); the shared light
    # and shade only tint them, so colour never turns to mud.
    if d >= 0.35:                                   # lit wall
        return _mix(base, _LIGHT_IVORY, 0.26)
    if d >= -0.2:                                   # half shade
        return _mix(_darken(base, 0.28), _SHADE_TAUPE, 0.06)
    return _mix(_darken(base, 0.54), _SHADE_TAUPE, 0.10)   # deep shade


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


OIL_STRENGTH = 0.45    # broken-colour brushwork on every form and the ground
_LOAD_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _stroke_load_tile(bucket: int) -> np.ndarray:
    """Paint load of overlapping oil strokes in one direction: where the
    brush was full the paint is a touch lighter and richer, where it dragged
    a touch thinner.  Zero-mean float32 in about [-1, 1], seamless."""
    rng = _py_random.Random(4471 + bucket)
    size = _STROKE_TILE
    img = Image.new("L", (size, size), 128)
    draw = ImageDraw.Draw(img)
    base = math.radians(bucket * 15.0)
    # Hopper, not Van Gogh: long, broad, smoothly laid strokes of nearly
    # the same value, so a plane reads as clean paint with a quiet trace of
    # the brush.
    for _ in range(420):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        length, width = rng.uniform(90, 260), rng.randint(16, 34)
        ang = base + math.radians(rng.gauss(0, 3))
        dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
        v = int(_clamp(rng.gauss(128, 22), 60, 196))
        for ox in (-size, 0, size):
            for oy in (-size, 0, size):
                draw.line([(cx - dx + ox, cy - dy + oy), (cx + dx + ox, cy + dy + oy)],
                          fill=v, width=width)
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(3.5)), dtype=np.float32)
    return (a - a.mean()) / 64.0


def _load_field(size: tuple[int, int], angle_deg: float, canvas_scale: int,
                scale: float = 1.0) -> np.ndarray:
    bucket = int(round((angle_deg % 180.0) / 15.0)) % 12
    tile_px = max(96, int(_STROKE_TILE * canvas_scale / 1024 * 0.75 * scale))
    key = (bucket, tile_px)
    tile = _LOAD_CACHE.get(key)
    if tile is None:
        t = _stroke_load_tile(bucket)
        if tile_px != _STROKE_TILE:
            im = Image.fromarray(((t + 2) * 63.75).clip(0, 255).astype(np.uint8), "L")
            t = np.asarray(im.resize((tile_px, tile_px), Image.BICUBIC),
                           dtype=np.float32) / 63.75 - 2
        if len(_LOAD_CACHE) > 36:
            _LOAD_CACHE.clear()
        _LOAD_CACHE[key] = t
        tile = t
    w, h = size
    ox, oy = _py_random.randrange(tile_px), _py_random.randrange(tile_px)
    reps_y, reps_x = (h + oy) // tile_px + 2, (w + ox) // tile_px + 2
    big = np.tile(tile, (reps_y, reps_x))
    return big[oy:oy + h, ox:ox + w]


def _oil_paint(arr: np.ndarray, angle_deg: float, canvas_scale: int, amount: float,
               scale: float = 1.0) -> np.ndarray:
    """Broken colour, in place on an (h, w, 3) float array: value follows the
    paint load of each stroke, and each channel drifts a little on its own
    so neighbouring strokes differ slightly in hue, as oil never lies as
    one flat colour."""
    if amount <= 0:
        return arr
    h, w = arr.shape[:2]
    load = _load_field((w, h), angle_deg, canvas_scale, scale)
    arr += (load * np.float32(9.0 * amount))[..., None]
    for c in range(3):
        drift = _load_field((w, h), angle_deg + 4, canvas_scale, scale * 0.8)
        arr[..., c] += drift * np.float32(1.2 * amount)
    return arr


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


def _tooth(h: int, w: int, ref: int) -> np.ndarray:
    """Zero-mean paint/paper tooth, (h, w) float32, unit amplitude; grains
    sized (~2 output px) to survive the supersample downscale."""
    cell = max(1, int(round(ref / 700)))
    small = np.random.default_rng().standard_normal(
        (max(1, h // cell), max(1, w // cell))).astype(np.float32)
    noise = Image.fromarray(((small + 4) * 32).clip(0, 255).astype(np.uint8), "L")
    return np.asarray(noise.resize((w, h), Image.BILINEAR), dtype=np.float32) / 32 - 4


def _texture_element(layer: Image.Image, shape: Shape, grain: float = 8.0) -> None:
    """Give one element its own surface, in place: a brush relief following
    the element's orientation, a tonal mottle with its own random phase, and
    a fine paint tooth.  The Grain setting scales all three, so neighbouring
    blocks never share one continuous texture and the slider is visible on
    the forms, not only on the paper."""
    bbox = layer.getbbox()
    if bbox is None:
        return
    k = 0.5 + max(0.0, grain) / 6.0           # grain 0 -> 0.5x, 8 -> 1.8x, 25 -> 4.7x
    relief, mottle_amp = _MATERIAL_TEXTURE.get(shape.material, _MATERIAL_TEXTURE["concrete"])
    angle = shape.rotation + _py_random.gauss(0, 4)
    _apply_impasto(layer, angle, min(0.9, relief * k + 0.03 * OIL_STRENGTH), bbox)
    region = layer.crop(bbox)
    mottle = _noise_image(region.size, cells=3, octaves=3, amp=min(0.3, mottle_amp * k))
    rgb = ImageChops.soft_light(region.convert("RGB"), Image.merge("RGB", (mottle,) * 3))
    if OIL_STRENGTH > 0:
        arr = _oil_paint(np.asarray(rgb, dtype=np.float32), angle, min(layer.size),
                         OIL_STRENGTH)
        rgb = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    if grain > 0:
        arr = np.asarray(rgb, dtype=np.float32)
        tooth = _tooth(arr.shape[0], arr.shape[1], min(layer.size))
        arr += (tooth * np.float32(grain * 0.9))[..., None]
        rgb = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    rgb.putalpha(region.getchannel("A"))
    layer.paste(rgb, bbox[:2])


def _sweep_mask(mask: Image.Image, dx: int, dy: int) -> Image.Image:
    """Union of ``mask`` shifted by every offset from (0, 0) to (dx, dy):
    doubling, so it is exact at any length in ~log2(length) steps."""
    covered, total = 0, max(abs(dx), abs(dy), 1)
    while covered < total:
        step = min(covered + 1, total - covered)
        sx, sy = int(round(dx * step / total)), int(round(dy * step / total))
        moved = Image.new("L", mask.size, 0)
        moved.paste(mask, (sx, sy))
        mask = ImageChops.lighter(mask, moved)
        covered += step
    return mask


def _shift_mask(mask: Image.Image, dx: int, dy: int) -> Image.Image:
    out = Image.new("L", mask.size, 0)
    out.paste(mask, (dx, dy))
    return out


def _shadow_dir() -> tuple[float, float]:
    lx, ly = _LIGHT_DIR
    n = math.hypot(lx, ly) or 1.0
    return -lx / n, -ly / n


def _paint_cast_shadow(base: Image.Image, layer: Image.Image, shape: Shape,
                       iw: int, ih: int, gap: float = 0.0, far: bool = False) -> None:
    """A Proun-style cast shadow, in place on ``base``: the volume of the
    form swept away from the light onto whatever lies beneath it.  ``gap``
    is how far the form's foot stands above that surface: a form standing
    on it casts a shadow from its foot; a form held high casts one that
    lands detached, further off, so the height between levels reads.
    Painted as a darker value of the surface it falls on.  Hard-edged."""
    bbox = layer.getbbox()
    if bbox is None:
        return
    ux, uy = _shadow_dir()
    size = min(iw, ih)
    near = _clamp(gap * 0.9, 0.0, 0.2) * size
    far = near + _clamp(shape.depth * 0.9, 0.007, 0.14) * size
    nx, ny = int(round(ux * near)), int(round(uy * near))
    fx, fy = int(round(ux * far)), int(round(uy * far))
    x0, y0 = max(0, min(bbox[0], bbox[0] + fx)), max(0, min(bbox[1], bbox[1] + fy))
    x1, y1 = min(iw, max(bbox[2], bbox[2] + fx)), min(ih, max(bbox[3], bbox[3] + fy))
    if x1 <= x0 or y1 <= y0:
        return
    m = layer.getchannel("A").crop((x0, y0, x1, y1))
    if nx or ny:
        m = _shift_mask(m, nx, ny)
    sweep = _sweep_mask(m, fx - nx, fy - ny)
    sweep = sweep.point(lambda v: 255 if v > 110 else 0)       # one hard edge
    if shape.material == "glass":
        sweep = sweep.point(lambda v: v // 2)
    region = base.crop((x0, y0, x1, y1))
    a = region.getchannel("A")
    rgb = region.convert("RGB")
    k = 0.50 + (0.16 * GROUND_RECESSION / 0.55 if far else 0.0)   # far ground: a lighter shadow
    shade = Image.blend(rgb.point(lambda v: int(v * k)),
                        Image.new("RGB", region.size, _SHADE_TAUPE), 0.10)
    out = Image.composite(shade, rgb, sweep)
    out.putalpha(a)
    base.paste(out, (x0, y0))


def _paint_relief(sheet: Image.Image, top: Image.Image, level: int, unit: float,
                  pdir: tuple[float, float]) -> None:
    """Paint a ground field (``top``, RGBA, full size) onto ``sheet`` at a
    relief ``level``: raised plates show a wall along the depth axis and
    cast a hard shadow on the ground; a sunken field carries a shadow band
    inside its light-facing rim.  Level 0 is simply laid flat."""
    mask = top.getchannel("A").point(lambda v: 255 if v > 127 else 0)
    if level == 0:
        sheet.alpha_composite(top)
        return
    ux, uy = _shadow_dir()
    h = abs(level) * unit
    if level > 0:
        sx, sy = int(round(ux * h * 1.2)), int(round(uy * h * 1.2))
        shadow = _sweep_mask(mask, sx, sy)
        rgb = sheet.convert("RGB")
        dark = Image.blend(rgb.point(lambda v: int(v * 0.58)),
                           Image.new("RGB", rgb.size, _SHADE_TAUPE), 0.08)
        rgb = Image.composite(dark, rgb, shadow)
        wx, wy = int(round(pdir[0] * h)), int(round(pdir[1] * h))
        wall = _sweep_mask(mask, wx, wy)
        tone = top.convert("RGB").point(lambda v: int(v * 0.66))
        # Spread the field colour into its wall by sweeping the coloured
        # top too, then darken: walls keep each field's own hue.
        wall_rgb = Image.new("RGB", rgb.size)
        steps = max(1, min(32, abs(wx) + abs(wy)))
        for k in range(steps, 0, -1):
            ox, oy = int(round(wx * k / steps)), int(round(wy * k / steps))
            wall_rgb.paste(tone, (ox, oy), _shift_mask(mask, ox, oy) if (ox or oy) else mask)
        rgb = Image.composite(wall_rgb, rgb, wall)
        rgb = Image.composite(top.convert("RGB"), rgb, mask)
        sheet.paste(rgb.convert("RGBA"), (0, 0))
        return
    sheet.alpha_composite(top)
    d = int(round(h * 1.2))
    inner = ImageChops.subtract(mask, _shift_mask(mask, int(round(ux * d)), int(round(uy * d))))
    rgb = sheet.convert("RGB")
    dark = rgb.point(lambda v: int(v * 0.6))
    sheet.paste(Image.composite(dark, rgb, inner).convert("RGBA"), (0, 0))


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


def _paint_float_shadow(stack: Image.Image, layer: Image.Image, shape: Shape,
                        genome: ShapeGenome, iw: int, ih: int) -> None:
    """The rare floating element: a flat, hard-edged painted shadow of its
    roof dropped onto whatever lies below (never onto the bare ground), so
    the gap between them reads.  Not an airbrushed drop shadow."""
    px, py = _projection_vector(shape, iw, ih, genome)
    drop = (shape.depth + shape.float_gap) * min(iw, ih)
    dx, dy = int(round(px * drop)), int(round(py * drop))
    shadow = Image.new("L", (iw, ih), 0)
    shadow.paste(layer.getchannel("A"), (dx, dy))
    mask = ImageChops.multiply(shadow, stack.getchannel("A")).point(lambda v: int(v * 0.42))
    tone = Image.new("RGBA", (iw, ih), _SHADE_TAUPE + (0,))
    tone.putalpha(mask)
    a = stack.getchannel("A")
    stack.alpha_composite(tone)
    stack.putalpha(a)


def _ground_rgb(genome: ShapeGenome, palette: list[tuple]) -> tuple[int, int, int]:
    base = _period_pigment(_palette_rgb(palette, genome.bg_color_idx))
    lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    return _mix(base, _PAPER, 0.45 if lum > 150 else 0.06)


def _build_stack_3d(genome: ShapeGenome, iw: int, ih: int,
                    palette: list[tuple], base: Image.Image | None = None,
                    grain: float = 8.0) -> Image.Image:
    """Render all forms back-to-front (painter's algorithm by height), each
    with its own depth cue: the top of its roof relative to the
    composition's lowest and highest roofs.  Painted onto ``base`` (the
    ground) when given, so cast shadows can fall on it."""
    shapes = genome.flatten()
    shapes.sort(key=lambda s: s.elevation)
    stack = base if base is not None else Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    ground = _ground_rgb(genome, palette)
    tops = [s.elevation for s in shapes if not s.bleed]
    lo, hi = (min(tops), max(tops)) if tops else (0.0, 0.0)
    from src.shapes.composition import _plan_overlap
    drawn: list[Shape] = []
    for shape in shapes:
        cue = 0.0 if hi - lo < 1e-3 else _clamp(2 * (shape.elevation - lo) / (hi - lo) - 1, -1, 1)
        layer = _render_cuboid_layer(shape, iw, ih, palette, genome, cue, ground)
        if base is not None:
            # Height of this form's foot above whatever is really beneath it
            # (a lower roof, or the ground): the shadow lands that far off.
            foot = shape.elevation - shape.depth
            below = max((o.elevation for o in drawn
                         if o.elevation <= foot + 1e-4 and _plan_overlap(o, shape) > 1e-6),
                        default=0.0)
            on_ground = below <= 1e-6
            drop = GROUND_DROP * GROUND_RECESSION / 0.55 if on_ground else 0.0
            _paint_cast_shadow(stack, layer, shape, iw, ih, max(0.0, foot - below) + drop,
                               far=on_ground and GROUND_RECESSION > 0)
        elif shape.float_gap > 0 and stack.getbbox() is not None:
            _paint_float_shadow(stack, layer, shape, genome, iw, ih)
        drawn.append(shape)
        _hand_edge(layer)
        if shape.material == "glass":
            layer = _glassify(layer)
        else:
            _texture_element(layer, shape, grain)
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
    ground = _ground_rgb(genome, palette)
    sheet = Image.new("RGBA", (w, h), ground + (255,))

    plane_pure = _period_pigment(_palette_rgb(palette, genome.bg_sec_idx))
    plane_rgb = _recede(plane_pure, ground, 0.35)
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
    if genome.bg_style == 8:
        # A great curved field (Hadid, "Malevich's Tektonik"): the edge of a
        # huge disc centred off the sheet, with a narrow pale rim.
        a = math.radians(genome.bg_angle)
        big = float(max(w, h))
        ccx, ccy = w * 0.5 + math.cos(a) * big * 0.95, h * 0.5 + math.sin(a) * big * 0.95
        rad = big * 1.02
        rim = _recede(_period_pigment(_palette_rgb(palette, 15 if len(palette) > 15 else 2)), ground, 0.0)
        band = big * 0.045
        level = 0 if GROUND_RECESSION > 0 else int(getattr(getattr(genome, "comp", None), "ground_level", 0) or 0)
        top = Image.new("RGBA", (w, h), (0, 0, 0, 0)) if level else None
        d2 = ImageDraw.Draw(top) if top is not None else draw
        d2.ellipse([ccx - rad - band, ccy - rad - band, ccx + rad + band, ccy + rad + band],
                   fill=rim + (255,))
        d2.ellipse([ccx - rad, ccy - rad, ccx + rad, ccy + rad], fill=_recede(plane_pure, ground, 0.1) + (255,))
        if top is not None:
            unit, pdir = _ground_relief_axes(genome, w, h)
            _paint_relief(sheet, top, level, unit * 1.4, pdir)
    if genome.bg_style in (3, 7):
        # A second, narrower plane crossing the first.
        a2 = ang + math.radians(90 if genome.bg_style == 7 else 34)
        ux, uy = math.cos(a2), -math.sin(a2)
        px, py = -uy, ux
        band = max(w, h) * 0.16
        sec = _recede(_period_pigment(_palette_rgb(palette, (genome.bg_sec_idx + 2) % len(palette))),
                      ground, 0.60)
        draw.polygon([(cx + ux * ext, cy + uy * ext), (cx - ux * ext, cy - uy * ext),
                      (cx - ux * ext + px * band, cy - uy * ext + py * band),
                      (cx + ux * ext + px * band, cy + uy * ext + py * band)],
                     fill=sec + (255,))

    _paint_marks(draw, genome, palette, ground, w, h, sheet)

    # Paper tooth (fibres + a little grain) and very soft uneven wash.
    rgb = _apply_finish(sheet.convert("RGB"), 5.0, grain * 0.5, 0.0)
    wash = _noise_image((w, h), cells=4, octaves=3, amp=0.06)
    rgb = ImageChops.soft_light(rgb, Image.merge("RGB", (wash,) * 3))
    if OIL_STRENGTH > 0:
        # Broad, quieter strokes laid across the ground in its own direction.
        arr = _oil_paint(np.asarray(rgb, dtype=np.float32), genome.bg_angle + 90.0,
                         min(w, h), OIL_STRENGTH * 0.5, scale=2.2)
        rgb = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    if grain > 0:
        # The ground is painted too: same tooth as the forms, a touch finer.
        arr = np.asarray(rgb, dtype=np.float32)
        arr += (_tooth(h, w, min(w, h)) * np.float32(grain * 0.7))[..., None]
        rgb = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    canvas.paste(rgb.convert("RGBA"), (0, 0))


def _ground_relief_axes(genome: ShapeGenome, w: int, h: int) -> tuple[float, tuple[float, float]]:
    """(px per relief step, unit depth direction) for ground plates."""
    from types import SimpleNamespace
    px, py = _projection_vector(SimpleNamespace(x=0.5, y=0.5), w, h, genome)
    n = math.hypot(px, py) or 1.0
    return min(w, h) * 0.011, (px / n, py / n)


def _paint_marks(draw: ImageDraw.ImageDraw, genome: ShapeGenome, palette: list[tuple],
                 ground: tuple[int, int, int], w: int, h: int,
                 sheet: Image.Image | None = None) -> None:
    """Flat ground primitives from the composition genes (wedges, bands,
    discs, then Malevich lines on top).  Fields are eased a little toward
    the ground so they sit behind the structure; lines stay pure."""
    comp = getattr(genome, "comp", None)
    marks = getattr(comp, "marks", None) or []
    if not marks:
        return
    sz = float(min(w, h))

    def col(mk, ease):
        c = _period_pigment(_palette_rgb(palette, int(mk.get("color", 5))))
        return _recede(c, ground, ease) + (255,)

    def bar(mk, width, length, offset=0.0):
        a = math.radians(mk.get("angle", 0.0))
        ux, uy = math.cos(a), math.sin(a)
        px, py = -uy, ux
        cx = mk["x"] * w + px * offset
        cy = mk["y"] * h + py * offset
        hl, hw = length * sz / 2, width / 2
        return [(cx - ux * hl - px * hw, cy - uy * hl - py * hw),
                (cx + ux * hl - px * hw, cy + uy * hl - py * hw),
                (cx + ux * hl + px * hw, cy + uy * hl + py * hw),
                (cx - ux * hl + px * hw, cy - uy * hl + py * hw)]

    order = {"triangle": 0, "band": 1, "disc": 2, "line": 3}
    unit, pdir = _ground_relief_axes(genome, w, h)
    real_draw = draw
    # Lower plates first, raised ones over them, lines on top.
    for mk in sorted(marks, key=lambda mk: (order.get(mk.get("kind"), 9) == 3,
                                            mk.get("level", 0),
                                            order.get(mk.get("kind"), 9))):
        kind = mk.get("kind")
        level = int(mk.get("level", 0)) if kind != "line" and GROUND_RECESSION <= 0 else 0
        top = None
        if level and sheet is not None:
            top = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(top)
        else:
            draw = real_draw
        if kind == "triangle":
            a = math.radians(mk.get("angle", 0.0))
            half = math.radians(mk.get("spread", 30.0) / 2)
            ln = mk.get("length", 1.0) * sz
            ax, ay = mk["x"] * w, mk["y"] * h
            pts = [(ax, ay),
                   (ax + math.cos(a - half) * ln, ay + math.sin(a - half) * ln),
                   (ax + math.cos(a + half) * ln, ay + math.sin(a + half) * ln)]
            draw.polygon(pts, fill=col(mk, 0.18))
        elif kind == "disc":
            rr = mk.get("radius", 0.15) * sz
            cx, cy = mk["x"] * w, mk["y"] * h
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col(mk, 0.15))
        elif kind == "band":
            draw.polygon(bar(mk, mk.get("width", 0.06) * sz, mk.get("length", 2.0)),
                         fill=col(mk, 0.2))
        elif kind == "line":
            n = int(mk.get("count", 1))
            width = max(1.0, mk.get("width", 0.004) * sz)
            gap = mk.get("gap", 0.02) * sz
            for k in range(n):
                off = (k - (n - 1) / 2) * gap
                draw.polygon(bar(mk, width, mk.get("length", 2.0), off), fill=col(mk, 0.0))
        if top is not None:
            _paint_relief(sheet, top, level, unit, pdir)


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
                          genome: ShapeGenome,
                          depth_cue: float = 0.0,
                          ground: tuple[int, int, int] | None = None) -> Image.Image:
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
    # Depth by level, painted per element: higher roofs come forward a
    # touch lighter, lower ones recede a touch toward the ground colour.
    if depth_cue > 0:
        base = _mix(base, _LIGHT_IVORY, 0.07 * depth_cue)
    elif depth_cue < 0 and ground is not None:
        base = _mix(base, ground, 0.12 * -depth_cue)
    front_color = base + (alpha,)

    # Projection vectors (pixels per unit)
    scale = min(width, height)
    proj_dx, proj_dy = _projection_vector(shape, width, height, genome)

    depth_dx = shape.depth * scale * proj_dx
    depth_dy = shape.depth * scale * proj_dy

    elev_dx = shape.elevation * scale * proj_dx
    elev_dy = shape.elevation * scale * proj_dy

    if shape.kind in ("sphere", "cone", "pyramid", "tetra"):
        return _render_solid_layer(shape, width, height, base, alpha,
                                   (proj_dx, proj_dy), scale)

    # Front face dimensions and center (with elevation offset)
    min_px = max(6, int(min(width, height) * 0.014))
    pw = max(min_px, int(shape.w * width))
    ph = max(min_px, int(shape.h * height))
    # Elevation is the height of the roof: the roof is drawn raised toward
    # the viewer (against the extrusion direction) and the walls run back
    # down from it.  Higher roofs are also painted later, so position and
    # occlusion agree.
    cx = shape.x * width - elev_dx
    cy = shape.y * height - elev_dy

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

    # Final safety fit: keep projected volumes fully inside render bounds
    # (except a bleeding spine, which is meant to run off the sheet).
    pad = -1e9 if shape.bleed else 2.0
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

        painted = material != "glass"
        line_w = max(1, int(round(min(width, height) / 750)))
        for face, fc in faces_to_draw:
            if not painted:
                draw.polygon(_ipoly(face), fill=fc, outline=fc)
                continue
            # Painted gradation down the wall in three quiet steps (lighter
            # under the roof edge, darker toward the base), as on Lissitzky's
            # Proun walls.  Each strip is outlined in its own colour so no
            # seams show.
            p1, p2, b2, b1 = face
            steps = ((0.0, 1 / 3, _mix(fc[:3], _LIGHT_IVORY, 0.05)),
                     (1 / 3, 2 / 3, fc[:3]),
                     (2 / 3, 1.0, _darken(fc[:3], 0.07)))
            for t0, t1, col in steps:
                a0 = (p1[0] + (b1[0] - p1[0]) * t0, p1[1] + (b1[1] - p1[1]) * t0)
                a1 = (p2[0] + (b2[0] - p2[0]) * t0, p2[1] + (b2[1] - p2[1]) * t0)
                c1 = (p2[0] + (b2[0] - p2[0]) * t1, p2[1] + (b2[1] - p2[1]) * t1)
                c0 = (p1[0] + (b1[0] - p1[0]) * t1, p1[1] + (b1[1] - p1[1]) * t1)
                col = col + (alpha,)
                draw.polygon(_ipoly([a0, a1, c1, c0]), fill=col, outline=col)
            # Contact line: where the wall meets whatever it stands on.
            draw.line(_ipoly([b1, b2]), fill=_darken(fc[:3], 0.38) + (alpha,), width=line_w)

    # Front face (always visible, drawn last)
    front_pts = _ipoly(front)
    draw.polygon(front_pts, fill=front_color)

    # Lit arrises: a crisp, thin ivory edge where the roof meets the light
    # (Lissitzky's sharp highlight), the one thing that makes a painted
    # volume look finished rather than cut from card.
    if material != "glass" and shape.depth > 0.002:
        edge_w = max(1, int(round(min(width, height) / 600)))
        hi = _mix(base, _LIGHT_IVORY, 0.65) + (alpha,)
        n_pts = len(front)
        for i in range(n_pts):
            p1, p2 = front[i], front[(i + 1) % n_pts]
            nx, ny = p2[1] - p1[1], -(p2[0] - p1[0])
            nn = math.hypot(nx, ny)
            if nn < 1e-6:
                continue
            if (nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1]) / nn > 0.3:
                draw.line(_ipoly([p1, p2]), fill=hi, width=edge_w)

    # Architectural detail patterns on front face
    if shape.detail != "none":
        # Incised, low-contrast panelling rather than black "vents".
        det_dark = _darken(front_color[:3], 0.16) + (alpha,)
        det_light = _lighten(front_color[:3], 0.10) + (alpha,)
        _draw_face_details(draw, front, shape, det_dark, det_light, width, height)

    return layer


def _construction_ink(base: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Graphite on light forms, pale chalk on dark ones, so the ruled lines
    always read."""
    lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    return (_GRAPHITE if lum > 110 else (226, 220, 204)) + (235,)


def _dashed(draw: ImageDraw.ImageDraw, p0, p1, fill, width: int, dash: float) -> None:
    """A ruled dashed line (construction / hidden edge)."""
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    if dash >= length:
        draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=fill, width=width)
        return
    n = int(length / dash)
    for k in range(0, n, 2):
        t0, t1 = k * dash / length, min(1.0, (k + 1) * dash / length)
        draw.line([(int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                   (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1))],
                  fill=fill, width=width)


def _half_plane(poly, origin, normal):
    """Clip a polygon to the side where (p - origin) . normal >= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da = (a[0] - origin[0]) * normal[0] + (a[1] - origin[1]) * normal[1]
        db = (b[0] - origin[0]) * normal[0] + (b[1] - origin[1]) * normal[1]
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _hull(points):
    pts = sorted(set((round(x, 2), round(y, 2)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _render_solid_layer(shape: Shape, width: int, height: int,
                        base: tuple[int, int, int], alpha: int,
                        proj: tuple[float, float], scale: int) -> Image.Image:
    """Sphere, cone, pyramid or tetrahedron, painted in flat hard-edged
    values (lit / half / shade faces, no gradient, no outline)."""
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    pdx, pdy = proj
    r = shape.w * width / 2.0
    lx, ly = _LIGHT_DIR
    ln = math.hypot(lx, ly) or 1.0
    lx, ly = lx / ln, ly / ln
    lit = _mix(base, _LIGHT_IVORY, 0.14) + (alpha,)
    shade = _face_shade(base, -lx, -ly) + (alpha,)

    def ip(pts):
        return [tuple(map(int, p)) for p in pts]

    if shape.kind == "sphere":
        roof = (shape.x * width - shape.elevation * scale * pdx,
                shape.y * height - shape.elevation * scale * pdy)
        cx = roof[0] + shape.depth * scale * pdx / 2
        cy = roof[1] + shape.depth * scale * pdy / 2
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=shade)
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        litm = Image.new("L", (width, height), 0)
        ox, oy = lx * r * 0.42, ly * r * 0.42
        ImageDraw.Draw(litm).ellipse([cx - r * 0.86 + ox, cy - r * 0.86 + oy,
                                      cx + r * 0.86 + ox, cy + r * 0.86 + oy], fill=255)
        layer.paste(Image.new("RGBA", (width, height), lit), (0, 0),
                    ImageChops.multiply(mask, litm))
        return layer

    floor = shape.elevation - shape.depth
    bc = (shape.x * width - floor * scale * pdx, shape.y * height - floor * scale * pdy)
    apex = (shape.x * width - shape.elevation * scale * pdx,
            shape.y * height - shape.elevation * scale * pdy)
    if shape.kind == "cone":
        hull = _hull([(bc[0] + r * math.cos(2 * math.pi * k / 64),
                       bc[1] + r * math.sin(2 * math.pi * k / 64)) for k in range(64)] + [apex])
        draw.polygon(ip(hull), fill=lit)
        ax, ay = apex[0] - bc[0], apex[1] - bc[1]
        nx, ny = -ay, ax
        if nx * lx + ny * ly > 0:
            nx, ny = -nx, -ny                      # the half turned from the light
        if math.hypot(nx, ny) > 1e-6:
            half = _half_plane(hull, bc, (nx, ny))
            if len(half) >= 3:
                draw.polygon(ip(half), fill=shade)
        return layer

    # Pyramid (square base) or tetrahedron (triangular base): one flat
    # value per face, from its outward normal, painted back to front.
    rad = math.radians(shape.rotation)
    c, sn = math.cos(rad), math.sin(rad)
    hw, hh = shape.w * width / 2, shape.h * height / 2
    if shape.kind == "pyramid":
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    else:
        local = [(0.0, -hh), (hw, hh), (-hw, hh)]
    basep = [(bc[0] + x * c - y * sn, bc[1] + x * sn + y * c) for x, y in local]
    faces = []
    n = len(basep)
    for i in range(n):
        p1, p2 = basep[i], basep[(i + 1) % n]
        mx, my = (p1[0] + p2[0]) / 2 - bc[0], (p1[1] + p2[1]) / 2 - bc[1]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        nx, ny = ey, -ex
        if nx * mx + ny * my < 0:
            nx, ny = -nx, -ny                      # outward
        faces.append(([p1, p2, apex], nx, ny))
    # Faces turned toward the extrusion direction lie in front.
    faces.sort(key=lambda f: f[1] * pdx + f[2] * pdy)
    for pts, nx, ny in faces:
        col = _face_shade(base, nx, ny) + (alpha,)
        draw.polygon(ip(pts), fill=col, outline=col)
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

    if getattr(genome, "city", None) is not None:
        from src.shapes.citymap import render_city
        return render_city(genome.city, width, height, palette, grain)

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
    _build_stack_3d(genome, iw, ih, palette, base=canvas, grain=grain)

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

    if getattr(genome, "city", None) is not None:
        # A map just shows more of the city on a wider/taller sheet.
        from src.shapes.citymap import render_city
        w, h = ((int(square_size * 1.5), square_size) if orientation == "landscape"
                else (square_size, int(square_size * 1.5)))
        return render_city(genome.city, w, h, palette, grain)

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
        box = (off_x, off_y, off_x + sq_iw, off_y + sq_ih)
        shapes_layer = canvas.crop(box)
        _build_stack_3d(genome, sq_iw, sq_ih, palette, base=shapes_layer, grain=grain)
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

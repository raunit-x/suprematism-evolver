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
from PIL import Image, ImageDraw, ImageFilter

from src.shapes.genome import ShapeGenome, ShapeGroup, Shape
from src.art.palettes import SUPREMATIST_PALETTE

DEFAULT_PALETTE = list(SUPREMATIST_PALETTE.values())

_SUPERSAMPLE = 2

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
    if kind == "triangle":
        # Top-center, bottom-right, bottom-left
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
        # 12 vertices clockwise
        return [
            (-arm, -hh), (arm, -hh), (arm, -arm),
            (hw, -arm), (hw, arm), (arm, arm),
            (arm, hh), (-arm, hh), (-arm, arm),
            (-hw, arm), (-hw, -arm), (-arm, -arm)
        ]
    else:
        # Default to bounding box (rect, square, circle, etc.)
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


def _render_cuboid_layer(shape: Shape, width: int, height: int,
                          palette: list[tuple],
                          genome: ShapeGenome) -> Image.Image:
    """Render a shape as an oblique-projected 3D extruded polygon with face shading.
    
    Generates local vertices based on shape kind, extrudes them, 
    and dynamically draws visible side faces using edge normals.
    """
    rgb = _palette_rgb(palette, shape.color_idx)
    alpha = int(shape.opacity * 255)

    is_glass = shape.opacity < 0.68
    if is_glass:
        front_color = _lighten(rgb, 0.34) + (max(70, int(alpha * 0.66)),)
        top_color = _lighten(rgb, 0.48) + (max(55, int(alpha * 0.46)),)
        side_color = _lighten(rgb, 0.22) + (max(45, int(alpha * 0.38)),)
        edge_color = _lighten(rgb, 0.60) + (max(80, int(alpha * 0.60)),)
        highlight_edge = (245, 248, 255, min(255, int(alpha * 0.78) + 68))
        shadow_edge = _darken(rgb, 0.35) + (max(50, int(alpha * 0.42)),)
        seam_color = _lighten(rgb, 0.32) + (max(50, int(alpha * 0.48)),)
    else:
        front_color = rgb + (alpha,)
        top_color = _lighten(rgb, 0.30) + (alpha,)
        side_color = _darken(rgb, 0.25) + (alpha,)
        edge_color = _darken(rgb, 0.50) + (min(255, alpha + 20),)
        highlight_edge = _lighten(rgb, 0.52) + (min(255, alpha + 30),)
        shadow_edge = _darken(rgb, 0.62) + (min(255, alpha + 30),)
        seam_color = _darken(rgb, 0.45) + (max(80, int(alpha * 0.75)),)

    # Projection vectors (pixels per unit)
    scale = min(width, height)
    proj_dx, proj_dy = _projection_vector(shape, width, height, genome)

    depth_dx = shape.depth * scale * proj_dx
    depth_dy = shape.depth * scale * proj_dy

    elev_dx = shape.elevation * scale * proj_dx
    elev_dy = shape.elevation * scale * proj_dy

    # Front face dimensions and center (with elevation offset)
    pw = max(1, int(shape.w * width))
    ph = max(1, int(shape.h * height))
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
        # Soft projected shadow to ground the forms and reduce toy-block look.
        shadow_poly = [
            (x + depth_dx * 1.3 - elev_dx * 0.2, y + depth_dy * 1.3 - elev_dy * 0.2)
            for x, y in front
        ]
        shadow_alpha = int(18 + alpha * 0.09) if is_glass else int(30 + alpha * 0.12)
        draw.polygon(_ipoly(shadow_poly), fill=(0, 0, 0, shadow_alpha))

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
                
                # Shading based on normal direction
                # ny < 0 means pointing UP.
                if abs(ny) > abs(nx):
                    fc = top_color
                else:
                    fc = side_color
                    
                faces_to_draw.append((face, fc))
        
        # Sort faces by depth (furthest along extrusion vector drawn first)
        faces_to_draw.sort(key=lambda item: (
            sum(p[0] for p in item[0]) * depth_dx + 
            sum(p[1] for p in item[0]) * depth_dy
        ), reverse=True)

        for face, fc in faces_to_draw:
            pts = _ipoly(face)
            draw.polygon(pts, fill=fc)
            # Architectural edge lines
            for i in range(len(pts)):
                draw.line([pts[i], pts[(i + 1) % len(pts)]],
                          fill=edge_color, width=1)

    # Front face (always visible, drawn last)
    front_pts = _ipoly(front)
    draw.polygon(front_pts, fill=front_color)

    if len(front) >= 3:
        # Subtle front-face panel modulation for a more finished architectural feel.
        if is_glass:
            inner = _ipoly(_inset_polygon(front, 0.90))
            draw.polygon(inner, fill=_lighten(rgb, 0.52) + (max(20, int(alpha * 0.24)),))
        else:
            inner = _ipoly(_inset_polygon(front, 0.92))
            core = _ipoly(_inset_polygon(front, 0.84))
            draw.polygon(inner, fill=_lighten(rgb, 0.10) + (int(alpha * 0.20),))
            draw.polygon(core, fill=_darken(rgb, 0.08) + (int(alpha * 0.10),))

    if shape.depth > 0.002:
        # Directional edge tinting simulates bevel highlights/shadows.
        light_dir = (-0.68, -0.74)
        for i in range(len(front)):
            p1 = front[i]
            p2 = front[(i + 1) % len(front)]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = ey
            ny = -ex
            dot = nx * light_dir[0] + ny * light_dir[1]
            ec = highlight_edge if dot > 0 else shadow_edge
            draw.line(
                [tuple(map(int, p1)), tuple(map(int, p2))],
                fill=ec,
                width=1,
            )

        # Material-specific surface finish.
        cfx, cfy = _centroid(front)
        if is_glass:
            axis = math.radians(shape.rotation - 28)
            dx = math.cos(axis)
            dy = math.sin(axis)
            half_len = min(pw, ph) * 0.34
            p1 = (cfx - dx * half_len, cfy - dy * half_len)
            p2 = (cfx + dx * half_len, cfy + dy * half_len)
            draw.line([tuple(map(int, p1)), tuple(map(int, p2))], fill=highlight_edge, width=1)
            p1b = (p1[0] + 2.0, p1[1] + 2.0)
            p2b = (p2[0] + 2.0, p2[1] + 2.0)
            draw.line([tuple(map(int, p1b)), tuple(map(int, p2b))], fill=seam_color, width=1)
        elif max(pw, ph) >= 34 and _py_random.random() < 0.72:
            seam_count = 1 + (1 if max(pw, ph) >= 72 and _py_random.random() < 0.45 else 0)
            axis = math.radians(shape.rotation + (90 if pw > ph else 0))
            dx = math.cos(axis)
            dy = math.sin(axis)
            nx = -dy
            ny = dx
            half_len = min(pw, ph) * 0.28
            for s_idx in range(seam_count):
                off = (s_idx - (seam_count - 1) / 2.0) * min(pw, ph) * 0.18
                p1 = (cfx + nx * off - dx * half_len, cfy + ny * off - dy * half_len)
                p2 = (cfx + nx * off + dx * half_len, cfy + ny * off + dy * half_len)
                draw.line([tuple(map(int, p1)), tuple(map(int, p2))], fill=seam_color, width=1)

    return layer


def _render_background_fields(canvas: Image.Image,
                               genome: ShapeGenome,
                               palette: list[tuple]) -> None:
    """Render dramatic Hadid-style backgrounds based on bg_style.

    Styles:
    0 = Bold geometric split field
    1 = Secondary color to black gradient
    2 = Clean solid background
    3 = Multi-band directional fields
    4 = Atmospheric radial sweep
    """
    if genome.bg_style == 2:
        return  # Keep clean solid canvas from base fill.

    w, h = canvas.size

    if genome.bg_style == 1:
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        sec_rgb = _palette_rgb(palette, genome.bg_sec_idx)
        rad = math.radians(genome.bg_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        xs = np.linspace(-1, 1, w)
        ys = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(xs, ys)
        proj = xx * cos_a + yy * sin_a
        proj = (proj - proj.min()) / (proj.max() - proj.min())

        arr[:, :, 0] = (proj * sec_rgb[0]).astype(np.uint8)
        arr[:, :, 1] = (proj * sec_rgb[1]).astype(np.uint8)
        arr[:, :, 2] = (proj * sec_rgb[2]).astype(np.uint8)
        arr[:, :, 3] = 255

        grad_img = Image.fromarray(arr, "RGBA")
        canvas.paste(Image.alpha_composite(canvas, grad_img), (0, 0))
        return

    if genome.bg_style == 3:
        draw = ImageDraw.Draw(canvas)
        sec_rgb = _palette_rgb(palette, genome.bg_sec_idx)
        alt_rgb = _palette_rgb(palette, (genome.bg_sec_idx + 2) % len(palette))
        field_angle = math.radians(genome.bg_angle)
        ux = math.cos(field_angle)
        uy = -math.sin(field_angle)
        ext = float(max(w, h)) * 2.2
        center_x = genome.bg_cx * w
        center_y = genome.bg_cy * h
        for band_idx in range(3):
            shift = (band_idx - 1) * max(w, h) * _py_random.uniform(0.10, 0.24)
            px = center_x + shift * ux
            py = center_y + shift * uy
            la = (px + ux * ext, py + uy * ext)
            lb = (px - ux * ext, py - uy * ext)
            perp_x, perp_y = uy, -ux
            off = float(max(w, h)) * _py_random.uniform(0.45, 0.9)
            lc = (lb[0] + perp_x * off, lb[1] + perp_y * off)
            ld = (la[0] + perp_x * off, la[1] + perp_y * off)
            color = sec_rgb if band_idx != 1 else alt_rgb
            alpha = int(70 + band_idx * 35)
            draw.polygon(
                [tuple(map(int, p)) for p in [la, lb, lc, ld]],
                fill=color + (alpha,),
            )
        return

    if genome.bg_style == 4:
        sec_rgb = np.array(_palette_rgb(palette, genome.bg_sec_idx), dtype=np.float32)
        alt_rgb = np.array(
            _palette_rgb(palette, (genome.bg_sec_idx + 3) % len(palette)),
            dtype=np.float32,
        )
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        cx = np.float32(genome.bg_cx)
        cy = np.float32(genome.bg_cy)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        r = np.clip(r / np.float32(np.sqrt(2.0)), 0.0, 1.0)

        ang = np.float32(math.radians(genome.bg_angle))
        beam = np.clip(1.0 - np.abs(dx * np.cos(ang) + dy * np.sin(ang)) * 2.2, 0.0, 1.0)
        halo = np.clip(1.0 - r * 1.35, 0.0, 1.0)
        mix = np.clip(0.58 * halo + 0.42 * beam, 0.0, 1.0)

        arr = np.zeros((h, w, 4), dtype=np.uint8)
        for c in range(3):
            col = sec_rgb[c] * (0.25 + 0.75 * mix) + alt_rgb[c] * (0.35 * (1.0 - mix))
            arr[:, :, c] = np.clip(col, 0, 255).astype(np.uint8)
        arr[:, :, 3] = 255
        canvas.paste(Image.alpha_composite(canvas, Image.fromarray(arr, "RGBA")), (0, 0))

        # Add one translucent directional slice for extra compositional variety.
        draw = ImageDraw.Draw(canvas)
        slice_ang = math.radians(genome.bg_angle + 20.0)
        ux = math.cos(slice_ang)
        uy = -math.sin(slice_ang)
        ext = float(max(w, h)) * 2.2
        px = w * genome.bg_cx
        py = h * genome.bg_cy
        la = (px + ux * ext, py + uy * ext)
        lb = (px - ux * ext, py - uy * ext)
        perp_x, perp_y = uy, -ux
        off = float(max(w, h)) * 0.55
        lc = (lb[0] + perp_x * off, lb[1] + perp_y * off)
        ld = (la[0] + perp_x * off, la[1] + perp_y * off)
        draw.polygon(
            [tuple(map(int, p)) for p in [la, lb, lc, ld]],
            fill=tuple(sec_rgb.astype(np.uint8)) + (70,),
        )
        return

    # Style 0: Geometric split field.
    draw = ImageDraw.Draw(canvas)
    field_angle = math.radians(genome.bg_angle)
    cos_a = math.cos(field_angle)
    sin_a = math.sin(field_angle)
    px = w * genome.bg_cx
    py = h * genome.bg_cy

    ext = float(max(w, h)) * 2.0
    la = (px + cos_a * ext, py - sin_a * ext)
    lb = (px - cos_a * ext, py + sin_a * ext)

    perp_x, perp_y = -sin_a, -cos_a
    off = float(max(w, h)) * 2.0
    lc = (lb[0] + perp_x * off, lb[1] + perp_y * off)
    ld = (la[0] + perp_x * off, la[1] + perp_y * off)

    sec_idx = genome.bg_sec_idx
    n = len(palette)
    if sec_idx >= n:
        sec_idx = sec_idx % n
    sec_rgb = _palette_rgb(palette, sec_idx)

    poly = [tuple(map(int, p)) for p in [la, lb, lc, ld]]
    draw.polygon(poly, fill=sec_rgb + (220,))


def _apply_atmospheric_blend(image: Image.Image,
                             genome: ShapeGenome,
                             palette: list[tuple]) -> Image.Image:
    """Softly blend background color into rendered structure planes."""
    blend = _clamp(genome.bg_blend, 0.0, 1.0)
    if blend <= 0.01:
        return image

    arr = np.array(image).astype(np.float32)
    h, w = arr.shape[:2]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    rad = math.radians(genome.bg_angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx = (genome.bg_cx * 2.0) - 1.0
    cy = (genome.bg_cy * 2.0) - 1.0

    directed = (xx - cx) * cos_a + (yy - cy) * sin_a
    directed = (directed - directed.min()) / max(1e-6, directed.max() - directed.min())
    directed = np.power(directed, 0.7 if genome.bg_style in (1, 3, 4) else 1.0)

    vignette = np.sqrt(xx * xx + yy * yy)
    vignette = np.clip((vignette - 0.35) / 0.9, 0.0, 1.0)

    mask = np.clip(0.75 * directed + 0.35 * vignette, 0.0, 1.0)
    amount = mask * (0.08 + 0.32 * blend)
    tint = np.array(_palette_rgb(palette, genome.bg_sec_idx), dtype=np.float32)

    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - amount) + tint[c] * amount

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), image.mode)


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


def _tile_fiber(width: int, height: int) -> np.ndarray:
    """Tile the fiber texture to cover (width, height).

    Returns a float32 array in [-1, 1] range representing
    the deviation from neutral.
    """
    tile = _get_fiber_tile()
    tile_arr = np.array(tile).astype(np.float32)
    ts = _FIBER_TILE_SIZE

    rows_needed = (height + ts - 1) // ts
    cols_needed = (width + ts - 1) // ts
    tiled = np.tile(tile_arr, (rows_needed, cols_needed))
    tiled = tiled[:height, :width]

    return (tiled - 128.0) / 128.0


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
    lo = rng.normal(0, 1, (lo_h, lo_w)).astype(np.float32)
    lo_up = np.array(
        Image.fromarray(((lo + 3) / 6 * 255).clip(0, 255).astype(np.uint8), "L")
        .resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    matte = (lo_up - 0.5) * strength * 0.6

    combined = fibers + matte

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
    noise = rng.uniform(0, 1, (h, w)).astype(np.float32)
    alpha_jitter = edge * noise * 55
    new_alpha = np.clip(alpha - alpha_jitter, 0, 255)

    # 2) Tiny overspill speckles just outside the boundary
    outside = (alpha < 5) & (edge_raw > 0.1)
    speckle_chance = rng.uniform(0, 1, (h, w)) < 0.12
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
    darken = edge * rng.uniform(0.3, 1.0, (h, w)).astype(np.float32) * 18
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        arr[:, :, c] = np.clip(ch - darken, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


# ------------------------------------------------------------------
# Shape layer rendering
# ------------------------------------------------------------------

def _render_shape_layer(shape: Shape, width: int, height: int,
                        palette: list[tuple]) -> Image.Image:
    rgb = _palette_rgb(palette, shape.color_idx)
    alpha = int(shape.opacity * 255)
    color = rgb + (alpha,)

    pw = max(1, int(shape.w * width))
    ph = max(1, int(shape.h * height))
    cx = int(shape.x * width)
    cy = int(shape.y * height)

    diag = int(math.ceil(math.hypot(pw, ph))) + 6
    half = diag
    tmp_size = 2 * half
    tmp = Image.new("RGBA", (tmp_size, tmp_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)

    drawer = _SHAPE_DRAWERS.get(shape.kind, _draw_rect)
    drawer(draw, half, half, pw, ph, color)

    tmp = _apply_shape_texture(tmp, strength=14.0)
    tmp = _apply_edge_roughness(tmp)

    if shape.rotation % 360 != 0:
        tmp = tmp.rotate(-shape.rotation, resample=Image.BICUBIC,
                         center=(half, half))

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paste_x = cx - half
    paste_y = cy - half
    layer.paste(tmp, (paste_x, paste_y), tmp)
    return layer


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


# Render settings (module-level defaults, updated by server)
GRAIN_STRENGTH: float = 8.0
GRADIENT_STRENGTH: float = 10.0


def render_genome(genome: ShapeGenome, width: int, height: int,
                  palette: list[tuple] | None = None) -> Image.Image:
    if palette is None:
        palette = DEFAULT_PALETTE

    if genome.projection_strength > 0.01:
        return _render_genome_3d(genome, width, height, palette)

    iw = width * _SUPERSAMPLE
    ih = height * _SUPERSAMPLE

    bg_rgb = _palette_rgb(palette, genome.bg_color_idx)
    canvas = Image.new("RGBA", (iw, ih), bg_rgb + (255,))

    for shape in genome.flatten():
        layer = _render_shape_layer(shape, iw, ih, palette)
        canvas = Image.alpha_composite(canvas, layer)

    result = canvas.convert("RGB")
    result = _apply_canvas_texture(result, strength=14.0)
    result = _apply_grain(result, strength=GRAIN_STRENGTH)
    result = _apply_color_gradient(result, strength=GRADIENT_STRENGTH)

    if _SUPERSAMPLE > 1:
        result = result.resize((width, height), Image.LANCZOS)

    return result


def _render_genome_3d(genome: ShapeGenome, width: int, height: int,
                       palette: list[tuple]) -> Image.Image:
    """Render genome with oblique 3D projection (Hadid / Architecton style).

    Shapes are sorted by elevation (painter's algorithm) and rendered
    as cuboids with face shading and architectural edge lines.
    """
    iw = width * _SUPERSAMPLE
    ih = height * _SUPERSAMPLE

    bg_rgb = _palette_rgb(palette, genome.bg_color_idx)
    canvas = Image.new("RGBA", (iw, ih), bg_rgb + (255,))

    # Dramatic diagonal background fields
    _render_background_fields(canvas, genome, palette)

    # Collect all shapes and sort by elevation (ascending = back to front)
    all_shapes = genome.flatten()
    all_shapes.sort(key=lambda s: s.elevation)

    for shape in all_shapes:
        layer = _render_cuboid_layer(
            shape, iw, ih, palette, genome,
        )
        canvas = Image.alpha_composite(canvas, layer)

    result = canvas.convert("RGB")
    result = _apply_atmospheric_blend(result, genome, palette)
    # Subtle canvas texture (lighter than flat mode for cleaner architectural look)
    result = _apply_canvas_texture(result, strength=6.0)
    result = _apply_grain(result, strength=GRAIN_STRENGTH)
    result = _apply_color_gradient(result, strength=GRADIENT_STRENGTH)

    if _SUPERSAMPLE > 1:
        result = result.resize((width, height), Image.LANCZOS)

    return result


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

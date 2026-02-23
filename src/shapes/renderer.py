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

def render_genome(genome: ShapeGenome, width: int, height: int,
                  palette: list[tuple] | None = None) -> Image.Image:
    if palette is None:
        palette = DEFAULT_PALETTE

    iw = width * _SUPERSAMPLE
    ih = height * _SUPERSAMPLE

    bg_rgb = _palette_rgb(palette, genome.bg_color_idx)
    canvas = Image.new("RGBA", (iw, ih), bg_rgb + (255,))

    for shape in genome.flatten():
        layer = _render_shape_layer(shape, iw, ih, palette)
        canvas = Image.alpha_composite(canvas, layer)

    result = canvas.convert("RGB")
    result = _apply_canvas_texture(result, strength=14.0)

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

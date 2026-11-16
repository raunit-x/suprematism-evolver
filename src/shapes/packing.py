"""Polygon packing inside Suprematist planes, and treatment of overlaps.

Two kinds of gene for the 2D shapes engine:

* ``Shape.pack`` — a plane's interior is packed with smaller polygons
  (Substrate-style splits, city-like lots, bands, wedge rings or concentric
  rings), optionally with round/angled primitives and a second, finer level
  of packing inside large cells.  Cells left unpainted show the plane's own
  colour, so the plane still reads as one form.
* ``ShapeGenome.overlap`` — where a large plane overlaps earlier large planes,
  the shared region is painted differently: a third flat colour (as in
  Lissitzky), its own packing, a knock-out to the ground, or a darker glaze.

Both are evolvable and grow through mutation (``mutate_packing``), so
lineages become more intricate over generations.  Geometry reuses the city-map
packers in ``citymap``.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from src.shapes import citymap as cm

# Basic geometry only: straight splits, simple block grids, bands and
# concentric rings.  ("radial" wedge rings still render for old genomes.)
PACK_MODES = ("split", "grid", "strips", "rings")
_MODE_WEIGHTS = (40, 22, 24, 14)
BASIC_PRIMS = ("disc", "tri", "semi")      # circles, triangles, half-circles
GAP_COLS = ("base", "base", "ground", "dark")
OVERLAP_MODES = ("tint", "pack", "knock", "shade")
STRUCTURE_AREA = 0.012          # planes at least this big (w*h) take part in overlaps
MAX_DEPTH = 3


# ----------------------------------------------------------------------
# Genes
# ----------------------------------------------------------------------

def new_pack(colors: list[int], n_colors: int, kind: str = "rect") -> dict:
    mode = random.choices(PACK_MODES, weights=_MODE_WEIGHTS)[0]
    if kind in ("circle", "ellipse") and random.random() < 0.6:
        mode = "rings"
    pool = list(dict.fromkeys(colors)) or list(range(n_colors))
    k = random.randint(2, 3)
    cols = random.sample(pool, min(k, len(pool)))
    if random.random() < 0.5:
        cols.append(random.randrange(n_colors))
    return {
        "mode": mode, "seed": random.randrange(1 << 30),
        "cell": random.uniform(0.07, 0.3),         # x the plane's short side
        "gap": random.uniform(0.05, 0.22),         # x the cell size
        "share": random.uniform(0.35, 0.85),       # cells painted (rest show the plane)
        "colors": cols,
        "gap_col": random.choice(GAP_COLS),
        "angle": random.choice([0.0, 0.0, 90.0, 45.0, random.uniform(-30, 30)]),
        "depth": 1,
        "prims": [],
    }


def new_overlap(colors: list[int], n_colors: int) -> dict:
    pool = list(dict.fromkeys(colors)) or list(range(n_colors))
    return {"mode": random.choices(OVERLAP_MODES, weights=[34, 30, 18, 18])[0],
            "color": random.randrange(n_colors) if random.random() < 0.5 else random.choice(pool),
            "pack": new_pack(pool, n_colors)}


def _used_colors(genome) -> list[int]:
    return [s.color_idx for s in genome.flatten()]


def seed_packing(genome, n_colors: int) -> None:
    """Sparse initial packing: at most one plane, coarse and partly filled,
    in about a quarter of new genomes; overlaps start as flat treatments
    only.  Evolution (``mutate_packing``) grows the rest."""
    used = _used_colors(genome)
    big = [g.anchor for g in genome.groups if g.anchor.w * g.anchor.h > 0.03]
    if big and random.random() < 0.25:
        s = max(big, key=lambda s: s.w * s.h)
        s.pack = new_pack(used, n_colors, s.kind)
        s.pack["cell"] = random.uniform(0.22, 0.4)          # few, large cells
        s.pack["share"] = random.uniform(0.3, 0.55)
    if random.random() < 0.35:
        genome.overlap = new_overlap(used, n_colors)
        genome.overlap["mode"] = random.choice(["tint", "knock", "shade"])


def _mutate_pack(p: dict, n_colors: int, strength: float) -> None:
    r = random.random()
    if r < 0.2:
        p["cell"] = max(0.04, min(0.45, p["cell"] * math.exp(random.gauss(0, 0.25 * strength))))
    elif r < 0.32:
        p["gap"] = max(0.02, min(0.35, p["gap"] + random.gauss(0, 0.04 * strength)))
    elif r < 0.44:
        p["share"] = max(0.15, min(1.0, p["share"] + random.gauss(0, 0.12 * strength)))
    elif r < 0.6:
        i = random.randrange(len(p["colors"])) if p["colors"] else 0
        if p["colors"]:
            p["colors"][i] = random.randrange(n_colors)
    elif r < 0.68:
        p["seed"] = random.randrange(1 << 30)
    elif r < 0.76:
        p["angle"] = (p["angle"] + random.choice([45.0, 90.0, random.gauss(0, 12)])) % 180
    elif r < 0.84:
        p["gap_col"] = random.choice(GAP_COLS)
    elif r < 0.9:
        p["mode"] = random.choices(PACK_MODES, weights=_MODE_WEIGHTS)[0]
    else:
        if len(p["colors"]) < 5:
            p["colors"].append(random.randrange(n_colors))


def _deepen(p: dict) -> None:
    """Complexify a pack: finer second level, a new primitive, a colour."""
    opts = []
    if p.get("depth", 1) < MAX_DEPTH:
        opts += ["depth"] * 2
    if len(p.get("prims", [])) < len(BASIC_PRIMS):
        opts += ["prim"] * 2
    if p["share"] < 0.95:
        opts += ["share"]
    if not opts:
        return
    op = random.choice(opts)
    if op == "depth":
        p["depth"] = p.get("depth", 1) + 1
    elif op == "prim":
        p.setdefault("prims", []).append(
            random.choice([k for k in BASIC_PRIMS if k not in p.get("prims", [])]))
    else:
        p["share"] = min(1.0, p["share"] + 0.1)


def mutate_packing(genome, n_colors: int, strength: float = 1.0) -> None:
    """Packing and overlap mutation, biased toward adding structure."""
    shapes = [s for s in genome.flatten() if s.w * s.h > 0.006 and s.kind != "line"]
    packed = [s for s in shapes if s.pack]
    used = _used_colors(genome)
    rate = min(1.0, strength)
    if shapes and random.random() < 0.35 * rate:            # pack another plane
        bare = [s for s in shapes if not s.pack]
        if bare:
            s = max(bare, key=lambda s: s.w * s.h * random.uniform(0.5, 1.5))
            s.pack = new_pack(used, n_colors, s.kind)
    if packed and random.random() < 0.3 * rate:
        _mutate_pack(random.choice(packed).pack, n_colors, strength)
    if packed and random.random() < 0.3 * rate:             # elaborate one
        _deepen(random.choice(packed).pack)
    if packed and random.random() < 0.04 * rate:
        random.choice(packed).pack = None
    if genome.overlap is None:
        if random.random() < 0.12 * rate:
            genome.overlap = new_overlap(used, n_colors)
    else:
        r = random.random()
        if r < 0.1 * rate:
            genome.overlap["mode"] = random.choice(OVERLAP_MODES)
        elif r < 0.2 * rate:
            genome.overlap["color"] = random.randrange(n_colors)
        elif r < 0.3 * rate:
            _mutate_pack(genome.overlap["pack"], n_colors, strength)
        elif r < 0.36 * rate:
            _deepen(genome.overlap["pack"])
        elif r < 0.38 * rate:
            genome.overlap = None


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

def _frame_rect(cx: float, cy: float, pw: float, ph: float, rot_deg: float) -> cm.Poly:
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    pts = [(-pw / 2, -ph / 2), (pw / 2, -ph / 2), (pw / 2, ph / 2), (-pw / 2, ph / 2)]
    return cm._ccw([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts])


def pack_cells(p: dict, region: cm.Poly, center: tuple[float, float], short: float,
               rot_deg: float) -> list[tuple[cm.Poly, int]]:
    """Cells for pack gene ``p`` over convex ``region`` (pixel units).
    Returns (polygon, colour slot) where slot -1 = leave the plane's colour,
    -2 = gap colour, otherwise an index into ``p["colors"]``."""
    rng = random.Random(p["seed"])
    cell = max(3.0, p["cell"] * short)
    gap = max(1.0, p["gap"] * cell)
    ang = math.radians(rot_deg + p.get("angle", 0.0))
    raw: list[tuple[cm.Poly, str]] = []
    mode = p["mode"]
    cx, cy = center
    if mode == "split":
        cm._lots_split(region, ang, cell * cell,
                       cm._Fill(lot=cell, street=gap, solid=0.0, voids=0.0, slabs=0.0, rng=rng),
                       raw)
    elif mode == "grid":
        cm._lots_grid(region, ang, cell * rng.uniform(2.2, 3.4), cell * rng.uniform(1.0, 1.6),
                      cm._Fill(lot=cell * 0.6, street=gap, solid=0.08, voids=0.0,
                               slabs=0.0, rng=rng), raw)
    elif mode == "strips":
        u = (math.cos(ang), math.sin(ang))
        lo, hi = cm._extent(region, *u)
        x = lo
        while x < hi:
            w = cell * rng.uniform(0.25, 1.4)
            band = cm._clip_half(cm._clip_half(region, u[0], u[1], x), -u[0], -u[1], -(x + w))
            if len(band) >= 3:
                raw.append((band, "lot"))
            x += w + gap
    elif mode == "radial":
        rmax = max(math.hypot(x - cx, y - cy) for x, y in region)
        r = cell * rng.uniform(0.5, 1.0)
        ring = 0
        while r < rmax and len(raw) < 3000:
            dr = cell * rng.uniform(0.7, 1.3)
            m = max(3, int(2 * math.pi * (r + dr / 2) / (cell * rng.uniform(1.0, 2.2))))
            for i in range(m):
                a0 = ang + 2 * math.pi * i / m + gap / (2 * max(r, 1))
                a1 = ang + 2 * math.pi * (i + 1) / m - gap / (2 * max(r, 1))
                if a1 <= a0:
                    continue
                k = 5
                outer = [(cx + (r + dr - gap / 2) * math.cos(a0 + (a1 - a0) * j / k),
                          cy + (r + dr - gap / 2) * math.sin(a0 + (a1 - a0) * j / k)) for j in range(k + 1)]
                inner = [(cx + (r + gap / 2) * math.cos(a1 - (a1 - a0) * j / k),
                          cy + (r + gap / 2) * math.sin(a1 - (a1 - a0) * j / k)) for j in range(k + 1)]
                raw.append((outer + inner, "lot"))
            r += dr
            ring += 1
        raw.insert(0, (cm._circle(cx, cy, cell * 0.45, cell * 0.45, (1.0, 0.0)), "disc"))
    else:                                                  # rings: drawn outside-in
        rmax = max(math.hypot(x - cx, y - cy) for x, y in region)
        r = rmax
        out: list[tuple[cm.Poly, int]] = []
        k = 0
        while r > cell * 0.3:
            slot = rng.randrange(len(p["colors"])) if rng.random() < p["share"] else -1
            out.append((cm._circle(cx, cy, r, r, (1.0, 0.0), k=48), slot))
            r -= cell * rng.uniform(0.35, 1.0)
            if r > gap:
                out.append((cm._circle(cx, cy, r, r, (1.0, 0.0), k=48), -2))
                r -= gap
            k += 1
        return out

    if p.get("prims"):
        raw = [cm._primitive(pl, rng.choice(p["prims"]), ang, rng)
               if kd != "slab" and rng.random() < 0.3 else (pl, kd) for pl, kd in raw]

    cells: list[tuple[cm.Poly, int]] = []
    n = len(p["colors"])
    weights = [2.0] + [1.0] * (n - 1) if n else []
    for pl, kd in raw:
        if not n or rng.random() >= p["share"]:
            cells.append((pl, -1))
        else:
            cells.append((pl, rng.choices(range(n), weights=weights)[0]))
            if kd == "ring":
                c = cm._circle(*_centre(pl), *(2 * [_radius(pl) * 0.5]), (1.0, 0.0))
                cells.append((c, -1))
        # A second, finer level inside large cells.
        if p.get("depth", 1) >= 2 and abs(cm._area(pl)) > (cell * 1.4) ** 2 \
                and rng.random() < 0.55 and kd not in ("disc", "ring") and len(pl) <= 8:
            sub: list[tuple[cm.Poly, str]] = []
            sc = cell * (0.42 if p["depth"] == 2 else 0.3)
            cm._lots_split(cm._ccw(pl), ang + (math.pi / 2 if rng.random() < 0.4 else 0), sc * sc,
                           cm._Fill(lot=sc, street=max(1.0, gap * 0.55), solid=0.0, voids=0.0,
                                    slabs=0.0, rng=rng), sub)
            for spl, _ in sub:
                if n and rng.random() < 0.6:
                    cells.append((spl, rng.randrange(n)))
    return cells


def _centre(pl):
    return (sum(x for x, _ in pl) / len(pl), sum(y for _, y in pl) / len(pl))


def _radius(pl):
    cx, cy = _centre(pl)
    return max(math.hypot(x - cx, y - cy) for x, y in pl)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _paint_cells(size: tuple[int, int], origin: tuple[int, int], cells, p: dict,
                 base_rgb, ground_rgb, dark_rgb, palette_rgb, alpha: int,
                 fill_first: tuple | None) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = origin
    if fill_first is not None:
        d.rectangle([0, 0, size[0], size[1]], fill=fill_first + (alpha,))
    gap_rgb = {"ground": ground_rgb, "dark": dark_rgb}.get(p.get("gap_col"), None)
    for pl, slot in cells:
        if slot == -1:
            if fill_first is None:
                continue
            rgb = base_rgb
        elif slot == -2:
            rgb = gap_rgb or base_rgb
        else:
            rgb = palette_rgb(p["colors"][slot])
        d.polygon([(x - ox, y - oy) for x, y in pl], fill=rgb + (alpha,))
    return layer


def shape_mask(shape, iw: int, ih: int, drawers: dict, default_drawer) -> Image.Image:
    """Full-canvas L mask of a shape, matching how the renderer draws it."""
    pw = max(1, int(shape.w * iw))
    ph = max(1, int(shape.h * ih))
    cx, cy = int(shape.x * iw), int(shape.y * ih)
    half = int(math.ceil(0.7 * math.hypot(pw, ph))) + 6
    tmp = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
    drawers.get(shape.kind, default_drawer)(ImageDraw.Draw(tmp), half, half, pw, ph,
                                            (255, 255, 255, 255))
    a = tmp.getchannel("A")
    if shape.rotation % 360 != 0:
        a = a.rotate(-shape.rotation, resample=Image.BILINEAR)
    mask = Image.new("L", (iw, ih), 0)
    mask.paste(a, (cx - half, cy - half))
    return mask.filter(ImageFilter.MinFilter(3))      # stay inside the painted edge


def paint_pack(canvas: Image.Image, shape, mask: Image.Image, palette_rgb, ground_rgb,
               dark_rgb, texture) -> None:
    """Pack the interior of ``shape`` (already painted) with its cells."""
    p = shape.pack
    bbox = mask.getbbox()
    if not p or bbox is None:
        return
    iw, ih = canvas.size
    pw, ph = shape.w * iw, shape.h * ih
    cx, cy = shape.x * iw, shape.y * ih
    region = _frame_rect(cx, cy, pw * 1.02, ph * 1.02, shape.rotation)
    cells = pack_cells(p, region, (cx, cy), min(pw, ph), shape.rotation)
    base = palette_rgb(shape.color_idx)
    fill_first = ground_rgb if p.get("gap_col") == "ground" else \
        dark_rgb if p.get("gap_col") == "dark" else None
    x0, y0, x1, y1 = bbox
    layer = _paint_cells((x1 - x0, y1 - y0), (x0, y0), cells, p, base, ground_rgb, dark_rgb,
                         palette_rgb, int(shape.opacity * 255), fill_first)
    layer = texture(layer)
    m = mask.crop(bbox)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), m))
    canvas.alpha_composite(layer, (x0, y0))


def paint_overlap(canvas: Image.Image, overlap: dict, region_mask: Image.Image, shape,
                  palette_rgb, ground_rgb, dark_rgb, texture) -> None:
    """Paint the region where ``shape`` overlaps earlier structures."""
    bbox = region_mask.getbbox()
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    if (x1 - x0) * (y1 - y0) < 16:
        return
    mode = overlap.get("mode", "tint")
    size = (x1 - x0, y1 - y0)
    m = region_mask.crop(bbox)
    if mode == "shade":
        under = canvas.crop(bbox)
        glaze = ImageChops.multiply(under, Image.new("RGBA", size, (150, 132, 120, 255)))
        glaze.putalpha(m)
        canvas.alpha_composite(glaze, (x0, y0))
        return
    if mode == "pack":
        p = dict(overlap["pack"])
        if p["mode"] in ("rings", "radial"):     # rings centred on an overlap read as glyphs
            p["mode"] = "split"
        region = cm._ccw([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        cells = pack_cells(p, region, ((x0 + x1) / 2, (y0 + y1) / 2),
                           max(8.0, min(shape.w * canvas.size[0], shape.h * canvas.size[1])),
                           shape.rotation)
        layer = _paint_cells(size, (x0, y0), cells, p, palette_rgb(overlap["color"]),
                             ground_rgb, dark_rgb, palette_rgb, 255,
                             palette_rgb(overlap["color"]))
    else:
        rgb = ground_rgb if mode == "knock" else palette_rgb(overlap["color"])
        if mode == "tint" and rgb == palette_rgb(shape.color_idx):
            rgb = dark_rgb
        layer = Image.new("RGBA", size, rgb + (255,))
    layer = texture(layer)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), m))
    canvas.alpha_composite(layer, (x0, y0))

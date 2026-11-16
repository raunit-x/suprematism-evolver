"""Paint a SceneGenome the way a panel painter would build it up.

Order of work, back to front, as on a real panel:

  imprimatura   toned ground that shows through wherever paint is thin
  sky           ultramarine/azurite worked down to a pale horizon, with
                clouds whose forms come from the genome's CPPN
  landscape     hill bands in aerial perspective, cypresses on the crests
  architecture  a one-point-perspective room (loggia or nave): tiled floor,
                pilastered side walls, coffered ceiling, arched openings
  figures       draped figures modelled from the light: a tonal
                underpainting (value) is turned into colour with
                Kubelka-Munk mixtures toward lead white in the lights and
                umber in the shadows, then the shadows are glazed with the
                drapery's own pigment, which is what makes them deep
  finish        aged varnish and a craquelure that only shows at size

Every element carries its own brushwork (strokes follow the element: level
in the sky and walls, vertical down the drapery) -- there is no texture laid
over the finished image.

All geometry is computed implicitly per pixel in numpy, so the same genome
renders identically at any resolution.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from src.cppn.network import CPPNNetwork
from src.renaissance import pigments as pg
from src.renaissance.genome import Figure, SceneGenome

P = pg.PIGMENTS
MAX_RENDER = 2048          # larger requests are rendered here and upscaled
SHADOW = pg.mix(P["burnt_umber"], P["bone_black"], 0.45)
STONE = pg.mix_many([("lead_white", 0.75), ("yellow_ochre", 0.2), ("raw_umber", 0.05)])
PIETRA = pg.mix_many([("lead_white", 0.7), ("bone_black", 0.16), ("azurite", 0.14)])


# ---------------------------------------------------------------------------
# Noise and fields
# ---------------------------------------------------------------------------

def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F")
                      .resize((w, h), Image.BICUBIC), dtype=np.float32)


def _noise(rng: np.random.Generator, w: int, h: int, cx: int, cy: int) -> np.ndarray:
    """Smooth value noise with ``cx`` x ``cy`` cells, in [0, 1]."""
    grid = rng.random((max(2, cy) + 1, max(2, cx) + 1)).astype(np.float32)
    return np.clip(_resize(grid, w, h), 0.0, 1.0)


def _fbm(rng, w, h, cx, cy, octaves=4) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += amp * _noise(rng, w, h, cx * 2 ** o, cy * 2 ** o)
        total += amp
        amp *= 0.5
    return out / total


def _cppn_fields(g: SceneGenome, w: int, h: int) -> np.ndarray:
    """Evaluate the genome's CPPN on a coarse grid and upsample: returns
    (3, h, w) fields in roughly [-1, 1] -- clouds, fold drift, terrain."""
    if g.cppn is None:
        return np.zeros((3, h, w), np.float32)
    gw, gh = 96, max(8, int(96 * h / w))
    xs = np.linspace(-1, 1, gw) * (w / h)
    ys = np.linspace(-1, 1, gh)
    xg, yg = np.meshgrid(xs, ys)
    inp = np.column_stack([xg.ravel(), yg.ravel() * 2.0,
                           np.sqrt(xg ** 2 + yg ** 2).ravel(), np.ones(xg.size)])
    out = CPPNNetwork.from_genome(g.cppn).forward(inp).T.reshape(3, gh, gw)
    fields = []
    for f in out:
        f = (f - f.mean()) / (f.std() + 1e-3)
        fields.append(_resize(np.clip(f / 2.0, -1, 1), w, h))
    return np.stack(fields)


def _smooth(x, edge_px):
    """Antialiased/sfumato coverage from a signed distance in pixels."""
    return np.clip(x / edge_px + 0.5, 0.0, 1.0)


class _Ctx:
    def __init__(self, g: SceneGenome, w: int, h: int, brush_scale: float):
        self.g, self.w, self.h = g, w, h
        self.aspect = w / h
        self.rng = np.random.default_rng(g.seed)
        self.feather = max(0.8, 0.0025 * h)
        lx, ly, lz = g.light_x, -g.light_elev, 0.75
        n = math.sqrt(lx * lx + ly * ly + lz * lz)
        self.L = (lx / n, ly / n, lz / n)
        self.brush = g.brush * brush_scale
        self.fields = _cppn_fields(g, w, h)
        # Brush textures: a fixed stroke count per painting, so strokes keep
        # their size relative to the picture at any resolution.
        self.strokes_h = _fbm(self.rng, w, h, 18, 140, 2) - 0.5   # level strokes
        self.strokes_v = _fbm(self.rng, w, h, 160, 22, 2) - 0.5   # vertical strokes
        self.mottle = _fbm(self.rng, w, h, 10, 10, 4) - 0.5
        # Thin passages: broad, gentle, only where a stroke ran dry.
        thin = np.clip(_fbm(self.rng, w, h, 7, 7, 3) - 0.6, 0, 1) * 2.0
        self.opacity = 1.0 - np.clip(thin * self.brush * 0.5, 0, 0.12)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.X = (xs + 0.5) / h          # canvas units: height = 1
        self.Y = (ys + 0.5) / h


def _paint(canvas, alpha, color, ctx: _Ctx, sl=None):
    """Lay ``color`` over ``canvas`` with coverage ``alpha``; thin passages
    let the layer below (ultimately the toned ground) show."""
    if sl is None:
        sl = (slice(None), slice(None))
    a = (alpha * ctx.opacity[sl])[..., None]
    canvas[sl] = canvas[sl] * (1.0 - a) + color * a


def _model(pig: np.ndarray, value: np.ndarray, ctx: _Ctx) -> np.ndarray:
    """Turn a tonal value (0 dark .. 1 light) into paint: lights are the
    pigment mixed toward lead white, shadows toward umber and then glazed
    with the pigment itself."""
    g = ctx.g
    mid = 0.55
    s = np.clip((mid - value) / mid, 0, 1)
    lt = np.clip((value - mid) / (1 - mid), 0, 1)
    pig = np.broadcast_to(pig, value.shape + (3,))
    r = pg.mix(pig, SHADOW, s * (0.25 + 0.6 * g.chiaroscuro))
    r = pg.glaze(r, pig, s * g.glaze * 1.6)
    return pg.mix(r, P["lead_white"], lt ** 1.4 * 0.62)


def _lambert(nx, ny, nz, ctx: _Ctx, ambient=None):
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    lx, ly, lz = ctx.L
    d = np.clip((nx * lx + ny * ly + nz * lz) / n, 0, 1)
    amb = (0.42 - 0.3 * ctx.g.chiaroscuro) if ambient is None else ambient
    return amb + (1 - amb) * d


# ---------------------------------------------------------------------------
# Sky and landscape
# ---------------------------------------------------------------------------

def _sky(canvas, ctx: _Ctx):
    g = ctx.g
    top = pg.mix(P[g.sky_top], P["lead_white"], 0.22)
    low = pg.mix(P[g.sky_low], P["lead_tin_yellow"], 0.15 + 0.3 * g.sky_warmth)
    low = pg.mix(low, P["vermilion"], 0.05 * g.sky_warmth)
    t = np.clip(ctx.Y / (g.horizon + 0.05), 0, 1)
    t = np.clip(t ** 1.3 + ctx.strokes_h * 0.12 * ctx.brush, 0, 1)
    sky = pg.mix(top, low, t)

    # Clouds: CPPN field plus noise, flattened into horizontal banks.
    c = 0.55 * ctx.fields[0] + 0.9 * (_fbm(ctx.rng, ctx.w, ctx.h, 5, 9, 4) - 0.5)
    thr = 0.55 - 0.75 * g.clouds
    cover = np.clip((c - thr) / 0.18, 0, 1) * np.clip((g.horizon - ctx.Y) / 0.08, 0, 1)
    gy, gx = np.gradient(c)
    lit = np.clip(0.6 - (gy * ctx.L[1] + gx * ctx.L[0]) * ctx.h * 0.08, 0, 1)
    cloud_light = pg.mix(P["lead_white"], P["lead_tin_yellow"], 0.15 + 0.2 * g.sky_warmth)
    cloud_dark = pg.mix_many([("lead_white", 0.55), (g.sky_top, 0.25), ("raw_umber", 0.2)])
    cloud = pg.mix(cloud_dark, cloud_light, lit)
    sky = sky * (1 - cover[..., None] * 0.92) + cloud * cover[..., None] * 0.92
    _paint(canvas, np.ones(canvas.shape[:2], np.float32), sky, ctx)

    if g.gloria > 0.05:
        cx = _focus_x(g) * ctx.aspect
        cy = max(0.08, g.horizon * 0.35)
        dx, dy = ctx.X - cx, ctx.Y - cy
        r = np.sqrt(dx * dx + dy * dy)
        R = 0.32
        band = np.floor(np.clip(r / R, 0, 1) * 5) / 5        # painted bands, not a glow
        a = np.where(r < R, (1 - band) * g.gloria * 0.85, 0)
        ang = np.arctan2(dy, dx)
        rays = (np.abs(np.sin(ang * 18)) < 0.06 + 0.002 / (r + 0.05)) & (r < R * 1.7)
        a = np.maximum(a, rays * 0.35 * g.gloria * np.clip(1.7 - r / R, 0, 1))
        a = a * np.clip((g.horizon - ctx.Y) / 0.05, 0, 1)
        gold = pg.mix(P["lead_tin_yellow"], P["lead_white"], 0.3 * band[..., None])
        _paint(canvas, a.astype(np.float32), gold, ctx)


def _focus_x(g: SceneGenome) -> float:
    if not g.figures:
        return 0.5
    main = max(g.figures, key=lambda f: f.scale - f.depth - abs(f.x - 0.5))
    return main.x


def _ridge(ctx: _Ctx, base: float, amp: float, layer: int) -> np.ndarray:
    """Ridge line y(x) for one hill band (per column)."""
    g = ctx.g
    w = ctx.w
    rng = np.random.default_rng(g.seed + 101 * (layer + 1))
    x = np.linspace(0, ctx.aspect, w, dtype=np.float32)
    p = np.zeros(w, np.float32)
    octaves = 2 + int(3 * g.hill_rough)
    amp_o, total, freq = 1.0, 0.0, rng.uniform(0.6, 1.2)
    for o in range(octaves):
        p += amp_o * (0.5 + 0.5 * np.sin(2 * np.pi * freq * x + rng.uniform(0, 2 * np.pi)))
        total += amp_o
        amp_o *= 0.5
        freq *= rng.uniform(1.8, 2.4)
    p /= total
    terr = ctx.fields[2][min(ctx.h - 1, int(base * ctx.h)), :]
    p = np.clip(p + 0.12 * terr, 0, 1.2)
    return base - amp * p


def _landscape(canvas, ctx: _Ctx):
    g = ctx.g
    n = g.hill_layers
    sky_low = pg.mix(P[g.sky_low], P["azurite"], 0.35)
    for i in range(n):
        t = i / max(1, n - 1)                    # 0 far .. 1 near
        base = g.horizon + 0.012 + 0.09 * t
        amp = g.hill_height * (1.1 - 0.6 * t)
        ridge = _ridge(ctx, base, amp, i)
        dist = (ctx.Y - ridge[None, :]) * ctx.h
        alpha = _smooth(dist, ctx.feather * (1.6 - t))
        slope = np.gradient(ridge) * ctx.w
        facing = np.clip(-slope * np.sign(ctx.L[0]) * 3.0, -1, 1)
        depth_below = np.clip((ctx.Y - ridge[None, :]) / 0.15, 0, 1)
        value = 0.58 + 0.22 * facing[None, :] * (1 - depth_below) - 0.12 * depth_below
        value = value + ctx.mottle * 0.25 + ctx.strokes_h * 0.1 * ctx.brush
        near = pg.mix_many([("green_earth", 0.6), ("yellow_ochre", 0.25), ("raw_umber", 0.15)])
        if i == n - 1 and g.setting == "landscape":
            near = pg.mix_many([("green_earth", 0.45), ("yellow_ochre", 0.3), ("burnt_umber", 0.25)])
        local = pg.mix(near, sky_low, 0.85 * (1 - t) ** 1.2)   # aerial perspective
        color = _model(local, np.clip(value, 0, 1).astype(np.float32), ctx)
        _paint(canvas, alpha, color, ctx)
        if 0 < i < n or n == 1:
            _cypresses(canvas, ctx, ridge, t, i)
    if g.setting == "landscape":
        # Foreground meadow running to the bottom edge.
        top = g.horizon + 0.11
        alpha = _smooth((ctx.Y - top) * ctx.h, ctx.feather * 3)
        val = 0.5 + 0.1 * ctx.mottle + 0.1 * ctx.strokes_h * ctx.brush \
            - 0.15 * np.clip((ctx.Y - top) / 0.4, 0, 1)
        meadow = pg.mix_many([("green_earth", 0.5), ("raw_umber", 0.3), ("yellow_ochre", 0.2)])
        _paint(canvas, alpha * 0.9, _model(meadow, val.astype(np.float32), ctx), ctx)


def _cypresses(canvas, ctx: _Ctx, ridge, t: float, layer: int):
    g = ctx.g
    rng = np.random.default_rng(g.seed + 7 * layer)
    count = int(round(g.trees * (0.3 + 0.7 * t) / max(1, g.hill_layers - 1)))
    for _ in range(count):
        x = rng.uniform(0.04, 0.96) * ctx.aspect
        col = min(ctx.w - 1, int(x * ctx.h))
        base = ridge[col] + 0.01
        hh = (0.04 + 0.07 * t) * rng.uniform(0.7, 1.3)
        ww = hh * 0.12
        x0, x1 = int((x - ww * 2) * ctx.h), int((x + ww * 2) * ctx.h) + 1
        y0, y1 = int((base - hh) * ctx.h), int(base * ctx.h) + 1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(ctx.w, x1), min(ctx.h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        X, Y = ctx.X[sl], ctx.Y[sl]
        v = np.clip((base - Y) / hh, 0, 1)            # 0 base .. 1 tip
        half = ww * np.sqrt(np.clip(1 - (2 * v - 0.8) ** 2, 0, 1)) * (1.15 - v * 0.6)
        d = (half - np.abs(X - x)) * ctx.h
        alpha = _smooth(d, ctx.feather) * (v < 0.999)
        nx = np.clip((X - x) / (half + 1e-4), -1, 1)
        val = _lambert(nx, -0.2 * np.ones_like(nx), np.sqrt(1 - nx * nx) + 0.1, ctx, 0.2)
        dark = pg.mix(P["verdigris"], P["bone_black"], 0.55)
        dark = pg.mix(dark, pg.mix(P["azurite"], P["lead_white"], 0.4), 0.5 * (1 - t))
        _paint(canvas, alpha.astype(np.float32), _model(dark, val * 0.8, ctx), ctx, sl)


# ---------------------------------------------------------------------------
# Architecture (one-point perspective box)
# ---------------------------------------------------------------------------

def _room(canvas, ctx: _Ctx):
    g = ctx.g
    vpx, hz = g.vp_x * ctx.aspect, g.horizon
    bw, rt, rf = g.room_width * ctx.aspect, g.room_top, g.room_floor
    dx, dy = ctx.X - vpx, ctx.Y - hz
    s_side = np.abs(dx) / bw
    s_floor = np.where(dy > 0, dy / rf, 0)
    s_ceil = np.where(dy < 0, -dy / rt, 0)
    s = np.maximum(np.maximum(s_side, s_floor), s_ceil)
    s = np.maximum(s, 1e-4)
    Z = 1.0 / s                                    # 1 = back wall, smaller = nearer
    back = s <= 1.0
    floor = (~back) & (s_floor >= s_side) & (s_floor >= s_ceil)
    ceil = (~back) & (s_ceil > s_side) & (s_ceil > s_floor)
    side = (~back) & (~floor) & (~ceil)
    left = side & (dx < 0)

    nave = g.setting == "nave"
    dim = 0.85 if nave else 1.0
    lit_right = ctx.L[0] > 0                       # light travels away from its source
    value = np.full(Z.shape, 0.6, np.float32)
    lit_side = left if not lit_right else (side & ~left)
    value = np.where(side, 0.48, value)
    value = np.where(lit_side, 0.74, value)
    value = np.where(ceil, 0.4, value)
    value = np.where(floor, 0.66 - 0.1 * Z, value)
    if nave:
        value = value - 0.12 * Z * (~floor)        # the far end of a nave falls dark
    value = value + ctx.mottle * 0.15 + ctx.strokes_h * 0.1 * ctx.brush

    color = _model(STONE, np.clip(value * dim, 0, 1).astype(np.float32), ctx)

    # Tiled pavement converging on the vanishing point.
    tiles = g.tiles
    tx = dx * Z / bw * tiles * 0.5
    tz = Z / g.room_width * tiles * 0.5
    px = np.minimum((bw / Z) * ctx.h * 2 / tiles, (rf / Z ** 2) * bw * ctx.h * 2 / tiles)
    q = np.sin(np.pi * tx) * np.sin(np.pi * tz)
    chk = 0.5 + 0.5 * np.clip(q * px * 0.6, -1, 1)
    fa = _model(P[g.floor_a], np.clip(value * dim, 0, 1).astype(np.float32), ctx)
    fb = _model(P[g.floor_b], np.clip(value * dim * 0.9, 0, 1).astype(np.float32), ctx)
    tile_col = fa * chk[..., None] + fb * (1 - chk[..., None])
    color = np.where(floor[..., None], tile_col, color)

    # Coffered ceiling.
    cu = np.abs(((dx * Z / bw) * 3) % 1 - 0.5)
    cv = np.abs((Z * 4) % 1 - 0.5)
    rib = (np.maximum(cu, cv) > 0.38)
    cval = np.where(rib, 0.62, 0.4) * dim
    color = np.where(ceil[..., None], _model(STONE, cval.astype(np.float32), ctx), color)

    # Pietra serena pilasters on the side walls and a cornice all round.
    if g.pilasters:
        zmin = bw / max(vpx, ctx.aspect - vpx)
        step = (1.0 - zmin) / g.pilasters
        zk = 1.0 - np.round((1.0 - Z) / step) * step
        pil = side & (np.abs(Z - zk) < 0.03 + 0.02 * Z) & (Z > zmin)
        pval = np.where(lit_side, 0.66, 0.44) * dim
        color = np.where(pil[..., None], _model(PIETRA, pval.astype(np.float32), ctx), color)
    corn_h = -rt * 0.42
    cornice = (side | back) & (np.abs(dy * Z - corn_h) < 0.008)
    color = np.where(cornice[..., None], _model(PIETRA, np.full(Z.shape, 0.45 * dim, np.float32), ctx), color)

    alpha = np.ones(Z.shape, np.float32)
    # Arched openings in the back wall show the landscape behind.
    n = g.openings
    if n:
        span = bw * 0.8
        ow = span / n * 0.55
        for k in range(n):
            xk = vpx - span + span * 2 * (k + 0.5) / n
            if nave:
                bot, oh = hz - rt * 0.05, rt * 0.6
                ow_k = ow * 0.55
            else:
                bot, oh = hz + rf, (rt + rf) * 0.78
                ow_k = ow
            spring = bot - oh + ow_k
            inside_rect = (np.abs(ctx.X - xk) < ow_k) & (ctx.Y > spring) & (ctx.Y < bot)
            inside_arch = ((ctx.X - xk) ** 2 + (ctx.Y - spring) ** 2 < ow_k ** 2) & (ctx.Y <= spring)
            hole = back & (inside_rect | inside_arch)
            alpha = np.where(hole, 0.0, alpha)
    _paint(canvas, alpha, color, ctx)


def _throne(canvas, ctx: _Ctx, f: Figure, fx: float, fy: float, h: float):
    g = ctx.g
    # Steps under the feet.
    for k, (wf, dh) in enumerate(((0.34, 0.035), (0.28, 0.035))):
        y0 = fy + 0.005 - k * dh
        rect = (np.abs(ctx.X - fx) < wf * h) & (ctx.Y < y0 + dh) & (ctx.Y >= y0)
        val = np.where(ctx.Y < y0 + dh * 0.35, 0.75, 0.5).astype(np.float32)
        _paint(canvas, rect.astype(np.float32), _model(STONE, val, ctx), ctx)
    # Back with a rounded top, framing a cloth of honour.
    half, top = 0.2 * h, fy - 0.98 * h
    inside = (np.abs(ctx.X - fx) < half) & (ctx.Y > top + half) & (ctx.Y < fy)
    cap = ((ctx.X - fx) ** 2 + (ctx.Y - top - half) ** 2 < half ** 2) & (ctx.Y <= top + half)
    shape = (inside | cap).astype(np.float32)
    nx = np.clip((ctx.X - fx) / half, -1, 1)
    val = _lambert(nx * 0.5, np.zeros_like(nx), np.ones_like(nx), ctx) * (1 - 0.4 * g.darkness)
    _paint(canvas, shape, _model(STONE, val.astype(np.float32), ctx), ctx)
    inner = ((np.abs(ctx.X - fx) < half * 0.78) & (ctx.Y > top + half) & (ctx.Y < fy)) | \
            (((ctx.X - fx) ** 2 + (ctx.Y - top - half) ** 2 < (half * 0.78) ** 2) & (ctx.Y <= top + half))
    pattern = 0.5 + 0.08 * np.sign(np.sin(ctx.X * 260 / h * 0.1) * np.sin(ctx.Y * 260 / h * 0.1))
    _paint(canvas, inner.astype(np.float32),
           _model(P[g.cloth_of_honor], (pattern * (1 - 0.3 * g.darkness)).astype(np.float32), ctx), ctx)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# (v, centre offset, half width) of the draped body, in figure heights.
_PROFILES = {
    "standing": ([(0.0, 0.025, 0.15), (0.25, 0.012, 0.122), (0.48, 0.0, 0.105),
                  (0.56, 0.0, 0.088), (0.72, 0.0, 0.10), (0.82, 0.0, 0.105),
                  (0.855, 0.0, 0.075), (0.875, 0.0, 0.03)], 0.0, 0.48),
    "seated":   ([(0.0, 0.04, 0.19), (0.22, 0.05, 0.18), (0.28, 0.04, 0.165),
                  (0.31, 0.0, 0.12), (0.34, 0.0, 0.088), (0.50, 0.0, 0.10),
                  (0.60, 0.0, 0.105), (0.635, 0.0, 0.075), (0.655, 0.0, 0.03)], -0.22, 0.30),
    "kneeling": ([(0.0, 0.07, 0.16), (0.12, 0.06, 0.15), (0.18, 0.0, 0.11),
                  (0.26, 0.0, 0.088), (0.42, 0.0, 0.10), (0.52, 0.0, 0.105),
                  (0.555, 0.0, 0.075), (0.575, 0.0, 0.03)], -0.30, 0.18),
}

# Arms: (shoulder, elbow, hand, in front of the body?) for back and front arm.
_GESTURES = {
    "down":     [((-0.085, 0.835), (-0.11, 0.64), (-0.10, 0.47), False),
                 ((0.085, 0.835), (0.11, 0.64), (0.10, 0.47), True)],
    "prayer":   [((-0.085, 0.835), (-0.06, 0.62), (0.035, 0.70), True),
                 ((0.085, 0.835), (0.08, 0.62), (0.045, 0.705), True)],
    "open":     [((-0.085, 0.835), (-0.14, 0.65), (-0.20, 0.60), False),
                 ((0.085, 0.835), (0.14, 0.65), (0.21, 0.62), True)],
    "raised":   [((-0.085, 0.835), (-0.10, 0.63), (-0.03, 0.54), True),
                 ((0.085, 0.835), (0.15, 0.74), (0.15, 0.93), True)],
    "pointing": [((-0.085, 0.835), (-0.11, 0.64), (-0.10, 0.47), False),
                 ((0.085, 0.835), (0.17, 0.78), (0.30, 0.80), True)],
    "cradle":   [((-0.085, 0.835), (-0.09, 0.63), (0.03, 0.60), True),
                 ((0.085, 0.835), (0.10, 0.64), (0.13, 0.63), True)],
}


def _seg_dist(u, v, a, b):
    """Distance from points to segment ab, plus the offset vector."""
    ax, ay = a
    bx, by = b
    ex, ey = bx - ax, by - ay
    t = np.clip(((u - ax) * ex + (v - ay) * ey) / (ex * ex + ey * ey + 1e-9), 0, 1)
    px, py = u - (ax + t * ex), v - (ay + t * ey)
    return np.sqrt(px * px + py * py), px, py


class _FigureFrame:
    def __init__(self, ctx: _Ctx, f: Figure, fx: float, fy: float, h: float):
        self.ctx, self.f, self.h = ctx, f, h
        hp = h * ctx.h
        self.hp = hp
        x0 = int((fx - 0.5 * h) * ctx.h)
        x1 = int((fx + 0.5 * h) * ctx.h) + 1
        y0 = int((fy - 1.08 * h) * ctx.h)
        y1 = int((fy + 0.02 * h) * ctx.h) + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(ctx.w, x1), min(ctx.h, y1)
        self.empty = x1 <= x0 or y1 <= y0
        self.sl = (slice(y0, y1), slice(x0, x1))
        X, Y = ctx.X[self.sl], ctx.Y[self.sl]
        self.v = (fy - Y) / h
        self.u = ((X - fx) / h - f.lean * self.v) * f.facing
        self.edge = ctx.feather / hp           # feather in figure units

    def cover(self, d):
        """Coverage from a signed distance in figure units."""
        return np.clip(d / self.edge + 0.5, 0, 1).astype(np.float32)

    def to_image_normal(self, nx, ny, nz):
        return nx * self.f.facing, ny, nz


def _shade_drapery(fr: _FigureFrame, pig, nx_l, phase, amp, ctx: _Ctx):
    f = fr.f
    fold = amp * np.sin(2 * np.pi * phase)
    valley = 1.0 - 0.28 * amp * (0.5 + 0.5 * np.cos(2 * np.pi * phase))
    nx = np.clip(nx_l + 0.55 * fold, -1.2, 1.2)
    nz = np.sqrt(np.clip(1 - nx_l * nx_l, 0.03, 1))
    ix, iy, iz = fr.to_image_normal(nx, -0.15 * np.ones_like(nx), nz)
    val = _lambert(ix, iy, iz, ctx) * valley
    val = val + ctx.strokes_v[fr.sl] * 0.14 * ctx.brush + 0.06 * (fr.v - 0.5)
    return _model(P[pig], np.clip(val, 0, 1).astype(np.float32), ctx)


def _sphere(fr: _FigureFrame, a, b, rx, ry, pig_rgb, ctx: _Ctx, amb=None):
    nx, ny = a / rx, b / ry
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.02, 1))
    ix, iy, iz = fr.to_image_normal(nx, -ny, nz)
    val = _lambert(ix, iy, iz, ctx, amb)
    return _model(pig_rgb, np.clip(val + ctx.mottle[fr.sl] * 0.05, 0, 1).astype(np.float32), ctx)


def _ellipse_d(fr, cu, cv, rx, ry, tilt=0.0):
    a, b = fr.u - cu, fr.v - cv
    if tilt:
        ca, sa = math.cos(tilt), math.sin(tilt)
        a, b = a * ca - b * sa, a * sa + b * ca
    r = np.sqrt((a / rx) ** 2 + (b / ry) ** 2)
    return (1 - r) * min(rx, ry), a, b


def _cast_shadow(canvas, ctx: _Ctx, f: Figure, fx: float, fy: float, h: float):
    g = ctx.g
    sx = -g.light_x * 0.2 * h
    rx = (0.17 * f.bulk + 0.16 * abs(g.light_x)) * h
    ry = (0.025 + 0.02 * (1 - g.light_elev)) * h
    d = 1 - np.sqrt(((ctx.X - fx - sx) / rx) ** 2 + ((ctx.Y - fy + 0.004) / ry) ** 2)
    alpha = _smooth(d * ry * ctx.h, 1.2) * (0.3 + 0.35 * g.chiaroscuro)
    canvas[:] = pg.mix(canvas, SHADOW, alpha[..., None] * ctx.opacity[..., None])


def _figure(canvas, ctx: _Ctx, f: Figure, fx: float, fy: float, h: float):
    fr = _FigureFrame(ctx, f, fx, fy, h)
    if fr.empty:
        return
    sl = fr.sl
    u, v = fr.u, fr.v
    table, dv, hip = _PROFILES[f.pose]
    vs = np.array([p[0] for p in table])
    cs = np.array([p[1] for p in table])
    hws = np.array([p[2] for p in table]) * f.bulk
    hws = np.where(vs < hip, hws * f.flare, hws)
    vtop = vs[-1]
    c = np.interp(v, vs, cs)
    hw = np.interp(v, vs, hws)
    hem = 0.006 * np.sin(u * 70 + f.folds)
    body_d = np.minimum(np.minimum(hw - np.abs(u - c), v - hem), vtop - v)
    seed_phase = (ctx.g.seed % 97) / 97.0
    drift = 0.5 * ctx.fields[1][sl] + 0.1 * np.sin(v * 9 + seed_phase * 6)
    nx_l = np.clip((u - c) / np.maximum(hw, 1e-3), -1, 1)
    amp = f.fold_depth * (0.2 + 0.8 * np.clip(1 - v / max(hip + 0.1, 0.2), 0, 1))

    head_u, head_v = 0.015, vtop + 0.05
    if f.pose == "standing":
        head_v = 0.925
    rx, ry = 0.05, 0.062
    sc = lambda p: (p[0], p[1] + dv)                         # noqa: E731

    def arm(spec):
        sh, el, hand, _ = spec
        sh, el, hand = sc(sh), sc(el), sc(hand)
        r = 0.032 * f.bulk
        d1, p1x, p1y = _seg_dist(u, v, sh, el)
        d2, p2x, p2y = _seg_dist(u, v, el, hand)
        use2 = d2 < d1
        dist = np.where(use2, d2, d1)
        px = np.where(use2, p2x, p1x)
        py = np.where(use2, p2y, p1y)
        alpha = fr.cover(r - dist)
        nx, ny = px / r, py / r
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.02, 1))
        ix, iy, iz = fr.to_image_normal(nx, -ny, nz)
        val = _lambert(ix, iy, iz, ctx) + ctx.strokes_v[sl] * 0.1 * ctx.brush
        _paint(canvas, alpha, _model(P[f.robe], np.clip(val, 0, 1).astype(np.float32), ctx), ctx, sl)
        # hand
        hd, ha, hb = _ellipse_d(fr, hand[0], hand[1], 0.024, 0.024)
        _paint(canvas, fr.cover(hd), _sphere(fr, ha, hb, 0.024, 0.024, skin, ctx), ctx, sl)

    skin = pg.flesh(f.skin)
    arms = _GESTURES.get(f.gesture, _GESTURES["down"])

    # Halo (gold leaf with tooled rays) behind everything else.
    if f.halo:
        hr = 0.105
        hd, ha, hb = _ellipse_d(fr, head_u - 0.008, head_v + dv * 0 + 0.012, hr, hr)
        ang = np.arctan2(hb, ha)
        rr = np.sqrt(ha * ha + hb * hb) / hr
        val = 0.68 + 0.12 * np.cos(ang * 28) * (rr > 0.55) - 0.25 * (rr > 0.9)
        val = val + ctx.mottle[sl] * 0.1
        gold = pg.mix(P["gold_leaf"], P["lead_tin_yellow"], 0.25)
        _paint(canvas, fr.cover(hd), _model(gold, np.clip(val, 0, 1).astype(np.float32), ctx), ctx, sl)

    for spec in arms:
        if not spec[3]:
            arm(spec)

    # Robe.
    phase = nx_l * f.folds * 0.5 + drift
    _paint(canvas, fr.cover(body_d), _shade_drapery(fr, f.robe, nx_l, phase, amp, ctx), ctx, sl)

    # Neck.
    nd, na, nb = _ellipse_d(fr, 0.008, vtop + 0.005, 0.024, 0.04)
    _paint(canvas, fr.cover(nd), _sphere(fr, na, nb, 0.024, 0.04, skin, ctx, 0.15), ctx, sl)

    # Mantle: the body silhouette widened a little, open down the front to
    # show the robe at the breast and a strip at the hem.
    if f.mantle:
        mhw = hw + 0.014
        mbody = np.minimum(np.minimum(mhw - np.abs(u - c), v - hem - 0.004), vtop - 0.01 - v)
        vn = v / vtop
        open_w = f.mantle_open * 1.7 * np.clip((vn - 0.5) / 0.3, 0, 1)
        opening = open_w - np.abs(u - c - 0.025)
        md = np.minimum(mbody, -opening)
        mphase = nx_l * f.folds * 0.4 + v * 2.5 + drift
        _paint(canvas, fr.cover(md), _shade_drapery(fr, f.mantle, nx_l, mphase, amp * 1.1, ctx), ctx, sl)
        if f.veil:
            vd, va, vb = _ellipse_d(fr, head_u - 0.012, head_v + 0.004, rx * 1.38, ry * 1.28, f.head_tilt * 0.5)
            fall, _, _ = _seg_dist(u, v, (head_u - 0.035, head_v), (-0.09, vtop - 0.06))
            veil_d = np.maximum(vd, 0.05 - fall)
            _paint(canvas, fr.cover(veil_d), _sphere(fr, va, vb, rx * 1.38, ry * 1.28, P[f.mantle], ctx), ctx, sl)

    # Infant held in the arms.
    if f.gesture == "cradle":
        bd, ba, bb = _ellipse_d(fr, 0.04, 0.625 + dv, 0.085, 0.036, -0.35)
        _paint(canvas, fr.cover(bd), _sphere(fr, ba, bb, 0.085, 0.036, skin, ctx), ctx, sl)
        if f.halo:
            cd, ca, cb = _ellipse_d(fr, 0.12, 0.67 + dv, 0.05, 0.05)
            _paint(canvas, fr.cover(cd) * 0.9,
                   _model(P["gold_leaf"], np.full(u.shape, 0.7, np.float32), ctx), ctx, sl)
        kd, ka, kb = _ellipse_d(fr, 0.115, 0.665 + dv, 0.03, 0.03)
        _paint(canvas, fr.cover(kd), _sphere(fr, ka, kb, 0.03, 0.03, skin, ctx), ctx, sl)

    for spec in arms:
        if spec[3]:
            arm(spec)

    # Head, hair, face.
    hd, ha, hb = _ellipse_d(fr, head_u, head_v + 0, rx, ry, f.head_tilt)
    _paint(canvas, fr.cover(hd), _sphere(fr, ha, hb, rx, ry, skin, ctx), ctx, sl)
    if f.veil and f.mantle:
        cap_d, ca, cb = _ellipse_d(fr, head_u - 0.012, head_v + 0.004, rx * 1.38, ry * 1.28, f.head_tilt * 0.5)
        cap = np.minimum(cap_d, (cb - ry * 0.42) * 0.5)
        _paint(canvas, fr.cover(cap), _sphere(fr, ca, cb, rx * 1.38, ry * 1.28, P[f.mantle], ctx), ctx, sl)
    else:
        hd2, ha2, hb2 = _ellipse_d(fr, head_u - 0.006, head_v + 0.008, rx * 1.1, ry * 1.07, f.head_tilt)
        hair = np.minimum(hd2, np.maximum((hb2 - ry * 0.3) * 0.5, (-ha2 - rx * 0.4) * 0.5))
        _paint(canvas, fr.cover(hair), _sphere(fr, ha2, hb2, rx * 1.1, ry * 1.07, P[f.hair], ctx), ctx, sl)

    if rx * fr.hp >= 6:
        dark = pg.mix(P["burnt_umber"], skin, 0.25)
        for eu in (-0.1, 0.48):
            ed, _, _ = _ellipse_d(fr, head_u + eu * rx, head_v + 0.06 * ry, rx * 0.12, ry * 0.055, f.head_tilt)
            _paint(canvas, fr.cover(ed) * 0.7, dark, ctx, sl)
        md_, _, _ = _ellipse_d(fr, head_u + 0.25 * rx, head_v - 0.5 * ry, rx * 0.16, ry * 0.05, f.head_tilt)
        _paint(canvas, fr.cover(md_) * 0.45, pg.mix(P["madder_lake"], skin, 0.4), ctx, sl)


def _place(g: SceneGenome, f: Figure, ctx: _Ctx) -> tuple[float, float, float]:
    """Feet position and figure height from the ground plane: a figure's
    height is proportional to how far its feet sit below the horizon, so
    heads of standing figures align near the horizon (isocephaly)."""
    near, far = 0.955, g.horizon + 0.07
    if g.setting != "landscape":
        far = max(far, g.horizon + g.room_floor + 0.03)
    fy = near - f.depth * (near - far)
    h = (fy - g.horizon) * 1.12 * f.scale
    return f.x * ctx.aspect, fy, h


# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------

def _finish(canvas, ctx: _Ctx):
    g = ctx.g
    if g.craquelure > 0.02:
        vis = np.clip((ctx.h - 300) / 900, 0.12, 1.0)
        lines = np.zeros(canvas.shape[:2], np.float32)
        for cells in (28, 61):
            n = _fbm(ctx.rng, ctx.w, ctx.h, int(cells * ctx.aspect), cells, 2)
            gy, gx = np.gradient(n)
            dist = np.abs(n - 0.5) / (np.sqrt(gx * gx + gy * gy) + 1e-6)
            lines = np.maximum(lines, np.clip(1 - dist / 0.7, 0, 1))
        canvas *= (1 - 0.4 * g.craquelure * vis * lines)[..., None]
    yellow = np.array([1.0, 0.975, 0.88], np.float32) ** (g.varnish * 1.6)
    canvas *= yellow


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render_scene(g: SceneGenome, width: int, height: int | None = None,
                 brush_scale: float = 1.0) -> Image.Image:
    height = height or width
    scale = min(1.0, MAX_RENDER / max(width, height))
    w, h = max(16, int(width * scale)), max(16, int(height * scale))
    ctx = _Ctx(g, w, h, brush_scale)

    ground = pg.mix(P[g.ground], P["lead_white"], 0.3)
    canvas = np.empty((h, w, 3), np.float32)
    canvas[:] = ground

    _sky(canvas, ctx)
    _landscape(canvas, ctx)
    if g.setting in ("loggia", "nave"):
        _room(canvas, ctx)
    if g.darkness > 0.01:
        canvas[:] = pg.mix(canvas, P["bone_black"], 0.6 * g.darkness)

    placed = sorted(((f,) + _place(g, f, ctx) for f in g.figures),
                    key=lambda t: t[2])                 # far (small fy) first
    if g.throne and placed:
        main = max(placed, key=lambda t: t[0].scale - abs(t[0].x - 0.5))
        _throne(canvas, ctx, *main)
    for f, fx, fy, fh in placed:
        _cast_shadow(canvas, ctx, f, fx, fy, fh)
    for f, fx, fy, fh in placed:
        _figure(canvas, ctx, f, fx, fy, fh)

    _finish(canvas, ctx)
    rgb = (pg.linear_to_srgb(canvas) * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    if scale < 1.0:
        img = img.resize((width, height), Image.LANCZOS)
    return img


def render_scene_extended(g: SceneGenome, square_size: int,
                          orientation: str = "square",
                          brush_scale: float = 1.0) -> Image.Image:
    if orientation == "landscape":
        return render_scene(g, int(square_size * 1.5), square_size, brush_scale)
    if orientation == "portrait":
        return render_scene(g, square_size, int(square_size * 1.5), brush_scale)
    return render_scene(g, square_size, square_size, brush_scale)

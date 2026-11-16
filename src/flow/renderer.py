"""Paint a FlowGenome as an oil sketch made of flow-following brushstrokes.

Order of work, as at the easel:

  ground      a toned canvas (raw sienna, ochre, rose, ...) with its weave;
              it shows through wherever the paint is thin
  fields      the primitives are resolved into a region map, a colour field
              and a flow field (vectors the strokes follow): sky ribbons
              and vortices, crest lines, furrows raking to the vanishing
              point, ripples, climbing cypress flames, curling crowns
  lay-in      a thin scrubbed underpainting of the colour field
  strokes     three passes, broad to fine; every stroke is a streamline
              traced through the flow field, carries its own bristle
              ridges and impasto height, and stops at the edge of its shape
              (unless the painting is still raw)
  contour     a dark painted line wherever a shape sits in front of another
  impasto     the paint's own relief, lit from the side

Maturity drives the handling from neo-expressionist (turbulent flow,
strokes crossing shapes, clashing colour, bare ground, drips, scrawls) to
a clean, resolved oil painting.  Nothing is laid over the finished image:
all texture belongs to a stroke or to the ground.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from src.flow import pigments as pg
from src.flow.genome import Element, FlowGenome
from src.renaissance.renderer import _cppn_fields, _fbm, _resize

P = pg.PIGMENTS
MAX_RENDER = 3072        # larger requests are rendered here and upscaled
FIELD_MAX = 768          # region/colour/flow fields are solved at most here
TAU = 2 * math.pi


def _norm(vx, vy):
    n = np.sqrt(vx * vx + vy * vy) + 1e-6
    return vx / n, vy / n


def _crest(b: Element, X: np.ndarray, aspect: float, terrain: np.ndarray | None,
           wobble: float) -> tuple[np.ndarray, np.ndarray]:
    """Crest height of a band at canvas x ``X`` and its slope dY/dX."""
    x = X / aspect
    a1 = TAU * b.freq * x + b.phase
    a2 = TAU * b.freq * 2.3 * x + 1.7 * b.phase
    y = b.y + b.amp * (np.sin(a1) + 0.35 * np.sin(a2)) + b.tilt * (x - 0.5)
    dy = (b.amp * TAU * b.freq * (np.cos(a1) + 0.35 * 2.3 * np.cos(a2)) + b.tilt) / aspect
    if terrain is not None:
        y = y + wobble * terrain
    return y, dy


class _Fields:
    """Region map, colour, accent and flow fields on a (fh, fw) grid."""

    def __init__(self, g: FlowGenome, fw: int, fh: int, aspect: float):
        self.g, self.fw, self.fh, self.aspect = g, fw, fh, aspect
        m = g.maturity
        ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float32)
        X = (xs + 0.5) / fh
        Y = (ys + 0.5) / fh
        self.X, self.Y = X, Y
        cf = _cppn_fields(g, fw, fh)               # turbulence, value, terrain
        rng = np.random.default_rng(g.seed ^ 0x5F3759DF)
        self.rid = np.zeros((fh, fw), np.int16)
        self.col = np.zeros((fh, fw, 3), np.float32)
        self.acc = np.zeros((fh, fw, 3), np.float32)
        self.vx = np.ones((fh, fw), np.float32)
        self.vy = np.zeros((fh, fw), np.float32)
        self.region_pig: list[str] = ["sky"]       # id -> pigment name

        orbs = [f for f in g.features if f.kind == "orb"]
        main = max(orbs, key=lambda f: f.size) if orbs else None
        self.light = _norm(np.float32(-0.6 if main is None or main.x < 0.5 else 0.6),
                           np.float32(-0.8))
        hz = g.horizon
        self._sky(X, Y, hz)

        for f in g.features:
            if f.kind == "swirl":
                self._swirl(f, X, Y)
        for f in orbs:
            self._halo(f, X, Y)
        for f in orbs:
            self._orb(f, X, Y)

        wobble = 0.03 * (1.0 - 0.75 * m) * (0.4 + g.turbulence)
        crests = [_crest(b, X, aspect, cf[2], wobble) for b in g.bands]
        for i, b in enumerate(g.bands):
            nxt = crests[i + 1][0] if i + 1 < len(g.bands) else np.full_like(X, 1.05)
            self._band(b, i, X, Y, *crests[i], nxt, hz)
        for f in g.features:
            if f.kind == "road":
                self._road(f, X, Y, hz)
        for f in g.features:
            if f.kind == "cypress":
                self._cypress(f, X, Y)
            elif f.kind == "tree":
                self._tree(f, X, Y)

        # Value variation from the CPPN; raw paintings swing harder.
        self.col *= (1.0 + 0.1 * (0.4 + 1.2 * (1 - m)) * cf[1])[..., None]

        # Turbulence: rotate the flow.  Raw paintings churn, mature ones
        # keep only a little expressive wobble.
        chaos = g.turbulence * (1.0 - m) ** 1.3
        noise = _fbm(rng, fw, fh, 6, 6, 3) - 0.5
        ang = chaos * (cf[0] * 1.9 + noise * 2.4) + 0.18 * g.turbulence * cf[0]
        ca, sa = np.cos(ang), np.sin(ang)
        self.vx, self.vy = _norm(self.vx * ca - self.vy * sa, self.vx * sa + self.vy * ca)

        # Distance to the nearest edge between shapes, canvas units: a
        # careful painter switches to a smaller brush near edges.
        r = self.rid
        edge = np.zeros(r.shape, bool)
        edge[:-1] |= r[:-1] != r[1:]
        edge[1:] |= r[:-1] != r[1:]
        edge[:, :-1] |= r[:, :-1] != r[:, 1:]
        edge[:, 1:] |= r[:, :-1] != r[:, 1:]
        self.edge_dist = (distance_transform_edt(~edge) / fh).astype(np.float32) \
            if edge.any() else np.full(r.shape, 1.0, np.float32)

    # ---- helpers --------------------------------------------------------

    def _new_region(self, pig: str) -> int:
        self.region_pig.append(pig)
        return len(self.region_pig) - 1

    def _set(self, mask, rid, col, acc, vx, vy):
        self.rid[mask] = rid
        self.col[mask] = col[mask] if col.ndim == 3 else col
        self.acc[mask] = acc[mask] if acc.ndim == 3 else acc
        vx, vy = _norm(vx, vy)
        self.vx[mask] = vx[mask] if np.ndim(vx) else vx
        self.vy[mask] = vy[mask] if np.ndim(vy) else vy

    # ---- primitives -----------------------------------------------------

    def _sky(self, X, Y, hz):
        g = self.g
        t = np.clip(Y / max(hz, 0.05), 0, 1) ** 1.3
        col = pg.mix(P[g.sky_top], P[g.sky_low], t[..., None])
        # Munch: banded, undulating sky -- strokes follow the bands.
        A = 0.015 + 0.06 * g.sky_wave
        k = TAU * (0.6 + 0.8 * g.sky_wave) / self.aspect
        wave = Y + A * np.sin(k * X + g.seed % 7)
        stripe = 0.5 + 0.5 * np.sin(TAU * g.sky_freq * wave)
        band = np.clip((stripe - 0.55) * 3, 0, 1) * g.sky_wave * 0.55
        col = pg.mix(col, P[g.sky_accent], band[..., None])
        self.col[:] = col
        self.acc[:] = P[g.sky_accent]
        self.vx[:], self.vy[:] = _norm(np.ones_like(X), A * k * np.cos(k * X + g.seed % 7))

    def _vortex(self, f: Element, X, Y, reach: float, inward: float):
        cx, cy = f.x * self.aspect, f.y
        dx, dy = X - cx, Y - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        w = np.exp(-(r / (f.size * reach)) ** 2) * (self.rid == 0)
        tx, ty = -dy / r * f.spin, dx / r * f.spin
        tx, ty = tx - inward * dx / r, ty - inward * dy / r
        self.vx, self.vy = _norm((1 - w) * self.vx + w * tx, (1 - w) * self.vy + w * ty)
        return r, w

    def _swirl(self, f: Element, X, Y):
        r, w = self._vortex(f, X, Y, 1.25, 0.15 + 0.35 * f.twist)
        ring = 0.5 + 0.5 * np.cos(TAU * r / f.size * (1.6 + 1.6 * f.twist) + f.phase)
        t = (w * ring * 0.5)[..., None]
        self.col[:] = pg.mix(self.col, P[f.accent], t)

    def _halo(self, f: Element, X, Y):
        cx, cy = f.x * self.aspect, f.y
        dx, dy = X - cx, Y - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        R = f.size * f.halo
        w = np.clip(1 - (r - f.size) / (R - f.size + 1e-6), 0, 1) ** 1.4 * (r > f.size)
        tx, ty = -dy / r * f.spin, dx / r * f.spin
        rx, ry = dx / r, dy / r
        hx, hy = _norm((1 - f.rays) * tx + f.rays * rx, (1 - f.rays) * ty + f.rays * ry)
        self.vx, self.vy = _norm((1 - w) * self.vx + w * hx, (1 - w) * self.vy + w * hy)
        rings = 0.5 + 0.5 * np.cos(TAU * (r - f.size) / (f.size * 0.55))
        rays = 0.5 + 0.5 * np.cos(np.arctan2(dy, dx) * (14 + int(f.halo * 3)))
        pat = (1 - f.rays) * rings + f.rays * rays
        t = (w * (0.35 + 0.45 * pat))[..., None]
        glow = pg.mix(P[f.pigment], P[f.accent], 0.3)
        self.col[:] = pg.mix(self.col, glow, t * 0.85)

    def _orb(self, f: Element, X, Y):
        cx, cy = f.x * self.aspect, f.y
        dx, dy = X - cx, Y - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        mask = r < f.size
        rid = self._new_region(f.pigment)
        col = pg.mix(P[f.pigment], P["zinc_white"], (0.35 * (1 - r / f.size))[..., None])
        tx, ty = -dy / r * f.spin, dx / r * f.spin
        self._set(mask, rid, col, P[f.accent], tx, ty)

    def _band(self, b: Element, i: int, X, Y, crest, slope, nxt, hz):
        g, m = self.g, self.g.maturity
        mask = Y > crest
        rid = self._new_region(b.pigment)
        depth = np.clip((Y - crest) / np.maximum(nxt - crest, 0.03), 0, 1.5)
        col = pg.mix(P[b.pigment], P["zinc_white"], (0.14 * (1 - np.clip(depth, 0, 1)))[..., None])
        col = pg.mix(col, pg.darker(b.pigment, 0.5), (0.22 * np.clip(depth, 0, 1))[..., None])
        # Aerial perspective for the far bands (painters of the flat
        # Matisse kind skip it).
        far = np.clip(1 - (b.y - hz) / 0.25, 0, 1) * (1 - 0.8 * g.matisse) * 0.3
        if far > 0.01:
            col = pg.mix(col, P[g.sky_low], far)
        vx, vy = np.ones_like(X), slope.copy()
        acc = P[b.accent]
        if b.kind == "field":
            vpX, vpY = g.vp_x * self.aspect, hz
            fx, fy = _norm(X - vpX, Y - vpY + 1e-3)
            flip = (fx * vx + fy * vy) < 0
            fx, fy = np.where(flip, -fx, fx), np.where(flip, -fy, fy)
            vx, vy = _norm(vx, vy)
            vx = (1 - b.furrows) * vx + b.furrows * fx
            vy = (1 - b.furrows) * vy + b.furrows * fy
            rows = 0.5 + 0.5 * np.cos(np.arctan2(X - vpX, Y - vpY + 1e-3) * (18 + 20 * b.freq))
            t = (np.clip((rows - 0.6) * 2.5, 0, 1) * b.furrows * 0.45)[..., None]
            col = pg.mix(col, acc, t)
        elif b.kind == "water":
            rip = np.sin(TAU * (Y * 34 + 0.25 * np.sin(X * 11 + b.phase)))
            vy = vy + 0.12 * rip
            t = (np.clip((rip - 0.5) * 2, 0, 1) * 0.3)[..., None]
            col = pg.mix(col, P[g.sky_low], t)
        self._set(mask, rid, col, acc, vx, vy)

    def _road(self, f: Element, X, Y, hz):
        a = self.aspect
        p0 = np.array([f.x * a, 1.06])
        p2 = np.array([f.end_x * a, hz + 0.004])
        p1 = np.array([(p0[0] + p2[0]) / 2 + f.tilt * a * 0.5, (p0[1] + p2[1]) / 2 + 0.05])
        ts = np.linspace(0, 1, 40)
        pts = ((1 - ts)[:, None] ** 2 * p0 + 2 * ((1 - ts) * ts)[:, None] * p1
               + ts[:, None] ** 2 * p2)
        # Distance to the centre line, segment by segment (point samples
        # would bead apart where the road narrows).
        best = np.full(X.shape, 1e9, np.float32)
        tpos = np.zeros(X.shape, np.float32)
        seg_dir = np.zeros(X.shape + (2,), np.float32)
        for k in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            ex, ey = bx - ax, by - ay
            ll = ex * ex + ey * ey + 1e-12
            u = np.clip(((X - ax) * ex + (Y - ay) * ey) / ll, 0, 1)
            d = (X - ax - u * ex) ** 2 + (Y - ay - u * ey) ** 2
            closer = d < best
            best = np.where(closer, d, best)
            tpos = np.where(closer, ts[k] + u * (ts[k + 1] - ts[k]), tpos)
            seg_dir[closer] = (ex, ey)
        width = f.size * (1 - tpos) ** 1.15 + 0.004
        mask = (np.sqrt(best) < width) & (Y > hz)
        rid = self._new_region(f.pigment)
        tt = tpos[..., None]
        col = pg.mix(P[f.pigment], P["zinc_white"], 0.18 * (1 - tt))
        col = pg.mix(col, P[self.g.sky_low], 0.25 * tt)
        self._set(mask, rid, col, P[f.accent], seg_dir[..., 0], seg_dir[..., 1])

    def _cypress(self, f: Element, X, Y):
        H = f.size
        W = f.amp * H
        s = (f.y - Y) / H
        sc = np.clip(s, 0, 1)
        sway = f.twist * W * 0.6 * np.sin(TAU * 1.2 * sc + f.phase) * sc
        xc = f.x * self.aspect + sway
        half = W * np.clip(np.sin(np.pi * sc ** 0.6), 0, 1) ** 0.85
        side = (X - xc) / (half + 1e-4)
        mask = (s > 0) & (s < 1) & (np.abs(side) < 1)
        rid = self._new_region(f.pigment)
        lit = np.clip(-side * np.sign(self.light[0]), -1, 1)
        col = pg.mix(P[f.pigment], pg.darker(f.pigment, 0.6), (0.3 * (1 - lit) / 2)[..., None])
        col = pg.mix(col, P[f.accent], (0.25 * np.clip(lit, 0, 1))[..., None])
        dsway = f.twist * W * 0.6 * TAU * 1.2 * np.cos(TAU * 1.2 * sc + f.phase) * sc / H
        flick = 0.7 * f.twist * np.sin(TAU * (3.5 * sc + 0.6 * side) + f.phase)
        vx, vy = _norm(dsway, -np.ones_like(X))
        c, sn = np.cos(flick), np.sin(flick)
        self._set(mask, rid, col, P[f.accent], vx * c - vy * sn, vx * sn + vy * c)

    def _tree(self, f: Element, X, Y):
        H = f.size
        R = H * 0.4
        cx, cy = f.x * self.aspect, f.y - H * 0.62
        trunk = (np.abs(X - cx) < R * 0.13) & (Y > cy) & (Y < f.y)
        dx, dy = X - cx, Y - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        th = np.arctan2(dy, dx)
        edge = R * (1 + f.twist * (0.6 * np.sin(5 * th + f.phase) + 0.4 * np.sin(9 * th + 2 * f.phase)))
        crown = r < edge
        rid = self._new_region(f.pigment)
        bark = pg.mix(P["burnt_sienna"], P["ivory_black"], 0.3)
        lit = np.clip(-(dx * self.light[0] + dy * self.light[1]) / R, -1, 1)
        col = pg.mix(P[f.pigment], P["zinc_white"], (0.22 * np.clip(lit, 0, 1))[..., None])
        col = pg.mix(col, pg.darker(f.pigment, 0.6), (0.3 * np.clip(-lit, 0, 1))[..., None])
        col = np.where(crown[..., None], col, bark)
        tx, ty = -dy / r * f.spin, dx / r * f.spin
        vx = np.where(crown, tx + 0.3 * dx / r, 0.05)
        vy = np.where(crown, ty + 0.3 * dy / r, -1.0)
        self._set(crown | trunk, rid, col, P[f.accent], vx, vy)

    # ---- sampling -------------------------------------------------------

    def cell(self, X, Y):
        i = np.clip((Y * self.fh).astype(np.int32), 0, self.fh - 1)
        j = np.clip((X * self.fh).astype(np.int32), 0, self.fw - 1)
        return i, j


# ---------------------------------------------------------------------------
# Strokes
# ---------------------------------------------------------------------------

def _trace(F: _Fields, X0, Y0, steps: int, step: float, sign, cross):
    """Streamlines from (X0, Y0) through the flow field.  Returns (N, steps+1,
    2) canvas points and the number of valid points per stroke."""
    n = X0.shape[0]
    pts = np.empty((n, steps + 1, 2), np.float32)
    pts[:, 0, 0], pts[:, 0, 1] = X0, Y0
    i, j = F.cell(X0, Y0)
    rid0 = F.rid[i, j]
    px, py = F.vx[i, j] * sign, F.vy[i, j] * sign
    x, y = X0.copy(), Y0.copy()
    alive = np.ones(n, bool)
    count = np.ones(n, np.int32)
    for s in range(steps):
        i, j = F.cell(x, y)
        dx, dy = F.vx[i, j], F.vy[i, j]
        flip = dx * px + dy * py < 0
        dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
        # A brush cannot turn on a pin: blend with the previous heading.
        dx, dy = _norm(0.55 * dx + 0.45 * px, 0.55 * dy + 0.45 * py)
        nx, ny = x + dx * step, y + dy * step
        ni, nj = F.cell(nx, ny)
        ok = ((F.rid[ni, nj] == rid0) | cross) & (nx > -0.02) & (nx < F.aspect + 0.02) \
            & (ny > -0.02) & (ny < 1.02)
        alive &= ok
        x, y = np.where(alive, nx, x), np.where(alive, ny, y)
        px, py = np.where(alive, dx, px), np.where(alive, dy, py)
        count += alive
        pts[:, s + 1, 0], pts[:, s + 1, 1] = x, y
    return pts, count


def _stroke_passes(g: FlowGenome):
    """(width, length, density) per pass in canvas units, broad to fine.
    Raw paintings are worked with a bigger, cruder brush."""
    st = g.stroke_scale * (1 + 0.6 * (1 - g.maturity))
    lm = 0.8 * g.vg + 2.3 * g.munch + 1.4 * g.matisse
    wm = 1.0 * g.vg + 0.8 * g.munch + 1.5 * g.matisse
    base = [(0.042, 0.10, 1.1), (0.021, 0.058, 1.25),
            (0.0105, 0.032, 1.45 * (1 - 0.55 * g.matisse))]
    return [(w * wm * st, L * lm * st, d) for w, L, d in base]


def _to_srgb8(lin: np.ndarray) -> np.ndarray:
    return (pg.linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)


def _angle8(dx: float, dy: float) -> int:
    """Stroke direction (mod pi) packed into a byte."""
    return int((math.atan2(dy, dx) % math.pi) / math.pi * 255.0 + 0.5) % 256


def _paint_strokes(img: Image.Image, ang: Image.Image, F: _Fields, g: FlowGenome,
                   rng, S: int):
    """All stroke passes.  ``img`` is RGBA (alpha = paint height), ``ang``
    records each stroke's direction so its bristle marks can follow it."""
    m = g.maturity
    draw, dang = ImageDraw.Draw(img), ImageDraw.Draw(ang)
    a = F.aspect
    palette = np.stack([P[k] for k in pg.NAMES])
    sv = 0.04 + 0.06 * g.vg + 0.12 * (1 - m)
    p_broken = 0.06 + 0.22 * g.vg + 0.06 * g.munch
    p_clash = 0.22 * (1 - m) ** 2
    p_cross = 0.85 * (1 - m) ** 1.5
    cover = 0.6 + 0.4 * m
    for w, L, dens in _stroke_passes(g):
        n = int(min(9000, dens * cover * a / (w * L)))
        X0 = rng.random(n).astype(np.float32) * a
        Y0 = rng.random(n).astype(np.float32)
        steps = 4
        cross = rng.random(n) < p_cross
        sign = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        reach = rng.uniform(0.6, 1.3, n).astype(np.float32)
        fwd, cf = _trace(F, X0, Y0, steps, L * reach / (2 * steps), sign, cross)
        bwd, cb = _trace(F, X0, Y0, steps, L * reach / (2 * steps), -sign, cross)

        i, j = F.cell(X0, Y0)
        col = F.col[i, j] * np.exp(rng.normal(0, sv, (n, 1))).astype(np.float32)
        broken = rng.random(n) < p_broken
        t = rng.uniform(0.5, 0.85, (n, 1)).astype(np.float32)   # side by side, not mud
        col = np.where(broken[:, None], pg.mix(col, F.acc[i, j], t), col)
        clash = rng.random(n) < p_clash
        pick = palette[rng.integers(0, len(palette), n)]
        col = np.where(clash[:, None], pg.mix(col, pick, 0.7), col)
        rgb = _to_srgb8(col)
        height = rng.integers(70, 190, n)
        wk = w * rng.uniform(0.7, 1.3, n)
        care = (rng.random(n) < m ** 1.5) & ~cross
        wk = np.where(care, np.minimum(wk, 2.2 * F.edge_dist[i, j] + 0.003), wk)
        widths = np.maximum(1, np.round(wk * S)).astype(int)

        for k in range(n):
            nb, nf = cb[k], cf[k]
            if nb + nf < 3:
                continue
            seg = np.concatenate([bwd[k, nb - 1:0:-1], fwd[k, :nf]]) * S
            xy = [(float(p[0]), float(p[1])) for p in seg]
            fill = (int(rgb[k, 0]), int(rgb[k, 1]), int(rgb[k, 2]), int(height[k]))
            pw = int(widths[k])
            draw.line(xy, fill=fill, width=pw, joint="curve")
            # Rounded heel and toe, a touch narrower than the body.
            r = pw * 0.45
            for x, y in (xy[0], xy[-1]):
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            d = seg[-1] - seg[0]
            dang.line(xy, fill=_angle8(d[0], d[1]), width=pw + 2)


def _contours(img: Image.Image, ang: Image.Image, F: _Fields, g: FlowGenome, rng, S: int, W: int, H: int):
    m = g.maturity
    strength = g.contour * (0.25 + 0.75 * g.matisse + 0.35 * g.munch) + 0.3 * (1 - m) ** 2
    width = S * (0.0012 + 0.0065 * strength)
    if width < 0.8:
        return
    draw, dang = ImageDraw.Draw(img), ImageDraw.Draw(ang)
    sx, sy = W / F.fw, H / F.fh
    rid = F.rid
    pad = np.pad(rid, 2, mode="edge")
    wob = 0.004 * S * (0.3 + 2.0 * (1 - m))
    for k in range(1, len(F.region_pig)):
        mask = np.pad(rid == k, 2, mode="edge").astype(np.float32)
        if not mask.any():
            continue
        pig = F.region_pig[k]
        c = pg.mix(pg.darker(pig, 0.7), P["ivory_black"], 0.25 + 0.4 * (1 - m))
        if g.munch > 0.5 and m > 0.4:
            c = pg.mix(c, P[pig], 0.25)
        cc = tuple(int(v) for v in _to_srgb8(c))
        for line in find_contours(mask, 0.5):
            # Skip specks: tiny islands would read as beads of outline.
            if len(line) < 4 or np.abs(np.diff(line, axis=0)).sum() < 0.04 * F.fh:
                continue
            # Keep only the stretches where a shape behind shows across.
            ii = np.clip(line[:, 0].astype(int), 0, pad.shape[0] - 1)
            jj = np.clip(line[:, 1].astype(int), 0, pad.shape[1] - 1)
            behind = np.zeros(len(line), bool)
            for di, dj in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nb = pad[np.clip(ii + di, 0, pad.shape[0] - 1),
                         np.clip(jj + dj, 0, pad.shape[1] - 1)]
                behind |= nb < k
            pts = np.stack([(line[:, 1] - 2) * sx, (line[:, 0] - 2) * sy], 1)
            kern = np.ones(7) / 7
            if len(pts) > 9:
                pts = np.stack([np.convolve(np.pad(pts[:, d], 3, mode="edge"), kern, "valid")
                                for d in (0, 1)], 1)
            ph = rng.uniform(0, 6.283)
            tvec = np.gradient(pts, axis=0)
            tl = np.linalg.norm(tvec, axis=1, keepdims=True) + 1e-6
            nrm = np.stack([-tvec[:, 1], tvec[:, 0]], 1) / tl
            arc = np.cumsum(tl[:, 0]) / S
            off = wob * (np.sin(arc * 40 + ph) + 0.5 * np.sin(arc * 97 + 2 * ph))
            pts = pts + nrm * off[:, None]
            # Split into brush-length runs; a few skip (dry brush).
            run: list[tuple[float, float]] = []
            seg_len = rng.integers(25, 70)
            for q in range(len(pts)):
                if behind[q]:
                    run.append((float(pts[q, 0]), float(pts[q, 1])))
                if (not behind[q] or len(run) >= seg_len or q == len(pts) - 1) and run:
                    if len(run) > 2 and rng.random() > 0.1 * m:
                        wv = max(1, int(round(width * rng.uniform(0.7, 1.25))))
                        draw.line(run, fill=cc + (200,), width=wv, joint="curve")
                        dang.line(run, fill=_angle8(run[-1][0] - run[0][0],
                                                    run[-1][1] - run[0][1]), width=wv + 2)
                    run = run[-1:] if behind[q] else []
                    seg_len = rng.integers(25, 70)


def _raw_marks(img: Image.Image, ang: Image.Image, F: _Fields, g: FlowGenome, rng,
               S: int, W: int, H: int):
    """Neo-expressionist residue -- drips and loaded gestural marks laid
    across the shapes -- gone by the time a lineage matures."""
    m = g.maturity
    draw, dang = ImageDraw.Draw(img), ImageDraw.Draw(ang)
    for _ in range(int(12 * (1 - m) ** 1.6)):
        x, y = rng.uniform(0, W), rng.uniform(0.15, 0.85) * H
        i, j = F.cell(np.float32(x / H), np.float32(y / H))
        c = tuple(int(v) for v in _to_srgb8(F.col[i, j] * 0.85))
        ln = rng.uniform(0.03, 0.12) * S
        wd = max(1, int(rng.uniform(0.002, 0.005) * S))
        draw.line([(x, y), (x + rng.normal(0, 1), y + ln)], fill=c + (150,), width=wd)
        draw.ellipse([x - wd, y + ln - wd * 0.6, x + wd, y + ln + wd * 1.2], fill=c + (170,))
        dang.line([(x, y), (x, y + ln)], fill=128, width=wd + 2)
    inks = [P["ivory_black"], P["zinc_white"], P["cadmium_red"], P["prussian_blue"]]
    for _ in range(int(5 * (1 - m) ** 2)):
        # A loaded brush dragged in one arc, sometimes doubled back.
        cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        rad = rng.uniform(0.04, 0.14) * S
        a0 = rng.uniform(0, TAU)
        sweep = rng.uniform(1.2, 3.5) * rng.choice([-1, 1])
        ts = np.linspace(0, 1, 14)
        th = a0 + sweep * ts
        rr = rad * (1 + 0.25 * np.sin(ts * 7 + a0))
        xs, ys = cx + rr * np.cos(th), cy + 0.7 * rr * np.sin(th)
        c = tuple(int(v) for v in _to_srgb8(inks[rng.integers(0, len(inks))]))
        wd = max(1, int(rng.uniform(0.006, 0.014) * S))
        pts = list(zip(xs.tolist(), ys.tolist()))
        for q in range(len(pts) - 1):
            seg = pts[q:q + 2]
            draw.line(seg, fill=c + (210,), width=max(1, int(wd * (1 - 0.6 * ts[q]))))
            dang.line(seg, fill=_angle8(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]),
                      width=wd + 2)


def _ground(g: FlowGenome, W: int, H: int, rng) -> np.ndarray:
    base = pg.mix(P[g.ground], P["zinc_white"], 0.25)
    weave = (np.sin(np.arange(W) * (TAU / max(2.0, H / 260)))[None, :]
             + np.sin(np.arange(H) * (TAU / max(2.0, H / 260)))[:, None]) * 0.5
    tex = 1 + 0.035 * weave + 0.05 * (_fbm(rng, W, H, 9, 9, 3) - 0.5)
    return base[None, None, :] * tex[..., None].astype(np.float32)


def _bristles(ang: np.ndarray, W: int, H: int, S: int, rng) -> np.ndarray:
    """Bristle marks: noise smeared along each stroke's own direction (a
    line-integral convolution).  The noise grain is fixed in canvas units,
    so a brush leaves the same marks at any resolution.  Returns ~[-1, 1]."""
    gap = 0.0017 * S                           # bristle spacing, pixels
    gw, gh = max(8, int(W / gap)), max(8, int(H / gap))
    noise = _resize(rng.random((gh, gw)).astype(np.float32), W, H)
    th = ang.astype(np.float32) * (math.pi / 255.0)
    ux, uy = np.cos(th), np.sin(th)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    steps = 6
    ds = 0.016 * S / steps
    acc = noise.copy()
    for sgn in (1.0, -1.0):
        x, y = xs.copy(), ys.copy()
        px, py = ux * sgn, uy * sgn
        for _ in range(steps):
            xi = np.clip(x.astype(np.int32), 0, W - 1)
            yi = np.clip(y.astype(np.int32), 0, H - 1)
            dx, dy = ux[yi, xi], uy[yi, xi]
            flip = dx * px + dy * py < 0
            dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
            x += dx * ds
            y += dy * ds
            px, py = dx, dy
            acc += noise[np.clip(y.astype(np.int32), 0, H - 1),
                         np.clip(x.astype(np.int32), 0, W - 1)]
    acc /= 2 * steps + 1
    acc -= acc.mean()
    return acc / (acc.std() + 1e-6) * 0.5


def _paint(g: FlowGenome, W: int, H: int, impasto_scale: float) -> np.ndarray:
    S = H                                       # canvas unit -> pixels
    aspect = W / H
    fh = min(H, FIELD_MAX if W <= H else int(FIELD_MAX / aspect))
    fw = max(8, int(round(fh * aspect)))
    F = _Fields(g, fw, fh, aspect)
    rng = np.random.default_rng(g.seed)
    m = g.maturity

    ground = _ground(g, W, H, np.random.default_rng(g.seed + 1))
    lay = np.stack([_resize(F.col[..., c], W, H) for c in range(3)], -1)
    scrub = _fbm(np.random.default_rng(g.seed + 2), W, H, 8, 8, 3)
    cover = np.clip(0.45 + 0.5 * m + 0.25 * g.matisse + (scrub - 0.5) * 0.9, 0.1, 1.0)
    canvas = pg.mix(ground, lay, cover[..., None])

    rgba = np.concatenate([_to_srgb8(canvas), np.full((H, W, 1), 30, np.uint8)], -1)
    img = Image.fromarray(rgba, "RGBA")
    # Direction layer, seeded from the flow field (the lay-in's scrubbing).
    th0 = np.arctan2(F.vy, F.vx) % math.pi
    ang = Image.fromarray((th0 / math.pi * 255).astype(np.uint8), "L").resize((W, H), Image.NEAREST)
    _paint_strokes(img, ang, F, g, rng, S)
    _contours(img, ang, F, g, rng, S, W, H)
    _raw_marks(img, ang, F, g, rng, S, W, H)

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    painted = np.asarray(img.getchannel("A"), np.float32) / 255.0
    bris = _bristles(np.asarray(ang), W, H, S, np.random.default_rng(g.seed + 3))
    load = np.clip((painted - 0.15) * 4, 0, 1)        # bare ground has no bristles
    # Bristle streaks carry a little value, more in thick (Van Gogh) paint.
    rgb = rgb * (1 + (0.06 + 0.06 * g.vg) * bris * load)[..., None]

    # Impasto: bristle ridges plus the soft rim of each stroke, lit from
    # the sun's side.
    k = g.impasto * (0.3 + 0.9 * g.vg) * impasto_scale
    if k > 0.01:
        r = max(1.0, S / 700)
        rim = np.asarray(img.getchannel("A").filter(ImageFilter.GaussianBlur(r)),
                         np.float32) / 255.0
        hgt = 0.35 * rim + 0.18 * bris * load
        gy, gx = np.gradient(hgt)
        lx, ly = F.light
        shade = 1 - k * (gx * lx + gy * ly) * (S / 1000) * 2.0
        rgb = rgb * np.clip(shade, 0.85, 1.15)[..., None]
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def render_flow(g: FlowGenome, width: int, height: int | None = None,
                impasto_scale: float = 1.0) -> Image.Image:
    height = height or width
    # Small renders are painted at 2x so stroke edges stay clean.
    ss = 2 if max(width, height) <= 640 else 1
    scale = min(1.0, MAX_RENDER / max(width * ss, height * ss))
    w, h = max(16, int(width * ss * scale)), max(16, int(height * ss * scale))
    rgb = _paint(g, w, h, impasto_scale)
    img = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")
    if (w, h) != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img


def render_flow_extended(g: FlowGenome, square_size: int, orientation: str = "square",
                         impasto_scale: float = 1.0) -> Image.Image:
    if orientation == "landscape":
        return render_flow(g, int(square_size * 1.5), square_size, impasto_scale)
    if orientation == "portrait":
        return render_flow(g, square_size, int(square_size * 1.5), impasto_scale)
    return render_flow(g, square_size, square_size, impasto_scale)

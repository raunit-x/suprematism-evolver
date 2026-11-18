"""Malevich's own two ways of showing an arkhitekton.

``drawing``: a pencil axonometric on kraft paper (the 1923-24 "planit"
sheets).  Fine graphite outlines of every visible edge, the darker roofs
hatched solid with graphite along the long axis of each form, faint
construction lines running off along the axonometric axes, and a ruled
frame.

``plaster``: a photograph of the white plaster model on black: one hard
raking light, true cast shadows, and nothing but form.

Both are true axonometric renders of the same expressed genome the painted
Hadid style uses (plan footprint, ``elevation`` = roof height, ``depth`` =
height of the block).  Visibility comes from a z-buffer rather than a
painter's algorithm, so interlocking blocks occlude correctly, and plaster
shadows come from a shadow map in the light's own view.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.shapes.genome import Shape, ShapeGenome

STYLES = ("painted", "drawing", "plaster")

_ELEV = math.radians(34.0)       # camera tilt above the ground plane
_Z_SCALE = 1.25                  # heights read a touch taller, as in the models
_LIGHT_TURN = math.radians(62.0) # key light swung this far left of the camera
_LIGHT_ELEV = math.radians(42.0)

_KRAFT = (196, 166, 116)
_GRAPHITE = np.array([58, 52, 48], np.float32)
_PLASTER = np.array([236, 234, 228], np.float32)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _footprint(shape: Shape) -> list[tuple[float, float]]:
    """Plan polygon in canvas units, counter-clockwise in a y-up sense."""
    from src.shapes.renderer import _get_shape_corners
    k = 1000.0  # corner helper works in pixels (min thicknesses etc.)
    pts = _get_shape_corners(shape.kind, shape.w * k, shape.h * k)
    a = math.radians(shape.rotation)
    ca, sa = math.cos(a), math.sin(a)
    out = [(shape.x + (px * ca - py * sa) / k, shape.y + (px * sa + py * ca) / k)
           for px, py in pts]
    # Plan y grows downward; flip to a right-handed (y-up) frame.
    out = [(x, -y) for x, y in out]
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(out, out[1:] + out[:1]))
    if area < 0:
        out.reverse()
    return out


def _prisms(genome: ShapeGenome) -> list[tuple[Shape, list[tuple[float, float]], float, float]]:
    out = []
    for s in genome.flatten():
        if s.depth <= 0 and s.elevation <= 0:
            continue
        top = max(s.elevation, 1e-3) * _Z_SCALE
        bot = max(0.0, s.elevation - max(s.depth, 0.004)) * _Z_SCALE
        poly = _footprint(s)
        if len(poly) >= 3:
            out.append((s, poly, bot, top))
    return out


class _View:
    """Orthographic camera: plan rotated by ``az`` then tilted by ``elev``.

    Screen x = u, screen y (down) = v sin e - z cos e; larger ``d`` is
    nearer the viewer."""

    def __init__(self, az: float, elev: float):
        self.az = az
        self.ca, self.sa = math.cos(az), math.sin(az)
        self.ce, self.se = math.cos(elev), math.sin(elev)

    def project(self, p: np.ndarray) -> np.ndarray:
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        u = x * self.ca - y * self.sa
        v = x * self.sa + y * self.ca          # depth into the scene
        sy = v * self.se - z * self.ce          # screen y, downward
        d = v * self.ce + z * self.se           # toward the viewer
        return np.stack([u, sy, d], -1)

    def unproject(self, u: np.ndarray, sy: np.ndarray, d: np.ndarray) -> np.ndarray:
        v = sy * self.se + d * self.ce
        z = -sy * self.ce + d * self.se
        x = u * self.ca + v * self.sa
        y = -u * self.sa + v * self.ca
        return np.stack([x, y, z], -1)

    def toward(self) -> np.ndarray:
        """World-space direction from the scene toward the viewer."""
        # d = v ce + z se, with v = x sa + y ca
        return np.array([self.sa * self.ce, self.ca * self.ce, self.se])


def _faces(prisms):
    """Yield (prism index, kind, normal, 3D polygon) for every face that can
    be seen from above: roofs and the walls."""
    for i, (s, poly, bot, top) in enumerate(prisms):
        pts = np.array(poly, np.float64)
        yield i, "top", np.array([0.0, 0.0, 1.0]), np.column_stack([pts, np.full(len(pts), top)])
        n = len(pts)
        for j in range(n):
            (x0, y0), (x1, y1) = pts[j], pts[(j + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            L = math.hypot(ex, ey)
            if L < 1e-6:
                continue
            nrm = np.array([ey / L, -ex / L, 0.0])   # outward for CCW polygons
            quad = np.array([[x0, y0, bot], [x1, y1, bot], [x1, y1, top], [x0, y0, top]])
            yield i, "wall", nrm, quad


class _Raster:
    """Z-buffer over a screen rectangle; ``fit`` maps view units to pixels."""

    def __init__(self, w: int, h: int, scale: float, ox: float, oy: float):
        self.w, self.h = w, h
        self.scale, self.ox, self.oy = scale, ox, oy
        self.z = np.full((h, w), -np.inf, np.float32)
        self.fid = np.full((h, w), -1, np.int32)

    def to_px(self, q: np.ndarray) -> np.ndarray:
        return np.column_stack([q[:, 0] * self.scale + self.ox, q[:, 1] * self.scale + self.oy])

    def draw(self, q: np.ndarray, fid: int) -> None:
        """Rasterise a planar polygon given its projected (u, sy, d) verts."""
        px = self.to_px(q)
        x0 = max(int(math.floor(px[:, 0].min())), 0)
        x1 = min(int(math.ceil(px[:, 0].max())) + 1, self.w)
        y0 = max(int(math.floor(px[:, 1].min())), 0)
        y1 = min(int(math.ceil(px[:, 1].max())) + 1, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        # Depth is linear in screen space for a planar face (orthographic).
        A = np.column_stack([px, np.ones(len(px))])
        coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        if not np.all(np.isfinite(coef)):
            return
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in px], fill=255)
        mask = np.asarray(m) > 0
        if not mask.any():
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dep = (coef[0] * (xx + 0.5) + coef[1] * (yy + 0.5) + coef[2]).astype(np.float32)
        zb = self.z[y0:y1, x0:x1]
        win = mask & (dep > zb)
        zb[win] = dep[win]
        self.fid[y0:y1, x0:x1][win] = fid


def _render_view(prisms, view: _View, w: int, h: int, margin: float,
                 fit_pts: np.ndarray | None = None):
    faces = []
    toward = view.toward()
    for i, kind, nrm, poly in _faces(prisms):
        if kind == "wall" and float(nrm @ toward) <= 1e-4:
            continue    # back-facing wall
        faces.append((i, kind, nrm, poly))
    allpts = np.concatenate([f[3] for f in faces]) if faces else np.zeros((1, 3))
    q_all = view.project(allpts if fit_pts is None else fit_pts)
    umin, umax = q_all[:, 0].min(), q_all[:, 0].max()
    vmin, vmax = q_all[:, 1].min(), q_all[:, 1].max()
    span_u, span_v = max(umax - umin, 1e-6), max(vmax - vmin, 1e-6)
    scale = min(w * (1 - 2 * margin) / span_u, h * (1 - 2 * margin) / span_v)
    ox = w / 2 - scale * (umin + umax) / 2
    oy = h / 2 - scale * (vmin + vmax) / 2
    r = _Raster(w, h, scale, ox, oy)
    for k, (_, _, _, poly) in enumerate(faces):
        r.draw(view.project(poly), k)
    return r, faces


# ---------------------------------------------------------------------------
# Plaster model on black
# ---------------------------------------------------------------------------

def _light_view(view: _View) -> _View:
    """The lamp as an orthographic 'camera', fixed relative to the
    viewer so every composition is lit the same way: high, from the left
    and slightly in front."""
    return _View(view.az - _LIGHT_TURN, _LIGHT_ELEV)


def _shadow_mask(prisms, r: _Raster, view: _View, world: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """1 where a visible point is in the light, 0 in cast shadow."""
    lv = _light_view(view)
    n = 1400
    lr, _ = _render_view(prisms, lv, n, n, 0.02)
    q = lv.project(world[valid])
    lx = np.clip((q[:, 0] * lr.scale + lr.ox).astype(int), 0, n - 1)
    ly = np.clip((q[:, 1] * lr.scale + lr.oy).astype(int), 0, n - 1)
    bias = 6.0 / lr.scale   # generous: grazing walls otherwise stripe with acne
    occluder = lr.z[ly, lx]
    lit = np.ones(r.fid.shape, np.float32)
    lit[valid] = (q[:, 2] >= occluder - bias).astype(np.float32)
    return lit


def _light_dir(view: _View) -> np.ndarray:
    # Unit vector from the scene toward the light.
    return _light_view(view).toward()


def render_plaster(genome: ShapeGenome, width: int, height: int) -> Image.Image:
    ss = 2 if width * height <= 1600 * 1600 else 1
    w, h = width * ss, height * ss
    prisms = _prisms(genome)
    if not prisms:
        return Image.new("RGB", (width, height), (8, 8, 8))
    az = math.radians(genome.projection_angle * 0.6 + 18.0)
    view = _View(az, _ELEV)
    r, faces = _render_view(prisms, view, w, h, 0.11)

    valid = r.fid >= 0
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx + 0.5 - r.ox) / r.scale
    sy = (yy + 0.5 - r.oy) / r.scale
    world = view.unproject(u, sy, np.where(valid, r.z, 0.0))

    lit = _shadow_mask(prisms, r, view, world, valid)
    L = _light_dir(view)
    normals = np.array([f[2] for f in faces] + [[0, 0, 1]], np.float32)
    nrm = normals[np.where(valid, r.fid, len(faces))]
    lam = np.clip(nrm @ L.astype(np.float32), 0.0, 1.0)
    # Plaster under one lamp: lit faces near white, turned faces in a deep
    # grey, cast shadow nearly as dark as the turned faces.
    # Roofs take the lamp full on; walls step down by how far they turn
    # from it, so every plane of the model reads as its own value.
    up = nrm[..., 2] > 0.5
    face_lit = np.where(up, 0.97, 0.18 + 0.62 * np.clip(lam / 0.75, 0, 1))
    tone = 0.12 + (face_lit - 0.12) * lit
    # Bounce from the lit roofs into the walls and shadows, very slight.
    tone = tone + 0.06 * (1 - lit) * (nrm[..., 2] > 0.5)
    # The lamp falls off across the model (it was photographed under one).
    fall = 1.0 - 0.22 * np.clip((world[..., 0] * L[0] + world[..., 1] * L[1]) * -1.4 + 0.3, 0, 1)
    tone = tone * np.where(valid, fall, 1.0)
    rgb = _PLASTER[None, None, :] * tone[..., None]

    bg = np.full((h, w, 3), 7.0, np.float32)
    out = np.where(valid[..., None], rgb, bg)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    img = img.resize((width, height), Image.LANCZOS)
    # A touch of lens softness and film grain.
    img = img.filter(ImageFilter.GaussianBlur(0.35 * width / 640))
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(abs(hash(round(genome.projection_angle, 3))) % 2**32)
    grain = rng.normal(0, 4.5, arr.shape[:2])[..., None]
    arr = arr + grain * (0.4 + 0.6 * arr / 255.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Pencil drawing on kraft
# ---------------------------------------------------------------------------

def _noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    small = rng.random((max(2, h // cells), max(2, w // cells))).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im).astype(np.float32) / 255.0


def _hatch(h: int, w: int, angle: float, pitch: float, rng: np.random.Generator) -> np.ndarray:
    """Graphite laid in close parallel strokes, uneven in pressure."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ca, sa = math.cos(angle), math.sin(angle)
    across = -xx * sa + yy * ca
    along = xx * ca + yy * sa
    idx = np.floor(across / pitch)
    # Each stroke has its own pressure, and pressure wavers along it.
    seed = rng.random(4096).astype(np.float32)
    press = seed[(idx.astype(np.int64) % 4096)]
    phase = seed[((idx.astype(np.int64) * 7) % 4096)] * 50
    waver = 0.5 + 0.5 * np.sin(along / (pitch * 9.0) + phase)
    frac = (across / pitch) - idx
    core = np.clip(1.0 - np.abs(frac - 0.5) * 2.2, 0, 1) ** 0.6
    return np.clip(core * (0.55 + 0.45 * press) * (0.75 + 0.25 * waver), 0, 1)


def render_drawing(genome: ShapeGenome, width: int, height: int,
                   palette: list[tuple] | None = None) -> Image.Image:
    ss = 2 if width * height <= 1600 * 1600 else 1
    w, h = width * ss, height * ss
    rng = np.random.default_rng(abs(hash((round(genome.projection_angle, 4),
                                          genome.total_shapes()))) % 2**32)
    unit = min(w, h) / 1280.0   # line weights scale with the sheet

    # Kraft paper: warm, mottled, with a little fibre.
    paper = np.array(_KRAFT, np.float32)[None, None, :] * (
        0.965 + 0.035 * _noise(h, w, int(160 * unit) + 4, rng)[..., None]
        + 0.03 * _noise(h, w, int(5 * unit) + 2, rng)[..., None]
        + 0.02 * _noise(h, w, 2, rng)[..., None])
    sheet = paper.copy()

    prisms = _prisms(genome)
    # Planit sheets look down from higher, and stretch the plan out long.
    az = math.radians(genome.projection_angle * 0.6 + 24.0)
    view = _View(az, math.radians(40.0))
    margin = 0.12
    r, faces = _render_view(prisms, view, w, h, margin) if prisms else (None, [])

    # Frame: a single ruled border, drawn before the forms.
    fr = Image.new("L", (w, h), 0)
    fd = ImageDraw.Draw(fr)
    inset = int(0.035 * min(w, h))
    lw = max(1, int(round(1.6 * unit)))
    fd.rectangle([inset, inset, w - inset, h - inset - int(0.05 * h)], outline=150, width=lw)
    fd.line([(inset, h - inset), (w - inset, h - inset)], fill=70, width=lw)

    if r is not None:
        # Construction lines: long, faint rules along the plan axes, struck
        # through a few of the corners and run off toward the frame.
        cl = ImageDraw.Draw(fr)
        dirs = [view.project(np.array([[0, 0, 0], [c, s, 0]], np.float64))
                for c, s in ((math.cos(math.radians(prisms[0][0].rotation)),
                              -math.sin(math.radians(prisms[0][0].rotation))),)]
        d0 = dirs[0]
        du = (d0[1] - d0[0])[:2]
        du = du / (np.linalg.norm(du) + 1e-9)
        big = sorted(prisms, key=lambda p: -(p[0].w * p[0].h))[:3]
        for s, poly, bot, top in big:
            for vx, vy in poly[:2]:
                p = view.project(np.array([[vx, vy, bot]]))
                px = r.to_px(p)[0]
                a = px - du * 3 * max(w, h)
                b = px + du * 3 * max(w, h)
                cl.line([tuple(a), tuple(b)], fill=55, width=max(1, int(unit)))
        frame = np.asarray(fr).astype(np.float32) / 255.0
        # Rules stop at the border.
        clip = np.zeros_like(frame)
        clip[inset:h - inset - int(0.05 * h), inset:w - inset] = 1
        frame_rules = np.where(clip > 0, frame, 0)
        frame_rules[np.asarray(fr) >= 140] = np.asarray(fr)[np.asarray(fr) >= 140] / 255.0
        frame = frame_rules
    else:
        frame = np.asarray(fr).astype(np.float32) / 255.0

    graphite = np.zeros((h, w), np.float32)
    lines = np.zeros((h, w), np.float32)

    if r is not None:
        valid = r.fid >= 0
        L = _light_dir(view)
        pal = palette or []
        # Per face: how much graphite.  Dark and saturated roofs are laid in
        # solid; pale roofs stay as bare paper; walls turned from the light
        # get a thin grey wash of strokes.
        face_tone = np.zeros(len(faces) + 1, np.float32)
        face_ang = np.zeros(len(faces) + 1, np.float32)
        for k, (i, kind, nrm, poly) in enumerate(faces):
            s = prisms[i][0]
            q = r.to_px(view.project(poly))
            if kind == "top":
                c = pal[s.color_idx % len(pal)] if pal else (0.5, 0.5, 0.5)
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                sat = max(c) - min(c)
                dark = lum < 0.62 or sat > 0.35
                face_tone[k] = (0.97 - 0.2 * lum) if dark else 0.0
                # Strokes follow the long side of the roof.
                e = [np.linalg.norm(q[(j + 1) % len(q)] - q[j]) for j in range(len(q))]
                j = int(np.argmax(e))
                dv = q[(j + 1) % len(q)] - q[j]
                face_ang[k] = math.atan2(dv[1], dv[0]) + rng.normal(0, 0.05)
            else:
                lam = float(nrm @ L)
                face_tone[k] = 0.0 if lam > -0.05 else 0.16
                dv = q[2] - q[1]
                face_ang[k] = math.atan2(dv[1], dv[0]) + math.pi / 2 * 0.0
        fid = np.where(valid, r.fid, len(faces))
        tone = face_tone[fid]
        # Hatch each toned face in its own direction.
        for k in np.nonzero(face_tone[:-1] > 0)[0]:
            m = r.fid == k
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            hh = _hatch(y1 - y0, x1 - x0, float(face_ang[k]),
                        max(1.6, 2.4 * unit), rng)
            sub = m[y0:y1, x0:x1]
            t = face_tone[k]
            # Dense roofs: strokes overlap to near-solid, light paper
            # showing through between; walls: open strokes.
            val = np.clip(t * (0.45 + 0.75 * hh), 0, 1) if t > 0.5 else t * hh * 1.4
            g = graphite[y0:y1, x0:x1]
            g[sub] = np.maximum(g[sub], val[sub])

        # Outlines: every place the visible face changes, plus the silhouette.
        f = fid
        edge = np.zeros((h, w), bool)
        edge[:, 1:] |= f[:, 1:] != f[:, :-1]
        edge[1:, :] |= f[1:, :] != f[:-1, :]
        # Depth steps between faces of one plane don't need drawing; any id
        # change does in this style.
        el = Image.fromarray((edge * 255).astype(np.uint8))
        lw = max(1, int(round(1.1 * unit)))
        if lw > 1:
            el = el.filter(ImageFilter.MaxFilter(2 * (lw // 2) + 1))
        el = el.filter(ImageFilter.GaussianBlur(0.5 * unit))
        lines = np.asarray(el).astype(np.float32) / 255.0
        # Pencil pressure varies along each line.
        lines = np.clip(lines * 1.6, 0, 1) * (0.7 + 0.3 * _noise(h, w, int(40 * unit) + 3, rng))

    # Graphite on kraft: grain of the paper breaks up every stroke.
    tooth = 0.75 + 0.25 * _noise(h, w, 2, rng)
    dark = np.clip(np.maximum(np.maximum(graphite * tooth, lines * 0.95), frame * 0.7), 0, 1)
    out = sheet * (1 - dark[..., None]) + _GRAPHITE[None, None, :] * dark[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return img.resize((width, height), Image.LANCZOS)


def render_arkhitekton(genome: ShapeGenome, width: int, height: int,
                       style: str, palette: list[tuple] | None = None) -> Image.Image:
    if style == "plaster":
        return render_plaster(genome, width, height)
    return render_drawing(genome, width, height, palette)

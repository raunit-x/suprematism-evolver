"""Composition genome for Hadid / architecton mode.

The old architecton generator evolved the parameters of one central clump,
so every individual looked alike and selection had little to work with.
Here the genome describes a *composition* and is expressed into the
existing ``ShapeGenome`` shape lists (so the renderer, server and JSON
format keep working):

* Global genes: dominant axis, counter-axis, spread (clump -> dispersed),
  size falloff, extrusion depth, ground colour / ground plane, palette
  (chroma + accent), oblique projection.
* Group genes: 1-3 sub-clusters, each with a centre, angle offset, scale.
* Element genes: kind, group, position along / across the group axis,
  size, aspect, rotation offset, colour role, layer, depth, glass.

Harmony is built into expression rather than hoped for: rotations snap to
the axis, its perpendicular and one counter-direction; colours keep a
minimum contrast against the ground; the whole composition is fitted to
the frame by scaling, never by clamping single elements.

Evolution operators work on the genes: Gaussian nudges plus structural
mutations (add / remove / duplicate / split / throw), and crossover that
swaps whole groups or splits the canvas spatially.  ``select_diverse``
pre-filters candidates with cheap metrics and does a MAP-Elites style
pick over (spread, element count) so a generation never collapses into
one look.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field

from PIL import Image, ImageDraw

from src.shapes.genome import (Shape, ShapeGenome, ShapeGroup, architecton_harmony,
                               create_architecton)

# HADID_PALETTE indices.
WHITE, WARM_WHITE, LIGHT_GREY, MID_GREY, GRAPHITE, BLACK = 0, 1, 2, 3, 4, 5
DEEP_BLUE, COBALT, RED, OCHRE, PALE_BLUE = 6, 7, 8, 9, 10

_LIGHT_NEUTRALS = (WHITE, WARM_WHITE, LIGHT_GREY)
_DARK_NEUTRALS = (GRAPHITE, BLACK)
_CHROMAS = (DEEP_BLUE, COBALT, RED, OCHRE)

PLANE_KINDS = ("rect", "trapezoid", "parallelogram")
ACCENT_KINDS = ("bar", "bar", "square", "circle", "cross", "triangle")
ROLES = ("lead", "light", "dark", "chroma", "accent")

MAX_LONG = 0.62           # long side of the dominant element (canvas units)
MIN_ELEMENTS, MAX_ELEMENTS = 4, 22
MARGIN = 0.045


# ----------------------------------------------------------------------
# Genes
# ----------------------------------------------------------------------

@dataclass
class Group:
    cx: float
    cy: float
    angle: float = 0.0     # offset from the dominant axis (degrees)
    scale: float = 1.0     # spatial extent multiplier


@dataclass
class Element:
    kind: str
    group: int
    t: float               # position along the group axis, ~[-1, 1]
    u: float               # position across the group axis, ~[-1, 1]
    size: float            # (0, 1], relative to MAX_LONG
    aspect: float          # long / short side, >= 1
    rot: float             # raw offset from the group axis; snapped on expression
    role: str              # colour role, see ROLES
    layer: float           # 0..1 stacking height
    depth: float = 0.5     # 0..1 extrusion multiplier
    glass: bool = False
    color: int = -1        # explicit palette index; -1 = derive from role
    detail: str = "none"   # face detail pattern (architecton family)


@dataclass
class CompGenome:
    axis: float
    counter: float               # counter-direction offset from the axis
    spread: float                # 0 clump .. 1 dispersed
    falloff: float               # steepness of size distribution for new accents
    extrusion: float             # base extrusion depth (canvas units)
    ground: int
    ground_plane: int            # 0 none, 1 one flat plane, 2 plane + band
    plane_color: int
    chroma: int
    accent: int
    proj_angle: float
    proj_strength: float
    family: str = "composition"  # "composition" or "architecton"
    lift: float = 0.10           # elevation per unit of element layer
    bg_style: int = -1           # explicit renderer bg_style; -1 = from ground_plane
    bg_angle: float | None = None
    bg_cx: float | None = None
    bg_cy: float | None = None
    camera: dict | None = None   # extra projection fields passed to the renderer
    groups: list[Group] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)

    def copy(self) -> CompGenome:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CompGenome:
        d = dict(d)
        groups = [Group(**g) for g in d.pop("groups", [])]
        elements = [Element(**e) for e in d.pop("elements", [])]
        return cls(groups=groups, elements=elements, **d)


# ----------------------------------------------------------------------
# Expression: genes -> shapes
# ----------------------------------------------------------------------

def _lum(idx: int, palette: list[tuple]) -> float:
    r, g, b = palette[idx % len(palette)]
    return (0.299 * r + 0.587 * g + 0.114 * b) * 255.0


def _harmonic_angles(comp: CompGenome) -> list[float]:
    return [0.0, 90.0, comp.counter, comp.counter + 90.0]


def _snap(rot: float, comp: CompGenome) -> float:
    """Snap a raw rotation offset to the nearest harmonic direction, keeping
    the half-turn (so asymmetric shapes like triangles keep pointing the
    same way)."""
    deltas = [(rot - a + 90.0) % 180.0 - 90.0 for a in _harmonic_angles(comp)]
    return rot - min(deltas, key=abs)


def _role_color(role: str, comp: CompGenome, i: int) -> int:
    """Colour for a role; neutrals vary by element index so they're stable
    under mutation of unrelated genes."""
    if role == "lead":
        return comp.chroma
    if role == "light":
        return _LIGHT_NEUTRALS[i % len(_LIGHT_NEUTRALS)]
    if role == "dark":
        return _DARK_NEUTRALS[i % len(_DARK_NEUTRALS)]
    if role == "chroma":
        return comp.chroma
    return comp.accent


def _group_frame(comp: CompGenome, g: Group) -> tuple[float, float, float, float, float, float]:
    ang = math.radians(comp.axis + g.angle)
    dx, dy = math.cos(ang), math.sin(ang)
    along = (0.10 + 0.40 * comp.spread) * g.scale
    across = (0.05 + 0.24 * comp.spread) * g.scale
    return g.cx, g.cy, dx, dy, along, across


def _element_extent(s: Shape, proj: tuple[float, float]) -> tuple[float, float, float, float]:
    rad = math.radians(s.rotation)
    ca, sa = abs(math.cos(rad)), abs(math.sin(rad))
    hw = (s.w * ca + s.h * sa) / 2
    hh = (s.w * sa + s.h * ca) / 2
    ex = (s.depth + s.elevation) * proj[0]
    ey = (s.depth + s.elevation) * proj[1]
    return (s.x - hw + min(0.0, ex), s.y - hh + min(0.0, ey),
            s.x + hw + max(0.0, ex), s.y + hh + max(0.0, ey))


def express(comp: CompGenome, palette: list[tuple] | None = None,
            genome: ShapeGenome | None = None) -> ShapeGenome:
    """Build (or refresh) a ShapeGenome from the composition genes."""
    from src.art.palettes import HADID_PALETTE
    palette = palette or list(HADID_PALETTE.values())
    g = genome or ShapeGenome()

    ground_lum = _lum(comp.ground, palette)
    lead_color = comp.chroma
    if abs(_lum(lead_color, palette) - ground_lum) < 55:
        lead_color = _DARK_NEUTRALS[0] if ground_lum > 128 else WARM_WHITE
    lead_lum = _lum(lead_color, palette)

    def _legible(color: int, is_lead: bool) -> int:
        """First colour that reads against the ground and (for everything
        but the lead) against the lead plane it may sit on."""
        if is_lead:
            return lead_color
        # Neutrals before colour: red stays a rare accent, never a fallback.
        for c in (color, MID_GREY, LIGHT_GREY, GRAPHITE, WHITE, BLACK, comp.chroma, comp.accent):
            lc = _lum(c, palette)
            if abs(lc - ground_lum) >= 55 and abs(lc - lead_lum) >= 40:
                return c
        return color

    shapes_by_group: dict[int, list[Shape]] = {}
    for i, el in enumerate(comp.elements):
        gi = el.group if 0 <= el.group < len(comp.groups) else 0
        grp = comp.groups[gi]
        cx, cy, dx, dy, along, across = _group_frame(comp, grp)
        x = cx + dx * el.t * along - dy * el.u * across
        y = cy + dy * el.t * along + dx * el.u * across

        long_side = MAX_LONG * max(0.03, el.size)
        aspect = _clamp(el.aspect, 0.04, 30.0)
        if el.kind == "bar":
            aspect = max(aspect, 6.0)
        if el.kind in ("square", "circle", "cross") and comp.family == "composition":
            aspect = 1.0
        short = max(0.007, long_side / aspect)   # aspect < 1: taller than wide
        kind = "rect" if el.kind == "bar" else el.kind

        if el.color >= 0:
            color = el.color
        else:
            color = _legible(_role_color(el.role, comp, i), el.role == "lead")

        depth = comp.extrusion * max(0.0, 0.25 + 1.3 * el.depth)
        shape = Shape(
            kind=kind, x=x, y=y, w=long_side, h=short,
            rotation=(comp.axis + grp.angle + _snap(el.rot, comp)) % 360.0,
            color_idx=color, opacity=1.0,
            material="glass" if el.glass else "concrete",
            depth=depth, elevation=comp.lift * el.layer, detail=el.detail,
        )
        shapes_by_group.setdefault(gi, []).append(shape)

    _set_globals(g, comp)

    # Fit the whole composition to the frame (scale about its centre).
    allshapes = [s for ss in shapes_by_group.values() for s in ss]
    if allshapes:
        proj = (math.cos(math.radians(comp.proj_angle)) * comp.proj_strength,
                -math.sin(math.radians(comp.proj_angle)) * comp.proj_strength)
        exts = [_element_extent(s, proj) for s in allshapes]
        x0 = min(e[0] for e in exts); y0 = min(e[1] for e in exts)
        x1 = max(e[2] for e in exts); y1 = max(e[3] for e in exts)
        avail = 1.0 - 2 * MARGIN
        f = min(1.0, avail / max(x1 - x0, 1e-6), avail / max(y1 - y0, 1e-6))
        bx, by = (x0 + x1) / 2, (y0 + y1) / 2
        nx0, nx1 = bx + (x0 - bx) * f, bx + (x1 - bx) * f
        ny0, ny1 = by + (y0 - by) * f, by + (y1 - by) * f
        sx = max(0.0, MARGIN - nx0) - max(0.0, nx1 - (1 - MARGIN))
        sy = max(0.0, MARGIN - ny0) - max(0.0, ny1 - (1 - MARGIN))
        for s in allshapes:
            s.x = bx + (s.x - bx) * f + sx
            s.y = by + (s.y - by) * f + sy
            s.w *= f
            s.h *= f
            s.depth *= f
            s.elevation *= f

    groups = []
    for gi in sorted(shapes_by_group):
        ss = sorted(shapes_by_group[gi], key=lambda s: s.w * s.h, reverse=True)
        grp = comp.groups[gi]
        groups.append(ShapeGroup(anchor=ss[0], members=ss[1:], cx=grp.cx, cy=grp.cy,
                                 angle=comp.axis + grp.angle))
    g.groups = groups
    g.satellites = []
    g.comp = comp
    return g


def _set_globals(g: ShapeGenome, comp: CompGenome) -> None:
    g.engine_tag = "hadid"
    g.armature_angle = comp.axis
    g.bg_color_idx = comp.ground
    g.bg_sec_idx = comp.plane_color
    g.bg_style = comp.bg_style if comp.bg_style >= 0 else {0: 2, 1: 0, 2: 3}.get(comp.ground_plane, 2)
    g.bg_angle = comp.bg_angle if comp.bg_angle is not None else (
        (comp.axis + 90.0 + comp.counter * 0.5) % 360.0)
    g.bg_cx = comp.bg_cx if comp.bg_cx is not None else (
        0.5 + 0.35 * math.cos(math.radians(comp.axis + comp.counter)))
    g.bg_cy = comp.bg_cy if comp.bg_cy is not None else (
        0.5 + 0.35 * math.sin(math.radians(comp.axis + comp.counter)))
    g.bg_blend = 0.0
    g.projection_angle = comp.proj_angle
    g.projection_strength = comp.proj_strength
    g.projection_secondary_strength = 0.0
    g.camera_mode = 0
    g.perspective_bias = 0.0
    for k, v in (comp.camera or {}).items():
        setattr(g, k, v)


# ----------------------------------------------------------------------
# Random creation
# ----------------------------------------------------------------------

def _random_accent(comp: CompGenome, group: int, far: bool = False) -> Element:
    size = 0.04 + 0.30 * random.random() ** comp.falloff
    kind = random.choice(ACCENT_KINDS)
    reach = random.uniform(1.1, 1.8) if far else random.uniform(0.0, 1.1)
    return Element(
        kind=kind, group=group,
        t=random.choice((-1, 1)) * reach if far else random.gauss(0, 0.6),
        u=random.gauss(0, 0.9 if far else 0.6),
        size=size, aspect=random.uniform(6, 16) if kind == "bar" else random.uniform(1, 2.5),
        rot=random.choice((0.0, 0.0, 90.0, comp.counter)),
        role=random.choices(("dark", "light", "accent", "chroma"), weights=(26, 38, 20, 16))[0],
        layer=random.uniform(0.4, 1.0), depth=random.uniform(0.2, 1.0),
    )


def create_composition() -> CompGenome:
    axis = random.choice((random.uniform(-38, -18), random.uniform(18, 38),
                          random.uniform(-70, -52), random.uniform(52, 70)))
    ground = random.choices(
        (WARM_WHITE, WHITE, LIGHT_GREY, BLACK, GRAPHITE, DEEP_BLUE, OCHRE),
        weights=(36, 18, 10, 12, 6, 10, 8))[0]
    chroma = random.choice([c for c in _CHROMAS if c != ground and c != RED] or [COBALT])
    comp = CompGenome(
        axis=axis,
        counter=random.choice((-1, 1)) * random.uniform(24, 62),
        spread=random.random(),
        falloff=random.uniform(0.8, 2.4),
        extrusion=random.uniform(0.025, 0.06),
        ground=ground,
        ground_plane=random.choices((0, 1, 2), weights=(62, 30, 8))[0],
        plane_color=random.choice([c for c in (LIGHT_GREY, MID_GREY, WARM_WHITE, PALE_BLUE, OCHRE)
                                   if c != ground]),
        chroma=chroma,
        accent=RED if ground != RED else BLACK,
        proj_angle=random.uniform(25, 60) * random.choice((1, -1)) + random.choice((0, 180)),
        proj_strength=random.uniform(0.55, 1.0),
        lift=random.uniform(0.07, 0.14),
    )

    n_groups = random.choices((1, 2, 3), weights=(45, 38, 17))[0]
    cx = 0.5 + random.uniform(-0.1, 0.1)
    cy = 0.5 + random.uniform(-0.1, 0.1)
    comp.groups.append(Group(cx=cx, cy=cy, angle=0.0, scale=1.0))
    for _ in range(n_groups - 1):
        a = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.22, 0.36)
        comp.groups.append(Group(cx=0.5 + r * math.cos(a), cy=0.5 + r * math.sin(a),
                                 angle=random.choice((0.0, 0.0, 90.0, comp.counter)),
                                 scale=random.uniform(0.35, 0.65)))

    # One dominant plane.
    comp.elements.append(Element(
        kind=random.choices(("rect", "trapezoid", "parallelogram", "circle", "square"),
                            weights=(50, 15, 15, 10, 10))[0],
        group=0, t=random.gauss(0, 0.15), u=random.gauss(0, 0.15),
        size=1.0, aspect=random.uniform(1.3, 3.2), rot=0.0, role="lead",
        layer=0.0, depth=random.uniform(0.0, 0.4),
    ))
    # Two to four secondaries.
    for _ in range(random.randint(2, 4)):
        kind = random.choice(PLANE_KINDS + ("bar", "triangle"))
        comp.elements.append(Element(
            kind=kind, group=0, t=random.gauss(0, 0.55), u=random.gauss(0, 0.5),
            size=random.uniform(0.32, 0.65),
            aspect=random.uniform(7, 14) if kind == "bar" else random.uniform(1.4, 4.0),
            rot=random.choice((0.0, 0.0, 90.0)), role=random.choice(("light", "dark", "chroma")),
            layer=random.uniform(0.15, 0.6), depth=random.uniform(0.1, 0.8),
            glass=False,
        ))
    # Counter-movement: one long bar crossing the dominant diagonal.
    comp.elements.append(Element(
        kind="bar", group=0, t=random.gauss(0, 0.3), u=random.gauss(0, 0.3),
        size=random.uniform(0.6, 0.95), aspect=random.uniform(12, 22), rot=comp.counter,
        role=random.choice(("dark", "accent")), layer=random.uniform(0.6, 1.0),
        depth=random.uniform(0.1, 0.4),
    ))
    # Accents: some near the mass, some thrown far, some in satellite groups.
    for _ in range(random.randint(2, 9)):
        gi = random.randrange(len(comp.groups))
        comp.elements.append(_random_accent(comp, gi, far=random.random() < 0.3))
    # A translucent plane, sometimes.
    if random.random() < 0.45:
        secondaries = [e for e in comp.elements[1:] if e.kind in PLANE_KINDS]
        if secondaries:
            random.choice(secondaries).glass = True
    return comp


def _role_of(idx: int) -> str:
    if idx in _LIGHT_NEUTRALS:
        return "light"
    if idx in _DARK_NEUTRALS or idx == MID_GREY:
        return "dark"
    if idx == RED:
        return "accent"
    return "chroma"


def from_architecton(sg: ShapeGenome) -> CompGenome:
    """Re-express an architecton from the original generator as composition
    genes, reproducing its look (explicit colours, absolute depth and
    stacking, original ground and camera) so it can evolve with the same
    operators as everything else."""
    shapes = sg.flatten()
    axis = sg.armature_angle
    offs = [((s.rotation - axis + 90.0) % 180.0) - 90.0 for s in shapes]
    off_axis = [o for o in offs if 8.0 < abs(o) < 82.0]
    counter = max(set(round(o) for o in off_axis), key=lambda v: sum(
        1 for o in off_axis if abs(o - v) < 4)) if off_axis else 28.0
    max_depth = max((s.depth for s in shapes), default=0.03) or 0.03
    max_elev = max((s.elevation for s in shapes), default=0.06) or 0.06
    chromas = [s.color_idx for s in shapes if s.color_idx in _CHROMAS]
    comp = CompGenome(
        axis=axis, counter=float(counter), spread=0.5, falloff=1.4,
        extrusion=max_depth / 1.55, ground=sg.bg_color_idx, ground_plane=1,
        plane_color=sg.bg_sec_idx,
        chroma=max(set(chromas), key=chromas.count) if chromas else COBALT,
        accent=RED, proj_angle=sg.projection_angle, proj_strength=sg.projection_strength,
        family="architecton", lift=max_elev, bg_style=sg.bg_style,
        bg_angle=sg.bg_angle, bg_cx=sg.bg_cx, bg_cy=sg.bg_cy,
        camera={k: getattr(sg, k) for k in (
            "camera_mode", "perspective_bias", "projection_secondary_angle",
            "projection_secondary_strength", "projection_vanishing_x",
            "projection_vanishing_y")},
        groups=[Group(cx=0.5, cy=0.5)],
    )
    cx, cy, dx, dy, along, across = _group_frame(comp, comp.groups[0])
    lead = max(shapes, key=lambda s: s.w * s.h) if shapes else None
    for s in shapes:
        vx, vy = s.x - cx, s.y - cy
        comp.elements.append(Element(
            kind=s.kind, group=0,
            t=(vx * dx + vy * dy) / along, u=(-vx * dy + vy * dx) / across,
            size=s.w / MAX_LONG, aspect=s.w / max(s.h, 1e-6),
            rot=(s.rotation - axis) % 360.0,
            role="lead" if s is lead else _role_of(s.color_idx),
            layer=s.elevation / comp.lift,
            depth=(s.depth / comp.extrusion - 0.25) / 1.3,
            glass=s.material == "glass", color=s.color_idx, detail=s.detail,
        ))
    return comp


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mutate_composition(comp: CompGenome, strength: float = 1.0) -> None:
    s = max(0.05, strength)
    r = random.random

    # Global genes.
    if r() < 0.20 * s:
        comp.axis += random.gauss(0, 10 * s)
    if r() < 0.12 * s:
        comp.counter = _clamp(abs(comp.counter) + random.gauss(0, 8 * s), 18, 70) * (
            1 if comp.counter >= 0 else -1)
    if r() < 0.25 * s:
        comp.spread = _clamp(comp.spread + random.gauss(0, 0.15 * s), 0.0, 1.0)
    if r() < 0.12 * s:
        comp.falloff = _clamp(comp.falloff + random.gauss(0, 0.3 * s), 0.6, 3.0)
    if r() < 0.15 * s:
        comp.extrusion = _clamp(comp.extrusion * math.exp(random.gauss(0, 0.3 * s)), 0.008, 0.09)
    if r() < 0.10 * s:
        comp.proj_angle += random.gauss(0, 10 * s)
        comp.proj_strength = _clamp(comp.proj_strength + random.gauss(0, 0.1 * s), 0.2, 1.0)
    if r() < 0.05 * s:
        comp.ground = random.choice((WARM_WHITE, WHITE, LIGHT_GREY, BLACK, GRAPHITE, DEEP_BLUE, OCHRE))
    if r() < 0.06 * s:
        comp.ground_plane = random.choices((0, 1, 2), weights=(60, 30, 10))[0]
    if r() < 0.06 * s:
        comp.chroma = random.choice([c for c in _CHROMAS if c != comp.ground] or [COBALT])

    # Group genes.
    for grp in comp.groups:
        if r() < 0.2 * s:
            grp.cx = _clamp(grp.cx + random.gauss(0, 0.04 * s), 0.1, 0.9)
            grp.cy = _clamp(grp.cy + random.gauss(0, 0.04 * s), 0.1, 0.9)
        if r() < 0.08 * s:
            grp.scale = _clamp(grp.scale * math.exp(random.gauss(0, 0.2 * s)), 0.25, 1.4)

    # Element genes.
    for el in comp.elements:
        if r() < 0.3 * s:
            el.t += random.gauss(0, 0.15 * s)
            el.u += random.gauss(0, 0.15 * s)
        if r() < 0.2 * s:
            el.size = _clamp(el.size * math.exp(random.gauss(0, 0.15 * s)), 0.03, 1.0)
        if r() < 0.1 * s:
            el.aspect = _clamp(el.aspect * math.exp(random.gauss(0, 0.2 * s)), 0.04, 30.0)
        if r() < 0.06 * s:
            el.rot = random.choice(_harmonic_angles(comp))
        if r() < 0.1 * s:
            el.layer = _clamp(el.layer + random.gauss(0, 0.2 * s), 0.0, 1.5)
        if r() < 0.05 * s and el.role != "lead":
            el.role = random.choice(("light", "dark", "chroma", "accent"))
            el.color = -1          # recolour by role from now on

    # One structural mutation, at most.
    n = len(comp.elements)
    pick = r()
    if pick < 0.14 * s and n < MAX_ELEMENTS:
        comp.elements.append(_random_accent(comp, random.randrange(len(comp.groups)),
                                            far=r() < 0.35))
    elif pick < 0.26 * s and n > MIN_ELEMENTS:
        victim = random.randrange(1, n)
        del comp.elements[victim]
    elif pick < 0.36 * s and n < MAX_ELEMENTS:
        # Duplicate an element a step along the axis: rhythm.
        src = random.choice(comp.elements[1:] or comp.elements)
        dup = copy.deepcopy(src)
        dup.role = "dark" if src.role == "lead" else src.role
        step = random.choice((-1, 1)) * random.uniform(0.15, 0.35)
        dup.t += step
        dup.u += random.gauss(0, 0.05)
        comp.elements.append(dup)
    elif pick < 0.42 * s and n < MAX_ELEMENTS:
        # Split a plane into two along its long axis.
        planes = [e for e in comp.elements if e.kind in PLANE_KINDS and e.size > 0.25]
        if planes:
            p = random.choice(planes)
            p.size *= 0.62
            twin = copy.deepcopy(p)
            twin.role = "dark" if p.role != "dark" else "light"
            twin.t += 0.28
            p.t -= 0.08
            comp.elements.append(twin)
    elif pick < 0.50 * s:
        # Throw an accent far from the mass.
        small = [e for e in comp.elements if e.size < 0.35]
        if small:
            e = random.choice(small)
            e.t = random.choice((-1, 1)) * random.uniform(1.2, 1.9)
            e.u = random.gauss(0, 1.0)
    elif pick < 0.54 * s:
        planes = [e for e in comp.elements[1:] if e.kind in PLANE_KINDS]
        if planes:
            e = random.choice(planes)
            e.glass = not e.glass
    elif pick < 0.58 * s and len(comp.groups) < 3:
        a = random.uniform(0, 2 * math.pi)
        rr = random.uniform(0.22, 0.36)
        comp.groups.append(Group(cx=0.5 + rr * math.cos(a), cy=0.5 + rr * math.sin(a),
                                 angle=random.choice((0.0, 90.0, comp.counter)),
                                 scale=random.uniform(0.35, 0.65)))
        gi = len(comp.groups) - 1
        movers = [e for e in comp.elements[1:] if e.size < 0.4]
        for e in random.sample(movers, min(len(movers), random.randint(1, 3))):
            e.group = gi
            e.t, e.u = random.gauss(0, 0.5), random.gauss(0, 0.5)
    elif pick < 0.61 * s and len(comp.groups) > 1:
        gone = random.randrange(1, len(comp.groups))
        del comp.groups[gone]
        for e in comp.elements:
            if e.group == gone:
                e.group = 0
            elif e.group > gone:
                e.group -= 1

    # Keep exactly one lead: the largest element.
    lead = max(comp.elements, key=lambda e: e.size)
    for e in comp.elements:
        if e.role == "lead" and e is not lead:
            e.role = "chroma"
    lead.role = "lead"


# ----------------------------------------------------------------------
# Crossover
# ----------------------------------------------------------------------

def _absolute(comp: CompGenome, el: Element) -> tuple[float, float, float]:
    grp = comp.groups[el.group if 0 <= el.group < len(comp.groups) else 0]
    cx, cy, dx, dy, along, across = _group_frame(comp, grp)
    return (cx + dx * el.t * along - dy * el.u * across,
            cy + dy * el.t * along + dx * el.u * across,
            comp.axis + grp.angle + _snap(el.rot, comp))


def _rehome(el: Element, src: CompGenome, dst: CompGenome) -> Element:
    """Re-express an element of ``src`` in ``dst``'s main-group frame,
    keeping its absolute position; its rotation is re-snapped to ``dst``'s
    harmonic directions."""
    x, y, rot = _absolute(src, el)
    cx, cy, dx, dy, along, across = _group_frame(dst, dst.groups[0])
    vx, vy = x - cx, y - cy
    out = copy.deepcopy(el)
    out.group = 0
    out.t = (vx * dx + vy * dy) / along
    out.u = (-vx * dy + vy * dx) / across
    out.rot = (rot - dst.axis) % 180.0
    if out.role == "lead":
        out.role = "chroma"
    return out


def crossover_composition(a: CompGenome, b: CompGenome) -> CompGenome:
    child = a.copy()
    # Palette block from either parent, as a unit.
    if random.random() < 0.5:
        child.ground, child.ground_plane, child.plane_color = b.ground, b.ground_plane, b.plane_color
        child.chroma, child.accent = b.chroma, b.accent
    if random.random() < 0.3:
        child.spread = (a.spread + b.spread) / 2

    if random.random() < 0.5 and len(b.groups) > 1:
        # Group swap: replace / add one satellite group from B.
        gb = random.randrange(1, len(b.groups))
        if len(child.groups) > 1 and random.random() < 0.6:
            gi = random.randrange(1, len(child.groups))
            child.elements = [e for e in child.elements if e.group != gi]
        else:
            child.groups.append(copy.deepcopy(b.groups[gb]))
            gi = len(child.groups) - 1
        child.groups[gi] = copy.deepcopy(b.groups[gb])
        for e in b.elements:
            if e.group == gb:
                ne = copy.deepcopy(e)
                ne.group = gi
                if ne.role == "lead":
                    ne.role = "chroma"
                child.elements.append(ne)
    else:
        # Spatial crossover: A's elements on one side of a line through the
        # centre perpendicular to A's axis, B's elements on the other.
        ang = math.radians(a.axis + random.uniform(-30, 30))
        nx, ny = math.cos(ang), math.sin(ang)

        def side(comp, el):
            x, y, _ = _absolute(comp, el)
            return (x - 0.5) * nx + (y - 0.5) * ny

        keep = [e for e in a.elements if e.role == "lead" or side(a, e) < 0]
        take = [_rehome(e, b, child) for e in b.elements if side(b, e) >= 0]
        child.elements = copy.deepcopy(keep) + take

    child.elements = child.elements[:MAX_ELEMENTS]
    if len(child.elements) < MIN_ELEMENTS:
        child.elements += [_random_accent(child, 0) for _ in range(MIN_ELEMENTS - len(child.elements))]
    used = sorted({e.group for e in child.elements})
    remap = {old: i for i, old in enumerate(used)}
    child.groups = [child.groups[i] for i in used if i < len(child.groups)] or [Group(0.5, 0.5)]
    for e in child.elements:
        e.group = remap.get(e.group, 0)
        if e.group >= len(child.groups):
            e.group = 0
    lead = max(child.elements, key=lambda e: e.size)
    for e in child.elements:
        if e.role == "lead" and e is not lead:
            e.role = "chroma"
    lead.role = "lead"
    return child


# ----------------------------------------------------------------------
# Cheap metrics, harmony, diversity
# ----------------------------------------------------------------------

_RASTER = 64


def _polygon(s: Shape) -> list[tuple[float, float]]:
    rad = math.radians(s.rotation)
    c, sn = math.cos(rad), math.sin(rad)
    hw, hh = s.w / 2, s.h / 2
    return [((s.x + lx * c - ly * sn) * _RASTER, (s.y + lx * sn + ly * c) * _RASTER)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def metrics(genome: ShapeGenome) -> dict:
    shapes = genome.flatten()
    if not shapes:
        return {"viable": False}
    union = Image.new("L", (_RASTER, _RASTER), 0)
    draw = ImageDraw.Draw(union)
    total = 0.0
    for s in shapes:
        draw.polygon(_polygon(s), fill=1)
        total += s.w * s.h
    covered = sum(union.getdata()) / (_RASTER * _RASTER)
    areas = sorted((s.w * s.h for s in shapes), reverse=True)
    wsum = sum(areas) or 1e-9
    comx = sum(s.x * s.w * s.h for s in shapes) / wsum
    comy = sum(s.y * s.w * s.h for s in shapes) / wsum
    centroid = (sum(s.x for s in shapes) / len(shapes), sum(s.y for s in shapes) / len(shapes))
    spread = sum(math.hypot(s.x - centroid[0], s.y - centroid[1]) for s in shapes) / len(shapes)
    dirs = {round(s.rotation % 90.0) // 6 for s in shapes}
    lead_d = math.sqrt(areas[0])
    far = sum(1 for s in shapes
              if s.w * s.h < areas[0] * 0.08 and math.hypot(s.x - comx, s.y - comy) > 0.75 * lead_d)
    return {
        "viable": True,
        "occupancy": covered,
        "overlap": total / max(covered, 1e-6),
        "com_offset": math.hypot(comx - 0.5, comy - 0.5),
        "lead_share": areas[0] / wsum,
        "second_ratio": (areas[1] / areas[0]) if len(areas) > 1 else 0.0,
        "spread": spread,
        "count": len(shapes),
        "directions": len(dirs),
        "far_accents": far,
        "family": getattr(genome.comp, "family", "composition"),
        "architecton_harmony": architecton_harmony(genome),
    }


def _band(v: float, lo: float, hi: float, soft: float) -> float:
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return max(0.0, 1.0 - d / soft)


def harmony(m: dict) -> float:
    if not m.get("viable"):
        return 0.0
    if m["family"] == "architecton":
        return m["architecton_harmony"]
    return (0.18 * _band(m["occupancy"], 0.17, 0.30, 0.10)
            + 0.16 * _band(m["overlap"], 1.05, 1.45, 0.5)
            + 0.12 * _band(m["com_offset"], 0.03, 0.12, 0.08)
            + 0.16 * _band(m["lead_share"], 0.35, 0.55, 0.15)
            + 0.10 * _band(m["second_ratio"], 0.15, 0.45, 0.2)
            + 0.14 * _band(m["directions"], 2, 3, 1.5)
            + 0.14 * _band(m["far_accents"], 1, 3, 1.5))


def viable(m: dict) -> bool:
    if not m.get("viable", False):
        return False
    if m["family"] == "architecton":   # dense stacked structures by design
        return 0.08 <= m["occupancy"] <= 0.75 and m["com_offset"] <= 0.3
    return 0.06 <= m["occupancy"] <= 0.55 and m["overlap"] <= 2.6 and m["com_offset"] <= 0.3


def _features(m: dict, g: ShapeGenome) -> list[float]:
    from src.art.palettes import HADID_PALETTE
    pal = list(HADID_PALETTE.values())
    ax = math.radians(2 * g.armature_angle)
    return [m["spread"] * 4, m["count"] / 8, m["occupancy"] * 4, m["com_offset"] * 5,
            math.cos(ax), math.sin(ax), _lum(g.bg_color_idx, pal) / 128,
            2.0 if m["family"] == "architecton" else 0.0, m["lead_share"] * 3]


def select_diverse(candidates: list[ShapeGenome], k: int) -> list[ShapeGenome]:
    """Pre-filter with cheap metrics, keep the more harmonious part of the
    pool, then pick ``k`` greedily so each new pick is as different as
    possible (max-min distance over composition features) from those
    already chosen."""
    if k <= 0:
        return []
    scored = []
    for g in candidates:
        m = metrics(g)
        if viable(m):
            scored.append((harmony(m), _features(m, g), g))
    if len(scored) < k:          # not enough survivors: fall back to everyone
        for g in candidates:
            if all(g is not t[2] for t in scored):
                m = metrics(g)
                scored.append((0.0, _features(m, g) if m.get("viable") else [0.0] * 9, g))
    scored.sort(key=lambda t: -t[0])
    pool = scored[:max(k, int(len(scored) * 0.6))]
    chosen = [pool[0]]
    rest = pool[1:]
    while len(chosen) < k and rest:
        def gap(t):
            return min(sum((a - b) ** 2 for a, b in zip(t[1], c[1])) for c in chosen)
        best = max(rest, key=lambda t: gap(t) * (0.6 + 0.4 * t[0]))
        chosen.append(best)
        rest.remove(best)
    return [t[2] for t in chosen]


def new_individual(palette: list[tuple] | None = None) -> ShapeGenome:
    """Half the newcomers come from the original architecton generator
    (dense stacked 3D structures), half are dispersed compositions."""
    if random.random() < 0.5:
        return express(from_architecton(create_architecton(11)), palette)
    return express(create_composition(), palette)

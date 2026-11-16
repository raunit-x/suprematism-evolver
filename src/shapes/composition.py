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
ULTRA, VERMILION, CARMINE, CHROME, LAVENDER, PRUSSIAN = 11, 12, 13, 14, 15, 16

_LIGHT_NEUTRALS = (WHITE, WARM_WHITE, WHITE, LIGHT_GREY)
_DARK_NEUTRALS = (GRAPHITE, BLACK)
_CHROMAS = (DEEP_BLUE, COBALT, RED, OCHRE, ULTRA, VERMILION, CARMINE, CHROME, PRUSSIAN)
_ACCENTS = (RED, VERMILION, CARMINE)
_HUE_FAMILY = {DEEP_BLUE: "blue", COBALT: "blue", ULTRA: "blue", PRUSSIAN: "blue",
               PALE_BLUE: "blue", RED: "red", VERMILION: "red", CARMINE: "red",
               OCHRE: "yellow", CHROME: "yellow"}


def _same_hue(a: int, b: int) -> bool:
    fa = _HUE_FAMILY.get(a)
    return fa is not None and fa == _HUE_FAMILY.get(b)
# Flat, confident grounds: paper, lamp-black and the primaries themselves.
GROUNDS = (WARM_WHITE, WHITE, LIGHT_GREY, BLACK, GRAPHITE, DEEP_BLUE, ULTRA, OCHRE, PRUSSIAN,
           RED, VERMILION, CHROME, COBALT)
_GROUND_WEIGHTS = (26, 18, 4, 16, 3, 8, 5, 3, 4, 4, 1, 2, 2)
PLANE_COLORS = (WARM_WHITE, WHITE, CHROME, OCHRE, ULTRA, COBALT, BLACK, RED, VERMILION,
                DEEP_BLUE, LIGHT_GREY)

PLANE_KINDS = ("rect", "trapezoid", "parallelogram")
SOLIDS = ("sphere", "cylinder", "cone", "pyramid", "tetra")   # painted solids; nothing stands on them
STACK_BODIES = ("rect", "rect", "square", "trapezoid")
STACK_CAPS = (None, None, "pyramid", "tetra", "cylinder", "square")
ACCENT_KINDS = ("bar", "bar", "bar", "bar", "square", "triangle", "trapezoid", "circle",
                "sphere", "cylinder", "cone", "pyramid")
ROLES = ("lead", "light", "dark", "chroma", "accent")

MAX_LONG = 0.74           # long side of the dominant element (canvas units)
MIN_ELEMENTS, MAX_ELEMENTS = 4, 50
MAX_NEST = 4                       # blocks on blocks on blocks
THICKNESS_CAP = 0.38               # extrusion <= this x the element's shortest side
STACK_THICKNESS = 0.45             # stacked blocks may be up to this x their side
TOWER_GENE = 0.75                  # depth gene above this: a tower (4-9 storeys)
TOWER_CAP = 2.0                    # towers may rise to this x their short side
TOWER_MAX = 0.22                   # ... and never past this (canvas units)
DEPTH_GAIN = 2.2                   # all forms: solid slabs, not card
FLOAT_CHANCE = 0.04                # unresolvable edge overlap: kept & floated (<5%)
BOXY = ("rect", "square", "bar", "trapezoid", "parallelogram")
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
    stack: dict | None = None  # recursive stacking rule, see _new_stack_rule
    bleed: bool = False        # a spine that runs off the sheet
    parent: int = -1       # index of the block this one sits on (-1 = free);
                           # a nested element's t/u/size live in the parent's
                           # frame. Invariant: parent index < own index.


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
    ground_level: int = 0        # relief of the big ground field: -1 sunk .. 3 raised
    hover: float = 0.0           # altitude of the whole structure above the ground
    marks: list = field(default_factory=list)   # ground primitives, see _new_mark
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
    # Paper white and lamp-black carry the neutrals; grey only now and then.
    if role == "light":
        return (WHITE, WARM_WHITE, WHITE, WARM_WHITE, LIGHT_GREY)[i % 5]
    if role == "dark":
        return (BLACK, BLACK, BLACK, GRAPHITE)[i % 4]
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
    # Roof at -elevation along proj, floor at depth - elevation.
    offs = (-s.elevation, s.depth - s.elevation)
    xs = [o * proj[0] for o in offs]
    ys = [o * proj[1] for o in offs]
    return (s.x - hw + min(xs), s.y - hh + min(ys), s.x + hw + max(xs), s.y + hh + max(ys))


def express(comp: CompGenome, palette: list[tuple] | None = None,
            genome: ShapeGenome | None = None) -> ShapeGenome:
    """Build (or refresh) a ShapeGenome from the composition genes."""
    from src.art.palettes import HADID_PALETTE
    palette = palette or list(HADID_PALETTE.values())
    g = genome or ShapeGenome()

    ground_lum = _lum(comp.ground, palette)
    # No yellow-on-yellow or blue-on-blue: the lead colour and every
    # fallback must differ in hue from the ground and its second colour.
    clash = (comp.ground, comp.plane_color)
    chroma = comp.chroma
    if any(_same_hue(chroma, c) for c in clash):
        chroma = next((c for c in (ULTRA, VERMILION, CHROME, RED, COBALT)
                       if not any(_same_hue(c, k) for k in clash)), BLACK)
    lead_color = chroma
    if abs(_lum(lead_color, palette) - ground_lum) < 55:
        lead_color = _DARK_NEUTRALS[0] if ground_lum > 128 else WARM_WHITE
    lead_lum = _lum(lead_color, palette)

    def _legible(color: int, is_lead: bool) -> int:
        """First colour that reads against the ground and (for everything
        but the lead) against the lead plane it may sit on."""
        if is_lead:
            return lead_color
        # Neutrals before colour: red stays a rare accent, never a fallback.
        # Colour before grey: fall back to paper white, lamp-black or a
        # primary; greys only if nothing else reads.
        for c in (color, WHITE, BLACK, WARM_WHITE, LIGHT_GREY, GRAPHITE, chroma,
                  comp.accent):
            lc = _lum(c, palette)
            if abs(lc - ground_lum) >= 55 and abs(lc - lead_lum) >= 40 \
                    and not _same_hue(c, comp.ground) and not _same_hue(c, lead_color):
                return c
        return color

    proj = (math.cos(math.radians(comp.proj_angle)) * comp.proj_strength,
            -math.sin(math.radians(comp.proj_angle)) * comp.proj_strength)
    n = len(comp.elements)
    built: list[Shape | None] = [None] * n
    group_of: list[int] = [0] * n
    for i, el in enumerate(comp.elements):
        if el.color >= 0:
            color = el.color
        else:
            color = _legible(chroma if el.role == "chroma" else _role_color(el.role, comp, i), el.role == "lead")
        unit = comp.extrusion * 0.6 * DEPTH_GAIN          # one storey
        if comp.family == "composition":
            # Heights in whole storeys so roofs line up across the picture.
            dg = _clamp(el.depth, 0.0, 1.0)
            if dg > TOWER_GENE:
                depth = unit * (4 + round((dg - TOWER_GENE) / (1 - TOWER_GENE) * 5))
            else:
                depth = unit * (1 + round(dg / TOWER_GENE * 3))
        else:
            depth = comp.extrusion * DEPTH_GAIN * max(0.0, 0.25 + 1.3 * el.depth)
        kind = "rect" if el.kind == "bar" else el.kind

        parent = built[el.parent] if 0 <= el.parent < i else None
        if 0 <= el.parent < i and parent is None:
            continue                              # its support was dropped
        if parent is not None and parent.kind in SOLIDS:
            continue                              # nothing stands on a sphere or cone
        if parent is not None:
            shape = _nested_shape(el, parent, kind, color, depth * 0.8)
            gi = group_of[el.parent]
        else:
            gi = el.group if 0 <= el.group < len(comp.groups) else 0
            grp = comp.groups[gi]
            cx, cy, dx, dy, along, across = _group_frame(comp, grp)
            x = cx + dx * el.t * along - dy * el.u * across
            y = cy + dy * el.t * along + dx * el.u * across

            long_side = MAX_LONG * max(0.03, el.size)
            aspect = _clamp(el.aspect, 0.04, 80.0 if el.bleed else 30.0)
            if el.kind == "bar":
                aspect = max(aspect, 6.0)
            if el.kind in ("square", "circle", "cross") and comp.family == "composition":
                aspect = 1.0
            if el.kind in SOLIDS:
                aspect = 1.0
            short = max(0.007, long_side / aspect)   # aspect < 1: taller than wide
            # Thin, weightless planes: never thicker than a fraction of the
            # element's own shortest side.  Enforced here, at expression, so
            # it holds after generation, mutation and crossover alike.
            side = min(long_side, short)
            if el.depth > TOWER_GENE and not el.bleed and comp.family == "composition":
                # Variable heights: a minority of pieces rise as posts and
                # towers; everything else stays a slab.
                depth = min(depth, TOWER_CAP * side, TOWER_MAX)
            else:
                depth = min(depth, THICKNESS_CAP * side)
            if el.kind == "circle":
                depth = min(depth, 0.012)        # a painted disc, not a peg
            dg = _clamp(el.depth, 0.0, 1.0)
            if el.kind == "sphere":
                depth = side                     # as tall as it is wide
            elif el.kind == "cylinder":
                depth = min(side * (0.7 + 1.8 * dg), TOWER_MAX)
            elif el.kind == "cone":
                depth = min(side * (0.8 + 1.4 * dg), TOWER_MAX)
            if comp.family == "composition":
                # A few shared levels (ground, half, one, one and a half):
                # every free element stands on one of them.
                floor = comp.lift * round(_clamp(el.layer, 0.0, 1.5) * 2) / 2
            else:
                floor = comp.lift * el.layer
            shape = Shape(
                kind=kind, x=x, y=y, w=long_side, h=short,
                rotation=(comp.axis + grp.angle + _snap(el.rot, comp)) % 360.0,
                color_idx=color, opacity=1.0,
                material="glass" if el.glass else "concrete",
                depth=depth, elevation=floor + depth, detail=el.detail,
                bleed=el.bleed,
            )
        built[i] = shape
        group_of[i] = gi

    # Pass 2: no free element may cross another's edge in plan without
    # resting on it.  Connect, disconnect, or (rarely) float; else drop.
    _resolve_overlaps(comp, built)

    # Pass 3: grow stacks on what survived; a block that would collide with
    # anything but its own supports is not built.
    shapes_by_group: dict[int, list[Shape]] = {}
    placed = [s for s in built if s is not None and not s.bleed]
    for i, el in enumerate(comp.elements):
        shape = built[i]
        if shape is None:
            continue
        shapes_by_group.setdefault(group_of[i], []).append(shape)
        if el.stack and shape.kind in BOXY and not el.glass:
            blocks = _expand_stack(el.stack, shape, comp, proj, palette, placed)
            placed.extend(blocks)
            shapes_by_group[group_of[i]].extend(blocks)

    _set_globals(g, comp)

    # Fit the whole composition to the frame (scale about its centre).
    allshapes = [s for ss in shapes_by_group.values() for s in ss]
    # Altitude: the whole structure (spines included) can be held above the
    # ground, so nothing touches it and its shadows land well away, the
    # ground reading as a distant plane under a Proun floating in space.
    if comp.hover > 0:
        for s in allshapes:
            s.elevation += comp.hover
    framed = [s for s in allshapes if not s.bleed] or allshapes
    if allshapes:
        proj = (math.cos(math.radians(comp.proj_angle)) * comp.proj_strength,
                -math.sin(math.radians(comp.proj_angle)) * comp.proj_strength)
        exts = [_element_extent(s, proj) for s in framed]
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


def _nested_shape(el: Element, parent: Shape, kind: str, color: int, depth: float,
                  overhang: float = 0.0, solid: bool = False) -> Shape:
    """A block standing on ``parent``'s roof.

    Size and position are taken in the parent's frame.  ``t``/``u`` of
    +-1 put the block flush with a parent edge, 0 centres it; ``overhang``
    (0..0.6) lets it cantilever past the edge along its length by that
    fraction of itself.  Its floor is the parent's roof, so its elevation
    (roof height) is the parent's plus its own height.
    """
    turned = abs(((el.rot % 180.0) - 90.0)) < 45.0      # across the parent?
    span_x, span_y = (parent.h, parent.w) if turned else (parent.w, parent.h)
    long_side = min(MAX_LONG * max(0.02, el.size), (0.8 + overhang) * span_x)
    short = min(max(0.006, long_side / max(1.0, el.aspect)), 0.8 * span_y)
    if kind in ("square", "pyramid", "tetra", "cylinder", "sphere", "cone"):
        long_side = short = min(long_side, short)
    ext_x, ext_y = (short / 2, long_side / 2) if turned else (long_side / 2, short / 2)
    t = _clamp(el.t, -1.0, 1.0)
    u = _clamp(el.u, -1.0, 1.0)
    lx = t * abs(parent.w / 2 - ext_x)
    ly = u * abs(parent.h / 2 - ext_y)
    if overhang > 0 and not turned:
        lx += math.copysign(overhang * 2 * ext_x * abs(t), t)
    rad = math.radians(parent.rotation)
    qx = parent.x + lx * math.cos(rad) - ly * math.sin(rad)
    qy = parent.y + lx * math.sin(rad) + ly * math.cos(rad)
    side = min(long_side, short)
    if kind in ("pyramid", "tetra", "cone"):
        depth = side * (0.45 + 0.6 * _clamp(el.depth, 0.0, 1.0))   # a cap, not a spike
    elif solid:
        # Refinement blocks are volumes, not chips: roughly cubic or taller.
        depth = _clamp(depth, 0.20 * side, (STACK_THICKNESS + 1.4 * el.depth) * side)
    else:
        depth = min(depth, THICKNESS_CAP * side)
    return Shape(
        kind=kind if (kind in BOXY or kind in SOLIDS) else "rect", x=qx, y=qy,
        w=long_side, h=short, rotation=(parent.rotation + (90.0 if turned else 0.0)) % 360.0,
        color_idx=color, opacity=1.0, material="glass" if el.glass else "concrete",
        depth=depth, elevation=parent.elevation + depth, detail=el.detail, stacked=True,
    )


def _new_stack_rule(max_levels: int) -> dict:
    """A recursive stacking rule: how blocks grow on top of a block.

    mode     single: one block | terrace: a chain, each on the last |
             row: ``count`` blocks side by side, each growing a terrace
    levels   recursion depth (1..MAX_NEST)
    shrink   size of each level relative to the one below (0.4..0.88)
    t, u     where on the roof the first level stands (-1..1)
    drift    how far each further level steps along / across (-0.5..0.5)
    height   block height (0..1 of the composition's extrusion)
    contrast probability a level changes colour (0..1)
    turn     alternate levels turn 90 degrees
    overhang how far a level may cantilever past its support (0..0.6)
    branch   blocks each upper level splits into (1..3): stacks become trees
    vary     chance a block in a row/branch is a storey taller (skyline)
    articulate chance a block's roof carries an incised pattern
    On expression sizes snap to simple fractions of the block below and
    positions to flush / centred, so stacks read as built, not piled.
    seed     fixes the colour sequence
    """
    return {
        "mode": random.choices(("single", "terrace", "row"), weights=(30, 45, 25))[0],
        "levels": random.randint(1, max(1, max_levels)),
        "shrink": random.uniform(0.5, 0.8),
        "count": random.randint(2, 4),
        "t": random.uniform(-0.6, 0.6), "u": random.uniform(-0.6, 0.6),
        "drift": random.uniform(-0.35, 0.35),
        "height": random.uniform(0.1, 0.8),
        "contrast": random.uniform(0.0, 1.0),
        "turn": random.random() < 0.3,
        "overhang": random.choice((0.0, 0.0, random.uniform(0.1, 0.45))),
        "branch": 1,
        "vary": random.uniform(0.0, 0.4),
        "articulate": random.uniform(0.0, 0.3),
        "body": random.choice(STACK_BODIES),
        "cap": random.choice(STACK_CAPS),
        "seed": random.randrange(1 << 30),
    }


_FLUSH = (-1.0, -0.5, 0.0, 0.5, 1.0)
_ARTICULATIONS = ("grid_frame", "lines_h", "lines_v", "windows")
MAX_STACK_BLOCKS = 40          # per stack; keeps renders bounded
_FRACTIONS = (0.25, 1 / 3, 0.5, 2 / 3, 0.75)


def _snap_to(v: float, options: tuple[float, ...]) -> float:
    return min(options, key=lambda o: abs(o - v))


def _expand_stack(rule: dict, host: Shape, comp: CompGenome, proj: tuple[float, float],
                  palette: list[tuple], obstacles: list[Shape] | None = None) -> list[Shape]:
    """Grow the blocks described by ``rule`` on top of ``host``.  A block
    that would cross the footprint of anything other than its own chain of
    supports (``obstacles`` or a sibling) is not built, nor anything on it."""
    rng = random.Random(rule.get("seed", 0))
    out: list[Shape] = []
    others = [o for o in (obstacles or []) if o is not host]
    supports: dict[int, list[Shape]] = {id(host): [host]}
    levels = int(_clamp(rule.get("levels", 1), 1, MAX_NEST))
    mode = rule.get("mode", "terrace")
    if mode == "single":
        levels = 1
    # Mostly paper and lamp-black; colour only as the odd accent.
    options = [WHITE, WHITE, WARM_WHITE, BLACK, LIGHT_GREY, comp.accent]

    branch = int(_clamp(rule.get("branch", 1), 1, 3))

    def grow(parent: Shape, level: int) -> None:
        if level >= levels or len(out) >= MAX_STACK_BLOCKS:
            return
        row = mode == "row" and level == 0
        fan = level >= 1 and branch > 1
        n = int(_clamp(rule.get("count", 2), 2, 8)) if row else branch if fan else 1
        for k in range(n):
            if len(out) >= MAX_STACK_BLOCKS:
                return
            if row or fan:
                t = -0.85 + 1.7 * k / (n - 1)
                u = rule.get("u", 0.0) if row else _snap_to(rule.get("u", 0.0), _FLUSH)
                frac = rule.get("shrink", 0.6) * (0.9 / n if row else 0.5)
                aspect = rng.uniform(1.0, 2.2)
            else:
                t = _snap_to(rule.get("t", 0.0) + rule.get("drift", 0.0) * level, _FLUSH)
                u = _snap_to(rule.get("u", 0.0) - rule.get("drift", 0.0) * level, _FLUSH)
                frac = _snap_to(rule.get("shrink", 0.6), _FRACTIONS)
                aspect = max(1.0, parent.w / max(parent.h, 1e-6))
            color = parent.color_idx
            if rng.random() < rule.get("contrast", 0.3):
                pl = _lum(parent.color_idx, palette)
                choices = [c for c in options if abs(_lum(c, palette) - pl) >= 45] or options
                color = rng.choice(choices)
            syn = Element(kind="rect", group=0, t=t, u=u, size=(parent.w / MAX_LONG) * frac,
                          aspect=aspect, rot=90.0 if rule.get("turn") and level % 2 else 0.0,
                          role="", layer=0.0)
            hgt = _clamp(rule.get("height", 0.4), 0.0, 1.0)
            storeys = 1 + round(hgt * 3)                 # 1..4 storeys
            if (row or fan) and rng.random() < rule.get("vary", 0.0):
                storeys += rng.randint(1, 3)             # a stepped skyline
            depth = comp.extrusion * 0.6 * DEPTH_GAIN * storeys
            syn.depth = hgt
            # Vocabulary: body blocks below, an optional cap (pyramid,
            # tetrahedron, drum, cube) crowning the last level.
            kind = rule.get("body", "rect") or "rect"
            cap = rule.get("cap")
            if cap and level == levels - 1:
                kind = cap
            if kind in ("square", "pyramid", "tetra", "cylinder"):
                syn.aspect = 1.0
            child = _nested_shape(syn, parent, kind, color, depth,
                                  _clamp(rule.get("overhang", 0.0), 0.0, 0.6)
                                  if not (row or fan) else 0.0, solid=True)
            chain = supports[id(parent)] + [child]
            if any(_plan_overlap(child, o) > 1e-6 for o in others + out
                   if o not in chain):
                continue                      # would cross another edge: skip
            supports[id(child)] = chain
            if rng.random() < rule.get("articulate", 0.0):
                child.detail = rng.choice(_ARTICULATIONS)
            out.append(child)
            if child.kind not in SOLIDS:
                grow(child, level + 1)

    grow(host, 0)
    return out


# ----------------------------------------------------------------------
# Ground primitives
# ----------------------------------------------------------------------

MARK_KINDS = ("triangle", "line", "band", "disc")
_LINE_COLORS = (BLACK, RED, ULTRA, OCHRE, LAVENDER, GRAPHITE, VERMILION)
_FIELD_COLORS = (ULTRA, RED, BLACK, OCHRE, LAVENDER, DEEP_BLUE, LIGHT_GREY, CARMINE, CHROME)
MAX_MARKS = 6


def _new_level() -> int:
    """Relief of a ground field: -1 sunk into the ground, 0 flush, 1..3 a
    plate raised that many steps."""
    return random.choices((-1, 0, 1, 2, 3), weights=(15, 25, 30, 20, 10))[0]


def _new_mark(comp: CompGenome, kind: str | None = None) -> dict:
    """A flat ground primitive, all in normalised canvas units.

    triangle  a wedge: apex (x, y) off or near the sheet, pointing along
              ``angle``, opening ``spread`` degrees, ``length`` long
    line      a Malevich line through (x, y) at ``angle``: ``width`` thin,
              ``length`` (>= 1.5 crosses the sheet), ``count`` parallel
              lines ``gap`` apart
    band      a broad flat bar: like a line but ``width`` 0.03..0.14
    disc      a flat circle of ``radius``
    Angles default to the composition's axis / counter-axis for harmony.
    """
    kind = kind or random.choices(MARK_KINDS, weights=(30, 40, 15, 15))[0]
    ang = comp.axis + random.choice((0.0, 90.0, comp.counter, comp.counter + 90.0))
    if kind == "triangle":
        a = random.uniform(0, 2 * math.pi)
        return {"kind": kind, "x": 0.5 + 0.75 * math.cos(a), "y": 0.5 + 0.75 * math.sin(a),
                "angle": math.degrees(a) + 180.0 + random.gauss(0, 18),
                "spread": random.uniform(14, 48), "length": random.uniform(0.8, 1.7),
                "color": random.choice(_FIELD_COLORS), "level": _new_level()}
    if kind == "disc":
        return {"kind": kind, "x": random.uniform(0.15, 0.85), "y": random.uniform(0.15, 0.85),
                "radius": random.uniform(0.07, 0.3), "color": random.choice(_FIELD_COLORS),
                "level": _new_level()}
    band = kind == "band"
    return {"kind": kind, "x": random.uniform(0.2, 0.8), "y": random.uniform(0.2, 0.8),
            "angle": ang + random.gauss(0, 2),
            "width": random.uniform(0.03, 0.14) if band else random.uniform(0.0025, 0.009),
            "length": random.choice((2.0, 2.0, random.uniform(0.35, 0.9))),
            "count": 1 if band else random.choices((1, 2, 3), weights=(60, 25, 15))[0],
            "gap": random.uniform(0.012, 0.04),
            "color": random.choice(_FIELD_COLORS if band else _LINE_COLORS),
            "level": _new_level() if band else 0}


def _mutate_marks(comp: CompGenome, f: float, m: float, temp: float) -> None:
    r = random.random
    if r() < 0.08 * f and len(comp.marks) < MAX_MARKS:
        comp.marks.append(_new_mark(comp))
    if r() < 0.06 * f and comp.marks:
        del comp.marks[random.randrange(len(comp.marks))]
    for mk in comp.marks:
        if r() < 0.15 * f:
            mk["x"] = _clamp(mk["x"] + random.gauss(0, 0.06 * m), -0.4, 1.4)
            mk["y"] = _clamp(mk["y"] + random.gauss(0, 0.06 * m), -0.4, 1.4)
        if "angle" in mk and r() < 0.1 * f:
            mk["angle"] += random.gauss(0, 8 * m)
        if mk["kind"] == "triangle" and r() < 0.1 * f:
            mk["spread"] = _clamp(mk["spread"] + random.gauss(0, 6 * m), 6, 70)
            mk["length"] = _clamp(mk["length"] + random.gauss(0, 0.15 * m), 0.4, 2.2)
        if mk["kind"] == "disc" and r() < 0.1 * f:
            mk["radius"] = _clamp(mk["radius"] * math.exp(random.gauss(0, 0.2 * m)), 0.04, 0.4)
        if mk["kind"] == "line" and r() < 0.08 * f:
            mk["count"] = int(_clamp(mk["count"] + random.choice((-1, 1)), 1, 4))
        if r() < 0.05 * f * (0.5 + temp):
            mk["color"] = random.choice(_LINE_COLORS if mk["kind"] == "line" else _FIELD_COLORS)
        if mk["kind"] != "line" and r() < 0.1 * f:
            mk["level"] = int(_clamp(mk.get("level", 0) + random.choice((-1, 1)), -1, 3))


def _mark_signature(comp) -> list[float]:
    """Diversity features for the ground: how many of each primitive."""
    marks = getattr(comp, "marks", None) or []
    return [sum(1 for mk in marks if mk["kind"] == k) * 0.9 for k in MARK_KINDS]


# ----------------------------------------------------------------------
# Plan-view overlap rule
# ----------------------------------------------------------------------

def _footprint(s: Shape) -> list[tuple[float, float]]:
    """Plan footprint as an oriented rectangle (clockwise, y down)."""
    rad = math.radians(s.rotation)
    c, sn = math.cos(rad), math.sin(rad)
    hw, hh = s.w / 2, s.h / 2
    return [(s.x + lx * c - ly * sn, s.y + lx * sn + ly * c)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def _poly_area(p: list[tuple[float, float]]) -> float:
    return abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2


def _clip(subject: list, clip: list) -> list:
    """Sutherland-Hodgman: ``subject`` clipped to convex ``clip`` (either
    winding: "inside" an edge means on the same side as clip's centroid)."""
    cx = sum(p[0] for p in clip) / len(clip)
    cy = sum(p[1] for p in clip) / len(clip)
    out = subject
    for i in range(len(clip)):
        a, b = clip[i - 1], clip[i]

        def side(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        ref = side((cx, cy))

        def inside(p, ref=ref, side=side):
            return side(p) * ref >= 0

        def cross(p, q, a=a, b=b):
            x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
            x3, y3, x4, y4 = p[0], p[1], q[0], q[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-12:
                return q
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        src, out = out, []
        if not src:
            break
        for j in range(len(src)):
            cur, prev = src[j], src[j - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
    return out


def _plan_overlap(a: Shape, b: Shape) -> float:
    """Area where two footprints overlap in plan."""
    if abs(a.x - b.x) > (a.w + a.h + b.w + b.h) / 2:
        return 0.0
    inter = _clip(_footprint(a), _footprint(b))
    return _poly_area(inter) if len(inter) >= 3 else 0.0


def _inside(pt: tuple[float, float], s: Shape) -> bool:
    rad = math.radians(-s.rotation)
    dx, dy = pt[0] - s.x, pt[1] - s.y
    lx = dx * math.cos(rad) - dy * math.sin(rad)
    ly = dx * math.sin(rad) + dy * math.cos(rad)
    return abs(lx) <= s.w / 2 and abs(ly) <= s.h / 2


def _push_apart(b: Shape, a: Shape, gap: float) -> tuple[float, float]:
    """Smallest move of ``b`` that clears ``a`` (separating-axis test)."""
    best, axis = 1e9, (0.0, 0.0)
    for poly in (_footprint(a), _footprint(b)):
        for i in range(4):
            ex, ey = poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]
            ln = math.hypot(ex, ey) or 1.0
            nx, ny = -ey / ln, ex / ln
            pa = [p[0] * nx + p[1] * ny for p in _footprint(a)]
            pb = [p[0] * nx + p[1] * ny for p in _footprint(b)]
            over = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if over < best:
                side = 1.0 if (b.x - a.x) * nx + (b.y - a.y) * ny >= 0 else -1.0
                best, axis = over, (nx * side, ny * side)
    d = best + gap
    return axis[0] * d, axis[1] * d


def _nearest_clear(b: Shape, others: list[Shape]) -> tuple[float, float] | None:
    """Smallest move (12 directions x growing distances) that leaves ``b``
    clear of every footprint in ``others`` with its centre on the sheet."""
    probe = b.copy()
    for dist in (0.02, 0.04, 0.07, 0.1, 0.14, 0.19, 0.25, 0.32):
        for k in range(12):
            a = k * math.pi / 6
            dx, dy = math.cos(a) * dist, math.sin(a) * dist
            probe.x, probe.y = b.x + dx, b.y + dy
            if not (0.04 < probe.x < 0.96 and 0.04 < probe.y < 0.96):
                continue
            if all(_plan_overlap(probe, o) <= 1e-6 for o in others):
                return dx, dy
    return None


def _resolve_overlaps(comp: CompGenome, built: list[Shape | None]) -> None:
    """Mandatory structural rule for free elements, in plan view.

    A free element whose footprint crosses another's edge must either rest
    on it (its centre over the other's roof: it is set down onto that roof,
    so it may cantilever past the edge) or be moved clear of it.  If it
    can't be cleared, it is dropped -- except FLOAT_CHANCE of the time it
    is kept, raised a clear gap above, and painted with a hard shadow.
    Spines (bleed) pass under everything and are exempt.  Anything standing
    on a moved / dropped element moves / drops with it.
    """
    n = len(comp.elements)
    kids: dict[int, list[int]] = {}
    for i, el in enumerate(comp.elements):
        if 0 <= el.parent < i:
            kids.setdefault(el.parent, []).append(i)

    def family(i):
        out, todo = [], [i]
        while todo:
            j = todo.pop()
            out.append(j)
            todo.extend(kids.get(j, []))
        return out

    def shift(i, dx, dy, dz):
        for j in family(i):
            s = built[j]
            if s is not None:
                s.x += dx
                s.y += dy
                s.elevation += dz

    roots = [i for i in range(n) if built[i] is not None and comp.elements[i].parent < 0
             and not built[i].bleed]
    roots.sort(key=lambda i: -built[i].w * built[i].h)
    placed: list[int] = []
    rng = random.Random(int(abs(comp.axis) * 1000) + n)
    gap = comp.extrusion * 0.4
    for i in roots:
        b = built[i]
        rests_on: set[int] = set()
        ok = False
        for _ in range(6):
            hits = [j for j in placed if j not in rests_on and _plan_overlap(b, built[j]) > 1e-6]
            if not hits:
                ok = True
                break
            a = built[max(hits, key=lambda j: _plan_overlap(b, built[j]))]
            j = max(hits, key=lambda j: _plan_overlap(b, built[j]))
            if _inside((b.x, b.y), a) and b.w * b.h <= a.w * a.h and a.kind not in SOLIDS:
                # Connect: set it down on the highest roof it rests on
                # (cantilever allowed), never sinking into a lower support.
                rests_on.add(j)
                roof = max(built[k].elevation for k in rests_on)
                # Still a root gene, not a stack-rule block: it keeps
                # counting in metrics (stacked stays False).
                shift(i, 0.0, 0.0, (roof + b.depth) - b.elevation)
            else:
                # Disconnect: move it clear (and off anything it rested on).
                dx, dy = _push_apart(b, a, gap)
                shift(i, dx, dy, 0.0)
                if rests_on:
                    base = (b.elevation - b.depth)
                    shift(i, 0.0, 0.0, -base if base > 0 else 0.0)
                    rests_on.clear()
        if ok:
            placed.append(i)
            continue
        hits = [j for j in placed if j not in rests_on and _plan_overlap(b, built[j]) > 1e-6]
        if not hits:                       # the last move did clear it
            placed.append(i)
            continue
        # Before giving up: the nearest clear spot on the sheet, back on its
        # own level (disconnected from everything).
        spot = _nearest_clear(b, [built[j] for j in placed])
        if spot is not None:
            base = b.elevation - b.depth
            shift(i, spot[0], spot[1], -base if rests_on else 0.0)
            placed.append(i)
            continue
        if rng.random() < FLOAT_CHANCE:
            hits = [j for j in placed if _plan_overlap(b, built[j]) > 1e-6]
            top = max(built[j].elevation for j in hits)
            dz = top + gap + b.depth - b.elevation
            shift(i, 0.0, 0.0, dz)
            b.float_gap = gap
            placed.append(i)
        else:
            for j in family(i):
                built[j] = None


def _set_globals(g: ShapeGenome, comp: CompGenome) -> None:
    g.engine_tag = "hadid"
    g.armature_angle = comp.axis
    g.bg_color_idx = comp.ground
    g.bg_sec_idx = comp.plane_color
    # Plain ground or the great arc only: the hard two-tone diagonal splits
    # fought the figure, so they are no longer expressed (old genes included).
    g.bg_style = 8 if (comp.bg_style == 8 or (comp.bg_style < 0 and comp.ground_plane == 3)) else 2
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

def _new_camera() -> dict | None:
    """Most compositions use one oblique direction; some are seen in true
    perspective, walls fanning out from a vanishing point under the eye."""
    if random.random() < 0.7:
        return None
    return {"camera_mode": 2,
            "projection_vanishing_x": random.uniform(0.3, 0.7),
            "projection_vanishing_y": random.uniform(0.3, 0.7),
            "perspective_bias": random.uniform(0.0, 0.6)}


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
        role=random.choices(("dark", "light", "accent", "chroma"), weights=(34, 40, 14, 12))[0],
        layer=random.uniform(0.4, 1.0), depth=random.uniform(0.2, 1.0),
        stack=(_new_stack_rule(2) if kind in ("square", "trapezoid") and size > 0.1
               and random.random() < 0.45 else None),
    )


def create_composition() -> CompGenome:
    axis = random.choice((random.uniform(-38, -18), random.uniform(18, 38),
                          random.uniform(-70, -52), random.uniform(52, 70)))
    ground = random.choices(GROUNDS, weights=_GROUND_WEIGHTS)[0]
    chroma = random.choice([c for c in _CHROMAS if c != ground and c not in _ACCENTS]
                           or [COBALT])
    comp = CompGenome(
        axis=axis,
        counter=random.choice((-1, 1)) * random.uniform(24, 62),
        spread=random.random(),
        falloff=random.uniform(0.8, 2.4),
        extrusion=random.uniform(0.025, 0.06),
        ground=ground,
        ground_plane=random.choices((0, 3), weights=(62, 38))[0],
        plane_color=random.choice([c for c in PLANE_COLORS if c != ground]),
        chroma=chroma,
        accent=random.choice(_ACCENTS),
        proj_angle=random.uniform(12, 78) * random.choice((1, -1)) + random.choice((0, 180)),
        proj_strength=random.uniform(0.4, 1.35),
        camera=_new_camera(),
        lift=random.uniform(0.08, 0.20),
        ground_level=random.choices((-1, 0, 1, 2, 3), weights=(15, 25, 30, 20, 10))[0],
        hover=0.0 if random.random() < 0.3 else random.uniform(0.04, 0.2),
    )

    n_groups = random.choices((1, 2, 3), weights=(32, 40, 28))[0]
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
            rot=random.choice((0.0, 0.0, 90.0)), role=random.choice(("light", "light", "dark", "dark", "chroma")),
            layer=random.uniform(0.15, 0.6),
            depth=random.uniform(0.8, 1.0) if random.random() < 0.3 else random.uniform(0.1, 0.7),
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
    # Ground primitives: wedges, Malevich lines, bands, discs.
    for _ in range(random.choices((0, 1, 2, 3, 4), weights=(18, 30, 28, 16, 8))[0]):
        comp.marks.append(_new_mark(comp))
    # A spine: one long bar running off the sheet that ties the parts
    # together (the white bridge in Hadid's Tektonik).
    if random.random() < 0.35:
        comp.elements.append(Element(
            kind="bar", group=0, t=0.0, u=random.gauss(0, 0.25), size=random.uniform(2.0, 2.5),
            aspect=random.uniform(28, 60), rot=random.choice((0.0, 0.0, comp.counter)),
            role=random.choice(("light", "light", "dark")), layer=0.0, depth=0.05, bleed=True,
        ))
    # Recursive stacks, often.
    if random.random() < 0.6:
        for _ in range(random.randint(1, 3)):
            _nest(comp, max_levels=2)
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
        family="architecton", lift=max_elev, bg_style=8 if random.random() < 0.3 else 2,
        bg_angle=sg.bg_angle, bg_cx=sg.bg_cx, bg_cy=sg.bg_cy,
        camera={k: getattr(sg, k) for k in (
            "camera_mode", "perspective_bias", "projection_secondary_angle",
            "projection_secondary_strength", "projection_vanishing_x",
            "projection_vanishing_y")},
        groups=[Group(cx=0.5, cy=0.5)],
    )
    cx, cy, dx, dy, along, across = _group_frame(comp, comp.groups[0])
    lead = max(shapes, key=lambda s: s.w * s.h) if shapes else None

    def _ungrey(idx: int, k: int) -> int:
        # The old generator leans grey; repaint most greys as paper white or
        # lamp-black (one in four stays grey for a quiet middle value).
        if idx == LIGHT_GREY:
            return (WHITE, WARM_WHITE, WHITE, LIGHT_GREY)[k % 4]
        if idx in (MID_GREY, GRAPHITE):
            return (BLACK, WHITE, BLACK, idx)[k % 4]
        return idx

    light_bg = sg.bg_color_idx in (WHITE, WARM_WHITE, LIGHT_GREY)
    dark_bg = sg.bg_color_idx in (BLACK, GRAPHITE)
    for k, s in enumerate(shapes):
        c = _ungrey(s.color_idx, k)
        if (c in (WHITE, WARM_WHITE) and light_bg) or (c == BLACK and dark_bg):
            c = s.color_idx                  # would vanish into the ground
        s.color_idx = c
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
# Nesting helpers
# ----------------------------------------------------------------------

def _level(comp: CompGenome, i: int) -> int:
    lvl = 0
    while 0 <= comp.elements[i].parent < i and lvl < 8:
        i = comp.elements[i].parent
        lvl += 1
    return lvl


def _compact(elements: list[Element], keep: list[int]) -> list[Element]:
    """Keep ``elements[keep]`` (in order), remapping parent pointers; an
    element whose parent is dropped is dropped with it."""
    remap: dict[int, int] = {}
    out: list[Element] = []
    for j in sorted(set(keep)):
        e = elements[j]
        if e.parent >= 0 and e.parent not in remap:
            continue
        ne = copy.deepcopy(e)
        ne.parent = remap[e.parent] if e.parent >= 0 else -1
        remap[j] = len(out)
        out.append(ne)
    return out


def _subtree(elements: list[Element], root: int) -> list[int]:
    idx = [root]
    for j in range(root + 1, len(elements)):
        if elements[j].parent in idx:
            idx.append(j)
    return idx


def _add_satellite(comp: CompGenome, max_levels: int) -> None:
    """A small structure of its own orbiting the main mass."""
    gi = random.randrange(len(comp.groups)) if comp.groups else 0
    sat = _random_accent(comp, gi, far=random.random() < 0.5)
    sat.kind = random.choice(("rect", "square", "trapezoid"))
    sat.size = max(sat.size, 0.12)
    sat.aspect = random.uniform(1.0, 2.2)
    sat.stack = _new_stack_rule(min(2, max_levels))
    comp.elements.append(sat)


def _elaborate(comp: CompGenome, max_levels: int) -> None:
    """One step of ordered complexity on an existing structure."""
    stacked = [e for e in comp.elements if e.stack]
    op = random.choices(
        ("new", "deepen", "branch", "row", "vary", "articulate", "roof", "cap", "satellite"),
        weights=(3 if len(stacked) < 4 else 1, 3, 2.5, 2, 1.5, 1.5, 1, 2, 2))[0]
    if op == "satellite":
        _add_satellite(comp, max_levels)
        return
    if op == "new" or not stacked:
        _nest(comp, max_levels, allow_spine=True)
        return
    rule = _by_area(stacked).stack
    if op == "deepen":
        rule["levels"] = int(_clamp(rule["levels"] + 1, 1, max_levels))
        if rule["mode"] == "single":
            rule["mode"] = "terrace"
    elif op == "branch":
        rule["branch"] = int(_clamp(rule.get("branch", 1) + 1, 1, 3))
        rule["levels"] = max(rule["levels"], 2)
        if rule["mode"] == "single":
            rule["mode"] = "terrace"
    elif op == "row":
        rule["mode"] = "row"
        rule["count"] = int(_clamp(rule.get("count", 2) + random.randint(1, 2), 2, 8))
    elif op == "vary":
        rule["vary"] = _clamp(rule.get("vary", 0.0) + 0.2, 0.0, 1.0)
    elif op == "articulate":
        rule["articulate"] = _clamp(rule.get("articulate", 0.0) + 0.2, 0.0, 1.0)
    elif op == "cap":
        rule["cap"] = random.choice([c for c in STACK_CAPS if c])
        rule["body"] = random.choice(STACK_BODIES)
    else:   # incised pattern on a large free roof
        big = [e for e in comp.elements if e.parent < 0 and e.size > 0.3 and not e.glass
               and e.kind in BOXY and e.detail == "none"]
        if big:
            random.choice(big).detail = random.choice(_ARTICULATIONS)


def _nest(comp: CompGenome, max_levels: int = 2, allow_spine: bool = False) -> None:
    """Give a block a recursive stacking rule (or deepen an existing one)."""
    hosts = [e for e in comp.elements if e.kind in BOXY and not e.glass and e.parent < 0
             and (allow_spine or not e.bleed)]
    if not hosts:
        return
    bare = [e for e in hosts if not e.stack]
    if bare:
        _by_area(bare).stack = _new_stack_rule(max_levels)
    else:
        rule = _by_area(hosts).stack
        rule["levels"] = int(_clamp(rule["levels"] + 1, 1, max_levels))


def _by_area(els: list[Element]) -> Element:
    """Pick an element with probability proportional to its footprint, so
    structure grows on the large planes rather than on hairline bars."""
    return random.choices(els, weights=[e.size * e.size / max(1.0, e.aspect) + 1e-4
                                        for e in els])[0]


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _lead_index(els: list[Element]) -> int:
    """The dominant element: the largest one that isn't a bleeding spine."""
    cands = [j for j, e in enumerate(els) if not e.bleed] or list(range(len(els)))
    return max(cands, key=lambda j: els[j].size)


def _mutate_stack(rule: dict, f: float, m: float, max_levels: int) -> None:
    r = random.random
    if r() < 0.4 * f:
        rule["shrink"] = _clamp(rule["shrink"] + random.gauss(0, 0.06 * m), 0.4, 0.88)
    if r() < 0.3 * f:
        rule["t"] = _clamp(rule["t"] + random.gauss(0, 0.15 * m), -1, 1)
        rule["u"] = _clamp(rule["u"] + random.gauss(0, 0.15 * m), -1, 1)
    if r() < 0.25 * f:
        rule["drift"] = _clamp(rule["drift"] + random.gauss(0, 0.08 * m), -0.5, 0.5)
    if r() < 0.25 * f:
        rule["height"] = _clamp(rule["height"] + random.gauss(0, 0.15 * m), 0.02, 1.0)
    if r() < 0.2 * f:
        rule["contrast"] = _clamp(rule["contrast"] + random.gauss(0, 0.15 * m), 0.0, 1.0)
    if r() < 0.2 * f:
        rule["levels"] = int(_clamp(rule["levels"] + random.choice((-1, 1)), 1, max_levels))
    if r() < 0.12 * f:
        rule["count"] = int(_clamp(rule["count"] + random.choice((-1, 1)), 2, 8))
    if r() < 0.15 * f:
        rule["vary"] = _clamp(rule.get("vary", 0.0) + random.gauss(0, 0.15 * m), 0.0, 1.0)
    if r() < 0.15 * f:
        rule["articulate"] = _clamp(rule.get("articulate", 0.0) + random.gauss(0, 0.15 * m),
                                    0.0, 1.0)
    if r() < 0.06 * f:
        rule["mode"] = random.choice(("single", "terrace", "row"))
    if r() < 0.06 * f:
        rule["turn"] = not rule.get("turn", False)
    if r() < 0.15 * f:
        rule["overhang"] = _clamp(rule.get("overhang", 0.0) + random.gauss(0, 0.12 * m), 0.0, 0.6)
    if r() < 0.05 * f:
        rule["seed"] = random.randrange(1 << 30)
    if r() < 0.06 * f:
        rule["body"] = random.choice(STACK_BODIES)
    if r() < 0.08 * f:
        rule["cap"] = random.choice(STACK_CAPS)


_SIZE_LADDER = [1.0 / (2 ** (k / 2)) for k in range(12)]     # sqrt(2) steps


def _cleanup(comp: CompGenome) -> None:
    """One tidying step: the operators that turn accumulated randomness into
    a clean composition.  Run more often as the lineage cools."""
    els = comp.elements
    n = len(els)
    op = random.choices(("prune", "dedupe", "align", "sizes", "levels"),
                        weights=(3, 2, 3, 2, 0.5))[0]
    lead = _lead_index(els)
    if op == "prune" and n > MIN_ELEMENTS:
        # Remove the smallest free fragment (no stack, nothing on it).
        has_kids = {e.parent for e in els if e.parent >= 0}
        frags = [j for j, e in enumerate(els) if j != lead and e.parent < 0
                 and not e.stack and j not in has_kids and e.size < 0.12]
        if frags:
            victim = min(frags, key=lambda j: els[j].size)
            comp.elements = _compact(els, [j for j in range(n) if j != victim])
    elif op == "dedupe" and n > MIN_ELEMENTS:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = els[i], els[j]
                if (a.parent == b.parent and a.group == b.group and a.kind == b.kind
                        and abs(a.t - b.t) < 0.18 and abs(a.u - b.u) < 0.18
                        and 0.7 < a.size / max(b.size, 1e-6) < 1.4 and j != lead
                        and not any(e.parent == j for e in els)):
                    comp.elements = _compact(els, [k for k in range(n) if k != j])
                    return
    elif op == "align":
        # Snap a few elements onto a shared grid along / across the axis.
        for e in random.sample(els, min(n, random.randint(2, 4))):
            if e.parent < 0:
                e.t = round(e.t * 4) / 4
                e.u = round(e.u * 4) / 4
            e.rot = _snap(e.rot, comp)
    elif op == "sizes":
        for e in random.sample(els, min(n, random.randint(2, 5))):
            if e.size < 0.98:
                e.size = min(_SIZE_LADDER, key=lambda v: abs(math.log(v / max(e.size, 1e-3))))
    else:
        for e in els:
            e.layer = min((0.0, 0.5, 1.0), key=lambda v: abs(v - e.layer))
            e.depth = min((0.1, 0.45, 0.7, 0.95), key=lambda v: abs(v - e.depth))


def mutate_composition(comp: CompGenome, strength: float = 1.0, temp: float = 1.0,
                       detail: float | None = None) -> None:
    """Mutate in place.

    ``strength`` (the UI slider) mostly scales how big changes are; how
    many changes happen grows only with its square root.  ``temp`` (1 hot
    .. ~0.15 cold) anneals with lineage age: hot lineages get structural
    changes and big nudges, cold ones small nudges and cleanup.
    ``detail`` (0..1, the Detail knob) sets how much new structure each
    child grows -- deeper / branching stacks, rows, caps, satellites,
    articulated roofs -- independently of ``temp``.  None = automatic
    (grows as the lineage cools, the original behaviour).
    """
    temp = _clamp(temp, 0.05, 1.0)
    strength = max(0.05, strength)
    f = _clamp(math.sqrt(strength), 0.25, 1.5) * (0.35 + 0.65 * temp)   # frequency
    m = strength * (0.3 + 0.7 * temp)                                  # magnitude
    max_levels = 1 + int(round((MAX_NEST - 1) * (1.0 - temp)))         # complexity unlocks
    r = random.random

    # Global genes.
    if r() < 0.20 * f:
        comp.axis += random.gauss(0, 10 * m)
    if r() < 0.12 * f:
        comp.counter = _clamp(abs(comp.counter) + random.gauss(0, 8 * m), 18, 70) * (
            1 if comp.counter >= 0 else -1)
    if r() < 0.25 * f:
        comp.spread = _clamp(comp.spread + random.gauss(0, 0.15 * m), 0.0, 1.0)
    if r() < 0.12 * f:
        comp.falloff = _clamp(comp.falloff + random.gauss(0, 0.3 * m), 0.6, 3.0)
    if r() < 0.15 * f:
        comp.extrusion = _clamp(comp.extrusion * math.exp(random.gauss(0, 0.3 * m)), 0.008, 0.09)
    if r() < 0.10 * f:
        comp.proj_angle += random.gauss(0, 10 * m)
        comp.proj_strength = _clamp(comp.proj_strength + random.gauss(0, 0.1 * m), 0.25, 1.4)
    if r() < 0.04 * f * (0.5 + temp) and comp.family == "composition":
        comp.camera = _new_camera()
    elif comp.camera and comp.camera.get("camera_mode") == 2 and r() < 0.1 * f:
        comp.camera = dict(comp.camera)
        for k in ("projection_vanishing_x", "projection_vanishing_y"):
            comp.camera[k] = _clamp(comp.camera[k] + random.gauss(0, 0.06 * m), 0.15, 0.85)
    if r() < 0.06 * f * temp:
        comp.ground = random.choices(GROUNDS, weights=_GROUND_WEIGHTS)[0]
    if r() < 0.06 * f:
        comp.ground_plane = random.choices((0, 3), weights=(65, 35))[0]
    if r() < 0.06 * f:
        comp.chroma = random.choice([c for c in _CHROMAS if c != comp.ground] or [COBALT])
    if r() < 0.05 * f:
        comp.plane_color = random.choice(PLANE_COLORS)
    if r() < 0.08 * f:
        comp.hover = (0.0 if comp.hover > 0 and r() < 0.15 else
                      _clamp(max(comp.hover, 0.03) + random.gauss(0, 0.04 * m), 0.0, 0.24))
    if r() < 0.08 * f:
        comp.ground_level = int(_clamp(comp.ground_level + random.choice((-1, 1)), -1, 3))
    if r() < 0.08 * f:
        comp.lift = _clamp(comp.lift * math.exp(random.gauss(0, 0.25 * m)), 0.05, 0.26)
    _mutate_marks(comp, f, m, temp)

    # Group genes.
    for grp in comp.groups:
        if r() < 0.2 * f:
            grp.cx = _clamp(grp.cx + random.gauss(0, 0.04 * m), 0.1, 0.9)
            grp.cy = _clamp(grp.cy + random.gauss(0, 0.04 * m), 0.1, 0.9)
        if r() < 0.08 * f:
            grp.scale = _clamp(grp.scale * math.exp(random.gauss(0, 0.2 * m)), 0.25, 1.4)

    # Element genes.
    for el in comp.elements:
        if r() < 0.3 * f:
            el.t += random.gauss(0, 0.15 * m)
            el.u += random.gauss(0, 0.15 * m)
        if r() < 0.2 * f:
            el.size = _clamp(el.size * math.exp(random.gauss(0, 0.15 * m)), 0.03,
                             2.6 if el.bleed else 1.0)
        if r() < 0.1 * f:
            el.aspect = _clamp(el.aspect * math.exp(random.gauss(0, 0.2 * m)), 0.04,
                               80.0 if el.bleed else 30.0)
        if r() < 0.06 * f:
            el.rot = random.choice(_harmonic_angles(comp))
        if r() < 0.1 * f:
            el.layer = _clamp(el.layer + random.gauss(0, 0.2 * m), 0.0, 1.5)
        if r() < 0.12 * f:
            el.depth = _clamp(el.depth + random.gauss(0, 0.2 * m), 0.0, 1.0)
        if r() < 0.03 * f and el.role != "lead":
            el.depth = random.uniform(0.8, 1.0) if el.depth < TOWER_GENE else random.uniform(0.1, 0.6)
        if r() < 0.05 * f and el.role != "lead":
            el.role = random.choice(("light", "dark", "chroma", "accent"))
            el.color = -1          # recolour by role from now on
        if el.stack:
            _mutate_stack(el.stack, f, m, max_levels)

    # Structural change: frequent when hot, rare when cold.  Additions and
    # removals are balanced, and removal goes after fragments first.
    n = len(comp.elements)
    if r() < min(0.9, 0.75 * f):
        # With the Detail knob set, structure is meant to accumulate: removal
        # fades out as detail rises (cleanup below still clears noise).
        keep_bias = 1.0 - 0.85 * _clamp(detail, 0.0, 1.0) if detail is not None else 1.0
        ops = {"add": 12 * temp + 2, "remove": 12 * keep_bias, "duplicate": 8 * temp + 2,
               "split": 6 * temp, "throw": 8 * temp, "glass": 3,
               "group+": 4 * temp, "group-": 3, "stack": 16}
        op = random.choices(list(ops), weights=list(ops.values()))[0]
        if op == "add" and n < MAX_ELEMENTS:
            comp.elements.append(_random_accent(comp, random.randrange(len(comp.groups)),
                                                far=r() < 0.35))
        elif op == "remove" and n > MIN_ELEMENTS:
            lead = _lead_index(comp.elements)
            cands = [j for j in range(n) if j != lead]
            victim = random.choices(cands, weights=[1.0 / (comp.elements[j].size + 0.04)
                                                    for j in cands])[0]
            doomed = set(_subtree(comp.elements, victim))
            comp.elements = _compact(comp.elements, [j for j in range(n) if j not in doomed])
        elif op == "duplicate" and n < MAX_ELEMENTS:
            src = random.choice(comp.elements[1:] or comp.elements)
            dup = copy.deepcopy(src)
            dup.role = "dark" if src.role == "lead" else src.role
            dup.t += random.choice((-1, 1)) * random.uniform(0.15, 0.35)
            dup.u += random.gauss(0, 0.05)
            comp.elements.append(dup)
        elif op == "split" and n < MAX_ELEMENTS:
            planes = [e for e in comp.elements if e.kind in PLANE_KINDS and e.size > 0.25]
            if planes:
                p = random.choice(planes)
                p.size *= 0.62
                twin = copy.deepcopy(p)
                twin.stack = None
                twin.role = "dark" if p.role != "dark" else "light"
                twin.t += 0.28
                p.t -= 0.08
                comp.elements.append(twin)
        elif op == "throw":
            small = [e for e in comp.elements if e.size < 0.35 and e.parent < 0]
            if small:
                e = random.choice(small)
                e.t = random.choice((-1, 1)) * random.uniform(1.2, 1.9)
                e.u = random.gauss(0, 1.0)
        elif op == "glass":
            planes = [e for e in comp.elements[1:] if e.kind in PLANE_KINDS]
            if planes:
                e = random.choice(planes)
                e.glass = not e.glass
        elif op == "group+" and len(comp.groups) < 3:
            a = random.uniform(0, 2 * math.pi)
            rr = random.uniform(0.22, 0.36)
            comp.groups.append(Group(cx=0.5 + rr * math.cos(a), cy=0.5 + rr * math.sin(a),
                                     angle=random.choice((0.0, 90.0, comp.counter)),
                                     scale=random.uniform(0.35, 0.65)))
            gi = len(comp.groups) - 1
            movers = [e for e in comp.elements[1:] if e.size < 0.4 and e.parent < 0]
            for e in random.sample(movers, min(len(movers), random.randint(1, 3))):
                e.group = gi
                e.t, e.u = random.gauss(0, 0.5), random.gauss(0, 0.5)
        elif op == "group-" and len(comp.groups) > 1:
            gone = random.randrange(1, len(comp.groups))
            del comp.groups[gone]
            for e in comp.elements:
                if e.group == gone:
                    e.group = 0
                elif e.group > gone:
                    e.group -= 1
        elif op == "stack":
            _nest(comp, max_levels)

    # Complexification: ordered structure (recursive stacks) grows as the
    # lineage cools and deeper levels unlock, while free fragments are
    # cleaned away below.  Complexity rises, noise falls.
    cool = 1.0 - temp
    if detail is None:
        # Elaboration: as the lineage cools, structures get more intricate in
        # an ordered way (deeper / branching stacks, longer rows, skylines,
        # articulated roofs).  This is what "refine" means.
        if r() < 0.15 + 0.7 * cool:
            _elaborate(comp, max_levels)
            for _ in range(3):                 # cooler lineages elaborate more per step
                if r() < cool * 0.85:
                    _elaborate(comp, max_levels)
    else:
        # Detail knob: elaboration on its own dial.  0 adds nothing; 1 adds
        # up to fourteen ordered steps per child, unlocks the deepest
        # nesting, and seeds whole new satellite structures.
        d = _clamp(detail, 0.0, 1.0)
        levels = max(max_levels, 1 + int(round((MAX_NEST - 1) * d)))
        for k in range(14):
            if r() < d * (1.0 - 0.045 * k):
                _elaborate(comp, levels)
        for _ in range(int(round(3 * d * d))):
            if len(comp.elements) < MAX_ELEMENTS and r() < 0.7:
                _add_satellite(comp, levels)
    # Cleanup removes only noise (stray fragments, duplicates, misalignment).
    if r() < 0.1 + 0.5 * cool:
        _cleanup(comp)

    # Keep exactly one lead: the largest element.
    lead = comp.elements[_lead_index(comp.elements)]
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


def _root(elements: list[Element], j: int) -> int:
    hops = 0
    while 0 <= elements[j].parent < j and hops < 8:
        j = elements[j].parent
        hops += 1
    return j


def _regroup(e: Element, gi: int) -> Element:
    e.group = gi
    if e.role == "lead":
        e.role = "chroma"
    return e


def _append_tree(dst: list[Element], src: list[Element], fix) -> None:
    """Append a compacted element list, offsetting its parent pointers."""
    off = len(dst)
    for e in src:
        ne = fix(copy.deepcopy(e))
        ne.parent = e.parent + off if e.parent >= 0 else -1
        if ne.role == "lead":
            ne.role = "chroma"
        dst.append(ne)


def crossover_composition(a: CompGenome, b: CompGenome) -> CompGenome:
    child = a.copy()
    # Palette block from either parent, as a unit.
    if random.random() < 0.5:
        child.ground, child.ground_plane, child.plane_color = b.ground, b.ground_plane, b.plane_color
        child.chroma, child.accent = b.chroma, b.accent
        child.marks = copy.deepcopy(b.marks)
    if random.random() < 0.3:
        child.spread = (a.spread + b.spread) / 2

    if random.random() < 0.5 and len(b.groups) > 1:
        # Group swap: replace / add one satellite group from B.
        gb = random.randrange(1, len(b.groups))
        if len(child.groups) > 1 and random.random() < 0.6:
            gi = random.randrange(1, len(child.groups))
            child.elements = _compact(child.elements, [
                j for j in range(len(child.elements))
                if child.elements[_root(child.elements, j)].group != gi])
        else:
            child.groups.append(copy.deepcopy(b.groups[gb]))
            gi = len(child.groups) - 1
        child.groups[gi] = copy.deepcopy(b.groups[gb])
        moved = _compact(b.elements, [j for j in range(len(b.elements))
                                      if b.elements[_root(b.elements, j)].group == gb])
        _append_tree(child.elements, moved, lambda e: _regroup(e, gi))
    else:
        # Spatial crossover: A's elements on one side of a line through the
        # centre perpendicular to A's axis, B's elements on the other.
        ang = math.radians(a.axis + random.uniform(-30, 30))
        nx, ny = math.cos(ang), math.sin(ang)

        def side(comp, j):
            # Nested blocks travel with the block they stand on.
            x, y, _ = _absolute(comp, comp.elements[_root(comp.elements, j)])
            return (x - 0.5) * nx + (y - 0.5) * ny

        keep = _compact(a.elements, [j for j in range(len(a.elements))
                                     if a.elements[_root(a.elements, j)].role == "lead"
                                     or side(a, j) < 0])
        take = _compact(b.elements, [j for j in range(len(b.elements)) if side(b, j) >= 0])
        child.elements = keep
        _append_tree(child.elements, take,
                     lambda e: _rehome(e, b, child) if e.parent < 0 else e)

    child.elements = _compact(child.elements, list(range(min(len(child.elements), MAX_ELEMENTS))))
    if len(child.elements) < MIN_ELEMENTS:
        child.elements += [_random_accent(child, 0) for _ in range(MIN_ELEMENTS - len(child.elements))]
    used = sorted({e.group for e in child.elements})
    remap = {old: i for i, old in enumerate(used)}
    child.groups = [child.groups[i] for i in used if i < len(child.groups)] or [Group(0.5, 0.5)]
    for e in child.elements:
        e.group = remap.get(e.group, 0)
        if e.group >= len(child.groups):
            e.group = 0
    lead = child.elements[_lead_index(child.elements)]
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
    allshapes = [s for s in genome.flatten() if not s.bleed]
    n_stacked = sum(1 for s in allshapes if s.stacked)
    # Overlap, coverage, balance and the craft checks look at free elements
    # only: a block standing on another is ordered structure, not clutter.
    shapes = [s for s in allshapes if not s.stacked] or allshapes
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
    lead_shape = max(shapes, key=lambda s: s.w * s.h)

    # "Craft" signature 1: most elements knotted in one small region.
    knot = sum(1 for s in shapes if math.hypot(s.x - comx, s.y - comy) < 0.15) / len(shapes)

    # "Craft" signature 2: mirror symmetry about the main axis through the
    # centre of mass (a fuselage on the axis with paired wings).
    ax = math.radians(genome.armature_angle)
    ux, uy = math.cos(ax), math.sin(ax)
    matched = 0.0
    for s in shapes:
        vx, vy = s.x - comx, s.y - comy
        along, across = vx * ux + vy * uy, -vx * uy + vy * ux
        if abs(across) < 0.02:                       # sits on the axis
            matched += 0.5 * s.w * s.h
            continue
        mx = comx + along * ux + across * uy          # mirrored centre
        my = comy + along * uy - across * ux
        for o in shapes:
            if o is not s and o.kind == s.kind and math.hypot(o.x - mx, o.y - my) < 0.04 \
                    and 0.6 < (o.w * o.h) / max(s.w * s.h, 1e-9) < 1.6:
                matched += s.w * s.h
                break
    symmetry = matched / wsum
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
        "fragments": sum(1 for e in (genome.comp.elements if genome.comp else [])
                         if e.parent < 0 and not e.stack and e.size < 0.08),
        "genes": len(genome.comp.elements) if genome.comp else len(shapes),
        "stacked": n_stacked,
        "architecton_harmony": architecton_harmony(genome),
        "knot": knot,
        "symmetry": symmetry,
        "lead_x": lead_shape.x,
        "lead_y": lead_shape.y,
        "lead_color": lead_shape.color_idx,
        "arc": 1.0 if genome.bg_style == 8 else 0.0,
        "marks": _mark_signature(genome.comp),
    }


def _band(v: float, lo: float, hi: float, soft: float) -> float:
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return max(0.0, 1.0 - d / soft)


def harmony(m: dict) -> float:
    if not m.get("viable"):
        return 0.0
    tidy = (0.35 * _band(m["fragments"], 0, 3, 4) + 0.20 * _band(m["genes"], 5, 16, 10)
            + 0.45 * min(1.0, m["stacked"] / 16.0))       # ordered complexity, more is better
    if m["family"] == "architecton":
        return 0.7 * m["architecton_harmony"] + 0.3 * tidy
    return 0.25 * tidy + 0.75 * ( (0.18 * _band(m["occupancy"], 0.17, 0.30, 0.10)
            + 0.16 * _band(m["overlap"], 1.05, 1.45, 0.5)
            + 0.12 * _band(m["com_offset"], 0.03, 0.12, 0.08)
            + 0.16 * _band(m["lead_share"], 0.35, 0.55, 0.15)
            + 0.10 * _band(m["second_ratio"], 0.15, 0.45, 0.2)
            + 0.14 * _band(m["directions"], 2, 3, 1.5)
            + 0.14 * _band(m["far_accents"], 1, 3, 1.5)))


def is_craft(m: dict) -> bool:
    """The two traits that make a composition read as a vehicle or machine:
    a compact overlapping knot, or mirror symmetry along its axis."""
    dense = m["family"] == "architecton"
    knotted = m["knot"] > (0.85 if dense else 0.7) and m["overlap"] > (2.2 if dense else 1.5)
    return knotted or m["symmetry"] > 0.5


def viable(m: dict) -> bool:
    if not m.get("viable", False) or is_craft(m):
        return False
    if m["family"] == "architecton":   # dense stacked structures by design
        return 0.08 <= m["occupancy"] <= 0.75 and m["com_offset"] <= 0.3
    return 0.06 <= m["occupancy"] <= 0.55 and m["overlap"] <= 2.6 and m["com_offset"] <= 0.3


def _features(m: dict, g: ShapeGenome) -> list[float]:
    from src.art.palettes import HADID_PALETTE
    pal = list(HADID_PALETTE.values())
    ax = math.radians(2 * g.armature_angle)
    lc = math.radians(m["lead_color"] * 41.0)        # spread colour indices on a circle
    return [m["spread"] * 4, m["count"] / 8, m["occupancy"] * 4, m["com_offset"] * 5,
            math.cos(ax), math.sin(ax), _lum(g.bg_color_idx, pal) / 128,
            2.0 if m["family"] == "architecton" else 0.0, m["lead_share"] * 3,
            m["lead_x"] * 3, m["lead_y"] * 3, math.cos(lc), math.sin(lc),
            m["arc"] * 1.5, m["stacked"] / 6] + m["marks"]


# Minimum squared feature distance between members of one generation.
# Relaxed step by step only if the pool can't fill the generation otherwise.
SIBLING_MIN_GAP = 1.6


def select_diverse(candidates: list[ShapeGenome], k: int, refine: float = 0.0,
                   existing: list[ShapeGenome] | None = None) -> list[ShapeGenome]:
    """Pre-filter with cheap metrics, keep the more harmonious part of the
    pool, then pick ``k`` greedily so each new pick is as different as
    possible (max-min distance over composition features) from those
    already chosen.  ``refine`` (0 explore .. 1 refine) shrinks the pool
    to the most harmonious and weights harmony over difference.
    ``existing`` members (e.g. the kept picks) count for distance but are
    not returned, so no candidate can be a near-clone of them either."""
    refine = _clamp(refine, 0.0, 1.0)
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
    pool = scored[:max(k, int(len(scored) * (0.6 - 0.35 * refine)))]
    anchors = []
    for g in existing or []:
        m = metrics(g)
        if m.get("viable"):
            anchors.append((1.0, _features(m, g), g))
    # Seed with the best candidate only when there are no anchors; otherwise
    # every pick, the best included, must clear the distance check.
    chosen = anchors + ([] if anchors else [pool[0]])
    rest = pool[len(chosen) - len(anchors):]
    w = 0.3 + 0.45 * refine          # never all harmony: keep real choices
    while len(chosen) - len(anchors) < k and rest:
        def gap(t):
            return min(sum((a - b) ** 2 for a, b in zip(t[1], c[1])) for c in chosen)
        gaps = {id(t): gap(t) for t in rest}
        floor = SIBLING_MIN_GAP
        far = [t for t in rest if gaps[id(t)] >= floor]
        while not far and floor > 0.05:               # relax only when forced
            floor *= 0.6
            far = [t for t in rest if gaps[id(t)] >= floor]
        far = far or rest
        top = max(gaps[id(t)] for t in far) or 1.0
        best = max(far, key=lambda t: (1 - w) * gaps[id(t)] / top + w * t[0])
        chosen.append(best)
        rest.remove(best)
    return [t[2] for t in chosen[len(anchors):]]


def new_individual(palette: list[tuple] | None = None) -> ShapeGenome:
    """Half the newcomers come from the original architecton generator
    (dense stacked 3D structures), half are dispersed compositions."""
    if random.random() < 0.5:
        comp = from_architecton(create_architecton(11))
        for _ in range(random.choices((0, 1, 2, 3), weights=(25, 35, 28, 12))[0]):
            comp.marks.append(_new_mark(comp))
        if random.random() < 0.4:
            for _ in range(random.randint(1, 2)):
                _nest(comp, max_levels=2)
        return express(comp, palette)
    return express(create_composition(), palette)

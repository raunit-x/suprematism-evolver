"""Hierarchical shape-genome for Suprematist evolutionary art.

A genome is composed of *ShapeGroups* (compositional clusters, each
with an anchor shape and supporting members) plus *satellites* (small
isolated accent marks for rhythm and balance).  Groups are the unit
of crossover — a well-evolved cluster survives intact across
generations.

Rendering order: groups in list order (anchor first, then members
back-to-front within each group), then all satellites on top.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

SHAPE_KINDS = (
    "square", "rect", "circle", "cross", "trapezoid",
    "triangle", "line", "ellipse", "semicircle",
)

MIN_SIZE = 0.02
MAX_SIZE = 0.55
MAX_GROUPS = 5
MAX_SATELLITES = 8
MAX_MEMBERS_PER_GROUP = 6

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

_BAR_KINDS = ("rect", "rect", "rect", "line", "line", "rect", "square")
_ALL_KINDS_WEIGHTED = SHAPE_KINDS + ("rect", "square", "line")
_ANCHOR_KINDS = ("square", "rect", "rect", "circle")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _ensure_inside(shape: Shape) -> None:
    """Adjust position (and size if necessary) so the rotated bounding box
    stays fully within the [0, 1] x [0, 1] canvas."""
    margin = 0.01
    max_span = 1.0 - 2 * margin

    rad = math.radians(shape.rotation)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))

    bb_w = shape.w * cos_a + shape.h * sin_a
    bb_h = shape.w * sin_a + shape.h * cos_a

    if bb_w > max_span or bb_h > max_span:
        scale = min(max_span / max(bb_w, 1e-9), max_span / max(bb_h, 1e-9))
        shape.w *= scale
        shape.h *= scale
        bb_w *= scale
        bb_h *= scale

    shape.x = _clamp(shape.x, bb_w / 2 + margin, 1.0 - bb_w / 2 - margin)
    shape.y = _clamp(shape.y, bb_h / 2 + margin, 1.0 - bb_h / 2 - margin)


# ------------------------------------------------------------------
# Shape
# ------------------------------------------------------------------

@dataclass
class Shape:
    kind: str
    x: float
    y: float
    w: float
    h: float
    rotation: float
    color_idx: int
    opacity: float = 1.0

    def copy(self) -> Shape:
        return Shape(
            self.kind, self.x, self.y, self.w, self.h,
            self.rotation, self.color_idx, self.opacity,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "x": self.x, "y": self.y,
            "w": self.w, "h": self.h, "rotation": self.rotation,
            "color_idx": self.color_idx, "opacity": self.opacity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Shape:
        return cls(
            kind=d["kind"], x=d["x"], y=d["y"],
            w=d["w"], h=d["h"], rotation=d["rotation"],
            color_idx=d["color_idx"], opacity=d.get("opacity", 1.0),
        )


# ------------------------------------------------------------------
# ShapeGroup — a compositional cluster
# ------------------------------------------------------------------

@dataclass
class ShapeGroup:
    anchor: Shape
    members: list[Shape] = field(default_factory=list)
    cx: float = 0.5
    cy: float = 0.5
    angle: float = 0.0

    def copy(self) -> ShapeGroup:
        return ShapeGroup(
            anchor=self.anchor.copy(),
            members=[m.copy() for m in self.members],
            cx=self.cx, cy=self.cy, angle=self.angle,
        )

    def all_shapes(self) -> list[Shape]:
        return [self.anchor] + self.members

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor.to_dict(),
            "members": [m.to_dict() for m in self.members],
            "cx": self.cx, "cy": self.cy, "angle": self.angle,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShapeGroup:
        return cls(
            anchor=Shape.from_dict(d["anchor"]),
            members=[Shape.from_dict(m) for m in d.get("members", [])],
            cx=d.get("cx", 0.5), cy=d.get("cy", 0.5),
            angle=d.get("angle", 0.0),
        )


# ------------------------------------------------------------------
# ShapeGenome
# ------------------------------------------------------------------

@dataclass
class ShapeGenome:
    groups: list[ShapeGroup] = field(default_factory=list)
    satellites: list[Shape] = field(default_factory=list)
    bg_color_idx: int = 0
    fitness: float = 0.0

    def flatten(self) -> list[Shape]:
        """Return all shapes in rendering order (back-to-front)."""
        result: list[Shape] = []
        for g in self.groups:
            result.append(g.anchor)
            result.extend(g.members)
        result.extend(self.satellites)
        return result

    def total_shapes(self) -> int:
        n = len(self.satellites)
        for g in self.groups:
            n += 1 + len(g.members)
        return n

    def copy(self) -> ShapeGenome:
        return ShapeGenome(
            groups=[g.copy() for g in self.groups],
            satellites=[s.copy() for s in self.satellites],
            bg_color_idx=self.bg_color_idx,
            fitness=0.0,
        )

    def to_dict(self) -> dict:
        return {
            "bg_color_idx": self.bg_color_idx,
            "groups": [g.to_dict() for g in self.groups],
            "satellites": [s.to_dict() for s in self.satellites],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShapeGenome:
        if "groups" in d:
            groups = [ShapeGroup.from_dict(g) for g in d["groups"]]
            sats = [Shape.from_dict(s) for s in d.get("satellites", [])]
            return cls(groups=groups, satellites=sats,
                       bg_color_idx=d.get("bg_color_idx", 0))
        # Backward compat: old flat format with "shapes" key
        shapes = [Shape.from_dict(s) for s in d.get("shapes", [])]
        if not shapes:
            return cls(bg_color_idx=d.get("bg_color_idx", 0))
        anchor = shapes[0]
        members = shapes[1:] if len(shapes) > 1 else []
        group = ShapeGroup(anchor=anchor, members=members,
                           cx=anchor.x, cy=anchor.y, angle=anchor.rotation)
        return cls(groups=[group], bg_color_idx=d.get("bg_color_idx", 0))


# ======================================================================
# Shape construction helpers
# ======================================================================

def _random_shape(num_palette_colors: int,
                  kind_pool: tuple = _ALL_KINDS_WEIGHTED) -> Shape:
    kind = random.choice(kind_pool)
    w = random.uniform(MIN_SIZE, MAX_SIZE)
    h = random.uniform(MIN_SIZE, MAX_SIZE)
    if kind in ("line", "rect") and random.random() < 0.55:
        ratio = random.uniform(2.5, 7.0)
        h = min(h, w / ratio)
    s = Shape(
        kind=kind,
        x=random.random(), y=random.random(),
        w=w, h=h,
        rotation=random.uniform(0, 360),
        color_idx=random.randint(0, num_palette_colors - 1),
    )
    _ensure_inside(s)
    return s


def _make_anchor(cx: float, cy: float, angle: float,
                 sub_indices: list[int]) -> Shape:
    """Large dominant shape at a group's center."""
    kind = random.choice(_ANCHOR_KINDS)
    w = random.uniform(0.18, MAX_SIZE)
    h = random.uniform(0.18, MAX_SIZE)
    s = Shape(
        kind=kind, x=cx, y=cy, w=w, h=h,
        rotation=angle % 360,
        color_idx=random.choice(sub_indices),
    )
    _ensure_inside(s)
    return s


def _make_member(cx: float, cy: float, angle: float,
                 spread: float, sub_indices: list[int],
                 kind_pool: tuple = _ALL_KINDS_WEIGHTED) -> Shape:
    """Supporting shape placed near a group center with aligned direction."""
    ox = random.gauss(0, spread)
    oy = random.gauss(0, spread)
    x = _clamp01(cx + ox)
    y = _clamp01(cy + oy)

    kind = random.choice(kind_pool)
    w = random.uniform(MIN_SIZE, MAX_SIZE * 0.7)
    h = random.uniform(MIN_SIZE, MAX_SIZE * 0.7)
    if kind in ("line", "rect") and random.random() < 0.6:
        ratio = random.uniform(2.5, 8.0)
        h = min(h, w / ratio)

    s = Shape(
        kind=kind, x=x, y=y, w=w, h=h,
        rotation=(angle + random.gauss(0, 12)) % 360,
        color_idx=random.choice(sub_indices),
    )
    _ensure_inside(s)
    return s


def _make_satellite(groups: list[ShapeGroup],
                    num_palette_colors: int) -> Shape:
    """Small accent mark placed to balance the composition."""
    if groups:
        gcxs = [g.cx for g in groups]
        gcys = [g.cy for g in groups]
        mass_cx = sum(gcxs) / len(gcxs)
        mass_cy = sum(gcys) / len(gcys)
    else:
        mass_cx, mass_cy = 0.5, 0.5

    if random.random() < 0.4:
        # Counterweight: opposite side from mass center
        x = _clamp01(1.0 - mass_cx + random.gauss(0, 0.15))
        y = _clamp01(1.0 - mass_cy + random.gauss(0, 0.15))
    else:
        theta = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.15, 0.45)
        x = _clamp01(mass_cx + r * math.cos(theta))
        y = _clamp01(mass_cy + r * math.sin(theta))

    kind = random.choice(_ALL_KINDS_WEIGHTED)
    w = random.uniform(MIN_SIZE, 0.12)
    h = random.uniform(MIN_SIZE, 0.12)
    if kind in ("line", "rect") and random.random() < 0.6:
        ratio = random.uniform(3.0, 8.0)
        h = min(h, w / ratio)

    # Echo a color from an existing group if possible
    if groups and random.random() < 0.6:
        g = random.choice(groups)
        all_s = g.all_shapes()
        color_idx = random.choice(all_s).color_idx
    else:
        color_idx = random.randint(0, num_palette_colors - 1)

    dom_angle = groups[0].angle if groups else random.uniform(0, 360)
    s = Shape(
        kind=kind, x=x, y=y, w=w, h=h,
        rotation=(dom_angle + random.gauss(0, 20)) % 360,
        color_idx=color_idx,
    )
    _ensure_inside(s)
    return s


# ======================================================================
# Group placement strategies
# ======================================================================

def _place_groups_balanced(n_groups: int) -> list[tuple[float, float, float]]:
    """Return (cx, cy, angle) for each group using dynamic balance."""
    if n_groups == 1:
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.25, 0.75)
        return [(cx, cy, random.uniform(-45, 45))]

    results = []
    main_angle = random.uniform(-45, 45)
    first_cx = random.uniform(0.25, 0.6)
    first_cy = random.uniform(0.25, 0.6)
    results.append((first_cx, first_cy, main_angle))

    for i in range(1, n_groups):
        if random.random() < 0.5:
            # Counterbalance: opposite-ish from the first
            cx = _clamp01(1.0 - first_cx + random.gauss(0, 0.12))
            cy = _clamp01(1.0 - first_cy + random.gauss(0, 0.12))
        else:
            # Diagonal offset
            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.2, 0.45)
            cx = _clamp01(first_cx + r * math.cos(theta))
            cy = _clamp01(first_cy + r * math.sin(theta))
        ang = main_angle + random.gauss(0, 25)
        results.append((cx, cy, ang))

    return results


def _place_groups_vertical_zones(n_groups: int) -> list[tuple[float, float, float]]:
    """Groups arranged in vertical zones (like Malevich's multi-zone paintings)."""
    results = []
    for i in range(n_groups):
        cy = (i + 0.5) / n_groups * 0.8 + 0.1
        cx = random.uniform(0.2, 0.8)
        ang = random.uniform(-30, 30)
        results.append((cx, cy, ang))
    return results


def _place_groups_diagonal(n_groups: int) -> list[tuple[float, float, float]]:
    """Groups along a diagonal axis."""
    main_angle = random.uniform(0.3, 2.5)
    cx0 = random.uniform(0.2, 0.5)
    cy0 = random.uniform(0.2, 0.5)
    step = random.uniform(0.15, 0.3)

    results = []
    for i in range(n_groups):
        cx = _clamp01(cx0 + i * step * math.cos(main_angle))
        cy = _clamp01(cy0 + i * step * math.sin(main_angle))
        ang = math.degrees(main_angle) + random.gauss(0, 15)
        results.append((cx, cy, ang))
    return results


_GROUP_PLACERS = [
    _place_groups_balanced,
    _place_groups_vertical_zones,
    _place_groups_diagonal,
]


# ======================================================================
# Random initialization
# ======================================================================

def create_random(num_palette_colors: int = 16) -> ShapeGenome:
    """Create a genome via multi-stage hierarchical generation.

    Stage 1: decide number of groups, place them on the canvas.
    Stage 2: populate each group with an anchor + members.
    Stage 3: add satellite accent marks for rhythm and balance.
    """
    n_groups = random.choices([1, 2, 3, 4], weights=[25, 40, 25, 10])[0]

    palette_size = min(num_palette_colors, random.randint(3, 6))
    sub_indices = random.sample(range(num_palette_colors), palette_size)

    placer = random.choice(_GROUP_PLACERS)
    placements = placer(n_groups)

    groups: list[ShapeGroup] = []
    for cx, cy, angle in placements:
        anchor = _make_anchor(cx, cy, angle, sub_indices)
        n_members = random.randint(1, 5)

        # Choose member kind pool per group
        if random.random() < 0.3:
            kind_pool = _BAR_KINDS
        else:
            kind_pool = _ALL_KINDS_WEIGHTED

        spread = random.uniform(0.04, 0.12)
        members = [
            _make_member(cx, cy, angle, spread, sub_indices, kind_pool)
            for _ in range(n_members)
        ]
        groups.append(ShapeGroup(
            anchor=anchor, members=members,
            cx=cx, cy=cy, angle=angle,
        ))

    n_satellites = random.randint(0, 6)
    satellites = [_make_satellite(groups, num_palette_colors)
                  for _ in range(n_satellites)]

    if random.random() < 0.12:
        bg = random.randint(0, num_palette_colors - 1)
    else:
        bg = random.choice([0, 1]) if num_palette_colors > 1 else 0

    return ShapeGenome(groups=groups, satellites=satellites, bg_color_idx=bg)


# ======================================================================
# Mutation
# ======================================================================

MUTATION_RATES = {
    # Macro (group-level) — kept low to preserve composition
    "add_group": 0.04,
    "remove_group": 0.03,
    "shift_group": 0.10,
    "rotate_group": 0.08,
    "resize_group": 0.08,
    # Meso (within-group)
    "add_member": 0.10,
    "remove_member": 0.06,
    "duplicate_echo": 0.06,
    # Micro (individual shape)
    "perturb_position": 0.30,
    "perturb_size": 0.20,
    "perturb_rotation": 0.18,
    "change_color": 0.12,
    "change_kind": 0.06,
    "swap_order": 0.08,
    # Satellites
    "add_satellite": 0.08,
    "remove_satellite": 0.05,
}


def _pick_random_shape(genome: ShapeGenome) -> Shape | None:
    """Pick a uniformly random shape from anywhere in the genome."""
    all_shapes = genome.flatten()
    return random.choice(all_shapes) if all_shapes else None


def mutate(genome: ShapeGenome, num_palette_colors: int = 16,
           strength: float = 1.0) -> None:
    """Apply hierarchical mutation operators in-place.

    strength scales all mutation probabilities (1.0 = default, 0 = none, 2 = aggressive).
    """
    def _rate(key: str) -> float:
        return min(MUTATION_RATES[key] * strength, 1.0)

    groups = genome.groups
    sats = genome.satellites

    # ------ Macro: group-level ------

    if random.random() < _rate("add_group") and len(groups) < MAX_GROUPS:
        cx = random.random()
        cy = random.random()
        angle = random.uniform(-45, 45)
        sub = [random.randint(0, num_palette_colors - 1)
               for _ in range(random.randint(2, 4))]
        anchor = _make_anchor(cx, cy, angle, sub)
        n_mem = random.randint(1, 3)
        members = [_make_member(cx, cy, angle, 0.08, sub)
                   for _ in range(n_mem)]
        groups.append(ShapeGroup(anchor=anchor, members=members,
                                 cx=cx, cy=cy, angle=angle))

    if random.random() < _rate("remove_group") and len(groups) > 1:
        groups.pop(random.randrange(len(groups)))

    if groups and random.random() < _rate("shift_group"):
        g = random.choice(groups)
        dx = random.gauss(0, 0.035)
        dy = random.gauss(0, 0.035)
        g.cx = _clamp01(g.cx + dx)
        g.cy = _clamp01(g.cy + dy)
        g.anchor.x = _clamp01(g.anchor.x + dx)
        g.anchor.y = _clamp01(g.anchor.y + dy)
        for m in g.members:
            m.x = _clamp01(m.x + dx)
            m.y = _clamp01(m.y + dy)

    if groups and random.random() < _rate("rotate_group"):
        g = random.choice(groups)
        delta = random.gauss(0, 8)
        g.angle = (g.angle + delta) % 360
        g.anchor.rotation = (g.anchor.rotation + delta) % 360
        for m in g.members:
            m.rotation = (m.rotation + delta) % 360

    if groups and random.random() < _rate("resize_group"):
        g = random.choice(groups)
        factor = 1.0 + random.gauss(0, 0.10)
        factor = _clamp(factor, 0.7, 1.4)
        for s in g.all_shapes():
            s.w = _clamp(s.w * factor, MIN_SIZE, MAX_SIZE)
            s.h = _clamp(s.h * factor, MIN_SIZE, MAX_SIZE)

    # ------ Meso: within-group ------

    if groups and random.random() < _rate("add_member"):
        g = random.choice(groups)
        if len(g.members) < MAX_MEMBERS_PER_GROUP:
            sub = [g.anchor.color_idx] + [m.color_idx for m in g.members]
            new = _make_member(g.cx, g.cy, g.angle, 0.08, sub)
            g.members.append(new)

    if groups and random.random() < _rate("remove_member"):
        g = random.choice(groups)
        if g.members:
            g.members.pop(random.randrange(len(g.members)))

    if groups and random.random() < _rate("duplicate_echo"):
        g = random.choice(groups)
        src = random.choice(g.all_shapes())
        sat = src.copy()
        sat.w = _clamp(sat.w * random.uniform(0.2, 0.5), MIN_SIZE, 0.12)
        sat.h = _clamp(sat.h * random.uniform(0.2, 0.5), MIN_SIZE, 0.12)
        theta = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.15, 0.4)
        sat.x = _clamp01(g.cx + r * math.cos(theta))
        sat.y = _clamp01(g.cy + r * math.sin(theta))
        if len(sats) < MAX_SATELLITES:
            sats.append(sat)

    # ------ Micro: individual shape properties ------

    s = _pick_random_shape(genome)
    if s is None:
        return

    if random.random() < _rate("perturb_position"):
        s.x = _clamp01(s.x + random.gauss(0, 0.04))
        s.y = _clamp01(s.y + random.gauss(0, 0.04))

    if random.random() < _rate("perturb_size"):
        s.w = _clamp(s.w + random.gauss(0, 0.03), MIN_SIZE, MAX_SIZE)
        s.h = _clamp(s.h + random.gauss(0, 0.03), MIN_SIZE, MAX_SIZE)

    if random.random() < _rate("perturb_rotation"):
        s.rotation = (s.rotation + random.gauss(0, 10)) % 360

    if random.random() < _rate("change_color"):
        s.color_idx = random.randint(0, num_palette_colors - 1)

    if random.random() < _rate("change_kind"):
        s.kind = random.choice(SHAPE_KINDS)

    if groups and random.random() < _rate("swap_order") and len(groups) >= 2:
        i, j = random.sample(range(len(groups)), 2)
        groups[i], groups[j] = groups[j], groups[i]

    # ------ Satellites ------

    if random.random() < _rate("add_satellite"):
        if len(sats) < MAX_SATELLITES:
            sats.append(_make_satellite(groups, num_palette_colors))

    if random.random() < _rate("remove_satellite") and sats:
        sats.pop(random.randrange(len(sats)))

    for shape in genome.flatten():
        _ensure_inside(shape)


# ======================================================================
# Crossover
# ======================================================================

def crossover(parent_a: ShapeGenome, parent_b: ShapeGenome) -> ShapeGenome:
    """Group-level crossover: swap whole compositional clusters."""
    max_groups = max(len(parent_a.groups), len(parent_b.groups))
    child_groups: list[ShapeGroup] = []

    for i in range(max_groups):
        have_a = i < len(parent_a.groups)
        have_b = i < len(parent_b.groups)

        if have_a and have_b:
            donor = parent_a if random.random() < 0.5 else parent_b
        elif have_a:
            if random.random() < 0.5:
                donor = parent_a
            else:
                continue
        else:
            if random.random() < 0.5:
                donor = parent_b
            else:
                continue
        child_groups.append(donor.groups[i].copy())

    if not child_groups:
        donor = parent_a if random.random() < 0.5 else parent_b
        child_groups.append(donor.groups[0].copy())

    # Mix satellites from both parents
    child_sats: list[Shape] = []
    max_sats = max(len(parent_a.satellites), len(parent_b.satellites))
    for i in range(max_sats):
        have_a = i < len(parent_a.satellites)
        have_b = i < len(parent_b.satellites)
        if have_a and have_b:
            donor = parent_a if random.random() < 0.5 else parent_b
            child_sats.append(donor.satellites[i].copy())
        elif have_a and random.random() < 0.5:
            child_sats.append(parent_a.satellites[i].copy())
        elif have_b and random.random() < 0.5:
            child_sats.append(parent_b.satellites[i].copy())

    bg = parent_a.bg_color_idx if random.random() < 0.5 else parent_b.bg_color_idx
    child = ShapeGenome(groups=child_groups, satellites=child_sats, bg_color_idx=bg)
    for shape in child.flatten():
        _ensure_inside(shape)
    return child

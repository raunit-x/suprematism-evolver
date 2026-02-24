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
MAX_GROUPS = 10
MAX_SATELLITES = 8
MAX_MEMBERS_PER_GROUP = 12

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
    depth: float = 0.0       # 3D cuboid depth (0 = flat 2D shape)
    elevation: float = 0.0   # Height in 3D stack

    def copy(self) -> Shape:
        return Shape(
            self.kind, self.x, self.y, self.w, self.h,
            self.rotation, self.color_idx, self.opacity,
            self.depth, self.elevation,
        )

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind, "x": self.x, "y": self.y,
            "w": self.w, "h": self.h, "rotation": self.rotation,
            "color_idx": self.color_idx, "opacity": self.opacity,
        }
        if self.depth > 0:
            d["depth"] = self.depth
        if self.elevation > 0:
            d["elevation"] = self.elevation
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Shape:
        return cls(
            kind=d["kind"], x=d["x"], y=d["y"],
            w=d["w"], h=d["h"], rotation=d["rotation"],
            color_idx=d["color_idx"], opacity=d.get("opacity", 1.0),
            depth=d.get("depth", 0.0), elevation=d.get("elevation", 0.0),
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
    bg_style: int = 0
    bg_color_idx: int = 0
    bg_angle: float = 0.0
    bg_cx: float = 0.33
    bg_cy: float = 0.55
    bg_sec_idx: int = 1
    bg_blend: float = 0.0
    fitness: float = 0.0
    projection_angle: float = 30.0     # Oblique projection slant angle (degrees)
    projection_strength: float = 0.0   # Depth projection scale (0 = flat 2D)
    projection_secondary_angle: float = -35.0
    projection_secondary_strength: float = 0.0
    projection_vanishing_x: float = 0.5
    projection_vanishing_y: float = 0.5
    camera_mode: int = 0               # 0=oblique, 1=bifocal, 2=fan-perspective
    perspective_bias: float = 0.0      # mode-dependent camera skew

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
            bg_style=self.bg_style,
            bg_color_idx=self.bg_color_idx,
            bg_angle=self.bg_angle,
            bg_cx=self.bg_cx,
            bg_cy=self.bg_cy,
            bg_sec_idx=self.bg_sec_idx,
            bg_blend=self.bg_blend,
            fitness=0.0,
            projection_angle=self.projection_angle,
            projection_strength=self.projection_strength,
            projection_secondary_angle=self.projection_secondary_angle,
            projection_secondary_strength=self.projection_secondary_strength,
            projection_vanishing_x=self.projection_vanishing_x,
            projection_vanishing_y=self.projection_vanishing_y,
            camera_mode=self.camera_mode,
            perspective_bias=self.perspective_bias,
        )

    def to_dict(self) -> dict:
        d = {
            "bg_style": self.bg_style,
            "bg_color_idx": self.bg_color_idx,
            "bg_angle": self.bg_angle,
            "bg_cx": self.bg_cx,
            "bg_cy": self.bg_cy,
            "bg_sec_idx": self.bg_sec_idx,
            "bg_blend": self.bg_blend,
            "groups": [g.to_dict() for g in self.groups],
            "satellites": [s.to_dict() for s in self.satellites],
        }
        if self.projection_strength > 0:
            d["projection_angle"] = self.projection_angle
            d["projection_strength"] = self.projection_strength
            d["projection_secondary_angle"] = self.projection_secondary_angle
            d["projection_secondary_strength"] = self.projection_secondary_strength
            d["projection_vanishing_x"] = self.projection_vanishing_x
            d["projection_vanishing_y"] = self.projection_vanishing_y
            d["camera_mode"] = self.camera_mode
            d["perspective_bias"] = self.perspective_bias
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ShapeGenome:
        if "groups" in d:
            groups = [ShapeGroup.from_dict(g) for g in d["groups"]]
            sats = [Shape.from_dict(s) for s in d.get("satellites", [])]
            return cls(groups=groups, satellites=sats,
                       bg_style=d.get("bg_style", 0),
                       bg_color_idx=d.get("bg_color_idx", 0),
                       bg_angle=d.get("bg_angle", 0.0),
                       bg_cx=d.get("bg_cx", 0.33),
                       bg_cy=d.get("bg_cy", 0.55),
                       bg_sec_idx=d.get("bg_sec_idx", 1),
                       bg_blend=d.get("bg_blend", 0.0),
                       projection_angle=d.get("projection_angle", 30.0),
                       projection_strength=d.get("projection_strength", 0.0),
                       projection_secondary_angle=d.get("projection_secondary_angle", -35.0),
                       projection_secondary_strength=d.get("projection_secondary_strength", 0.0),
                       projection_vanishing_x=d.get("projection_vanishing_x", 0.5),
                       projection_vanishing_y=d.get("projection_vanishing_y", 0.5),
                       camera_mode=d.get("camera_mode", 0),
                       perspective_bias=d.get("perspective_bias", 0.0))
        # Backward compat: old flat format with "shapes" key
        shapes = [Shape.from_dict(s) for s in d.get("shapes", [])]
        if not shapes:
            return cls(bg_color_idx=d.get("bg_color_idx", 0), bg_style=0)
        anchor = shapes[0]
        members = shapes[1:] if len(shapes) > 1 else []
        group = ShapeGroup(anchor=anchor, members=members,
                           cx=anchor.x, cy=anchor.y, angle=anchor.rotation)
        return cls(groups=[group], bg_color_idx=d.get("bg_color_idx", 0), bg_style=0)


# ======================================================================
# 3D bounds helpers
# ======================================================================

def _max_projection_components(genome: ShapeGenome) -> tuple[float, float]:
    """Return conservative absolute projection components in canvas units."""
    s1 = max(0.0, genome.projection_strength)
    if s1 <= 1e-6:
        return 0.0, 0.0

    mode = int(genome.camera_mode) % 3
    if mode == 0:
        ang = math.radians(genome.projection_angle)
        return abs(math.cos(ang) * s1), abs(math.sin(ang) * s1)

    if mode == 1:
        s2 = max(0.0, genome.projection_secondary_strength)
        a1 = math.radians(genome.projection_angle)
        a2 = math.radians(genome.projection_secondary_angle)
        x_comp = max(abs(math.cos(a1) * s1), abs(math.cos(a2) * s2))
        y_comp = max(abs(math.sin(a1) * s1), abs(math.sin(a2) * s2))
        return x_comp, y_comp

    # Fan perspective can point in any direction; use conservative isotropic bound.
    mag = s1 * (1.55 + 0.80 * abs(genome.perspective_bias))
    return mag, mag


def _ensure_inside_3d(shape: Shape, genome: ShapeGenome) -> None:
    """Keep full projected 3D volume inside frame, not only front face."""
    _ensure_inside(shape)
    if genome.projection_strength <= 0.01:
        return

    rad = math.radians(shape.rotation)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    bb_w = shape.w * cos_a + shape.h * sin_a
    bb_h = shape.w * sin_a + shape.h * cos_a

    px, py = _max_projection_components(genome)
    extrude = max(0.0, shape.depth) + max(0.0, shape.elevation)
    safety = 1.25 + 0.25 * abs(genome.perspective_bias)
    ext_x = extrude * px * safety + 0.01
    ext_y = extrude * py * safety + 0.01
    margin = 0.015

    max_w = max(0.04, 1.0 - 2.0 * (margin + ext_x))
    max_h = max(0.04, 1.0 - 2.0 * (margin + ext_y))
    if bb_w > max_w or bb_h > max_h:
        scale = min(max_w / max(bb_w, 1e-9), max_h / max(bb_h, 1e-9))
        shape.w *= scale
        shape.h *= scale
        bb_w *= scale
        bb_h *= scale

    lo_x = bb_w / 2.0 + margin + ext_x
    hi_x = 1.0 - bb_w / 2.0 - margin - ext_x
    lo_y = bb_h / 2.0 + margin + ext_y
    hi_y = 1.0 - bb_h / 2.0 - margin - ext_y

    shape.x = 0.5 if lo_x > hi_x else _clamp(shape.x, lo_x, hi_x)
    shape.y = 0.5 if lo_y > hi_y else _clamp(shape.y, lo_y, hi_y)


def _enforce_3d_bounds(genome: ShapeGenome) -> None:
    for s in genome.flatten():
        _ensure_inside_3d(s, genome)


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

    bg = random.randint(0, num_palette_colors - 1) if random.random() < 0.12 else (random.choice([0, 1]) if num_palette_colors > 1 else 0)
    bg_sec_idx = random.randint(0, num_palette_colors - 1)
    if bg_sec_idx == bg:
        bg_sec_idx = (bg + 1) % num_palette_colors
    bg_angle = random.uniform(0, 360)
    bg_cx = random.uniform(0.1, 0.9)
    bg_cy = random.uniform(0.1, 0.9)

    return ShapeGenome(
        groups=groups, satellites=satellites, bg_color_idx=bg,
        bg_angle=bg_angle, bg_cx=bg_cx, bg_cy=bg_cy, bg_sec_idx=bg_sec_idx
    )


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
    "perturb_opacity": 0.10,
    "swap_order": 0.08,
    # Satellites
    "add_satellite": 0.08,
    "remove_satellite": 0.05,
    # Background
    "perturb_bg": 0.15,
    # 3D depth / projection (Hadid mode)
    "perturb_depth": 0.15,
    "perturb_elevation": 0.10,
    "perturb_projection": 0.05,
    "perturb_camera": 0.07,
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

    # ------ Background ------
    if random.random() < _rate("perturb_bg"):
        if random.random() < 0.15:
            genome.bg_style = random.choice([0, 1, 2, 3, 4])
        genome.bg_angle = (genome.bg_angle + random.gauss(0, 15)) % 360
        genome.bg_cx = _clamp01(genome.bg_cx + random.gauss(0, 0.05))
        genome.bg_cy = _clamp01(genome.bg_cy + random.gauss(0, 0.05))
        genome.bg_blend = _clamp(genome.bg_blend + random.gauss(0, 0.08), 0.0, 0.85)
        if random.random() < 0.2:
            genome.bg_sec_idx = random.randint(0, num_palette_colors - 1)

    # ------ 3D depth / projection (active when projection_strength > 0) ------

    if genome.projection_strength > 0.01:
        if s and random.random() < _rate("perturb_depth"):
            s.depth = max(0.0, s.depth + random.gauss(0, 0.015))

        if s and random.random() < _rate("perturb_elevation"):
            s.elevation = max(0.0, s.elevation + random.gauss(0, 0.012))

        if random.random() < _rate("perturb_projection"):
            genome.projection_angle = _clamp(
                genome.projection_angle + random.gauss(0, 3), 10, 70,
            )
            if random.random() < 0.3:
                genome.projection_strength = _clamp(
                    genome.projection_strength + random.gauss(0, 0.04),
                    0.1, 0.8,
                )
            if random.random() < 0.25:
                genome.projection_secondary_angle = _clamp(
                    genome.projection_secondary_angle + random.gauss(0, 8),
                    -85, 85,
                )
            if random.random() < 0.25:
                genome.projection_secondary_strength = _clamp(
                    genome.projection_secondary_strength + random.gauss(0, 0.05),
                    0.0, 0.8,
                )

        if random.random() < _rate("perturb_camera"):
            if random.random() < 0.3:
                genome.camera_mode = random.choice([0, 1, 2])
            genome.projection_vanishing_x = _clamp01(
                genome.projection_vanishing_x + random.gauss(0, 0.08),
            )
            genome.projection_vanishing_y = _clamp01(
                genome.projection_vanishing_y + random.gauss(0, 0.08),
            )
            genome.perspective_bias = _clamp(
                genome.perspective_bias + random.gauss(0, 0.10),
                -1.0, 1.0,
            )
        if s and random.random() < _rate("perturb_opacity"):
            if random.random() < 0.35:
                # Flip material family between solid and glass-like.
                if s.opacity >= 0.82:
                    # Solid -> glass is now intentionally rarer.
                    if random.random() < 0.10:
                        s.opacity = random.uniform(0.42, 0.74)
                    else:
                        s.opacity = random.uniform(0.90, 1.0)
                else:
                    # Glass tends to anneal into solid blocks over time.
                    if random.random() < 0.84:
                        s.opacity = random.uniform(0.90, 1.0)
                    else:
                        s.opacity = random.uniform(0.42, 0.74)
            else:
                # Gentle drift toward solidity.
                s.opacity = _clamp(s.opacity + random.gauss(0.08, 0.06), 0.30, 1.0)

    if genome.projection_strength > 0.01:
        _enforce_3d_bounds(genome)
    else:
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

    bg_style = parent_a.bg_style if random.random() < 0.5 else parent_b.bg_style
    bg = parent_a.bg_color_idx if random.random() < 0.5 else parent_b.bg_color_idx
    bg_angle = parent_a.bg_angle if random.random() < 0.5 else parent_b.bg_angle
    bg_cx = parent_a.bg_cx if random.random() < 0.5 else parent_b.bg_cx
    bg_cy = parent_a.bg_cy if random.random() < 0.5 else parent_b.bg_cy
    bg_sec_idx = parent_a.bg_sec_idx if random.random() < 0.5 else parent_b.bg_sec_idx
    bg_blend = parent_a.bg_blend if random.random() < 0.5 else parent_b.bg_blend
    proj_angle = parent_a.projection_angle if random.random() < 0.5 else parent_b.projection_angle
    proj_strength = parent_a.projection_strength if random.random() < 0.5 else parent_b.projection_strength
    proj_angle_b = (
        parent_a.projection_secondary_angle
        if random.random() < 0.5
        else parent_b.projection_secondary_angle
    )
    proj_strength_b = (
        parent_a.projection_secondary_strength
        if random.random() < 0.5
        else parent_b.projection_secondary_strength
    )
    vanish_x = (
        parent_a.projection_vanishing_x
        if random.random() < 0.5
        else parent_b.projection_vanishing_x
    )
    vanish_y = (
        parent_a.projection_vanishing_y
        if random.random() < 0.5
        else parent_b.projection_vanishing_y
    )
    camera_mode = parent_a.camera_mode if random.random() < 0.5 else parent_b.camera_mode
    perspective_bias = (
        parent_a.perspective_bias if random.random() < 0.5 else parent_b.perspective_bias
    )
    child = ShapeGenome(
        groups=child_groups, satellites=child_sats, 
        bg_style=bg_style, bg_color_idx=bg,
        bg_angle=bg_angle, bg_cx=bg_cx, bg_cy=bg_cy, bg_sec_idx=bg_sec_idx,
        bg_blend=bg_blend,
        projection_angle=proj_angle, projection_strength=proj_strength,
        projection_secondary_angle=proj_angle_b,
        projection_secondary_strength=proj_strength_b,
        projection_vanishing_x=vanish_x,
        projection_vanishing_y=vanish_y,
        camera_mode=camera_mode,
        perspective_bias=perspective_bias,
    )
    if child.projection_strength > 0.01:
        _enforce_3d_bounds(child)
    else:
        for shape in child.flatten():
            _ensure_inside(shape)
    return child


# ======================================================================
# Architecton creation (Hadid / Malevich 3D mode)
# ======================================================================

def create_architecton(num_palette_colors: int = 10) -> ShapeGenome:
    """Create a Hadid-style architecton with diverse structural archetypes.

    Scene archetypes intentionally vary the compositional logic so initialization
    doesn't collapse into one repeating template.
    """
    groups: list[ShapeGroup] = []
    satellites: list[Shape] = []

    palette_size = min(num_palette_colors, random.randint(5, 8))
    sub_indices = random.sample(range(num_palette_colors), palette_size)
    scene_style = random.choices(
        ["spine", "radial", "cantilever", "towerfield"],
        weights=[35, 20, 25, 20],
    )[0]

    camera_mode = random.choices([0, 1, 2], weights=[45, 35, 20])[0]
    proj_angle = random.choice([18, 24, 30, 36, 44, 54, 62])
    proj_strength = random.uniform(0.24, 0.72)
    proj2_angle = proj_angle + random.choice([-82, -68, -55, 55, 68, 82])
    proj2_strength = random.uniform(0.12, 0.64)
    vanish_x = random.uniform(0.15, 0.85)
    vanish_y = random.uniform(0.12, 0.88)
    perspective_bias = random.uniform(-0.65, 0.65)
    glass_ratio = random.uniform(0.01, 0.05)

    def _choose_opacity(kind: str, w: float, h: float, depth: float,
                        elevation: float, material: str | None = None) -> float:
        if material == "solid":
            return random.uniform(0.90, 1.0)
        if material == "glass":
            return random.uniform(0.40, 0.74)

        p_glass = glass_ratio
        if kind in ("line", "triangle"):
            p_glass *= 0.45
        if max(w, h) > 0.20:
            p_glass *= 0.28
        if depth > 0.045 and elevation < 0.25:
            p_glass *= 0.28

        return random.uniform(0.42, 0.78) if random.random() < p_glass else random.uniform(0.88, 1.0)

    def _new_shape(kind: str, x: float, y: float, w: float, h: float, rotation: float,
                   depth: float, elevation: float, color_idx: int | None = None,
                   material: str | None = None) -> Shape:
        opacity = _choose_opacity(kind, w, h, depth, elevation, material=material)
        s = Shape(
            kind=kind,
            x=_clamp01(x),
            y=_clamp01(y),
            w=_clamp(w, MIN_SIZE, MAX_SIZE),
            h=_clamp(h, MIN_SIZE, MAX_SIZE),
            rotation=rotation % 360,
            color_idx=color_idx if color_idx is not None else random.choice(sub_indices),
            opacity=opacity,
            depth=max(0.0, depth),
            elevation=max(0.0, elevation),
        )
        _ensure_inside(s)
        return s

    def _append_group(shapes: list[Shape], cx: float, cy: float, angle: float) -> None:
        if not shapes or len(groups) >= MAX_GROUPS:
            return
        groups.append(
            ShapeGroup(anchor=shapes[0], members=shapes[1:], cx=_clamp01(cx), cy=_clamp01(cy), angle=angle),
        )

    # --- 1) Main deck / ground plane ---
    if scene_style == "cantilever":
        deck_angle = random.uniform(-30, -8)
        deck_cx = random.uniform(0.35, 0.62)
        deck_cy = random.uniform(0.56, 0.75)
    elif scene_style == "spine":
        deck_angle = random.uniform(-16, 16)
        deck_cx = random.uniform(0.30, 0.55)
        deck_cy = random.uniform(0.52, 0.72)
    elif scene_style == "radial":
        deck_angle = random.uniform(-45, 45)
        deck_cx = random.uniform(0.40, 0.60)
        deck_cy = random.uniform(0.45, 0.65)
    else:  # towerfield
        deck_angle = random.uniform(-8, 8)
        deck_cx = random.uniform(0.42, 0.58)
        deck_cy = random.uniform(0.60, 0.78)

    deck_w = random.uniform(0.45, 0.9)
    deck_h = random.uniform(0.09, 0.22)
    deck_depth = random.uniform(0.02, 0.06)
    deck_shapes = [
        _new_shape(
            kind=random.choice(["rect", "trapezoid"]),
            x=deck_cx, y=deck_cy,
            w=deck_w, h=deck_h,
            rotation=deck_angle,
            depth=deck_depth,
            elevation=0.0,
            material="solid",
        ),
    ]
    elev_seed = deck_depth
    for _ in range(random.randint(1, 3)):
        sc = random.uniform(0.58, 0.85)
        d = random.uniform(0.01, 0.035)
        deck_shapes.append(
            _new_shape(
                kind=random.choice(["rect", "trapezoid"]),
                x=deck_cx + random.gauss(0, 0.045),
                y=deck_cy + random.gauss(0, 0.03),
                w=deck_w * sc,
                h=deck_h * sc,
                rotation=deck_angle + random.gauss(0, 6),
                depth=d,
                elevation=elev_seed,
                material="solid",
            ),
        )
        elev_seed += d * random.uniform(0.8, 1.05)
    _append_group(deck_shapes, deck_cx, deck_cy, deck_angle)

    # --- 2) Cluster placements based on scene archetype ---
    cluster_centers: list[tuple[float, float]] = []
    cluster_angles: list[float] = []

    if scene_style == "spine":
        n_clusters = random.randint(4, 6)
        axis = random.uniform(-1.0, 1.2)
        start_x = random.uniform(0.12, 0.3)
        start_y = random.uniform(0.3, 0.64)
        step = random.uniform(0.12, 0.18)
        for i in range(n_clusters):
            cx = _clamp01(start_x + i * step * math.cos(axis) + random.gauss(0, 0.03))
            cy = _clamp01(start_y + i * step * math.sin(axis) + random.gauss(0, 0.03))
            cluster_centers.append((cx, cy))
            cluster_angles.append(math.degrees(axis) + random.gauss(0, 12))
    elif scene_style == "radial":
        n_clusters = random.randint(4, 7)
        hub_x = random.uniform(0.35, 0.65)
        hub_y = random.uniform(0.35, 0.65)
        spin = random.uniform(0, 2 * math.pi)
        for i in range(n_clusters):
            theta = spin + (2 * math.pi * i / n_clusters) + random.gauss(0, 0.24)
            radius = random.uniform(0.12, 0.34)
            cx = _clamp01(hub_x + radius * math.cos(theta))
            cy = _clamp01(hub_y + radius * math.sin(theta))
            cluster_centers.append((cx, cy))
            cluster_angles.append(math.degrees(theta) + random.choice([0, 90, -90]))
    elif scene_style == "cantilever":
        n_clusters = random.randint(3, 6)
        spine_y = random.uniform(0.48, 0.72)
        for i in range(n_clusters):
            t = i / max(1, n_clusters - 1)
            cx = _clamp01(0.14 + 0.70 * t + random.gauss(0, 0.03))
            cy = _clamp01(spine_y + random.gauss(0, 0.045))
            cluster_centers.append((cx, cy))
            cluster_angles.append(deck_angle + random.choice([0, 0, 90, -90, random.uniform(-25, 25)]))
    else:  # towerfield
        rows = random.randint(2, 3)
        cols = random.randint(2, 3)
        x0 = random.uniform(0.20, 0.35)
        y0 = random.uniform(0.24, 0.40)
        sx = random.uniform(0.18, 0.26)
        sy = random.uniform(0.14, 0.21)
        for r in range(rows):
            for c in range(cols):
                if len(cluster_centers) >= 6:
                    break
                cx = _clamp01(x0 + c * sx + random.gauss(0, 0.028))
                cy = _clamp01(y0 + r * sy + random.gauss(0, 0.028))
                cluster_centers.append((cx, cy))
                cluster_angles.append(random.choice([0, 90, -90]) + random.gauss(0, 8))

    # --- 3) Vertical / angular structural clusters ---
    for i, (cx, cy) in enumerate(cluster_centers):
        if len(groups) >= MAX_GROUPS:
            break
        angle = cluster_angles[i]
        n_levels = random.randint(2, 8)
        base_w = random.uniform(0.055, 0.22)
        level_h = random.uniform(0.018, 0.075)
        base_depth = random.uniform(0.03, 0.11)
        elev = elev_seed + random.uniform(0.0, 0.2)
        taper = random.uniform(0.06, 0.22)

        cluster_shapes: list[Shape] = []
        top_x = cx
        top_y = cy
        top_elev = elev
        for lvl in range(n_levels):
            sc = max(0.28, 1.0 - taper * lvl)
            wobble = 0.004 + 0.006 * lvl
            sw = base_w * sc * random.uniform(0.86, 1.18)
            sh = level_h * random.uniform(0.75, 1.4)
            sd = base_depth * random.uniform(0.65, 1.1)
            lx = cx + random.gauss(0, wobble)
            ly = cy + random.gauss(0, wobble * 0.8)
            kind = random.choices(["rect", "trapezoid", "triangle"], weights=[60, 30, 10])[0]
            cluster_shapes.append(
                _new_shape(kind, lx, ly, sw, sh, angle + random.gauss(0, 4), sd, elev),
            )
            top_x, top_y, top_elev = lx, ly, elev + sd
            elev += sd * random.uniform(0.82, 1.02)

            if lvl > 0 and random.random() < 0.36:
                arm_w = random.uniform(0.06, 0.24)
                arm_h = sh * random.uniform(0.4, 0.95)
                arm_shift = random.choice([-1, 1]) * arm_w * random.uniform(0.25, 0.6)
                cluster_shapes.append(
                    _new_shape(
                        random.choice(["rect", "line", "triangle"]),
                        lx + arm_shift * math.cos(math.radians(angle)),
                        ly + arm_shift * math.sin(math.radians(angle)),
                        arm_w,
                        arm_h,
                        angle + random.gauss(0, 7),
                        sd * random.uniform(0.35, 0.75),
                        elev - sd * random.uniform(0.25, 0.8),
                    ),
                )

        if random.random() < 0.62:
            facade_rows = random.randint(2, 5)
            facade_cols = random.randint(2, 6)
            for rr in range(facade_rows):
                for cc in range(facade_cols):
                    if random.random() < 0.22:
                        continue
                    win_x = cx + (cc - (facade_cols - 1) / 2.0) * base_w * random.uniform(0.12, 0.22)
                    win_y = cy + (rr - (facade_rows - 1) / 2.0) * level_h * random.uniform(0.6, 1.0)
                    cluster_shapes.append(
                        _new_shape(
                            kind="rect",
                            x=win_x,
                            y=win_y,
                            w=random.uniform(0.009, 0.022),
                            h=random.uniform(0.008, 0.022),
                            rotation=angle + random.gauss(0, 2),
                            depth=random.uniform(0.003, 0.012),
                            elevation=top_elev + random.uniform(0.0, 0.08),
                            material="glass" if random.random() < 0.20 else "solid",
                        ),
                    )

        _append_group(cluster_shapes, cx, cy, angle)

    # --- 4) Bridges / spanning bars between clusters ---
    if len(cluster_centers) >= 2 and len(groups) < MAX_GROUPS:
        indexed = list(enumerate(cluster_centers))
        indexed.sort(key=lambda it: it[1][0])
        pair_pool = [(indexed[i][0], indexed[i + 1][0]) for i in range(len(indexed) - 1)]
        random.shuffle(pair_pool)
        for a_idx, b_idx in pair_pool[: random.randint(1, min(3, len(pair_pool)))]:
            a = cluster_centers[a_idx]
            b = cluster_centers[b_idx]
            bx = (a[0] + b[0]) * 0.5
            by = (a[1] + b[1]) * 0.5
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            bridge_elev = elev_seed + random.uniform(0.05, 0.35)
            bridge_shapes = [
                _new_shape(
                    kind=random.choice(["rect", "line"]),
                    x=bx, y=by,
                    w=max(0.04, dist * random.uniform(0.9, 1.35)),
                    h=random.uniform(0.01, 0.04),
                    rotation=angle,
                    depth=random.uniform(0.015, 0.05),
                    elevation=bridge_elev,
                ),
            ]
            for _ in range(random.randint(0, 2)):
                bridge_shapes.append(
                    _new_shape(
                        kind="rect",
                        x=bx + random.gauss(0, max(0.02, dist * 0.15)),
                        y=by + random.gauss(0, 0.02),
                        w=random.uniform(0.016, 0.06),
                        h=random.uniform(0.01, 0.038),
                        rotation=angle + random.gauss(0, 5),
                        depth=random.uniform(0.008, 0.03),
                        elevation=bridge_elev + random.uniform(0.0, 0.08),
                    ),
                )
            _append_group(bridge_shapes, bx, by, angle)

    # --- 5) Sweeping blades / wedges for energetic perspective ---
    n_sweeps = random.randint(1, 2)
    for _ in range(n_sweeps):
        if len(groups) >= MAX_GROUPS:
            break
        sw_cx = random.uniform(0.18, 0.82)
        sw_cy = random.uniform(0.18, 0.82)
        sw_angle = random.uniform(-70, 70) if scene_style != "radial" else random.uniform(0, 360)
        sweep = _new_shape(
            kind=random.choice(["triangle", "trapezoid", "rect"]),
            x=sw_cx, y=sw_cy,
            w=random.uniform(0.22, 0.55),
            h=random.uniform(0.03, 0.12),
            rotation=sw_angle,
            depth=random.uniform(0.012, 0.05),
            elevation=random.uniform(0.0, 0.35),
        )
        _append_group([sweep], sw_cx, sw_cy, sw_angle)

    # --- 6) Floating satellites / accents ---
    n_sats = random.randint(3, 9)
    for _ in range(n_sats):
        sat = _new_shape(
            kind=random.choice(["rect", "square", "line", "triangle"]),
            x=random.random(),
            y=random.random(),
            w=random.uniform(0.012, 0.09),
            h=random.uniform(0.008, 0.05),
            rotation=random.uniform(0, 360),
            depth=random.uniform(0.006, 0.04),
            elevation=random.uniform(0.0, 0.55),
        )
        satellites.append(sat)

    bg_style = random.choices([0, 1, 2, 3, 4], weights=[30, 22, 10, 20, 18])[0]
    bg_idx = random.choice([4, 5, 6, 0]) if num_palette_colors >= 7 else random.randint(0, num_palette_colors - 1)
    bg_sec_idx = random.randint(0, num_palette_colors - 1)
    if bg_sec_idx == bg_idx:
        bg_sec_idx = (bg_idx + 1) % num_palette_colors

    genome = ShapeGenome(
        groups=groups,
        satellites=satellites,
        bg_style=bg_style,
        bg_color_idx=bg_idx,
        bg_angle=random.uniform(0, 360),
        bg_cx=random.uniform(0.12, 0.88),
        bg_cy=random.uniform(0.12, 0.88),
        bg_sec_idx=bg_sec_idx,
        bg_blend=random.uniform(0.05, 0.65) if random.random() < 0.7 else 0.0,
        projection_angle=proj_angle,
        projection_strength=proj_strength,
        projection_secondary_angle=proj2_angle,
        projection_secondary_strength=proj2_strength,
        projection_vanishing_x=vanish_x,
        projection_vanishing_y=vanish_y,
        camera_mode=camera_mode,
        perspective_bias=perspective_bias,
    )
    _enforce_3d_bounds(genome)
    return genome

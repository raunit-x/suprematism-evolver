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
    material: str = "concrete"  # Hadid material family: concrete or marble
    depth: float = 0.0       # 3D cuboid depth (0 = flat 2D shape)
    elevation: float = 0.0   # Height in 3D stack
    detail: str = "none"     # Architectural detail pattern on face

    def copy(self) -> Shape:
        return Shape(
            self.kind, self.x, self.y, self.w, self.h,
            self.rotation, self.color_idx, self.opacity, self.material,
            self.depth, self.elevation, self.detail,
        )

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind, "x": self.x, "y": self.y,
            "w": self.w, "h": self.h, "rotation": self.rotation,
            "color_idx": self.color_idx, "opacity": self.opacity,
            "material": self.material,
        }
        if self.depth > 0:
            d["depth"] = self.depth
        if self.elevation > 0:
            d["elevation"] = self.elevation
        if self.detail != "none":
            d["detail"] = self.detail
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Shape:
        material = d.get("material", "concrete")
        if material not in ("concrete", "marble"):
            material = "concrete"
        return cls(
            kind=d["kind"], x=d["x"], y=d["y"],
            w=d["w"], h=d["h"], rotation=d["rotation"],
            color_idx=d["color_idx"], opacity=d.get("opacity", 1.0),
            material=material,
            depth=d.get("depth", 0.0), elevation=d.get("elevation", 0.0),
            detail=d.get("detail", "none"),
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
    # Compositional armature / focal fields
    armature_angle: float = 35.0       # dominant compositional axis (degrees)
    focal_x: float = 0.38             # eye entry point x (off-center)
    focal_y: float = 0.40             # eye entry point y
    density_falloff: float = 0.5      # complexity attenuation from focal (0=uniform, 1=sharp)
    engine_tag: str = ""              # "hadid", "maps", or "" (shapes/cppn)

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
            armature_angle=self.armature_angle,
            focal_x=self.focal_x,
            focal_y=self.focal_y,
            density_falloff=self.density_falloff,
            engine_tag=self.engine_tag,
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
            "armature_angle": self.armature_angle,
            "focal_x": self.focal_x,
            "focal_y": self.focal_y,
            "density_falloff": self.density_falloff,
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
        if self.engine_tag:
            d["engine_tag"] = self.engine_tag
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
                       perspective_bias=d.get("perspective_bias", 0.0),
                       armature_angle=d.get("armature_angle", 35.0),
                       focal_x=d.get("focal_x", 0.38),
                       focal_y=d.get("focal_y", 0.40),
                       density_falloff=d.get("density_falloff", 0.5),
                       engine_tag=d.get("engine_tag", ""))
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


def _solidify_architecton(genome: ShapeGenome) -> None:
    """Force Hadid 3D genomes toward complete, chunky solids.

    Prevents wire-like artifacts by converting line primitives and enforcing
    minimum thickness/depth for all projected shapes.
    """
    if genome.projection_strength <= 0.01:
        return

    for s in genome.flatten():
        if s.kind == "line":
            s.kind = "rect"

        # Keep all materials fully opaque/solid in Hadid mode.
        s.opacity = _clamp(max(s.opacity, 0.96), 0.96, 1.0)

        # Enforce minimum visible face thickness.
        min_face = 0.022
        s.w = max(s.w, min_face)
        s.h = max(s.h, min_face)

        # Reduce extreme rod-like aspect ratios that read as wireframes.
        major = max(s.w, s.h)
        minor = min(s.w, s.h)
        if minor > 1e-6 and major / minor > 6.0:
            target_minor = major / 6.0
            if s.w < s.h:
                s.w = max(s.w, target_minor)
            else:
                s.h = max(s.h, target_minor)

        # Ensure extrusion is visible and reads as mass.
        s.depth = max(s.depth, 0.022)


_DETAIL_TYPES = (
    "none", "windows", "checker", "ladder",
    "lines_h", "lines_v", "dots", "grid_frame", "cross_brace",
)


def _assign_detail(shape: Shape) -> None:
    """Assign an architectural detail pattern based on face area."""
    face_area = shape.w * shape.h
    if face_area < 0.0008:
        shape.detail = "none"
        return
    p_detail = min(0.75, 0.25 + face_area * 10.0)
    if random.random() < p_detail:
        shape.detail = random.choices(
            ["windows", "checker", "ladder", "lines_h", "lines_v",
             "dots", "grid_frame", "cross_brace"],
            weights=[25, 12, 12, 10, 8, 8, 15, 10],
        )[0]


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
                 kind_pool: tuple = _ALL_KINDS_WEIGHTED,
                 armature_angle: float | None = None) -> Shape:
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

    if armature_angle is not None and random.random() < 0.45:
        rot = armature_angle + random.gauss(0, 15)
    else:
        rot = angle + random.gauss(0, 12)

    s = Shape(
        kind=kind, x=x, y=y, w=w, h=h,
        rotation=rot % 360,
        color_idx=random.choice(sub_indices),
    )
    _ensure_inside(s)
    return s


def _make_satellite(groups: list[ShapeGroup],
                    num_palette_colors: int,
                    focal_x: float = 0.5, focal_y: float = 0.5) -> Shape:
    """Small accent mark placed to balance the composition.

    Biases placement away from the quadrant diagonally opposite the focal point,
    creating intentional negative space.
    """
    if groups:
        gcxs = [g.cx for g in groups]
        gcys = [g.cy for g in groups]
        mass_cx = sum(gcxs) / len(gcxs)
        mass_cy = sum(gcys) / len(gcys)
    else:
        mass_cx, mass_cy = 0.5, 0.5

    anti_fx = 1.0 - focal_x
    anti_fy = 1.0 - focal_y

    if random.random() < 0.3 and groups:
        anchor = random.choice(groups).anchor
        s_edge = _make_edge_member(anchor, anchor.rotation,
                                    list(range(num_palette_colors)))
        s_edge.w = _clamp(s_edge.w * random.uniform(0.4, 0.7), MIN_SIZE, 0.12)
        s_edge.h = _clamp(s_edge.h * random.uniform(0.4, 0.7), MIN_SIZE, 0.12)
        return s_edge

    if random.random() < 0.4:
        x = _clamp01(1.0 - mass_cx + random.gauss(0, 0.15))
        y = _clamp01(1.0 - mass_cy + random.gauss(0, 0.15))
    else:
        for _attempt in range(5):
            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0.15, 0.45)
            x = _clamp01(mass_cx + r * math.cos(theta))
            y = _clamp01(mass_cy + r * math.sin(theta))
            dist_to_antifocal = math.hypot(x - anti_fx, y - anti_fy)
            if dist_to_antifocal > 0.18:
                break

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


def _place_groups_armature(n_groups: int, armature_angle: float = 35.0,
                           focal_x: float = 0.38, focal_y: float = 0.40
                           ) -> list[tuple[float, float, float]]:
    """Place groups along a dominant compositional axis (armature).

    The primary group sits near the focal point; others distribute along
    the armature line with one counter-element on the perpendicular axis.
    """
    rad = math.radians(armature_angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    perp_cos, perp_sin = -sin_a, cos_a

    results = []
    results.append((focal_x, focal_y, armature_angle + random.gauss(0, 8)))

    if n_groups == 1:
        return results

    step = random.uniform(0.14, 0.26)
    for i in range(1, n_groups):
        if i == n_groups - 1 and n_groups >= 3 and random.random() < 0.6:
            # Counter-element on the perpendicular axis for tension
            r = random.uniform(0.15, 0.30)
            sign = random.choice([-1, 1])
            cx = _clamp01(focal_x + sign * r * perp_cos + random.gauss(0, 0.04))
            cy = _clamp01(focal_y + sign * r * perp_sin + random.gauss(0, 0.04))
            ang = armature_angle + 90 + random.gauss(0, 15)
        else:
            direction = random.choice([-1, 1]) if i == 1 else (1 if i % 2 == 0 else -1)
            dist = step * ((i + 1) // 2) + random.gauss(0, 0.03)
            cx = _clamp01(focal_x + direction * dist * cos_a)
            cy = _clamp01(focal_y + direction * dist * sin_a)
            ang = armature_angle + random.gauss(0, 12)
        results.append((cx, cy, ang))

    return results


_GROUP_PLACERS = [
    _place_groups_balanced,
    _place_groups_vertical_zones,
    _place_groups_diagonal,
]


# ======================================================================
# Compositional helpers: edge members, gravity, focal weighting
# ======================================================================

def _make_edge_member(anchor: Shape, angle: float,
                      sub_indices: list[int],
                      kind_pool: tuple = _ALL_KINDS_WEIGHTED) -> Shape:
    """Place a member along a random edge of the anchor's bounding box.

    Creates children that grow from parent edges rather than floating freely,
    reinforcing hierarchical nesting.
    """
    rad = math.radians(anchor.rotation)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    hw, hh = anchor.w / 2.0, anchor.h / 2.0

    edge = random.choice(["top", "bottom", "left", "right"])
    t = random.uniform(-0.8, 0.8)
    if edge == "top":
        lx, ly = t * hw, -hh
    elif edge == "bottom":
        lx, ly = t * hw, hh
    elif edge == "left":
        lx, ly = -hw, t * hh
    else:
        lx, ly = hw, t * hh

    wx = lx * cos_r - ly * sin_r + anchor.x
    wy = lx * sin_r + ly * cos_r + anchor.y

    kind = random.choice(kind_pool)
    scale = random.uniform(0.3, 0.65)
    w = _clamp(anchor.w * scale, MIN_SIZE, MAX_SIZE * 0.6)
    h = _clamp(anchor.h * scale, MIN_SIZE, MAX_SIZE * 0.6)
    if kind in ("line", "rect") and random.random() < 0.5:
        ratio = random.uniform(2.0, 6.0)
        h = min(h, w / ratio)

    rot = anchor.rotation + random.choice([0, 90]) + random.gauss(0, 8)
    s = Shape(
        kind=kind, x=_clamp01(wx), y=_clamp01(wy), w=w, h=h,
        rotation=rot % 360,
        color_idx=random.choice(sub_indices),
    )
    _ensure_inside(s)
    return s


def _apply_gravitational_alignment(genome: ShapeGenome) -> None:
    """Rotate smaller shapes toward alignment with nearby larger shapes.

    Mimics how Lissitzky's Prouns work -- forms feel like they exert forces
    on each other rather than occupying the same space accidentally.
    In 3D (Hadid) mode the effect is much stronger for architectural order.
    """
    shapes = genome.flatten()
    if len(shapes) < 2:
        return
    is_3d = genome.projection_strength > 0.01
    areas = [(s.w * s.h, s) for s in shapes]
    areas.sort(key=lambda t: t[0], reverse=True)

    radius = 0.20 if is_3d else 0.30
    base_influence = 0.55 if is_3d else 0.25

    for idx in range(1, len(areas)):
        _, s = areas[idx]
        best_dist = 999.0
        best_angle = s.rotation
        for j in range(idx):
            _, big = areas[j]
            dx = s.x - big.x
            dy = s.y - big.y
            dist = math.hypot(dx, dy)
            if dist < best_dist and (big.w * big.h) > (s.w * s.h):
                best_dist = dist
                best_angle = big.rotation

        if best_dist < radius:
            influence = base_influence * (1.0 - best_dist / radius)
            cur = s.rotation % 360
            tgt = best_angle % 360
            diff = tgt - cur
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            s.rotation = (cur + diff * influence) % 360

    if is_3d and genome.engine_tag != "maps":
        _snap_positions_to_grid(shapes)


def _snap_positions_to_grid(shapes: list[Shape], strength: float = 0.3) -> None:
    """Gently nudge shape positions toward a coarse implicit grid.

    This reduces the 'thrown together' look by introducing subtle regularity
    without making positions perfectly mechanical.
    """
    grid_step = 0.04
    for s in shapes:
        grid_x = round(s.x / grid_step) * grid_step
        grid_y = round(s.y / grid_step) * grid_step
        s.x = s.x + (grid_x - s.x) * strength
        s.y = s.y + (grid_y - s.y) * strength


def _apply_focal_weighting(genome: ShapeGenome) -> None:
    """Bias shape size and opacity toward the focal zone.

    Creates clear entry/exit points: large high-contrast forms near the focal
    point, trailing off into smaller, quieter forms elsewhere.
    """
    fx, fy = genome.focal_x, genome.focal_y
    falloff = genome.density_falloff
    if falloff < 0.05:
        return

    max_dist = math.sqrt(2.0)
    for s in genome.flatten():
        dist = math.hypot(s.x - fx, s.y - fy)
        t = min(dist / max_dist, 1.0)
        scale = 1.0 + (1.0 - t) * 0.18 * falloff - t * 0.12 * falloff
        s.w = _clamp(s.w * scale, MIN_SIZE, MAX_SIZE)
        s.h = _clamp(s.h * scale, MIN_SIZE, MAX_SIZE)
        opacity_mod = 1.0 - t * 0.12 * falloff
        s.opacity = _clamp(s.opacity * opacity_mod, 0.82, 1.0)


# ======================================================================
# Random initialization
# ======================================================================

def create_random(num_palette_colors: int = 16) -> ShapeGenome:
    """Create a genome via multi-stage hierarchical generation.

    Stage 1: decide number of groups, place them on the canvas.
    Stage 2: populate each group with an anchor + members.
    Stage 3: add satellite accent marks for rhythm and balance.
    Stage 4: apply compositional post-processing (armature, gravity, focal).
    """
    n_groups = random.choices([1, 2, 3, 4], weights=[25, 40, 25, 10])[0]

    armature_angle = random.choice([
        random.uniform(25, 55), random.uniform(125, 155),
        random.uniform(-55, -25), random.uniform(-155, -125),
    ])
    focal_x = random.gauss(0.38, 0.08)
    focal_y = random.gauss(0.40, 0.08)
    focal_x = _clamp(focal_x, 0.22, 0.62)
    focal_y = _clamp(focal_y, 0.22, 0.62)
    density_falloff = random.uniform(0.3, 0.8)

    palette_size = min(num_palette_colors, random.randint(3, 6))
    sub_indices = random.sample(range(num_palette_colors), palette_size)

    # ~40% armature placer, ~60% legacy placers
    if random.random() < 0.40:
        placements = _place_groups_armature(n_groups, armature_angle,
                                            focal_x, focal_y)
    else:
        placer = random.choice(_GROUP_PLACERS)
        placements = placer(n_groups)

    groups: list[ShapeGroup] = []
    for cx, cy, angle in placements:
        anchor = _make_anchor(cx, cy, angle, sub_indices)
        n_members = random.randint(1, 5)

        if random.random() < 0.3:
            kind_pool = _BAR_KINDS
        else:
            kind_pool = _ALL_KINDS_WEIGHTED

        spread = random.uniform(0.04, 0.12)
        members = []
        for _ in range(n_members):
            if random.random() < 0.45:
                members.append(_make_edge_member(anchor, angle,
                                                  sub_indices, kind_pool))
            else:
                members.append(_make_member(cx, cy, angle, spread,
                                            sub_indices, kind_pool,
                                            armature_angle=armature_angle))
        groups.append(ShapeGroup(
            anchor=anchor, members=members,
            cx=cx, cy=cy, angle=angle,
        ))

    n_satellites = random.randint(0, 6)
    satellites = [_make_satellite(groups, num_palette_colors,
                                  focal_x=focal_x, focal_y=focal_y)
                  for _ in range(n_satellites)]

    bg = random.randint(0, num_palette_colors - 1) if random.random() < 0.12 else (random.choice([0, 1]) if num_palette_colors > 1 else 0)
    bg_sec_idx = random.randint(0, num_palette_colors - 1)
    if bg_sec_idx == bg:
        bg_sec_idx = (bg + 1) % num_palette_colors
    bg_angle = random.uniform(0, 360)
    bg_cx = random.uniform(0.1, 0.9)
    bg_cy = random.uniform(0.1, 0.9)

    genome = ShapeGenome(
        groups=groups, satellites=satellites, bg_color_idx=bg,
        bg_angle=bg_angle, bg_cx=bg_cx, bg_cy=bg_cy, bg_sec_idx=bg_sec_idx,
        armature_angle=armature_angle, focal_x=focal_x, focal_y=focal_y,
        density_falloff=density_falloff,
    )
    _apply_gravitational_alignment(genome)
    _apply_focal_weighting(genome)
    return genome


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
    "switch_material": 0.07,
    "swap_order": 0.08,
    # Satellites
    "add_satellite": 0.08,
    "remove_satellite": 0.05,
    # Background
    "perturb_bg": 0.15,
    # Compositional armature / focal
    "perturb_armature": 0.05,
    "perturb_focal": 0.06,
    # 3D depth / projection (Hadid mode)
    "perturb_depth": 0.15,
    "perturb_elevation": 0.10,
    "perturb_projection": 0.05,
    "perturb_camera": 0.07,
    "change_detail": 0.08,
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
    if s.material not in ("concrete", "marble"):
        s.material = "concrete"

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
        if genome.projection_strength > 0.01:
            s.kind = random.choice(("rect", "square", "trapezoid", "triangle"))
        else:
            s.kind = random.choice(SHAPE_KINDS)

    if groups and random.random() < _rate("swap_order") and len(groups) >= 2:
        i, j = random.sample(range(len(groups)), 2)
        groups[i], groups[j] = groups[j], groups[i]

    # ------ Satellites ------

    if random.random() < _rate("add_satellite"):
        if len(sats) < MAX_SATELLITES:
            sats.append(_make_satellite(groups, num_palette_colors,
                                        focal_x=genome.focal_x,
                                        focal_y=genome.focal_y))

    if random.random() < _rate("remove_satellite") and sats:
        sats.pop(random.randrange(len(sats)))

    # ------ Compositional armature / focal ------

    if random.random() < _rate("perturb_armature"):
        genome.armature_angle = (genome.armature_angle + random.gauss(0, 5)) % 360

    if random.random() < _rate("perturb_focal"):
        genome.focal_x = _clamp(genome.focal_x + random.gauss(0, 0.04), 0.12, 0.88)
        genome.focal_y = _clamp(genome.focal_y + random.gauss(0, 0.04), 0.12, 0.88)
        genome.density_falloff = _clamp(
            genome.density_falloff + random.gauss(0, 0.06), 0.0, 1.0)

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
        _is_map = genome.engine_tag == "maps"
        if s and random.random() < _rate("perturb_depth"):
            _d_sigma = 0.003 if _is_map else 0.015
            s.depth = _clamp(s.depth + random.gauss(0, _d_sigma),
                             0.0, 0.012 if _is_map else 999.0)

        if s and random.random() < _rate("perturb_elevation"):
            _e_sigma = 0.002 if _is_map else 0.012
            s.elevation = _clamp(s.elevation + random.gauss(0, _e_sigma),
                                 0.0, 0.008 if _is_map else 999.0)

        if random.random() < _rate("perturb_projection"):
            _is_map = genome.engine_tag == "maps"
            genome.projection_angle = _clamp(
                genome.projection_angle + random.gauss(0, 3), 10, 70,
            )
            if random.random() < 0.3:
                _ps_max = 0.04 if _is_map else 0.8
                genome.projection_strength = _clamp(
                    genome.projection_strength + random.gauss(0, 0.01 if _is_map else 0.04),
                    0.005 if _is_map else 0.1, _ps_max,
                )
            if random.random() < 0.25:
                genome.projection_secondary_angle = _clamp(
                    genome.projection_secondary_angle + random.gauss(0, 8),
                    -85, 85,
                )
            if random.random() < 0.25:
                _ps2_max = 0.025 if _is_map else 0.8
                genome.projection_secondary_strength = _clamp(
                    genome.projection_secondary_strength + random.gauss(0, 0.01 if _is_map else 0.05),
                    0.0, _ps2_max,
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
        if s and random.random() < _rate("change_detail"):
            if random.random() < 0.25:
                s.detail = "none"
            else:
                s.detail = random.choices(
                    _DETAIL_TYPES[1:],
                    weights=[25, 12, 12, 10, 8, 8, 15, 10],
                )[0]
        if s and random.random() < _rate("perturb_opacity"):
            s.opacity = _clamp(s.opacity + random.gauss(0.10, 0.04), 0.92, 1.0)
        if s and random.random() < _rate("switch_material"):
            if s.material == "marble":
                s.material = "concrete" if random.random() < 0.65 else "marble"
            else:
                s.material = "marble" if random.random() < 0.30 else "concrete"
            # Keep both materials fully solid.
            s.opacity = random.uniform(0.94, 1.0)

    # ------ Compositional post-processing ------
    if random.random() < 0.35:
        _apply_gravitational_alignment(genome)

    if genome.projection_strength > 0.01:
        if genome.engine_tag != "maps":
            _solidify_architecton(genome)
        _enforce_3d_bounds(genome)
    else:
        for shape in genome.flatten():
            _ensure_inside(shape)


# ======================================================================
# Crossover
# ======================================================================

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _group_centroid(g: ShapeGroup) -> tuple[float, float]:
    shapes = g.all_shapes()
    if not shapes:
        return (g.cx, g.cy)
    sx = sum(s.x for s in shapes) / len(shapes)
    sy = sum(s.y for s in shapes) / len(shapes)
    return (sx, sy)


def _shift_group(g: ShapeGroup, dx: float, dy: float) -> None:
    """Translate every shape in a group by (dx, dy)."""
    for s in g.all_shapes():
        s.x = _clamp01(s.x + dx)
        s.y = _clamp01(s.y + dy)
    g.cx = _clamp01(g.cx + dx)
    g.cy = _clamp01(g.cy + dy)


def _blend_shape_properties(s: Shape, donor: Shape, t: float) -> None:
    """Interpolate continuous properties of *s* toward *donor*."""
    s.w = _clamp(_lerp(s.w, donor.w, t), MIN_SIZE, MAX_SIZE)
    s.h = _clamp(_lerp(s.h, donor.h, t), MIN_SIZE, MAX_SIZE)
    s.depth = max(0.0, _lerp(s.depth, donor.depth, t))
    s.elevation = max(0.0, _lerp(s.elevation, donor.elevation, t))
    s.opacity = _clamp(_lerp(s.opacity, donor.opacity, t), 0.0, 1.0)
    angle_diff = donor.rotation - s.rotation
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    s.rotation = (s.rotation + angle_diff * t) % 360


def _crossover_blend_genome_params(a: ShapeGenome, b: ShapeGenome,
                                    t: float) -> dict:
    """Interpolate genome-level parameters at ratio *t* (0→a, 1→b)."""
    return dict(
        bg_style=a.bg_style if random.random() < (1 - t) else b.bg_style,
        bg_color_idx=a.bg_color_idx if random.random() < (1 - t) else b.bg_color_idx,
        bg_angle=_lerp(a.bg_angle, b.bg_angle, t),
        bg_cx=_lerp(a.bg_cx, b.bg_cx, t),
        bg_cy=_lerp(a.bg_cy, b.bg_cy, t),
        bg_sec_idx=a.bg_sec_idx if random.random() < (1 - t) else b.bg_sec_idx,
        bg_blend=_lerp(a.bg_blend, b.bg_blend, t),
        projection_angle=_lerp(a.projection_angle, b.projection_angle, t),
        projection_strength=_lerp(a.projection_strength, b.projection_strength, t),
        projection_secondary_angle=_lerp(
            a.projection_secondary_angle, b.projection_secondary_angle, t),
        projection_secondary_strength=_lerp(
            a.projection_secondary_strength, b.projection_secondary_strength, t),
        projection_vanishing_x=_lerp(
            a.projection_vanishing_x, b.projection_vanishing_x, t),
        projection_vanishing_y=_lerp(
            a.projection_vanishing_y, b.projection_vanishing_y, t),
        camera_mode=a.camera_mode if random.random() < (1 - t) else b.camera_mode,
        perspective_bias=_lerp(a.perspective_bias, b.perspective_bias, t),
        armature_angle=_lerp(a.armature_angle, b.armature_angle, t),
        focal_x=_lerp(a.focal_x, b.focal_x, t),
        focal_y=_lerp(a.focal_y, b.focal_y, t),
        density_falloff=_lerp(a.density_falloff, b.density_falloff, t),
        engine_tag=a.engine_tag or b.engine_tag,
    )


def crossover(parent_a: ShapeGenome, parent_b: ShapeGenome) -> ShapeGenome:
    """Crossover with multiple strategies for structural variety.

    Picks one of several strategies:
      - spatial_splice: take left/top half from A, right/bottom from B
      - scaffold_graft: keep A's largest groups as skeleton, graft B's details
      - interleave: alternating groups A-B-A-B with spatial re-centering
      - blend: pair groups by proximity and interpolate shapes
    """
    is_maps = (parent_a.engine_tag == "maps" or
               parent_b.engine_tag == "maps")

    is_3d = (parent_a.projection_strength > 0.01 or
             parent_b.projection_strength > 0.01 or
             is_maps)

    if not is_3d:
        return _crossover_flat(parent_a, parent_b)

    strategy = random.choices(
        ["spatial_splice", "scaffold_graft", "interleave", "blend"],
        weights=[30, 25, 25, 20],
    )[0]

    if strategy == "spatial_splice":
        child = _crossover_spatial_splice(parent_a, parent_b)
    elif strategy == "scaffold_graft":
        child = _crossover_scaffold_graft(parent_a, parent_b)
    elif strategy == "interleave":
        child = _crossover_interleave(parent_a, parent_b)
    else:
        child = _crossover_blend(parent_a, parent_b)

    if not is_maps:
        _solidify_architecton(child)
    _enforce_3d_bounds(child)
    return child


def _crossover_flat(parent_a: ShapeGenome, parent_b: ShapeGenome) -> ShapeGenome:
    """Original coin-flip crossover for flat 2D genomes."""
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
        child_groups.append((parent_a if random.random() < 0.5 else parent_b).groups[0].copy())

    child_sats = _mix_satellites(parent_a, parent_b)
    params = _crossover_blend_genome_params(parent_a, parent_b, 0.5)
    child = ShapeGenome(groups=child_groups, satellites=child_sats, **params)
    for shape in child.flatten():
        _ensure_inside(shape)
    return child


def _mix_satellites(a: ShapeGenome, b: ShapeGenome) -> list[Shape]:
    sats: list[Shape] = []
    for s in a.satellites:
        if random.random() < 0.5:
            sats.append(s.copy())
    for s in b.satellites:
        if random.random() < 0.5:
            sats.append(s.copy())
    return sats


def _crossover_spatial_splice(parent_a: ShapeGenome,
                               parent_b: ShapeGenome) -> ShapeGenome:
    """Split canvas spatially and take groups from each parent by region."""
    use_vertical = random.random() < 0.5
    split = random.uniform(0.35, 0.65)
    t = random.uniform(0.3, 0.7)

    child_groups: list[ShapeGroup] = []
    for g in parent_a.groups:
        cx, cy = _group_centroid(g)
        in_region = (cx < split) if use_vertical else (cy < split)
        if in_region:
            child_groups.append(g.copy())
    for g in parent_b.groups:
        cx, cy = _group_centroid(g)
        in_region = (cx >= split) if use_vertical else (cy >= split)
        if in_region:
            child_groups.append(g.copy())

    if not child_groups:
        all_groups = parent_a.groups + parent_b.groups
        if all_groups:
            child_groups.append(random.choice(all_groups).copy())

    child_sats = _mix_satellites(parent_a, parent_b)
    params = _crossover_blend_genome_params(parent_a, parent_b, t)
    return ShapeGenome(groups=child_groups, satellites=child_sats, **params)


def _crossover_scaffold_graft(parent_a: ShapeGenome,
                                parent_b: ShapeGenome) -> ShapeGenome:
    """Use the largest groups from one parent as skeleton, graft detail
    shapes from the other parent onto those positions."""
    primary, donor = (parent_a, parent_b) if random.random() < 0.5 else (parent_b, parent_a)
    t = random.uniform(0.3, 0.7)

    scored = [(len(g.all_shapes()), i) for i, g in enumerate(primary.groups)]
    scored.sort(reverse=True)

    n_keep = max(1, len(scored) * 2 // 3)
    keep_indices = {idx for _, idx in scored[:n_keep]}

    child_groups: list[ShapeGroup] = []
    for i, g in enumerate(primary.groups):
        if i in keep_indices:
            child_groups.append(g.copy())

    donor_shapes_pool: list[Shape] = []
    for g in donor.groups:
        donor_shapes_pool.extend(g.all_shapes())

    if child_groups and donor_shapes_pool:
        n_graft = random.randint(1, min(len(donor_shapes_pool), 6))
        grafts = random.sample(donor_shapes_pool, n_graft)
        target_group = random.choice(child_groups)
        tcx, tcy = _group_centroid(target_group)
        for gs in grafts:
            grafted = gs.copy()
            grafted.x = _clamp01(tcx + random.gauss(0, 0.05))
            grafted.y = _clamp01(tcy + random.gauss(0, 0.04))
            grafted.rotation = (target_group.angle + random.gauss(0, 5)) % 360
            grafted.elevation = max(0.0, grafted.elevation + random.gauss(0, 0.03))
            target_group.members.append(grafted)

    if not child_groups:
        all_groups = primary.groups + donor.groups
        if all_groups:
            child_groups.append(all_groups[0].copy())

    child_sats = _mix_satellites(primary, donor)
    params = _crossover_blend_genome_params(parent_a, parent_b, t)
    return ShapeGenome(groups=child_groups, satellites=child_sats, **params)


def _crossover_interleave(parent_a: ShapeGenome,
                           parent_b: ShapeGenome) -> ShapeGenome:
    """Alternate groups A-B-A-B and re-center them along a shared axis."""
    all_a = [(g.copy(), 'a') for g in parent_a.groups]
    all_b = [(g.copy(), 'b') for g in parent_b.groups]
    t = random.uniform(0.3, 0.7)

    merged: list[ShapeGroup] = []
    ia, ib = 0, 0
    pick_a = random.random() < 0.5
    while ia < len(all_a) or ib < len(all_b):
        if pick_a and ia < len(all_a):
            merged.append(all_a[ia][0])
            ia += 1
        elif ib < len(all_b):
            merged.append(all_b[ib][0])
            ib += 1
        elif ia < len(all_a):
            merged.append(all_a[ia][0])
            ia += 1
        pick_a = not pick_a

    if not merged:
        all_groups = parent_a.groups + parent_b.groups
        if all_groups:
            merged.append(random.choice(all_groups).copy())

    axis_angle = _lerp(parent_a.armature_angle, parent_b.armature_angle, t)
    axis_rad = math.radians(axis_angle)
    center_x = _lerp(parent_a.focal_x, parent_b.focal_x, t)
    center_y = _lerp(parent_a.focal_y, parent_b.focal_y, t)
    n = len(merged)
    spacing = min(0.14, 0.55 / max(1, n))

    for i, g in enumerate(merged):
        offset = (i - (n - 1) / 2.0) * spacing
        target_x = center_x + offset * math.cos(axis_rad)
        target_y = center_y + offset * math.sin(axis_rad)
        gcx, gcy = _group_centroid(g)
        dx = _clamp(target_x - gcx, -0.3, 0.3)
        dy = _clamp(target_y - gcy, -0.3, 0.3)
        _shift_group(g, dx, dy)
        old_angle = g.angle
        blend_angle = axis_angle + random.gauss(0, 5)
        angle_diff = blend_angle - old_angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        g.angle = (old_angle + angle_diff * 0.4) % 360

    child_sats = _mix_satellites(parent_a, parent_b)
    params = _crossover_blend_genome_params(parent_a, parent_b, t)
    return ShapeGenome(groups=merged, satellites=child_sats, **params)


def _crossover_blend(parent_a: ShapeGenome,
                      parent_b: ShapeGenome) -> ShapeGenome:
    """Pair groups by spatial proximity and interpolate their shapes."""
    t = random.uniform(0.25, 0.75)
    groups_a = [g.copy() for g in parent_a.groups]
    groups_b = [g.copy() for g in parent_b.groups]

    child_groups: list[ShapeGroup] = []
    used_b: set[int] = set()

    for ga in groups_a:
        ca = _group_centroid(ga)
        best_j = -1
        best_dist = 999.0
        for j, gb in enumerate(groups_b):
            if j in used_b:
                continue
            cb = _group_centroid(gb)
            d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0 and best_dist < 0.4:
            used_b.add(best_j)
            gb = groups_b[best_j]
            shapes_a = ga.all_shapes()
            shapes_b = gb.all_shapes()
            n_out = max(len(shapes_a), len(shapes_b))
            blended: list[Shape] = []
            for k in range(n_out):
                if k < len(shapes_a) and k < len(shapes_b):
                    s = shapes_a[k].copy()
                    _blend_shape_properties(s, shapes_b[k], t)
                    if random.random() < t:
                        s.kind = shapes_b[k].kind
                        s.color_idx = shapes_b[k].color_idx
                        s.material = shapes_b[k].material
                        s.detail = shapes_b[k].detail
                    blended.append(s)
                elif k < len(shapes_a):
                    if random.random() < 0.7:
                        blended.append(shapes_a[k].copy())
                else:
                    if random.random() < 0.7:
                        blended.append(shapes_b[k].copy())
            if blended:
                child_groups.append(ShapeGroup(
                    anchor=blended[0],
                    members=blended[1:],
                    cx=_lerp(ga.cx, gb.cx, t),
                    cy=_lerp(ga.cy, gb.cy, t),
                    angle=_lerp(ga.angle, gb.angle, t),
                ))
        else:
            child_groups.append(ga)

    for j, gb in enumerate(groups_b):
        if j not in used_b and random.random() < 0.4:
            child_groups.append(gb)

    if not child_groups:
        all_groups = parent_a.groups + parent_b.groups
        if all_groups:
            child_groups.append(random.choice(all_groups).copy())

    child_sats = _mix_satellites(parent_a, parent_b)
    params = _crossover_blend_genome_params(parent_a, parent_b, t)
    return ShapeGenome(groups=child_groups, satellites=child_sats, **params)


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

    camera_mode = random.choices([0, 1, 2], weights=[45, 30, 25])[0]
    proj_angle = random.choice([20, 25, 30, 35, 40, 45, 55])
    proj_strength = random.uniform(0.25, 0.65)
    proj2_angle = proj_angle + random.choice([-90, -75, -60, 60, 75, 90])
    proj2_strength = random.uniform(0.12, 0.50)
    vanish_x = random.uniform(0.30, 0.70)
    vanish_y = random.uniform(0.25, 0.75)
    perspective_bias = random.uniform(-0.45, 0.45)

    def _choose_material(kind: str, w: float, h: float, depth: float,
                         elevation: float, material: str | None = None) -> str:
        """Pick a solid material family for Hadid objects."""
        if material in ("concrete", "marble"):
            return material

        p_marble = 0.30
        if max(w, h) > 0.20:
            p_marble *= 0.55
        if depth > 0.05 and elevation < 0.22:
            p_marble *= 0.50
        if kind in ("line", "triangle"):
            p_marble *= 1.10
        return "marble" if random.random() < p_marble else "concrete"

    def _choose_opacity(material: str) -> float:
        if material == "marble":
            return random.choices([1.0, random.uniform(0.94, 0.97)],
                                   weights=[20, 80])[0]
        return random.choices([1.0, random.uniform(0.94, 0.97),
                                random.uniform(0.97, 1.0)],
                               weights=[25, 50, 25])[0]

    def _new_shape(kind: str, x: float, y: float, w: float, h: float, rotation: float,
                   depth: float, elevation: float, color_idx: int | None = None,
                   material: str | None = None) -> Shape:
        mat = _choose_material(kind, w, h, depth, elevation, material=material)
        opacity = _choose_opacity(mat)
        s = Shape(
            kind=kind,
            x=_clamp01(x),
            y=_clamp01(y),
            w=_clamp(w, MIN_SIZE, MAX_SIZE),
            h=_clamp(h, MIN_SIZE, MAX_SIZE),
            rotation=rotation % 360,
            color_idx=color_idx if color_idx is not None else random.choice(sub_indices),
            opacity=opacity,
            material=mat,
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
        deck_angle = random.uniform(-20, -5)
        deck_cx = random.uniform(0.38, 0.58)
        deck_cy = random.uniform(0.55, 0.70)
    elif scene_style == "spine":
        deck_angle = random.uniform(-12, 12)
        deck_cx = random.uniform(0.35, 0.55)
        deck_cy = random.uniform(0.50, 0.65)
    elif scene_style == "radial":
        deck_angle = random.uniform(-25, 25)
        deck_cx = random.uniform(0.40, 0.60)
        deck_cy = random.uniform(0.42, 0.62)
    else:  # towerfield
        deck_angle = random.uniform(-6, 6)
        deck_cx = random.uniform(0.42, 0.58)
        deck_cy = random.uniform(0.55, 0.70)

    armature_angle = deck_angle
    focal_x = deck_cx + random.gauss(0, 0.06)
    focal_y = deck_cy + random.gauss(0, 0.06)
    focal_x = _clamp(focal_x, 0.15, 0.85)
    focal_y = _clamp(focal_y, 0.15, 0.85)
    density_falloff = random.uniform(0.3, 0.7)

    _DECK_KINDS = ["rect", "trapezoid", "parallelogram", "hexagon"]
    deck_w = random.uniform(0.40, 0.95)
    deck_h = random.uniform(0.08, 0.24)
    deck_depth = random.uniform(0.02, 0.07)
    deck_shapes = [
        _new_shape(
            kind=random.choices(_DECK_KINDS, weights=[40, 30, 18, 12])[0],
            x=deck_cx, y=deck_cy,
            w=deck_w, h=deck_h,
            rotation=deck_angle,
            depth=deck_depth,
            elevation=0.0,
            material="concrete",
        ),
    ]
    elev_seed = deck_depth
    for _ in range(random.randint(1, 4)):
        sc = random.uniform(0.50, 0.88)
        d = random.uniform(0.01, 0.04)
        deck_shapes.append(
            _new_shape(
                kind=random.choices(_DECK_KINDS, weights=[40, 30, 18, 12])[0],
                x=deck_cx + random.gauss(0, 0.045),
                y=deck_cy + random.gauss(0, 0.03),
                w=deck_w * sc,
                h=deck_h * sc,
                rotation=deck_angle + random.gauss(0, 7),
                depth=d,
                elevation=elev_seed,
                material="concrete",
            ),
        )
        elev_seed += d * random.uniform(0.8, 1.05)
    _append_group(deck_shapes, deck_cx, deck_cy, deck_angle)

    # --- 2) Cluster placements based on scene archetype ---
    cluster_centers: list[tuple[float, float]] = []
    cluster_angles: list[float] = []

    if scene_style == "spine":
        n_clusters = random.randint(3, 5)
        axis = random.uniform(-0.6, 0.6)
        start_x = random.uniform(0.22, 0.38)
        start_y = random.uniform(0.35, 0.55)
        step = random.uniform(0.08, 0.14)
        for i in range(n_clusters):
            cx = _clamp01(start_x + i * step * math.cos(axis) + random.gauss(0, 0.012))
            cy = _clamp01(start_y + i * step * math.sin(axis) + random.gauss(0, 0.012))
            cluster_centers.append((cx, cy))
            cluster_angles.append(math.degrees(axis) + random.gauss(0, 6))
    elif scene_style == "radial":
        n_clusters = random.randint(3, 5)
        hub_x = random.uniform(0.38, 0.62)
        hub_y = random.uniform(0.38, 0.62)
        spin = random.uniform(0, 2 * math.pi)
        for i in range(n_clusters):
            theta = spin + (2 * math.pi * i / n_clusters) + random.gauss(0, 0.15)
            radius = random.uniform(0.06, 0.18)
            cx = _clamp01(hub_x + radius * math.cos(theta))
            cy = _clamp01(hub_y + radius * math.sin(theta))
            cluster_centers.append((cx, cy))
            cluster_angles.append(math.degrees(theta) + random.gauss(0, 8))
    elif scene_style == "cantilever":
        n_clusters = random.randint(2, 4)
        spine_y = random.uniform(0.50, 0.68)
        for i in range(n_clusters):
            t = i / max(1, n_clusters - 1)
            cx = _clamp01(0.25 + 0.50 * t + random.gauss(0, 0.015))
            cy = _clamp01(spine_y + random.gauss(0, 0.025))
            cluster_centers.append((cx, cy))
            cluster_angles.append(deck_angle + random.gauss(0, 8))
    else:  # towerfield
        rows = random.randint(2, 3)
        cols = random.randint(2, 3)
        x0 = random.uniform(0.28, 0.38)
        y0 = random.uniform(0.30, 0.42)
        sx = random.uniform(0.10, 0.18)
        sy = random.uniform(0.08, 0.14)
        for r in range(rows):
            for c in range(cols):
                if len(cluster_centers) >= 5:
                    break
                cx = _clamp01(x0 + c * sx + random.gauss(0, 0.012))
                cy = _clamp01(y0 + r * sy + random.gauss(0, 0.012))
                cluster_centers.append((cx, cy))
                cluster_angles.append(deck_angle + random.gauss(0, 6))

    # Shape vocabulary for clusters — includes new primitives
    _CLUSTER_KINDS = ["rect", "rect", "trapezoid", "parallelogram",
                      "hexagon", "chevron", "lshape", "tshape"]
    _ARM_KINDS = ["rect", "rect", "trapezoid", "parallelogram", "arrow"]

    # --- 3) Vertical / angular structural clusters ---
    # Each cluster randomly picks a size class for variety.
    _SIZE_CLASSES = [
        {"w": (0.03, 0.09), "h": (0.012, 0.040), "d": (0.02, 0.06), "lvl": (3, 6)},   # small
        {"w": (0.06, 0.18), "h": (0.020, 0.070), "d": (0.035, 0.10), "lvl": (3, 9)},   # medium
        {"w": (0.14, 0.32), "h": (0.030, 0.095), "d": (0.05, 0.14), "lvl": (4, 11)},   # large
        {"w": (0.22, 0.45), "h": (0.025, 0.060), "d": (0.04, 0.10), "lvl": (2, 5)},    # wide platform
    ]

    for i, (cx, cy) in enumerate(cluster_centers):
        if len(groups) >= MAX_GROUPS:
            break
        angle = cluster_angles[i]

        sc_cls = random.choices(_SIZE_CLASSES, weights=[15, 45, 30, 10])[0]
        n_levels = random.randint(*sc_cls["lvl"])
        base_w = random.uniform(*sc_cls["w"])
        level_h = random.uniform(*sc_cls["h"])
        base_depth = random.uniform(*sc_cls["d"])
        elev = elev_seed + random.uniform(0.0, 0.08)
        taper = random.uniform(0.01, 0.08)

        cluster_shapes: list[Shape] = []
        top_x = cx
        top_y = cy
        top_elev = elev

        # Foundation / base platform
        if random.random() < 0.65:
            plat_kind = random.choices(
                ["rect", "trapezoid", "hexagon", "parallelogram"],
                weights=[45, 25, 15, 15],
            )[0]
            plat_w = base_w * random.uniform(1.15, 1.60)
            plat_h = level_h * random.uniform(0.35, 0.60)
            plat = _new_shape(
                plat_kind, cx, cy, plat_w, plat_h,
                angle + random.gauss(0, 0.5),
                random.uniform(0.015, 0.040), elev_seed,
            )
            plat.detail = "none"
            cluster_shapes.append(plat)

        for lvl in range(n_levels):
            sc = max(0.35, 1.0 - taper * lvl)
            wobble = 0.0005 + 0.0005 * lvl
            sw = base_w * sc * random.uniform(0.94, 1.06)
            sh = level_h * random.uniform(0.90, 1.10)
            sd = base_depth * random.uniform(0.82, 1.05)
            lx = cx + random.gauss(0, wobble)
            ly = cy + random.gauss(0, wobble * 0.5)
            kind = random.choices(
                _CLUSTER_KINDS, weights=[30, 30, 14, 8, 5, 5, 4, 4],
            )[0]
            cluster_shapes.append(
                _new_shape(kind, lx, ly, sw, sh, angle + random.gauss(0, 0.8), sd, elev),
            )
            top_x, top_y, top_elev = lx, ly, elev + sd
            elev += sd * random.uniform(0.92, 1.02)

            # Cantilever arms — stay close, share the elevation
            if lvl > 0 and random.random() < 0.30:
                arm_w = sw * random.uniform(0.5, 1.2)
                arm_h = sh * random.uniform(0.50, 0.85)
                arm_side = random.choice([-1, 1])
                arm_shift = arm_side * sw * random.uniform(0.45, 0.65)
                cluster_shapes.append(
                    _new_shape(
                        random.choice(_ARM_KINDS),
                        lx + arm_shift * math.cos(math.radians(angle)),
                        ly + arm_shift * math.sin(math.radians(angle)),
                        arm_w, arm_h,
                        angle + random.gauss(0, 1.5),
                        sd * random.uniform(0.40, 0.70),
                        elev - sd * 0.5,
                    ),
                )

        # Structural sub-elements: floor beams, columns, frames
        if n_levels >= 2:
            ang_rad = math.radians(angle)
            ang_cos, ang_sin = math.cos(ang_rad), math.sin(ang_rad)

            # Horizontal floor beams between levels
            for lvl_idx in range(1, min(n_levels, len(cluster_shapes))):
                if random.random() < 0.60:
                    s = cluster_shapes[lvl_idx]
                    beam_w = base_w * random.uniform(0.90, 1.20)
                    beam = _new_shape(
                        "rect", s.x, s.y, beam_w,
                        random.uniform(0.006, 0.014),
                        angle + random.gauss(0, 0.5),
                        random.uniform(0.010, 0.025), s.elevation,
                    )
                    beam.detail = "none"
                    cluster_shapes.append(beam)

            # Vertical columns on sides of taller structures
            if n_levels >= 3 and random.random() < 0.55:
                col_h = level_h * n_levels * random.uniform(0.4, 0.7)
                for side in [-1, 1]:
                    col_off = side * base_w * random.uniform(0.38, 0.50)
                    col_x = cx + col_off * ang_cos
                    col_y = cy + col_off * ang_sin
                    col = _new_shape(
                        "rect", col_x, col_y,
                        random.uniform(0.008, 0.018), col_h,
                        angle + random.gauss(0, 0.5),
                        random.uniform(0.008, 0.020), elev_seed,
                    )
                    col.detail = "none"
                    cluster_shapes.append(col)

            # Small antenna / chimney accent on top
            if random.random() < 0.35:
                ant = _new_shape(
                    "rect", top_x + random.gauss(0, 0.005),
                    top_y + random.gauss(0, 0.003),
                    random.uniform(0.006, 0.015),
                    random.uniform(0.03, 0.08),
                    angle + random.gauss(0, 3),
                    random.uniform(0.006, 0.012),
                    top_elev + random.uniform(0.01, 0.04),
                )
                ant.detail = "none"
                cluster_shapes.append(ant)

        for s in cluster_shapes:
            _assign_detail(s)

        _append_group(cluster_shapes, cx, cy, angle)

    # --- 4) Bridges / spanning bars between adjacent clusters ---
    if len(cluster_centers) >= 2 and len(groups) < MAX_GROUPS:
        indexed = list(enumerate(cluster_centers))
        indexed.sort(key=lambda it: it[1][0])
        pair_pool = [(indexed[i][0], indexed[i + 1][0]) for i in range(len(indexed) - 1)]
        random.shuffle(pair_pool)
        n_bridges = min(2, len(pair_pool))
        for a_idx, b_idx in pair_pool[:n_bridges]:
            if len(groups) >= MAX_GROUPS:
                break
            a = cluster_centers[a_idx]
            b = cluster_centers[b_idx]
            bx = (a[0] + b[0]) * 0.5
            by = (a[1] + b[1]) * 0.5
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            bridge_elev = elev_seed + random.uniform(0.04, 0.18)
            main_beam = _new_shape(
                kind="rect",
                x=bx, y=by,
                w=max(0.04, dist * random.uniform(0.95, 1.15)),
                h=random.uniform(0.012, 0.030),
                rotation=angle,
                depth=random.uniform(0.012, 0.035),
                elevation=bridge_elev,
            )
            main_beam.detail = random.choice(["lines_h", "cross_brace", "grid_frame", "none"])
            bridge_shapes = [main_beam]
            # Support piers at connection points
            for t in [0.15, 0.85]:
                px = a[0] + (b[0] - a[0]) * t
                py = a[1] + (b[1] - a[1]) * t
                pier = _new_shape(
                    "rect", px, py,
                    random.uniform(0.010, 0.020),
                    random.uniform(0.015, 0.030),
                    angle + 90,
                    random.uniform(0.008, 0.020),
                    elev_seed,
                )
                pier.detail = "none"
                bridge_shapes.append(pier)
            _append_group(bridge_shapes, bx, by, angle)

    # --- 5) Sweeping blades — anchored to clusters, not random ---
    n_sweeps = random.randint(1, 2)
    for _ in range(n_sweeps):
        if len(groups) >= MAX_GROUPS or not cluster_centers:
            break
        anchor_idx = random.randrange(len(cluster_centers))
        anchor = cluster_centers[anchor_idx]
        sw_angle = cluster_angles[anchor_idx] + random.gauss(0, 15)
        sw_cx = anchor[0] + random.gauss(0, 0.03)
        sw_cy = anchor[1] + random.gauss(0, 0.02)
        sw_kind = random.choices(
            ["rect", "trapezoid", "parallelogram", "chevron", "arrow"],
            weights=[35, 25, 18, 12, 10],
        )[0]
        sweep = _new_shape(
            kind=sw_kind,
            x=sw_cx, y=sw_cy,
            w=random.uniform(0.14, 0.45),
            h=random.uniform(0.020, 0.08),
            rotation=sw_angle,
            depth=random.uniform(0.008, 0.040),
            elevation=elev_seed + random.uniform(0.0, 0.15),
        )
        _assign_detail(sweep)
        sweep_group = [sweep]
        if random.random() < 0.40:
            acc = _new_shape(
                "rect", sw_cx + random.gauss(0, 0.015),
                sw_cy + random.gauss(0, 0.012),
                sweep.w * random.uniform(0.5, 0.8),
                random.uniform(0.006, 0.016),
                sw_angle + random.gauss(0, 2),
                random.uniform(0.006, 0.020),
                sweep.elevation + random.uniform(0.01, 0.04),
            )
            sweep_group.append(acc)
        _append_group(sweep_group, sw_cx, sw_cy, sw_angle)

    # --- 6) Satellites — tightly orbit their parent cluster ---
    n_sats = random.randint(0, 2)
    _SAT_KINDS = ["rect", "square", "trapezoid", "hexagon",
                  "parallelogram", "cross", "chevron"]
    for _ in range(n_sats):
        near_cx, near_cy = random.choice(cluster_centers) if cluster_centers else (0.5, 0.5)
        sat = _new_shape(
            kind=random.choice(_SAT_KINDS),
            x=near_cx + random.gauss(0, 0.04),
            y=near_cy + random.gauss(0, 0.04),
            w=random.uniform(0.020, 0.07),
            h=random.uniform(0.015, 0.045),
            rotation=deck_angle + random.gauss(0, 8),
            depth=random.uniform(0.012, 0.035),
            elevation=elev_seed + random.uniform(0.0, 0.15),
        )
        _assign_detail(sat)
        satellites.append(sat)

    # --- 7) Shared ground plane under the whole composition ---
    if cluster_centers and len(groups) < MAX_GROUPS:
        all_cx = [c[0] for c in cluster_centers]
        all_cy = [c[1] for c in cluster_centers]
        centroid_x = sum(all_cx) / len(all_cx)
        centroid_y = sum(all_cy) / len(all_cy)
        spread_x = max(all_cx) - min(all_cx)
        spread_y = max(all_cy) - min(all_cy)
        ground = _new_shape(
            kind="rect",
            x=centroid_x, y=centroid_y,
            w=spread_x + random.uniform(0.08, 0.18),
            h=spread_y + random.uniform(0.05, 0.12),
            rotation=deck_angle + random.gauss(0, 3),
            depth=random.uniform(0.004, 0.012),
            elevation=max(0.0, elev_seed - 0.02),
        )
        ground.opacity = random.uniform(0.15, 0.40)
        ground.detail = "none"
        groups.insert(0, ShapeGroup(
            anchor=ground,
            members=[],
            cx=centroid_x,
            cy=centroid_y,
            angle=ground.rotation,
        ))

    bg_style = random.choices(
        [0, 1, 2, 3, 4, 5, 6, 7],
        weights=[28, 6, 18, 8, 12, 10, 10, 8],
    )[0]
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
        bg_blend=random.uniform(0.02, 0.20) if random.random() < 0.35 else 0.0,
        projection_angle=proj_angle,
        projection_strength=proj_strength,
        projection_secondary_angle=proj2_angle,
        projection_secondary_strength=proj2_strength,
        projection_vanishing_x=vanish_x,
        projection_vanishing_y=vanish_y,
        camera_mode=camera_mode,
        perspective_bias=perspective_bias,
        armature_angle=armature_angle,
        focal_x=focal_x,
        focal_y=focal_y,
        density_falloff=density_falloff,
        engine_tag="hadid",
    )
    _apply_gravitational_alignment(genome)
    _apply_focal_weighting(genome)
    _solidify_architecton(genome)
    _enforce_3d_bounds(genome)
    return genome


# ======================================================================
# Map creation (abstract city-plan compositions)
# ======================================================================

def create_map(num_palette_colors: int = 12) -> ShapeGenome:
    """Create an abstract city-plan genome inspired by Lutyens' Delhi.

    Nearly flat plan view.  Thin road lines form a connected network;
    large flat building blocks and gardens fill the spaces between
    roads densely.  Minimal depth/elevation — just enough for subtle
    relief.
    """
    groups: list[ShapeGroup] = []
    satellites: list[Shape] = []

    palette_size = min(num_palette_colors, random.randint(6, 10))
    sub_indices = random.sample(range(num_palette_colors), palette_size)

    # Indices into MAPS_PALETTE:
    # 0=off_white 1=charcoal 2=silver 3=signal_red 4=cobalt_blue
    # 5=orange 6=white 7=ochre_yellow 8=deep_red 9=cool_grey
    # 10=pale_blue 11=graphite
    road_colors = [i for i in [2, 1, 8] if i < num_palette_colors] or [0]
    garden_colors = [i for i in [10, 4] if i < num_palette_colors] or [0]
    building_colors = [i for i in [3, 5, 6, 4, 9] if i < num_palette_colors] or sub_indices[:3]
    monument_colors = [i for i in [7, 3, 6] if i < num_palette_colors] or sub_indices[:2]
    bg_candidates = [i for i in [1, 11, 0] if i < num_palette_colors] or [0]

    _BK = ["rect", "rect", "rect", "lshape", "tshape", "cross",
           "hexagon", "square", "trapezoid"]

    scene_style = random.choices(
        ["axial_plan", "radial_plan", "grid_plan", "hexagonal_plan"],
        weights=[35, 25, 25, 15],
    )[0]

    if scene_style == "axial_plan":
        map_armature = random.uniform(-5, 5)
        map_focal_x = random.uniform(0.42, 0.58)
        map_focal_y = 0.5
    elif scene_style == "radial_plan":
        map_armature = random.uniform(0, 360)
        map_focal_x = random.uniform(0.40, 0.60)
        map_focal_y = random.uniform(0.40, 0.60)
    elif scene_style == "grid_plan":
        map_armature = random.uniform(-12, 12)
        map_focal_x = random.uniform(0.38, 0.62)
        map_focal_y = random.uniform(0.38, 0.62)
    else:
        map_armature = random.uniform(0, 60)
        map_focal_x = random.uniform(0.42, 0.58)
        map_focal_y = random.uniform(0.42, 0.58)
    map_density_falloff = random.uniform(0.15, 0.45)

    # Flat 2D plan view — no 3D projection at all
    camera_mode = 0
    proj_angle = 30.0
    proj_strength = 0.0
    proj2_angle = -35.0
    proj2_strength = 0.0
    vanish_x = 0.5
    vanish_y = 0.5
    perspective_bias = 0.0

    _MAP_MIN = 0.003

    # ---- flat factory helpers ----
    def _road(x, y, length, width, angle):
        s = Shape(kind="rect", x=_clamp01(x), y=_clamp01(y),
                  w=_clamp(length, _MAP_MIN, MAX_SIZE),
                  h=_clamp(width, _MAP_MIN, MAX_SIZE),
                  rotation=angle % 360,
                  color_idx=random.choice(road_colors),
                  opacity=random.uniform(0.85, 1.0), material="concrete",
                  depth=0.0, elevation=0.0)
        _ensure_inside(s)
        return s

    def _rb(x1, y1, x2, y2, w=0.005):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return _road(mx, my, math.hypot(x2 - x1, y2 - y1) + 0.004, w,
                     math.degrees(math.atan2(y2 - y1, x2 - x1)))

    def _bldg(x, y, w, h, angle):
        s = Shape(kind=random.choices(_BK, weights=[20, 20, 20, 8, 6, 5, 6, 8, 7])[0],
                  x=_clamp01(x), y=_clamp01(y),
                  w=_clamp(w, _MAP_MIN, MAX_SIZE),
                  h=_clamp(h, _MAP_MIN, MAX_SIZE),
                  rotation=angle % 360,
                  color_idx=random.choice(building_colors),
                  opacity=random.uniform(0.88, 1.0), material="concrete",
                  depth=0.0, elevation=0.0)
        _ensure_inside(s)
        return s

    def _park(x, y, w, h, angle):
        s = Shape(kind=random.choice(["rect", "ellipse", "rect"]),
                  x=_clamp01(x), y=_clamp01(y),
                  w=_clamp(w, _MAP_MIN, MAX_SIZE),
                  h=_clamp(h, _MAP_MIN, MAX_SIZE),
                  rotation=angle % 360,
                  color_idx=random.choice(garden_colors),
                  opacity=random.uniform(0.70, 0.90), material="marble",
                  depth=0.0, elevation=0.0)
        _ensure_inside(s)
        return s

    def _mon(x, y, size, angle):
        s = Shape(kind=random.choice(["cross", "hexagon", "circle", "square"]),
                  x=_clamp01(x), y=_clamp01(y),
                  w=_clamp(size, _MAP_MIN, MAX_SIZE),
                  h=_clamp(size * random.uniform(0.85, 1.0), _MAP_MIN, MAX_SIZE),
                  rotation=angle % 360,
                  color_idx=random.choice(monument_colors),
                  opacity=1.0, material="marble",
                  depth=0.0, elevation=0.0)
        _ensure_inside(s)
        return s

    def _ag(shapes, cx, cy, angle):
        if shapes and len(groups) < MAX_GROUPS:
            groups.append(ShapeGroup(
                anchor=shapes[0], members=shapes[1:],
                cx=_clamp01(cx), cy=_clamp01(cy), angle=angle))

    # ==================================================================
    # AXIAL PLAN  (Lutyens' Delhi reference)
    # ==================================================================
    if scene_style == "axial_plan":
        ax_cx = random.uniform(0.44, 0.56)
        ax_a = random.uniform(-5, 5)
        ar = math.radians(ax_a)
        aco, asi = math.cos(ar), math.sin(ar)
        # perpendicular unit
        pco, psi = -asi, aco

        # Central ceremonial avenue
        satellites.append(_road(ax_cx, 0.5, 0.88, 0.008, ax_a))
        # Flanking secondary lanes
        for side in [-1, 1]:
            off = random.uniform(0.025, 0.04)
            satellites.append(
                _road(ax_cx + side * off * pco, 0.5 + side * off * psi,
                      0.82, 0.004, ax_a))

        # Cross-streets at regular intervals
        n_cross = random.randint(5, 9)
        cross_ys: list[float] = []
        for i in range(n_cross):
            t = (i + 0.5) / n_cross
            cy = 0.06 + t * 0.88
            cross_ys.append(cy)
            cw = random.uniform(0.35, 0.90)
            satellites.append(_road(ax_cx, cy, cw, 0.005, ax_a + 90))

        # Grand monument at one terminus
        mon_cy = random.choice([0.06, 0.94])
        satellites.append(_mon(ax_cx, mon_cy, random.uniform(0.06, 0.10), ax_a))
        for side in [-1, 1]:
            satellites.append(
                _bldg(ax_cx + side * random.uniform(0.06, 0.10), mon_cy,
                      random.uniform(0.04, 0.07), random.uniform(0.03, 0.05),
                      ax_a))
            satellites.append(
                _rb(ax_cx, mon_cy,
                    ax_cx + side * 0.10, mon_cy, w=0.004))

        # Dense buildings + parks flanking the axis between cross-streets
        for i in range(len(cross_ys) - 1):
            yt, yb = cross_ys[i], cross_ys[i + 1]
            yc = (yt + yb) / 2
            span_y = yb - yt
            for side in [-1, 1]:
                # Near-axis garden
                gx = ax_cx + side * random.uniform(0.04, 0.08)
                satellites.append(
                    _park(gx, yc, random.uniform(0.04, 0.08),
                          span_y * random.uniform(0.55, 0.80), ax_a))

                # Building blocks in 2-3 bands
                for band in range(random.randint(2, 3)):
                    bx = ax_cx + side * (0.10 + band * random.uniform(0.07, 0.12))
                    bw = random.uniform(0.05, 0.10)
                    bh = span_y * random.uniform(0.50, 0.85)
                    satellites.append(_bldg(bx, yc, bw, bh, ax_a + random.gauss(0, 2)))
                    # Side-street connecting this block back to axis
                    satellites.append(_rb(bx, yc, ax_cx, yc, w=0.004))

                    # Sub-blocks within larger blocks
                    if random.random() < 0.5:
                        for _ in range(random.randint(1, 3)):
                            sx = bx + random.gauss(0, bw * 0.3)
                            sy = yc + random.gauss(0, bh * 0.3)
                            satellites.append(
                                _bldg(sx, sy, bw * random.uniform(0.25, 0.45),
                                      bh * random.uniform(0.2, 0.4),
                                      ax_a + random.gauss(0, 3)))

        # Diagonal avenues radiating from key intersections
        for _ in range(random.randint(3, 6)):
            da = ax_a + random.choice([30, 45, 60, -30, -45, -60])
            dr = math.radians(da)
            oy = random.choice(cross_ys)
            dlen = random.uniform(0.18, 0.40)
            ex = _clamp01(ax_cx + dlen * math.cos(dr))
            ey = _clamp01(oy + dlen * math.sin(dr))
            satellites.append(_rb(ax_cx, oy, ex, ey, w=0.005))
            satellites.append(
                _bldg(ex, ey, random.uniform(0.03, 0.06),
                      random.uniform(0.02, 0.04), da))

        # Outer residential blocks
        for _ in range(random.randint(8, 16)):
            bx = _clamp01(ax_cx + random.choice([-1, 1]) * random.uniform(0.30, 0.46))
            by = random.uniform(0.08, 0.92)
            satellites.append(
                _bldg(bx, by, random.uniform(0.03, 0.07),
                      random.uniform(0.02, 0.05),
                      ax_a + random.choice([0, 90]) + random.gauss(0, 5)))

    # ==================================================================
    # RADIAL PLAN
    # ==================================================================
    elif scene_style == "radial_plan":
        hx = random.uniform(0.40, 0.60)
        hy = random.uniform(0.40, 0.60)
        n_boul = random.randint(6, 10)
        spin = random.uniform(0, 360)
        hub_r = random.uniform(0.020, 0.035)

        satellites.append(_mon(hx, hy, hub_r * 2.5, 0))

        boul_angles: list[float] = []
        for i in range(n_boul):
            adeg = spin + 360.0 * i / n_boul + random.gauss(0, 2)
            boul_angles.append(adeg)
            arad = math.radians(adeg)
            blen = random.uniform(0.30, 0.46)
            ex = _clamp01(hx + math.cos(arad) * blen)
            ey = _clamp01(hy + math.sin(arad) * blen)
            satellites.append(_rb(hx, hy, ex, ey, w=0.006))

        # Concentric ring roads
        n_rings = random.randint(2, 5)
        ring_radii: list[float] = []
        for ri in range(1, n_rings + 1):
            rr = hub_r * 2.5 + ri * random.uniform(0.05, 0.09)
            ring_radii.append(rr)
            n_seg = 24
            pts = [(_clamp01(hx + rr * math.cos(2 * math.pi * s / n_seg)),
                     _clamp01(hy + rr * math.sin(2 * math.pi * s / n_seg)))
                    for s in range(n_seg)]
            for s in range(n_seg):
                p1, p2 = pts[s], pts[(s + 1) % n_seg]
                satellites.append(_rb(p1[0], p1[1], p2[0], p2[1], w=0.004))

        # Fill sectors between boulevards and rings with buildings/parks
        for bi in range(n_boul):
            a1 = math.radians(boul_angles[bi])
            a2 = math.radians(boul_angles[(bi + 1) % n_boul])
            mid_a = (a1 + a2) / 2.0
            sector_span = abs(a2 - a1)
            for rr in ring_radii:
                bx = _clamp01(hx + math.cos(mid_a) * rr)
                by = _clamp01(hy + math.sin(mid_a) * rr)
                arc_w = rr * sector_span * 0.6
                is_park = random.random() < 0.20
                bw = _clamp(arc_w, 0.04, 0.10)
                bh = random.uniform(0.03, 0.07)
                if is_park:
                    satellites.append(
                        _park(bx, by, bw, bh, math.degrees(mid_a)))
                else:
                    satellites.append(
                        _bldg(bx, by, bw, bh,
                              math.degrees(mid_a) + random.gauss(0, 5)))
                # Infill buildings in the wedge
                for _ in range(random.randint(1, 3)):
                    ja = random.uniform(a1, a2)
                    jr = rr * random.uniform(0.7, 1.2)
                    satellites.append(
                        _bldg(_clamp01(hx + math.cos(ja) * jr),
                              _clamp01(hy + math.sin(ja) * jr),
                              random.uniform(0.025, 0.055),
                              random.uniform(0.020, 0.045),
                              math.degrees(ja) + random.gauss(0, 8)))

        # Secondary roundabouts at boulevard ends
        for adeg in boul_angles[:random.randint(3, n_boul)]:
            arad = math.radians(adeg)
            r_end = ring_radii[-1] if ring_radii else 0.25
            ex = _clamp01(hx + math.cos(arad) * r_end)
            ey = _clamp01(hy + math.sin(arad) * r_end)
            satellites.append(_mon(ex, ey, random.uniform(0.012, 0.022), 0))

    # ==================================================================
    # GRID PLAN
    # ==================================================================
    elif scene_style == "grid_plan":
        gcx = random.uniform(0.38, 0.62)
        gcy = random.uniform(0.38, 0.62)
        ga = random.uniform(-12, 12)
        gar = math.radians(ga)
        gac, gas = math.cos(gar), math.sin(gar)
        rows = random.randint(5, 9)
        cols = random.randint(5, 9)
        sx = random.uniform(0.07, 0.11)
        sy = random.uniform(0.06, 0.09)
        x0 = gcx - (cols - 1) * sx / 2.0
        y0 = gcy - (rows - 1) * sy / 2.0

        # Horizontal streets
        rg: list[Shape] = []
        for r in range(rows + 1):
            ay = y0 - sy * 0.5 + r * sy
            rg.append(_road(gcx, ay, cols * sx + 0.04,
                            0.005 if r % 3 != 0 else 0.008, ga))
        _ag(rg, gcx, gcy, ga)

        # Vertical streets
        rg2: list[Shape] = []
        for c in range(cols + 1):
            ax = x0 - sx * 0.5 + c * sx
            rg2.append(_road(ax, gcy,
                             0.005 if c % 3 != 0 else 0.008,
                             rows * sy + 0.04, ga))
        _ag(rg2, gcx, gcy, ga)

        # Diagonals
        for _ in range(random.randint(1, 3)):
            da = ga + random.choice([30, 45, -30, -45])
            satellites.append(_road(gcx, gcy, random.uniform(0.30, 0.65), 0.006, da))

        # City blocks
        for r in range(rows):
            for c in range(cols):
                bx = x0 + c * sx
                by = y0 + r * sy
                is_park = random.random() < 0.15
                bw = sx * random.uniform(0.55, 0.88)
                bh = sy * random.uniform(0.55, 0.88)
                if is_park:
                    satellites.append(_park(bx, by, bw, bh, ga))
                else:
                    satellites.append(_bldg(bx, by, bw, bh, ga + random.gauss(0, 2)))
                    if random.random() < 0.50:
                        for _ in range(random.randint(1, 2)):
                            ox = random.uniform(-0.3, 0.3) * bw
                            oy = random.uniform(-0.3, 0.3) * bh
                            satellites.append(
                                _bldg(bx + ox * gac - oy * gas,
                                      by + ox * gas + oy * gac,
                                      bw * random.uniform(0.25, 0.45),
                                      bh * random.uniform(0.25, 0.45),
                                      ga + random.gauss(0, 3)))

        # Monuments
        for _ in range(random.randint(1, 3)):
            mx = x0 + random.randint(0, cols - 1) * sx
            my = y0 + random.randint(0, rows - 1) * sy
            satellites.append(_mon(mx, my, random.uniform(0.04, 0.08), ga))

    # ==================================================================
    # HEXAGONAL PLAN
    # ==================================================================
    else:
        hcx = random.uniform(0.42, 0.58)
        hcy = random.uniform(0.42, 0.58)
        n_rings = random.randint(2, 4)
        hspin = random.uniform(0, 60)

        satellites.append(_mon(hcx, hcy, random.uniform(0.035, 0.055), 0))

        for ring in range(1, n_rings + 1):
            radius = ring * random.uniform(0.09, 0.13)
            pts = [(_clamp01(hcx + math.cos(math.radians(hspin + v * 60)) * radius),
                     _clamp01(hcy + math.sin(math.radians(hspin + v * 60)) * radius))
                    for v in range(6)]

            for v in range(6):
                p1, p2 = pts[v], pts[(v + 1) % 6]
                satellites.append(_rb(p1[0], p1[1], p2[0], p2[1], w=0.005))
            for vx, vy in pts:
                satellites.append(_rb(hcx, hcy, vx, vy, w=0.004))

            for v in range(6):
                vx, vy = pts[v]
                p2 = pts[(v + 1) % 6]
                mx, my = (vx + p2[0]) / 2, (vy + p2[1]) / 2
                is_park = random.random() < 0.22
                bw = random.uniform(0.04, 0.08)
                bh = random.uniform(0.035, 0.07)
                if is_park:
                    satellites.append(_park(vx, vy, bw, bh, hspin + v * 60))
                else:
                    satellites.append(_bldg(vx, vy, bw, bh, hspin + v * 60 + random.gauss(0, 3)))
                satellites.append(
                    _bldg(mx, my, random.uniform(0.035, 0.065),
                          random.uniform(0.030, 0.055),
                          hspin + v * 60 + 30 + random.gauss(0, 3)))
                # Infill between ring and inner ring
                if ring > 1:
                    ir = (ring - 0.5) * random.uniform(0.09, 0.13)
                    mid_a = math.radians(hspin + v * 60 + 30)
                    ix = _clamp01(hcx + math.cos(mid_a) * ir)
                    iy = _clamp01(hcy + math.sin(mid_a) * ir)
                    satellites.append(
                        _bldg(ix, iy, random.uniform(0.03, 0.06),
                              random.uniform(0.025, 0.05),
                              hspin + v * 60 + 30 + random.gauss(0, 5)))

    # --- Background ---
    bg_style = random.choices([2, 2, 6, 1], weights=[40, 20, 20, 20])[0]
    bg_idx = random.choice(bg_candidates)
    bg_sec_idx = random.choice([i for i in sub_indices if i != bg_idx] or sub_indices)

    genome = ShapeGenome(
        groups=groups, satellites=satellites,
        bg_style=bg_style, bg_color_idx=bg_idx,
        bg_angle=random.uniform(0, 360),
        bg_cx=random.uniform(0.25, 0.75), bg_cy=random.uniform(0.25, 0.75),
        bg_sec_idx=bg_sec_idx,
        bg_blend=random.uniform(0.01, 0.08) if random.random() < 0.3 else 0.0,
        projection_angle=proj_angle, projection_strength=proj_strength,
        projection_secondary_angle=proj2_angle,
        projection_secondary_strength=proj2_strength,
        projection_vanishing_x=vanish_x, projection_vanishing_y=vanish_y,
        camera_mode=camera_mode, perspective_bias=perspective_bias,
        armature_angle=map_armature,
        focal_x=map_focal_x, focal_y=map_focal_y,
        density_falloff=map_density_falloff,
        engine_tag="maps",
    )
    for shape in genome.flatten():
        _ensure_inside(shape)
    return genome

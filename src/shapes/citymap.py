"""Abstract city maps: a dense tiling of painted lots, cut by streets.

In the spirit of modern abstract city maps and of Substrate: the sheet is
packed with small hard-edged parcels; the paper ground shows as the streets,
the water and a few open squares.  Colour is Constructivist — lamp-black,
vermilion, ultramarine, chrome yellow and warm greys on paper.

The genome is a small set of planning genes, not a list of shapes:

* ``plan``    the city type — ``grid`` (one street grid), ``patchwork``
              (districts with their own grids), ``radial`` (ring-and-radial,
              like Moscow), ``substrate`` (recursive irregular splits) or
              ``slabs`` (Constructivist superblocks of long slab buildings).
* ``cuts``    straight arterials.  Applied in order, each splits the districts
              its segment crosses (a later cut stops at earlier ones, giving
              T-junctions); each district gets its own street grid angle.
* ``hub``     ring roads and radial avenues (``radial`` plan).
* ``waters``  rivers (with bridges and islands), coasts (with piers) and
              lakes, painted over the tiling so blocks end against them.
* ``avenues``, ``rails``, ``plazas``  curved boulevards, railways and round
              or square landmark squares.
* scalar genes for scale, block / lot / street size, irregularity and
              colour weights.

``express_city`` turns the genes into polygons deterministically (seeded),
so a picked map renders identically in the next generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops, ImageDraw

# Indices into MAPS_PALETTE (src/art/palettes.py).
PAPER, BLACK, VERMILION, ULTRA, CHROME, WARM_GREY, DARK_GREY, IVORY, CARMINE, \
    COBALT, OCHRE, VIRIDIAN, TEAL, SALMON, PLUM, SIENNA, NAVY, SKY, KRAFT, \
    PALE_GREY, OLIVE, ROSE, ORANGE = range(23)
N_COLORS = 23
NEUTRALS = (BLACK, DARK_GREY, WARM_GREY, IVORY, PALE_GREY)
CHROMAS = (VERMILION, ULTRA, CHROME, CARMINE, COBALT, OCHRE, VIRIDIAN, TEAL,
           SALMON, PLUM, SIENNA, SKY, OLIVE, ROSE, ORANGE)

# Named palettes after Constructivist artists and media: (ground choices,
# lead, colour weights).  Neutrals carry most of each sheet.
PALETTE_SETS: dict[str, tuple[tuple[int, ...], int, dict[int, float]]] = {
    "rodchenko":  ((PAPER, IVORY), VERMILION,
                   {BLACK: 3.0, VERMILION: 2.4, WARM_GREY: 1.2, IVORY: 0.6}),
    "lissitzky":  ((PAPER, IVORY), VERMILION,
                   {BLACK: 3.0, DARK_GREY: 2.2, WARM_GREY: 2.0, IVORY: 0.6, VERMILION: 1.0}),
    "klutsis":    ((PAPER,), VERMILION,
                   {BLACK: 2.5, VERMILION: 1.8, ULTRA: 1.5, CHROME: 1.3, IVORY: 0.8,
                    WARM_GREY: 0.6}),
    "popova":     ((PAPER, KRAFT), SIENNA,
                   {BLACK: 2.0, SIENNA: 1.6, OCHRE: 1.4, VIRIDIAN: 1.1, NAVY: 1.0,
                    IVORY: 0.7, WARM_GREY: 0.8}),
    "stepanova":  ((PAPER, PALE_GREY), VIRIDIAN,
                   {BLACK: 2.4, VIRIDIAN: 1.4, SALMON: 1.3, IVORY: 1.0, DARK_GREY: 1.0}),
    "stenberg":   ((PAPER, IVORY), TEAL,
                   {BLACK: 2.2, TEAL: 1.5, VERMILION: 1.2, PLUM: 1.0, CHROME: 0.5,
                    PALE_GREY: 1.0}),
    "exter":      ((PAPER,), ULTRA,
                   {ULTRA: 1.8, VERMILION: 1.3, CHROME: 0.9, OCHRE: 1.0, SKY: 1.2,
                    BLACK: 1.6, IVORY: 0.6}),
    "suetin":     ((PAPER, IVORY), OCHRE,
                   {OCHRE: 1.4, SALMON: 1.1, VIRIDIAN: 0.8, BLACK: 1.2, IVORY: 1.3,
                    SIENNA: 1.0, SKY: 0.8, PALE_GREY: 1.0}),
    "kraft":      ((KRAFT,), VERMILION,
                   {BLACK: 3.0, VERMILION: 1.6, IVORY: 1.2, DARK_GREY: 1.0}),
    "blueprint":  ((NAVY, ULTRA), VERMILION,
                   {IVORY: 2.4, SKY: 1.8, PALE_GREY: 1.6, VERMILION: 0.5, CHROME: 0.3}),
    "night":      ((BLACK,), CHROME,
                   {IVORY: 2.4, WARM_GREY: 1.8, DARK_GREY: 0.8, CHROME: 1.0, VERMILION: 0.8}),
    "mono":       ((PAPER, IVORY, PALE_GREY), VERMILION,
                   {BLACK: 3.0, DARK_GREY: 2.5, WARM_GREY: 2.5, IVORY: 1.2, VERMILION: 0.2}),
    "poster":     ((CHROME, SALMON), BLACK,
                   {BLACK: 3.0, VERMILION: 1.2, IVORY: 1.0, ULTRA: 0.5}),
    "olive":      ((PAPER, KRAFT), OLIVE,
                   {OLIVE: 1.6, BLACK: 2.2, VERMILION: 0.9, PALE_GREY: 1.2, OCHRE: 0.8,
                    IVORY: 0.6}),
}
PALETTE_SETS.update({
    # Richer, more saturated harmonies.
    "rodchenko_pink": ((PAPER, IVORY), ROSE,
                       {BLACK: 2.6, ROSE: 1.6, VERMILION: 1.4, IVORY: 0.8, DARK_GREY: 0.8}),
    "cobalt_orange":  ((PAPER, IVORY), ORANGE,
                       {COBALT: 1.8, ULTRA: 1.0, ORANGE: 1.4, NAVY: 1.2, CHROME: 0.6,
                        IVORY: 0.8, BLACK: 1.0}),
    "teal_coral":     ((PAPER, PALE_GREY), SALMON,
                       {TEAL: 1.8, SALMON: 1.4, NAVY: 1.2, IVORY: 0.9, CHROME: 0.5,
                        BLACK: 0.8}),
    "plum_ochre":     ((PAPER, KRAFT), OCHRE,
                       {PLUM: 1.6, OCHRE: 1.4, ROSE: 1.0, BLACK: 1.4, PALE_GREY: 1.0}),
    "agitprop":       ((PAPER,), VERMILION,
                       {VERMILION: 2.0, CARMINE: 1.2, ORANGE: 1.0, BLACK: 2.4, IVORY: 0.8,
                        CHROME: 0.6}),
    "sea_and_sun":    ((IVORY, PAPER), CHROME,
                       {ULTRA: 1.5, SKY: 1.4, TEAL: 1.0, CHROME: 1.2, ORANGE: 0.7,
                        NAVY: 1.0}),
})
_PALETTE_WEIGHTS = {"rodchenko_pink": 7, "cobalt_orange": 7, "teal_coral": 7,
                    "plum_ochre": 6, "agitprop": 7, "sea_and_sun": 6,
                    "rodchenko": 10, "lissitzky": 8, "klutsis": 9, "popova": 7,
                    "stepanova": 6, "stenberg": 6, "exter": 6, "suetin": 6, "kraft": 5,
                    "blueprint": 4, "night": 3, "mono": 6, "poster": 3, "olive": 4}

PLANS = ("grid", "patchwork", "radial", "substrate", "slabs")
_PLAN_WEIGHTS = (22, 26, 16, 18, 18)
WATER_KINDS = ("river", "coast", "lake")

MAX_CUTS = 9
MAX_AVENUES = 4
MAX_WATERS = 3
MAX_PLAZAS = 4
MAX_ZONES = 7
MAX_COMPLEXES = 6
ZONE_KINDS = ("fine", "coarse", "substrate", "slabs", "solid", "tint", "turn")
ORN_KINDS = ("court", "split", "stripe", "dot", "frame")
PRIM_KINDS = ("disc", "semi", "tri", "ellipse", "ring", "notch", "wedge")
MAX_LOTS = 14000           # keeps worst-case renders bounded
DOMAIN_PAD = 0.03


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------
# Genome
# ----------------------------------------------------------------------

@dataclass
class CityGenome:
    seed: int = 0                      # block / lot geometry
    color_seed: int = 0                # colouring only
    plan: str = "patchwork"
    base_angle: float = 0.0            # main grid direction (degrees)
    angle_spread: float = 30.0         # how far district grids turn from it
    zoom: float = 1.0                  # overall scale of blocks and lots
    block: float = 0.09                # long side of a typical block
    aspect: float = 2.2                # block long / short side
    street: float = 0.0065             # street width
    lot: float = 0.022                 # typical lot frontage
    irregular: float = 0.25            # share of districts split Substrate-style
    slabs: float = 0.0                 # share of blocks built as slab rows
    solid: float = 0.12                # share of blocks painted as one mass
    voids: float = 0.03                # share of blocks left open (squares)
    region: float = 0.5                # colour clustering by district
    lead: int = VERMILION              # the strong colour
    lead_weight: float = 1.0
    ground: int = PAPER                # paper (streets)
    palette: str = "rodchenko"         # name of the palette set it came from
    water_color: int = -1              # -1 = same as the ground
    weights: list[float] = field(default_factory=list)
    cuts: list[dict] = field(default_factory=list)
    hub: dict | None = None
    waters: list[dict] = field(default_factory=list)
    avenues: list[dict] = field(default_factory=list)
    rails: list[dict] = field(default_factory=list)
    plazas: list[dict] = field(default_factory=list)
    complexes: list[dict] = field(default_factory=list)
    # Complexification genes: these start empty and accumulate down a
    # lineage (see ``complexify``), so picked maps grow more elaborate.
    zones: list[dict] = field(default_factory=list)    # neighbourhood character
    ornament: float = 0.0              # share of lots with inner detail
    orn_kinds: list[str] = field(default_factory=list)
    prims: list[str] = field(default_factory=list)     # non-rectangular buildings
    prim_share: float = 0.0
    relief: float = 0.0                # 0 flat plan .. 1.4 tall extruded blocks
    view: float = -60.0                # direction roofs are lifted (degrees)
    tonal: float = 0.0                 # light/dark range within each colour

    def copy(self) -> CityGenome:
        return CityGenome.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        def plain(v):
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            return v
        return {k: plain(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> CityGenome:
        g = cls()
        for k, v in d.items():
            if k == "river":                      # older single-river genomes
                if v:
                    g.waters = [dict(v, kind="river", bridges=0)]
                continue
            if hasattr(g, k):
                setattr(g, k, _deep_copy(v))
        if not g.weights:
            g.weights = _palette_weights("lissitzky", g)
        g.weights = (list(g.weights) + [0.0] * N_COLORS)[:N_COLORS]
        if g.plan not in PLANS:
            g.plan = "patchwork"
        return g


def _deep_copy(v):
    if isinstance(v, (list, tuple)):
        return [_deep_copy(x) for x in v]
    if isinstance(v, dict):
        return {k: _deep_copy(x) for k, x in v.items()}
    return v


# ----------------------------------------------------------------------
# Random genes
# ----------------------------------------------------------------------

def _palette_weights(name: str, g: CityGenome, ground: bool = True) -> list[float]:
    """Colour weights for a named palette (jittered); sets lead and ground."""
    grounds, lead, base = PALETTE_SETS[name]
    g.palette = name
    g.lead = lead
    if ground:
        g.ground = random.choice(grounds)
    w = [0.0] * N_COLORS
    for i, v in base.items():
        w[i] = v * random.uniform(0.6, 1.4)
    w[lead] = 0.0                       # the lead gets lead_weight instead
    g.lead_weight = base.get(lead, 1.0) * random.uniform(0.25, 0.6)
    w[g.ground] = 0.0
    if g.lead == g.ground:
        g.lead = BLACK if g.ground != BLACK else IVORY
    return w


def _new_cut(base_angle: float, spread: float, long: bool = True) -> dict:
    a = base_angle + random.choice([0, 90, spread, -spread, 90 + spread, 45]) \
        + random.gauss(0, 4)
    return {"x": random.uniform(0.1, 0.9), "y": random.uniform(0.1, 0.9),
            "angle": a % 180,
            "length": random.uniform(0.9, 1.6) if long else random.uniform(0.25, 0.7),
            "width": random.uniform(0.008, 0.016)}


def _new_curve(width: float, n: int = 4, wobble: float = 0.12) -> dict:
    """A bending curve entering and leaving the sheet."""
    side = random.random() < 0.5
    t0, t1 = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
    start = (t0, -0.1) if side else (-0.1, t0)
    end = (t1, 1.1) if side else (1.1, t1)
    pts = [start]
    for i in range(1, n - 1):
        f = i / (n - 1)
        pts.append((start[0] + (end[0] - start[0]) * f + random.gauss(0, wobble),
                    start[1] + (end[1] - start[1]) * f + random.gauss(0, wobble)))
    pts.append(end)
    return {"pts": pts, "width": width}


def _new_water(kind: str | None = None) -> dict:
    kind = kind or random.choices(WATER_KINDS, weights=[60, 22, 18])[0]
    if kind == "river":
        w = _new_curve(random.uniform(0.04, 0.12), n=random.randint(3, 6),
                       wobble=random.uniform(0.05, 0.16))
        w.update(kind="river", bridges=random.choice([0, 2, 3, 3, 4, 5, 6]),
                 bridge_style=random.choice([0, 0, 1]),
                 island=random.random() < 0.3)
        return w
    if kind == "coast":
        w = _new_curve(0.0, n=random.randint(3, 5), wobble=0.07)
        # Pull the coast toward one edge so the sea takes a quarter to a half.
        edge = random.uniform(0.15, 0.4)
        horiz = abs(w["pts"][0][0] - w["pts"][-1][0]) < 0.6   # runs top->bottom
        w["pts"] = [((edge if horiz else x) + (x - 0.5) * 0.4 if horiz else x,
                     y if horiz else edge + (y - 0.5) * 0.4)
                    for x, y in w["pts"]]
        w.update(kind="coast", side=random.choice([-1, 1]),
                 piers=random.random() < 0.7, pier_seed=random.randrange(1 << 20))
        return w
    return {"kind": "lake", "x": random.uniform(0.2, 0.8), "y": random.uniform(0.2, 0.8),
            "r": random.uniform(0.08, 0.2),
            "lobes": [random.uniform(0.7, 1.3) for _ in range(random.randint(5, 9))],
            "island": random.random() < 0.4, "bridges": random.choice([0, 0, 1])}


def _new_hub() -> dict:
    return {"x": random.uniform(0.3, 0.7), "y": random.uniform(0.3, 0.7),
            "rays": random.randint(5, 12), "rot": random.uniform(0, 360),
            "r0": random.uniform(0.04, 0.09), "ring": random.uniform(0.07, 0.14),
            "growth": random.uniform(1.0, 1.35)}


def _new_plaza() -> dict:
    return {"x": random.uniform(0.15, 0.85), "y": random.uniform(0.15, 0.85),
            "r": random.uniform(0.03, 0.09),
            "shape": random.choice(["circle", "circle", "square"]),
            "fill": random.choice(["void", "lead", "black", "ring"])}


def _new_water_color(g: CityGenome) -> int:
    c = random.choices([-1, ULTRA, BLACK, DARK_GREY, SKY, TEAL, NAVY],
                       weights=[12, 3, 2, 1, 2, 1, 1])[0]
    return -1 if c in (g.ground, g.lead) else c


COMPLEX_KINDS = ("courtyard", "stadium", "factory", "institute", "railyard", "park",
                 "circus", "gasworks", "planetarium")
ROUND_KINDS = ("circus", "gasworks", "planetarium", "stadium")


def _new_complex(kind: str | None = None) -> dict:
    kind = kind or random.choices(COMPLEX_KINDS,
                                  weights=[20, 14, 14, 14, 10, 12, 14, 12, 10])[0]
    w = random.uniform(0.14, 0.34)
    return {"kind": kind, "x": random.uniform(0.2, 0.8), "y": random.uniform(0.2, 0.8),
            "w": w, "h": w * random.uniform(0.45, 1.0),
            "angle": random.choice([0.0, 0.0, 90.0, random.uniform(0, 180)]),
            "seed": random.randrange(1 << 20)}


def _mutate_complex(g: CityGenome, s: float) -> None:
    r = random.random()
    if not g.complexes or (len(g.complexes) < 4 and r < 0.2):
        g.complexes.append(_new_complex())
    elif r < 0.27:
        g.complexes.pop(random.randrange(len(g.complexes)))
    else:
        c = random.choice(g.complexes)
        k = random.random()
        if k < 0.35:
            c["x"] = _clamp(c["x"] + random.gauss(0, 0.06 * s), 0.05, 0.95)
            c["y"] = _clamp(c["y"] + random.gauss(0, 0.06 * s), 0.05, 0.95)
        elif k < 0.6:
            f = math.exp(random.gauss(0, 0.2 * s))
            c["w"] = _clamp(c["w"] * f, 0.08, 0.5)
            c["h"] = _clamp(c["h"] * math.exp(random.gauss(0, 0.15 * s)), 0.06, 0.5)
        elif k < 0.75:
            c["angle"] = (c["angle"] + random.gauss(0, 15 * s)) % 180
        elif k < 0.9:
            c["seed"] = random.randrange(1 << 20)        # rebuild its insides
        else:
            c["kind"] = random.choice(COMPLEX_KINDS)


def random_city(plan: str | None = None, palette: str | None = None) -> CityGenome:
    g = CityGenome()
    g.seed = random.randrange(1 << 30)
    g.color_seed = random.randrange(1 << 30)
    g.plan = plan or random.choices(PLANS, weights=_PLAN_WEIGHTS)[0]
    g.base_angle = random.choice([0, 0, random.uniform(-25, 25), random.uniform(0, 180)])
    g.zoom = random.choice([0.65, 0.8, 1.0, 1.0, 1.3, 1.7]) * random.uniform(0.9, 1.1)
    g.block = random.uniform(0.05, 0.13)
    g.aspect = random.uniform(1.3, 3.6)
    g.street = random.uniform(0.0035, 0.011)
    g.lot = random.uniform(0.012, 0.032)
    g.angle_spread = random.uniform(15, 45)
    g.irregular = random.uniform(0.0, 0.25)
    g.solid = random.uniform(0.04, 0.2)
    g.voids = random.uniform(0.0, 0.06)
    n_cuts = random.randint(2, 6)

    if g.plan == "grid":
        g.angle_spread = 0.0
        g.irregular = 0.0
        g.aspect = random.uniform(2.2, 4.2)
        n_cuts = random.randint(0, 3)
    elif g.plan == "radial":
        g.hub = _new_hub()
        n_cuts = random.choice([0, 0, 1])
    elif g.plan == "substrate":
        g.irregular = random.uniform(0.75, 1.0)
        n_cuts = random.randint(1, 4)
    elif g.plan == "slabs":
        g.slabs = random.uniform(0.6, 1.0)
        g.block = random.uniform(0.1, 0.18)
        g.aspect = random.uniform(1.0, 1.8)
        g.voids = random.uniform(0.02, 0.07)
        g.solid = random.uniform(0.02, 0.1)
        n_cuts = random.randint(1, 4)
    if g.plan != "slabs" and random.random() < 0.15:
        g.slabs = random.uniform(0.1, 0.35)                 # a few slab blocks
    g.cuts = [_new_cut(g.base_angle, g.angle_spread, long=i < 2) for i in range(n_cuts)]

    g.region = random.uniform(0.2, 0.9)
    palette = palette or random.choices(list(_PALETTE_WEIGHTS),
                                        weights=list(_PALETTE_WEIGHTS.values()))[0]
    g.weights = _palette_weights(palette, g)
    g.water_color = _new_water_color(g)

    n_water = random.choices([0, 1, 2], weights=[30, 55, 15])[0]
    g.waters = [_new_water() for _ in range(n_water)]
    g.avenues = [_new_curve(random.uniform(0.010, 0.018))
                 for _ in range(random.choice([0, 0, 0, 1, 1, 2]))]
    g.rails = [_new_curve(0.0, n=random.randint(3, 4), wobble=0.06)
               for _ in range(random.choices([0, 1, 2], weights=[65, 28, 7])[0])]
    g.plazas = [_new_plaza() for _ in range(random.choices([0, 1, 2], weights=[65, 25, 10])[0])]
    # Start fairly plain: structure is grown by evolution (``complexify``).
    g.complexes = [_new_complex() for _ in range(random.choices([0, 1], weights=[60, 40])[0])]
    if random.random() < 0.5:
        g.prims = [random.choice(PRIM_KINDS)]
        g.prim_share = random.uniform(0.04, 0.12)
    if random.random() < 0.45:
        g.relief = random.uniform(0.35, 0.9)
    g.view = random.choice([-60.0, -120.0, -45.0, -135.0]) + random.gauss(0, 6)
    g.tonal = random.choice([0.0, 0.0, random.uniform(0.2, 0.6)])
    return g


def _new_zone(g: CityGenome) -> dict:
    kind = random.choice(ZONE_KINDS)
    z = {"kind": kind, "x": random.uniform(0.1, 0.9), "y": random.uniform(0.1, 0.9),
         "r": random.uniform(0.12, 0.3)}
    if kind == "tint":
        z["color"] = random.choice([c for c in CHROMAS + NEUTRALS
                                    if c != g.ground and g.weights[c] > 0] or [g.lead])
    if kind == "turn":
        z["angle"] = random.choice([45.0, 90.0, 30.0, -30.0]) + random.gauss(0, 5)
    return z


def complexity(g: CityGenome) -> int:
    """A rough count of structure, for display and tests."""
    return (len(g.cuts) + 2 * len(g.zones) + 3 * len(g.complexes)
            + sum(c.get("elab", 0) for c in g.complexes) + len(g.waters)
            + len(g.avenues) + len(g.rails) + len(g.plazas)
            + 2 * len(g.orn_kinds) + int(5 * g.ornament))


def complexify(g: CityGenome) -> None:
    """Add one piece of structure (the Picbreeder/NEAT 'add node' step).
    Small, local additions are favoured over new big features."""
    opts = []
    if len(g.cuts) < MAX_CUTS + 5:
        opts += ["local_cut"] * 4
    if len(g.zones) < MAX_ZONES:
        opts += ["zone"] * 4
    if len(g.orn_kinds) < len(ORN_KINDS):
        opts += ["orn_kind"] * 3
    if g.orn_kinds and g.ornament < 0.6:
        opts += ["ornament"] * 3
    if g.complexes and any(c.get("elab", 0) < 3 for c in g.complexes):
        opts += ["elab"] * 4
    if len(g.complexes) < MAX_COMPLEXES:
        opts += ["complex"] * 2
    if len(g.plazas) < MAX_PLAZAS:
        opts += ["plaza"]
    if len(g.avenues) < MAX_AVENUES:
        opts += ["avenue"]
    if len(g.rails) < 2:
        opts += ["rail"]
    rivers = [w for w in g.waters if w.get("kind") == "river"]
    if rivers:
        opts += ["bridge", "island"]
    if len(g.prims) < len(PRIM_KINDS):
        opts += ["prim"] * 3
    if g.prims and g.prim_share < 0.35:
        opts += ["prim_share"] * 2
    if g.relief < 1.4:
        opts += ["relief"] * 2
    if g.tonal < 0.7:
        opts += ["tonal"]
    if len(g.complexes) < MAX_COMPLEXES:
        opts += ["round_complex"] * 2
    if not opts:
        return
    op = random.choice(opts)
    if op == "local_cut":
        c = _new_cut(g.base_angle, g.angle_spread, long=False)
        c["length"] = random.uniform(0.12, 0.45)
        c["width"] = random.uniform(0.005, 0.011)
        g.cuts.append(c)
    elif op == "zone":
        g.zones.append(_new_zone(g))
    elif op == "orn_kind":
        g.orn_kinds.append(random.choice([k for k in ORN_KINDS if k not in g.orn_kinds]))
        g.ornament = max(g.ornament, 0.12)
    elif op == "ornament":
        g.ornament = min(0.6, g.ornament + random.uniform(0.06, 0.15))
    elif op == "elab":
        c = random.choice([c for c in g.complexes if c.get("elab", 0) < 3])
        c["elab"] = c.get("elab", 0) + 1
    elif op == "complex":
        g.complexes.append(_new_complex())
    elif op == "plaza":
        g.plazas.append(_new_plaza())
    elif op == "avenue":
        g.avenues.append(_new_curve(random.uniform(0.010, 0.018)))
    elif op == "rail":
        g.rails.append(_new_curve(0.0, n=random.randint(3, 4), wobble=0.06))
    elif op == "bridge":
        w = random.choice(rivers)
        w["bridges"] = min(9, int(w.get("bridges", 0)) + 1)
    elif op == "island":
        w = random.choice(rivers)
        w["island"] = True
    elif op == "prim":
        g.prims.append(random.choice([k for k in PRIM_KINDS if k not in g.prims]))
        g.prim_share = max(g.prim_share, 0.06)
    elif op == "prim_share":
        g.prim_share = min(0.35, g.prim_share + random.uniform(0.03, 0.08))
    elif op == "relief":
        g.relief = min(1.4, max(0.3, g.relief + random.uniform(0.15, 0.3)))
    elif op == "tonal":
        g.tonal = min(0.7, g.tonal + random.uniform(0.1, 0.2))
    elif op == "round_complex":
        g.complexes.append(_new_complex(random.choice(ROUND_KINDS)))


# ----------------------------------------------------------------------
# Variation
# ----------------------------------------------------------------------

def mutate_city(g: CityGenome, strength: float = 1.0, temp: float = 1.0) -> None:
    s = strength * (0.35 + 0.65 * temp)

    def gauss(v, sigma, lo, hi):
        return _clamp(v + random.gauss(0, sigma * s), lo, hi)

    def nudge(pt, sigma):
        return (pt[0] + random.gauss(0, sigma * s), pt[1] + random.gauss(0, sigma * s))

    # Complexification: a child usually gains structure, whatever the
    # temperature; refining makes parameter changes small, not structure.
    for _ in range(1 + (random.random() < 0.5)):
        complexify(g)

    for _ in range(random.randint(1, 1 + int(2 * temp))):
        roll = random.random()
        if roll < 0.08:
            g.seed = random.randrange(1 << 30)
        elif roll < 0.16:
            g.color_seed = random.randrange(1 << 30)
        elif roll < 0.26:
            k = random.randrange(4)
            if k == 0:
                g.block = gauss(g.block, 0.02, 0.035, 0.2)
                g.aspect = gauss(g.aspect, 0.4, 1.0, 4.5)
            elif k == 1:
                g.street = gauss(g.street, 0.0015, 0.0025, 0.014)
                g.lot = gauss(g.lot, 0.005, 0.008, 0.045)
            elif k == 2:
                g.zoom = gauss(g.zoom, 0.2, 0.5, 2.0)
            else:
                g.irregular = gauss(g.irregular, 0.2, 0.0, 1.0)
                g.slabs = gauss(g.slabs, 0.15, 0.0, 1.0)
        elif roll < 0.31:
            g.solid = gauss(g.solid, 0.06, 0.0, 0.4)
            g.voids = gauss(g.voids, 0.02, 0.0, 0.15)
        elif roll < 0.40:
            used = [i for i, w in enumerate(g.weights) if w > 0]
            i = (random.choice(used) if used and random.random() < 0.85
                 else random.choice(CHROMAS + NEUTRALS))
            if i != g.ground:
                g.weights[i] = gauss(g.weights[i] or 0.3, 0.6, 0.0, 5.0)
            g.lead_weight = gauss(g.lead_weight, 0.3, 0.05, 2.5)
            g.region = gauss(g.region, 0.15, 0.0, 1.0)
        elif roll < 0.42:
            lead = random.choice([c for c in CHROMAS + (BLACK,) if c != g.ground])
            g.weights[g.lead], g.weights[lead] = g.weights[lead], g.weights[g.lead]
            g.lead = lead
        elif roll < 0.45:                                    # a new palette
            g.weights = _palette_weights(random.choice(list(PALETTE_SETS)), g,
                                         ground=random.random() < 0.6)
        elif roll < 0.53 and g.cuts:
            c = random.choice(g.cuts)
            c["x"] = gauss(c["x"], 0.08, 0.0, 1.0)
            c["y"] = gauss(c["y"], 0.08, 0.0, 1.0)
            c["angle"] = (c["angle"] + random.gauss(0, 8 * s)) % 180
            c["length"] = gauss(c["length"], 0.2, 0.2, 1.8)
        elif roll < 0.58:
            if len(g.cuts) < MAX_CUTS and (random.random() < 0.75 or not g.cuts):
                g.cuts.append(_new_cut(g.base_angle, g.angle_spread,
                                       long=random.random() < 0.4))
            elif g.cuts:
                g.cuts.pop(random.randrange(len(g.cuts)))
        elif roll < 0.70:
            _mutate_water(g, s)
        elif roll < 0.74:
            if g.avenues and random.random() < 0.6:
                a = random.choice(g.avenues)
                k = random.randrange(1, len(a["pts"]) - 1) if len(a["pts"]) > 2 else 0
                a["pts"][k] = nudge(a["pts"][k], 0.07)
            elif len(g.avenues) < MAX_AVENUES and random.random() < 0.6:
                g.avenues.append(_new_curve(random.uniform(0.010, 0.018)))
            elif g.avenues:
                g.avenues.pop(random.randrange(len(g.avenues)))
        elif roll < 0.78:
            if g.rails and random.random() < 0.5:
                r = random.choice(g.rails)
                k = random.randrange(len(r["pts"]))
                r["pts"][k] = nudge(r["pts"][k], 0.06)
            elif len(g.rails) < 2 and random.random() < 0.6:
                g.rails.append(_new_curve(0.0, n=random.randint(3, 4), wobble=0.06))
            elif g.rails:
                g.rails.pop(random.randrange(len(g.rails)))
        elif roll < 0.83:
            if g.plazas and random.random() < 0.6:
                p = random.choice(g.plazas)
                p["x"], p["y"] = nudge((p["x"], p["y"]), 0.06)
                p["r"] = gauss(p["r"], 0.015, 0.02, 0.14)
                if random.random() < 0.3:
                    p["fill"] = random.choice(["void", "lead", "black", "ring"])
            elif len(g.plazas) < MAX_PLAZAS and random.random() < 0.6:
                g.plazas.append(_new_plaza())
            elif g.plazas:
                g.plazas.pop(random.randrange(len(g.plazas)))
        elif roll < 0.88 and g.hub:
            h = g.hub
            h["x"], h["y"] = nudge((h["x"], h["y"]), 0.05)
            h["rot"] = (h["rot"] + random.gauss(0, 10 * s)) % 360
            h["ring"] = gauss(h["ring"], 0.02, 0.05, 0.2)
            h["growth"] = gauss(h["growth"], 0.08, 0.9, 1.5)
            if random.random() < 0.3:
                h["rays"] = int(_clamp(h["rays"] + random.choice([-2, -1, 1, 2]), 3, 16))
        elif roll < 0.92 * (1.0 if temp > 0.4 else 0.97):
            g.base_angle = (g.base_angle + random.gauss(0, 10 * s)) % 180
            g.angle_spread = gauss(g.angle_spread, 8, 0.0, 50.0)
        elif roll < 0.93:
            g.water_color = _new_water_color(g)
        elif roll < 0.97:
            _mutate_complex(g, s)
        elif roll < 0.975 and (g.relief > 0 or g.tonal > 0):
            g.view = (g.view + random.gauss(0, 15 * s) + 180) % 360 - 180
            if g.relief > 0:
                g.relief = gauss(g.relief, 0.15, 0.2, 1.4)
            if g.tonal > 0:
                g.tonal = gauss(g.tonal, 0.1, 0.0, 0.7)
        elif roll < 0.985 and g.zones:
            z = random.choice(g.zones)
            z["x"] = _clamp(z["x"] + random.gauss(0, 0.06 * s), 0.0, 1.0)
            z["y"] = _clamp(z["y"] + random.gauss(0, 0.06 * s), 0.0, 1.0)
            z["r"] = _clamp(z["r"] + random.gauss(0, 0.04 * s), 0.06, 0.45)
        else:                                                # change city type
            g.plan = random.choice(PLANS)
            if g.plan == "radial" and g.hub is None:
                g.hub = _new_hub()
            if g.plan == "slabs":
                g.slabs = max(g.slabs, 0.6)
            elif g.plan == "substrate":
                g.irregular = max(g.irregular, 0.75)
            elif g.plan == "grid":
                g.angle_spread, g.irregular = 0.0, 0.0


def _mutate_water(g: CityGenome, s: float) -> None:
    if not g.waters or (len(g.waters) < MAX_WATERS and random.random() < 0.15):
        g.waters.append(_new_water())
        return
    if random.random() < 0.06:
        g.waters.pop(random.randrange(len(g.waters)))
        return
    w = random.choice(g.waters)
    r = random.random()
    if w["kind"] == "lake":
        if r < 0.5:
            w["x"] = _clamp(w["x"] + random.gauss(0, 0.06 * s), 0.0, 1.0)
            w["y"] = _clamp(w["y"] + random.gauss(0, 0.06 * s), 0.0, 1.0)
            w["r"] = _clamp(w["r"] + random.gauss(0, 0.03 * s), 0.05, 0.3)
        elif r < 0.8:
            k = random.randrange(len(w["lobes"]))
            w["lobes"][k] = _clamp(w["lobes"][k] + random.gauss(0, 0.2 * s), 0.5, 1.5)
        else:
            w["island"] = not w.get("island", False)
        return
    if r < 0.55:
        k = random.randrange(len(w["pts"]))
        x, y = w["pts"][k]
        w["pts"][k] = (x + random.gauss(0, 0.06 * s), y + random.gauss(0, 0.06 * s))
    elif w["kind"] == "river":
        if r < 0.7:
            w["width"] = _clamp(w["width"] + random.gauss(0, 0.015 * s), 0.025, 0.16)
        elif r < 0.88:
            w["bridges"] = int(_clamp(w.get("bridges", 0) + random.choice([-2, -1, 1, 2]), 0, 9))
        elif r < 0.94:
            w["island"] = not w.get("island", False)
        else:
            w["bridge_style"] = 1 - w.get("bridge_style", 0)
    else:
        if r < 0.8:
            w["piers"] = not w.get("piers", False)
        else:
            w["pier_seed"] = random.randrange(1 << 20)


def crossover_city(a: CityGenome, b: CityGenome) -> CityGenome:
    """Uniform crossover of scalar genes; streets and features mixed."""
    child = a.copy()
    for k in ("seed", "color_seed", "plan", "base_angle", "angle_spread", "zoom",
              "block", "aspect", "street", "lot", "irregular", "slabs", "solid",
              "voids", "region"):
        if random.random() < 0.5:
            setattr(child, k, getattr(b, k))
    # Colour travels as a whole palette, never mixed swatch by swatch.
    src = random.choice([a, b])
    for k in ("palette", "lead", "lead_weight", "ground", "water_color"):
        setattr(child, k, getattr(src, k))
    child.weights = list(src.weights)
    # Street network: a's cuts on one side of a random line, b's on the other.
    ang = random.uniform(0, math.pi)
    nx, ny = math.cos(ang), math.sin(ang)

    def side(c):
        return (c["x"] - 0.5) * nx + (c["y"] - 0.5) * ny

    cuts = [c for c in a.cuts if side(c) < 0] + [c for c in b.cuts if side(c) >= 0]
    child.cuts = _deep_copy(cuts[:MAX_CUTS] or a.cuts)
    child.hub = _deep_copy(a.hub if (a.hub and (random.random() < 0.5 or not b.hub)) else b.hub)
    if child.plan == "radial" and child.hub is None:
        child.hub = _new_hub()
    for k in ("waters", "avenues", "rails", "plazas", "complexes"):
        setattr(child, k, _deep_copy(getattr(random.choice([a, b]), k)))
    # Structure that has been grown is pooled, not dropped: a child of two
    # elaborate parents is at least as elaborate as the richer one.
    za = [z for z in a.zones if random.random() < 0.7]
    zb = [z for z in b.zones if random.random() < 0.7]
    child.zones = _deep_copy((za + zb)[:MAX_ZONES])
    child.orn_kinds = list(dict.fromkeys(a.orn_kinds + b.orn_kinds))
    child.ornament = max(a.ornament, b.ornament)
    child.prims = list(dict.fromkeys(a.prims + b.prims))
    child.prim_share = max(a.prim_share, b.prim_share)
    src3 = random.choice([a, b])
    child.relief, child.view = src3.relief, src3.view
    child.tonal = max(a.tonal, b.tonal)
    return child


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------

Poly = list[tuple[float, float]]


def _clip_half(poly: Poly, nx: float, ny: float, c: float) -> Poly:
    """Keep the part of a convex polygon where nx*x + ny*y >= c."""
    out: Poly = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = p[0] * nx + p[1] * ny - c
        dq = q[0] * nx + q[1] * ny - c
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def _clip_convex(poly: Poly, clip: Poly, inset: float = 0.0) -> Poly:
    """Clip ``poly`` by a convex counter-clockwise ``clip`` polygon, whose
    edges are optionally moved inward by ``inset``."""
    n = len(clip)
    for i in range(n):
        if len(poly) < 3:
            return []
        a, b = clip[i], clip[(i + 1) % n]
        nx, ny = -(b[1] - a[1]), b[0] - a[0]          # inward for CCW
        ln = math.hypot(nx, ny) or 1.0
        nx, ny = nx / ln, ny / ln
        poly = _clip_half(poly, nx, ny, nx * a[0] + ny * a[1] + inset)
    return poly


def _area(p: Poly) -> float:
    s = 0.0
    for i in range(len(p)):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % len(p)]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _ccw(p: Poly) -> Poly:
    return p if _area(p) >= 0 else p[::-1]


def _split(poly: Poly, nx: float, ny: float, c: float, gap: float) -> tuple[Poly, Poly]:
    return (_clip_half(poly, nx, ny, c + gap / 2),
            _clip_half(poly, -nx, -ny, -(c - gap / 2)))


def _extent(p: Poly, ux: float, uy: float) -> tuple[float, float]:
    d = [x * ux + y * uy for x, y in p]
    return min(d), max(d)


def _inside(poly: Poly, x: float, y: float) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0:
            return False
    return True


def _segment_hits(poly: Poly, cut: dict) -> bool:
    a = math.radians(cut["angle"])
    dx, dy = math.cos(a), math.sin(a)
    half = cut["length"] / 2
    for k in range(25):
        t = -half + 2 * half * k / 24
        if _inside(poly, cut["x"] + dx * t, cut["y"] + dy * t):
            return True
    return False


def _rect_poly(dom) -> Poly:
    x0, y0, x1, y1 = dom
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _ring_radii(hub: dict, dom) -> list[float]:
    far = max(math.hypot(x - hub["x"], y - hub["y"]) for x, y in _rect_poly(dom))
    radii = [hub["r0"]]
    w = hub["ring"]
    while radii[-1] < far and len(radii) < 40:
        radii.append(radii[-1] + w)
        w *= hub["growth"]
    return radii


def _radial_cells(hub: dict, dom, street: float) -> list[tuple[Poly, float | None]]:
    box = _ccw(_rect_poly(dom))
    radii = _ring_radii(hub, dom)
    n = int(hub["rays"])
    rot = math.radians(hub["rot"])
    cx, cy = hub["x"], hub["y"]
    cells: list[tuple[Poly, float | None]] = []
    for k in range(len(radii) - 1):
        r0, r1 = radii[k], radii[k + 1]
        rm = (r0 + r1) / 2
        sub = int(_clamp(round((2 * math.pi * rm / n) / ((r1 - r0) * 1.4)), 1, 8))
        m = n * sub
        for i in range(m):
            a0 = rot + 2 * math.pi * i / m
            a1 = rot + 2 * math.pi * (i + 1) / m
            quad = [(cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)),
                    (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
                    (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)),
                    (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1))]
            quad = _ccw(quad)
            cell = _clip_convex(quad, quad, street * 0.5)
            cell = _clip_convex(cell, box) if len(cell) >= 3 else []
            if len(cell) >= 3:
                mid = (a0 + a1) / 2
                cells.append((_ccw(cell), mid if k % 2 else mid + math.pi / 2))
    return cells


def _districts(g: CityGenome, dom) -> list[tuple[Poly, float | None]]:
    if g.plan == "radial" and g.hub:
        polys = _radial_cells(g.hub, dom, g.street)
    else:
        polys = [(_ccw(_rect_poly(dom)), None)]
    for cut in g.cuts:
        a = math.radians(cut["angle"])
        nx, ny = -math.sin(a), math.cos(a)
        c = cut["x"] * nx + cut["y"] * ny
        nxt: list[tuple[Poly, float | None]] = []
        for p, ang in polys:
            if _segment_hits(p, cut):
                for half in _split(p, nx, ny, c, cut["width"]):
                    if len(half) >= 3 and abs(_area(half)) > 1e-5:
                        nxt.append((_ccw(half), ang))
            else:
                nxt.append((p, ang))
        polys = nxt
    return polys


@dataclass
class Lot:
    poly: Poly
    color: int
    kind: str = "lot"


@dataclass
class _Fill:
    """Per-district fill parameters."""
    lot: float
    street: float
    solid: float
    voids: float
    slabs: float
    rng: random.Random


def _rect(u: tuple, v: tuple, a0, a1, b0, b1) -> Poly:
    return [(u[0] * a + v[0] * b, u[1] * a + v[1] * b)
            for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]


def _lots_grid(district: Poly, ang: float, bw: float, bh: float, f: _Fill,
               out: list[tuple[Poly, str]]) -> None:
    """Regular street grid at angle ``ang``; blocks split into lot strips."""
    rng = f.rng
    ux, uy = math.cos(ang), math.sin(ang)
    vx, vy = -uy, ux
    u0, u1 = _extent(district, ux, uy)
    v0, v1 = _extent(district, vx, vy)
    stagger = rng.random() < 0.35
    ph_u, ph_v = rng.uniform(0, bw), rng.uniform(0, bh)
    row = 0
    v = v0 - ph_v
    while v < v1 and len(out) < MAX_LOTS:
        u = u0 - ph_u - (bw * 0.5 if stagger and row % 2 else 0.0)
        while u < u1:
            _block(district, (ux, uy), (vx, vy),
                   u + f.street / 2, u + bw - f.street / 2,
                   v + f.street / 2, v + bh - f.street / 2, f, out)
            u += bw
        v += bh
        row += 1


def _block(district: Poly, u: tuple, v: tuple, a0, a1, b0, b1, f: _Fill,
           out: list[tuple[Poly, str]]) -> None:
    rng, lot = f.rng, f.lot
    whole = _clip_convex(_rect(u, v, a0, a1, b0, b1), district)
    if len(whole) < 3 or abs(_area(whole)) < (lot * 0.4) ** 2:
        return
    r = rng.random()
    if r < f.voids:
        return
    if r < f.voids + f.solid:
        out.append((whole, "solid"))
        return
    gap = f.street * 0.32
    long_u = (a1 - a0) >= (b1 - b0)
    span0, span1 = (a0, a1) if long_u else (b0, b1)
    d0, d1 = (b0, b1) if long_u else (a0, a1)

    def emit(s0, s1, e0, e1, kind):
        poly = _rect(u, v, s0, s1, e0, e1) if long_u else _rect(u, v, e0, e1, s0, s1)
        poly = _clip_convex(poly, district)
        if len(poly) >= 3 and abs(_area(poly)) > (gap * 2) ** 2:
            out.append((poly, kind))

    if rng.random() < f.slabs:
        # Constructivist superblock: long slab buildings in open ground.
        t = lot * rng.uniform(0.5, 0.9)
        pitch = t * rng.uniform(1.5, 2.3)
        margin = (span1 - span0) * rng.uniform(0.04, 0.15)
        across = rng.random() < 0.3                      # slabs run the short way
        if across:
            p = span0 + margin
            while p + t < span1 - margin:
                emit(p, p + t, d0 + margin * 0.5, d1 - margin * 0.5, "slab")
                p += pitch
        else:
            p = d0 + t * 0.3
            while p + t < d1:
                s_off = rng.uniform(0, margin)
                emit(span0 + s_off, span1 - margin + s_off * 0.5, p, p + t, "slab")
                p += pitch
        return

    # Lots front the long side; deep blocks get two back-to-back rows.
    depth = d1 - d0
    rows = 2 if depth > lot * 1.6 and rng.random() < 0.8 else 1
    cuts = [span0]
    pos = span0
    while True:
        pos += lot * rng.uniform(0.55, 1.6)
        if pos > span1 - lot * 0.4:
            break
        cuts.append(pos)
    cuts.append(span1)
    for r_i in range(rows):
        rd0 = d0 + depth * r_i / rows + (gap / 2 if r_i else 0)
        rd1 = d0 + depth * (r_i + 1) / rows - (gap / 2 if r_i < rows - 1 else 0)
        # Each row has its own frontage rhythm so rows don't align.
        jitter = cuts if not r_i else (
            [cuts[0]] + [c + rng.uniform(-0.3, 0.3) * lot for c in cuts[1:-1]] + [cuts[-1]])
        k = 0
        while k < len(jitter) - 1:
            # Occasionally a lot spans two plots (a bigger building).
            k2 = k + (2 if rng.random() < 0.15 and k + 2 < len(jitter) else 1)
            l0 = jitter[k] + (gap / 2 if k else 0)
            l1 = jitter[k2] - (gap / 2 if k2 < len(jitter) - 1 else 0)
            if l1 - l0 > gap:
                emit(l0, l1, rd0, rd1, "lot")
            k = k2


def _lots_split(poly: Poly, ang: float, target: float, f: _Fill,
                out: list[tuple[Poly, str]], level: int = 0) -> None:
    """Substrate-style recursive splitting along two orthogonal directions."""
    if len(out) > MAX_LOTS:
        return
    rng = f.rng
    area = abs(_area(poly))
    ux, uy = math.cos(ang), math.sin(ang)
    vx, vy = -uy, ux
    eu = _extent(poly, ux, uy)
    ev = _extent(poly, vx, vy)
    lu, lv = eu[1] - eu[0], ev[1] - ev[0]
    if area < target * rng.uniform(0.5, 1.6) or min(lu, lv) < f.street * 3:
        if area > (f.street * 1.2) ** 2:
            r = rng.random()
            if r < f.voids * 0.5:
                return
            out.append((poly, "solid" if area > target * 2.5 and r < 0.5 else "lot"))
        return
    # Cut across the longer extent, somewhere off-centre.
    nx, ny, lo, hi = (ux, uy, *eu) if lu >= lv else (vx, vy, *ev)
    t = rng.uniform(0.25, 0.75) if rng.random() < 0.7 else rng.uniform(0.1, 0.9)
    gap = f.street * (1.0 if level < 3 else 0.55 if level < 6 else 0.32)
    for half in _split(poly, nx, ny, lo + (hi - lo) * t, gap):
        if len(half) >= 3:
            _lots_split(half, ang, target, f, out, level + 1)


def _circle(cx: float, cy: float, rx: float, ry: float, u: tuple, a0: float = 0.0,
            a1: float = 2 * math.pi, k: int = 28) -> Poly:
    v = (-u[1], u[0])
    pts = []
    for i in range(k + 1 if a1 - a0 < 2 * math.pi - 1e-6 else k):
        t = a0 + (a1 - a0) * i / k
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + u[0] * x + v[0] * y, cy + u[1] * x + v[1] * y))
    return pts


def _primitive(poly: Poly, kind: str, ang: float, rng: random.Random) -> tuple[Poly, str]:
    """Turn a lot into a non-rectangular building: rotunda, half-round,
    triangle, oval, ring, notched L or a quarter-round wedge."""
    u = (math.cos(ang), math.sin(ang))
    v = (-u[1], u[0])
    u0, u1 = _extent(poly, *u)
    v0, v1 = _extent(poly, *v)
    if abs(_area(poly)) < 0.8 * (u1 - u0) * (v1 - v0):
        return poly, "lot"                     # only on (nearly) whole rectangles
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    hu, hv = (u1 - u0) / 2, (v1 - v0) / 2

    def P(a, b):
        return (u[0] * a + v[0] * b, u[1] * a + v[1] * b)

    cx, cy = P(um, vm)
    if kind in ("disc", "ring"):
        r = min(hu, hv) * 0.96
        return _circle(cx, cy, r, r, u), kind
    if kind == "ellipse":
        return _circle(cx, cy, hu * 0.97, hv * 0.97, u), "lot"
    if kind == "semi":
        r = min(hu, hv * 2) * 0.97
        flip = rng.random() < 0.5
        bx, by = P(um, v0 if not flip else v1)
        a0 = 0.0 if not flip else math.pi
        return _circle(bx, by, r, r, u, a0, a0 + math.pi, 24), "lot"
    if kind == "wedge":
        corner = rng.randrange(4)
        cu, cv, a0 = ((u0, v0, 0.0), (u1, v0, math.pi / 2), (u1, v1, math.pi),
                      (u0, v1, 1.5 * math.pi))[corner]
        r = min(2 * hu, 2 * hv) * 0.97
        bx, by = P(cu, cv)
        return [(bx, by)] + _circle(bx, by, r, r, u, a0, a0 + math.pi / 2, 12), "lot"
    if kind == "tri":
        c = rng.randrange(4)
        corners = [P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)]
        del corners[c]
        return _ccw(corners), "lot"
    # notch: an L-shaped block
    fu, fv = rng.uniform(0.35, 0.6), rng.uniform(0.35, 0.6)
    nu, nv = u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv
    return [P(u0, v0), P(u1, v0), P(u1, nv), P(nu, nv), P(nu, v1), P(u0, v1)], "lot"


def express_city(g: CityGenome, domain=(0.0, 0.0, 1.0, 1.0)) -> list[Lot]:
    """Genes -> painted lots (deterministic)."""
    dom = (domain[0] - DOMAIN_PAD, domain[1] - DOMAIN_PAD,
           domain[2] + DOMAIN_PAD, domain[3] + DOMAIN_PAD)
    rng = random.Random(g.seed)
    crng = random.Random(g.color_seed)
    base = math.radians(g.base_angle)
    spread = math.radians(g.angle_spread)
    z = g.zoom
    block = max(0.035, g.block * z)
    lot_size = max(0.0085, g.lot * z)
    street = max(0.0025, g.street * math.sqrt(z))

    weights = list(g.weights)
    weights[g.lead] += 3.0 * g.lead_weight
    weights[g.ground] = 0.0                                 # never ground on ground
    if sum(weights) <= 0:
        weights[BLACK if g.ground != BLACK else IVORY] = 1.0
    lots: list[Lot] = []
    for d_i, (district, hint) in enumerate(_districts(g, dom)):
        drng = random.Random(g.seed * 31 + d_i)
        if hint is not None:
            ang = hint + drng.gauss(0, 0.02)
        else:
            ang = base + drng.choice([0, 0, 0, math.pi / 2, spread, -spread]) + drng.gauss(0, 0.03)
        # Neighbourhood character from the zones covering this district.
        cx = sum(p[0] for p in district) / len(district)
        cy = sum(p[1] for p in district) / len(district)
        zs = [z for z in g.zones
              if math.hypot(cx - z["x"], cy - z["y"]) < z["r"] or _inside(district, z["x"], z["y"])]
        kinds = {z["kind"] for z in zs}
        zscale = (0.55 if "fine" in kinds else 1.0) * (1.6 if "coarse" in kinds else 1.0)
        for z in zs:
            if z["kind"] == "turn":
                ang += math.radians(z.get("angle", 45.0))
        bw = block * zscale * drng.uniform(0.7, 1.3)
        bh = bw / max(1.0, g.aspect * drng.uniform(0.8, 1.2))
        if drng.random() < 0.5:
            bw, bh = bh, bw
        fill = _Fill(lot=max(0.0075, lot_size * zscale * drng.uniform(0.8, 1.25)),
                     street=street * (0.8 if "fine" in kinds else 1.0),
                     solid=0.55 if "solid" in kinds else g.solid, voids=g.voids,
                     slabs=0.85 if "slabs" in kinds else
                     _clamp(g.slabs * drng.uniform(0.6, 1.4), 0, 1), rng=rng)
        shapes: list[tuple[Poly, str]] = []
        if "substrate" in kinds or (drng.random() < g.irregular and "slabs" not in kinds):
            _lots_split(district, ang, (fill.lot * 1.6) ** 2, fill, shapes)
        else:
            _lots_grid(district, ang, bw, bh, fill, shapes)
        # District colour bias: some districts lean to a few colours.
        dw = [w * (crng.uniform(1 - g.region, 1 + 2 * g.region) if w > 0 else 0)
              for w in weights]
        if sum(dw) <= 0:
            dw = weights
        for z in zs:
            if z["kind"] == "tint" and z.get("color", -1) != g.ground:
                dw[z["color"] % N_COLORS] = max(dw[z["color"] % N_COLORS], 1.0) * 4.0
        prev = -1
        if g.prims and g.prim_share > 0:
            prng = random.Random(g.seed * 7 + d_i * 131)
            unit2 = (lot_size * 0.9) ** 2
            shapes = [_primitive(pl, prng.choice(g.prims), ang, prng)
                      if kd != "slab" and abs(_area(pl)) > unit2 and prng.random() < g.prim_share
                      else (pl, kd) for pl, kd in shapes]
        for poly, kind in shapes:
            w = list(dw)
            if kind in ("solid", "slab"):
                # Large masses and slabs: mostly black or the lead colour.
                w = [x * (2.5 if i in (BLACK, g.lead, DARK_GREY, IVORY) else 0.3)
                     for i, x in enumerate(w)]
            if 0 <= prev < len(w) and sum(w) - w[prev] > 0:
                w[prev] *= 0.25                    # neighbours rarely match
            idx = crng.choices(range(N_COLORS), weights=w)[0]
            prev = idx
            lots.append(Lot(poly, idx, kind))
        if len(lots) > MAX_LOTS:
            break
    return lots


def _catmull(pts, steps: int = 24, closed: bool = False) -> Poly:
    pts = [tuple(p) for p in pts]
    if len(pts) < 3:
        return pts
    if closed:
        p = [pts[-1]] + pts + [pts[0], pts[1]]
    else:
        p = [pts[0]] + pts + [pts[-1]]
    out: Poly = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for s in range(steps):
            t = s / steps
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t
                                    + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                                    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                             for k in range(2)))
    if not closed:
        out.append(pts[-1])
    return out


def _frames(curve: Poly) -> list[tuple[tuple, tuple, tuple]]:
    """(point, unit tangent, unit normal) along a polyline."""
    out = []
    for i, p in enumerate(curve):
        a = curve[max(0, i - 1)]
        b = curve[min(len(curve) - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(tx, ty) or 1.0
        tx, ty = tx / ln, ty / ln
        out.append((p, (tx, ty), (-ty, tx)))
    return out


def _visible(p, margin: float = 0.02) -> bool:
    return -margin < p[0] < 1 + margin and -margin < p[1] < 1 + margin


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

_SHADOW = (118, 98, 86)          # warm umber glaze for cast shadows
_WALL_DARK = (26, 22, 20)


def _mixc(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _hull(pts: Poly) -> Poly:
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _pigment(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


def _vary(c: tuple[int, int, int], rng: random.Random, amt: float) -> tuple[int, int, int]:
    """Each lot is its own dab of gouache: a slightly different density."""
    f = 1.0 + rng.uniform(-amt, amt)
    w = rng.uniform(-amt, amt) * 40
    return tuple(int(_clamp(x * f + w * k, 0, 255)) for x, k in zip(c, (0.6, 0.3, -0.4)))


# ----------------------------------------------------------------------
# Siting landmarks: complexes and plazas belong to the plan.  They never
# straddle water, avenues, railways, district cuts or the radial rings and
# rays, and they turn with the street grid of the district they stand in.
# ----------------------------------------------------------------------

_SITE_DOM = (-0.25, -0.25, 1.25, 1.25)   # fixed, so every output size sites alike
_SITE_RES = 240                          # mask pixels per unit


def _district_angles(g: CityGenome) -> list[tuple[Poly, float, float, float]]:
    """Each district with its street-grid direction (radians) and centroid;
    the angle mirrors the one ``express_city`` gives that district's lots."""
    dom = (-DOMAIN_PAD, -DOMAIN_PAD, 1 + DOMAIN_PAD, 1 + DOMAIN_PAD)
    base = math.radians(g.base_angle)
    spread = math.radians(g.angle_spread)
    out = []
    for d_i, (district, hint) in enumerate(_districts(g, dom)):
        cx = sum(p[0] for p in district) / len(district)
        cy = sum(p[1] for p in district) / len(district)
        drng = random.Random(g.seed * 31 + d_i)
        if hint is not None:
            ang = hint + drng.gauss(0, 0.02)
        else:
            ang = base + drng.choice([0, 0, 0, math.pi / 2, spread, -spread]) + drng.gauss(0, 0.03)
        for z in g.zones:
            if z["kind"] == "turn" and (math.hypot(cx - z["x"], cy - z["y"]) < z["r"]
                                        or _inside(district, z["x"], z["y"])):
                ang += math.radians(z.get("angle", 45.0))
        out.append((district, ang, cx, cy))
    return out


def _angle_at(districts, x: float, y: float, default: float) -> float:
    best, best_d = default, float("inf")
    for poly, ang, cx, cy in districts:
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            if _inside(poly, x, y):
                return ang
            best, best_d = ang, d
    return best


class _SiteMask:
    """Coarse raster of where landmarks may not go."""

    def __init__(self):
        x0, y0, x1, y1 = _SITE_DOM
        self.w = int((x1 - x0) * _SITE_RES)
        self.h = int((y1 - y0) * _SITE_RES)
        self.img = Image.new("L", (self.w, self.h), 0)
        self.draw = ImageDraw.Draw(self.img)
        self._arr = None

    def px(self, p):
        return ((p[0] - _SITE_DOM[0]) * _SITE_RES, (p[1] - _SITE_DOM[1]) * _SITE_RES)

    def line(self, pts, width_u):
        self.draw.line([self.px(p) for p in pts], fill=255,
                       width=max(1, int(math.ceil(width_u * _SITE_RES))), joint="curve")
        self._arr = None

    def polygon(self, pts):
        self.draw.polygon([self.px(p) for p in pts], fill=255)
        self._arr = None

    def ellipse(self, c, r, width_u=None):
        x, y = self.px(c)
        q = r * _SITE_RES
        box = [x - q, y - q, x + q, y + q]
        if width_u is None:
            self.draw.ellipse(box, fill=255)
        else:
            self.draw.ellipse(box, outline=255, width=max(1, int(math.ceil(width_u * _SITE_RES))))
        self._arr = None

    def free(self, pts) -> bool:
        if self._arr is None:
            self._arr = np.asarray(self.img) > 0
        xs = [self.px(p) for p in pts]
        bx0 = max(0, int(min(x for x, _ in xs)) - 1)
        by0 = max(0, int(min(y for _, y in xs)) - 1)
        bx1 = min(self.w, int(max(x for x, _ in xs)) + 2)
        by1 = min(self.h, int(max(y for _, y in xs)) + 2)
        if bx1 <= bx0 or by1 <= by0:
            return False
        fp = Image.new("L", (bx1 - bx0, by1 - by0), 0)
        ImageDraw.Draw(fp).polygon([(x - bx0, y - by0) for x, y in xs], fill=255)
        return not (self._arr[by0:by1, bx0:bx1] & (np.asarray(fp) > 0)).any()


def _footprint(x, y, hw, hh, ang, round_: bool) -> Poly:
    t, n = (math.cos(ang), math.sin(ang)), (-math.sin(ang), math.cos(ang))
    if round_:
        return [(x + t[0] * hw * math.cos(a) + n[0] * hh * math.sin(a),
                 y + t[1] * hw * math.cos(a) + n[1] * hh * math.sin(a))
                for a in (2 * math.pi * k / 28 for k in range(28))]
    return [(x + t[0] * su * hw + n[0] * sv * hh, y + t[1] * su * hw + n[1] * sv * hh)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _site_landmarks(g: CityGenome, street: float) -> tuple[list[dict], list[dict]]:
    """Where complexes and plazas actually stand: the nearest spot to their
    gene position that is clear of the city's lines and water, turned to the
    local street grid.  A landmark with no room is left out."""
    if not g.complexes and not g.plazas:
        return [], []
    m = _SiteMask()
    districts = _district_angles(g)
    base = math.radians(g.base_angle)
    gap = street * 1.5                                   # breathing room

    for w in g.waters:
        kind = w.get("kind", "river")
        if kind == "lake":
            lobes = w["lobes"]
            ctrl = [(w["x"] + w["r"] * l * math.cos(2 * math.pi * k / len(lobes)),
                     w["y"] + w["r"] * l * math.sin(2 * math.pi * k / len(lobes)))
                    for k, l in enumerate(lobes)]
            shore = _catmull(ctrl, 16, closed=True)
            m.polygon(shore)
            m.line(shore + shore[:1], 2 * gap)
        elif kind == "coast":
            curve = _catmull(w["pts"], 40)
            sgn = w.get("side", 1)
            far = [(p[0] + n[0] * sgn * 3, p[1] + n[1] * sgn * 3) for p, _, n in _frames(curve)]
            m.polygon(curve + far[::-1])
            m.line(curve, 2 * gap + 0.08)                  # piers reach out ~0.08
        else:
            m.line(_catmull(w["pts"], 40), w["width"] * 1.15 + street * 4 + 2 * gap)
    for av in g.avenues:
        m.line(_catmull(av["pts"], 30), av["width"] + 2 * gap)
    for rl in g.rails:
        m.line(_catmull(rl["pts"], 40), street * 2.6 + 2 * gap)
    for cut in g.cuts:
        a = math.radians(cut["angle"])
        d = (math.cos(a) * cut["length"] / 2, math.sin(a) * cut["length"] / 2)
        m.line([(cut["x"] - d[0], cut["y"] - d[1]), (cut["x"] + d[0], cut["y"] + d[1])],
               cut["width"] + 2 * gap)
    if g.plan == "radial" and g.hub:
        h = g.hub
        radii = _ring_radii(h, _SITE_DOM)
        hub_c = (h["x"], h["y"])
        m.ellipse(hub_c, radii[0] + gap)
        for r in radii[1::max(1, len(radii) // 4)]:
            m.ellipse(hub_c, r, street * 1.8 + 2 * gap)
        for i in range(int(h["rays"])):
            a = math.radians(h["rot"]) + 2 * math.pi * i / int(h["rays"])
            m.line([(h["x"] + radii[0] * math.cos(a), h["y"] + radii[0] * math.sin(a)),
                    (h["x"] + radii[-1] * math.cos(a), h["y"] + radii[-1] * math.sin(a))],
                   street * 2.2 + 2 * gap)

    def site(x, y, hw, hh, turn, round_, seed):
        """First clear spot, spiralling out from (x, y) and shrinking if need be."""
        rng = random.Random(seed)
        offs = [(0.0, 0.0)]
        for ring in range(1, 11):
            r = ring * 0.025
            k = 8 + ring * 2
            a0 = rng.uniform(0, 2 * math.pi)
            offs += [(r * math.cos(a0 + 2 * math.pi * j / k), r * math.sin(a0 + 2 * math.pi * j / k))
                     for j in range(k)]
        for scale in (1.0, 0.85, 0.7, 0.55):
            for dx, dy in offs:
                cx, cy = _clamp(x + dx, 0.06, 0.94), _clamp(y + dy, 0.06, 0.94)
                ang = _angle_at(districts, cx, cy, base) + turn
                fp = _footprint(cx, cy, hw * scale, hh * scale, ang, round_)
                if m.free(fp):
                    return cx, cy, scale, ang, fp
        return None

    rail_frames = [_frames(_catmull(rl["pts"], 40)) for rl in g.rails]

    def site_by_rail(c, W, H):
        """A railyard lies alongside a railway, long side on the line, the
        nearest stretch to its gene position first."""
        cands = []
        for frames in rail_frames:
            for i in range(2, len(frames) - 2):
                p, t, n = frames[i]
                if 0.08 < p[0] < 0.92 and 0.08 < p[1] < 0.92:
                    cands.append((math.hypot(p[0] - c["x"], p[1] - c["y"]), p, t, n))
        cands.sort(key=lambda e: e[0])
        sides = (1, -1) if c.get("seed", 0) % 2 else (-1, 1)
        for scale in (1.0, 0.85, 0.7, 0.55):
            hw, hh = (W + street) * scale, (H + street) * scale
            off = hh + street * 1.3 + gap + 0.004
            for _, p, t, n in cands:
                for sd in sides:
                    cx, cy = p[0] + n[0] * sd * off, p[1] + n[1] * sd * off
                    ang = math.atan2(t[1], t[0])
                    fp = _footprint(cx, cy, hw, hh, ang, False)
                    if m.free(fp):
                        # rail_v: where the line runs, in the yard's own frame.
                        return cx, cy, scale, ang, fp, -sd * off
        return None

    placed_c: list[dict] = []
    for c in g.complexes:
        kind = c.get("kind", "courtyard")
        if kind == "railyard" and not g.rails:
            # No railway to serve: build something else on the site.
            kind = ("courtyard", "factory", "institute")[c.get("seed", 0) % 3]
            c = {**c, "kind": kind}
        W, H = c["w"] * 0.5, c["h"] * 0.5
        if kind == "railyard":
            H = min(H, W * 0.45)                           # yards are long and narrow
            s = site_by_rail(c, W, H)
            if s is None:
                continue
            cx, cy, scale, ang, fp, rail_v = s
            placed_c.append({**c, "x": cx, "y": cy, "w": 2 * W * scale, "h": 2 * H * scale,
                             "angle": math.degrees(ang), "rail_v": rail_v})
            m.polygon(fp)
            m.line(fp + fp[:1], 2 * gap)
            continue
        if kind in ("circus", "planetarium"):
            W = H = max(W, H)
        elif kind == "gasworks":
            H = W
        round_ = kind in ("circus", "planetarium")
        # The gene angle only says "along" or "across" the local grid.
        turn = math.pi / 2 if round((c.get("angle", 0.0) % 180) / 90) % 2 else 0.0
        s = site(c["x"], c["y"], W + street, H + street, turn, round_, c.get("seed", 0))
        if s is None:
            continue
        cx, cy, scale, ang, fp = s
        placed_c.append({**c, "x": cx, "y": cy, "w": c["w"] * scale, "h": c["h"] * scale,
                         "angle": math.degrees(ang)})
        m.polygon(fp)
        m.line(fp + fp[:1], 2 * gap)

    placed_p: list[dict] = []
    for i, pz in enumerate(g.plazas):
        sq = pz.get("shape") == "square"
        R = pz["r"] + street
        s = site(pz["x"], pz["y"], R, R, 0.0, not sq, i * 7919 + int(pz["r"] * 1e4))
        if s is None:
            continue
        cx, cy, scale, ang, fp = s
        placed_p.append({**pz, "x": cx, "y": cy, "r": pz["r"] * scale, "angle": math.degrees(ang)})
        m.polygon(fp)
        m.line(fp + fp[:1], 2 * gap)
    return placed_c, placed_p


def render_city(g: CityGenome, width: int, height: int, palette: list[tuple],
                grain: float = 8.0) -> Image.Image:
    """Paint the map.  The unit square maps to the centred square of the
    output; wider/taller outputs simply show more city."""
    ss = 2 if width * height <= 2048 * 3072 else 1
    iw, ih = width * ss, height * ss
    side = min(iw, ih)
    ox, oy = (iw - side) / 2, (ih - side) / 2
    domain = (-ox / side, -oy / side, (iw - ox) / side, (ih - oy) / side)

    pal = [_pigment(c) for c in palette]
    col = lambda i: pal[i % len(pal)]                         # noqa: E731
    ground = col(g.ground)
    water = ground if g.water_color < 0 else col(g.water_color)
    img = Image.new("RGB", (iw, ih), ground)
    draw = ImageDraw.Draw(img)
    rng = random.Random(g.color_seed ^ 0x5F3759DF)
    frng = random.Random(g.seed ^ 0x2545F491)                 # feature details
    z = g.zoom
    street = max(0.0025, g.street * math.sqrt(z))
    # Bridges and piers must read against the water: the street colour on
    # dark water (the road carries on across), black or ivory on light water.
    dark_on = BLACK if g.ground != BLACK else IVORY
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]     # noqa: E731
    if water != ground and abs(lum(ground) - lum(water)) > 70:
        bridge_col = ground
    else:
        bridge_col = col(dark_on) if abs(lum(col(dark_on)) - lum(water)) > 70 else col(IVORY)

    def px(p):
        return (ox + p[0] * side, oy + p[1] * side)

    def poly(pts, fill):
        draw.polygon([px(p) for p in pts], fill=fill)

    def stroke(pts, width_u, fill, round_ends=True):
        pp = [px(p) for p in pts]
        w = max(1, int(round(width_u * side)))
        draw.line(pp, fill=fill, width=w)
        if round_ends and w > 2:
            r = w / 2
            for x, y in pp:                     # round joins (no mitre spikes)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def bar(c, t, n, along, across, fill):
        """Rectangle centred at c: ``along`` on t, ``across`` on n."""
        a, b = along / 2, across / 2
        poly([(c[0] + t[0] * sa * a + n[0] * sb * b, c[1] + t[1] * sa * a + n[1] * sb * b)
              for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))], fill)

    # 1. Lots: a flat plan, or lifted into blocks seen obliquely under one
    # light (hard flat shadows, walls a darker shade of their own colour).
    orng = random.Random(g.seed ^ 0x9E3779B9)
    trng = random.Random(g.color_seed ^ 0x1234567)
    lot_unit = max(0.0085, g.lot * z)
    lots = express_city(g, domain)
    colours = []
    for lot in lots:
        c = _vary(col(lot.color), rng, 0.035)
        if g.tonal > 0:                       # light and dark tones of a hue
            t = trng.uniform(-1, 1) * g.tonal * 0.32
            c = _mixc(c, (250, 246, 236), t) if t > 0 else _mixc(c, (18, 17, 18), -t)
        colours.append(c)

    if g.relief > 0 and lots:
        va = math.radians(g.view)
        ev = (math.cos(va), math.sin(va))                  # roofs lift this way
        sa = va + math.radians(80)
        sv = (math.cos(sa), math.sin(sa))                  # shadows fall this way
        hrng = random.Random(g.seed ^ 0x51ED27)
        heights = []
        for lot in lots:
            f = min(3.5, math.exp(hrng.gauss(0, 0.55)))
            f *= {"solid": 1.8, "slab": 1.3, "disc": 1.7, "ring": 1.5}.get(lot.kind, 1.0)
            heights.append(f * g.relief * lot_unit * 0.85)
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        for lot, h in zip(lots, heights):
            pts = list(lot.poly) + [(x + sv[0] * h, y + sv[1] * h) for x, y in lot.poly]
            md.polygon([px(p) for p in _hull(pts)], fill=150)
        img.paste(Image.composite(ImageChops.multiply(img, Image.new("RGB", img.size, _SHADOW)),
                                  img, mask))
        order = sorted(range(len(lots)), key=lambda i: -sum(
            x * ev[0] + y * ev[1] for x, y in lots[i].poly) / len(lots[i].poly))
        for i in order:
            lot, h, c = lots[i], heights[i], colours[i]
            base = lot.poly if _area(lot.poly) >= 0 else lot.poly[::-1]
            dx, dy = ev[0] * h, ev[1] * h
            for k in range(len(base)):
                a, b = base[k], base[(k + 1) % len(base)]
                nx, ny = b[1] - a[1], -(b[0] - a[0])           # outward (CCW)
                ln = math.hypot(nx, ny) or 1.0
                nx, ny = nx / ln, ny / ln
                if nx * ev[0] + ny * ev[1] >= 0:
                    continue                                 # hidden behind the roof
                lit = max(0.0, -(nx * sv[0] + ny * sv[1]))
                wall = _mixc(c, _WALL_DARK, 0.5 - 0.32 * lit)
                poly([a, b, (b[0] + dx, b[1] + dy), (a[0] + dx, a[1] + dy)], wall)
            roof = Lot([(x + dx, y + dy) for x, y in lot.poly], lot.color, lot.kind)
            poly(roof.poly, c)
            if lot.kind == "ring":
                _ring_well(roof, _mixc(c, _WALL_DARK, 0.55), poly)
            if g.orn_kinds and lot.kind != "slab" and orng.random() < g.ornament:
                _ornament(roof, orng.choice(g.orn_kinds), orng, g, col, poly, ground,
                          dark_on, street, lot_unit, side, draw, px)
    else:
        for lot, c in zip(lots, colours):
            poly(lot.poly, c)
            if lot.kind == "ring":
                _ring_well(lot, ground, poly)
            if g.orn_kinds and lot.kind != "slab" and orng.random() < g.ornament:
                _ornament(lot, orng.choice(g.orn_kinds), orng, g, col, poly, ground,
                          dark_on, street, lot_unit, side, draw, px)

    # 2. Ring-and-radial avenues.
    if g.plan == "radial" and g.hub:
        h = g.hub
        radii = _ring_radii(h, domain)
        for r in radii[1::max(1, len(radii) // 4)] + [radii[0]]:
            stroke([(h["x"] + r * math.cos(2 * math.pi * k / 96),
                     h["y"] + r * math.sin(2 * math.pi * k / 96)) for k in range(97)],
                   street * 1.8, ground, round_ends=False)
        for i in range(int(h["rays"])):
            a = math.radians(h["rot"]) + 2 * math.pi * i / int(h["rays"])
            stroke([(h["x"] + radii[0] * math.cos(a), h["y"] + radii[0] * math.sin(a)),
                    (h["x"] + radii[-1] * math.cos(a), h["y"] + radii[-1] * math.sin(a))],
                   street * 2.2, ground, round_ends=False)
        r0 = radii[0]
        cx, cy = px((h["x"], h["y"]))
        rp = r0 * side
        draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], fill=ground)
        rp *= 0.62
        draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], fill=col(g.lead))

    # 3. Avenues.
    for av in g.avenues:
        stroke(_catmull(av["pts"], 30), av["width"], ground)

    # 3b. Large complexes, each with its own internal structure, sited
    # clear of water, avenues, railways and ring roads, on the local grid.
    sited_complexes, sited_plazas = _site_landmarks(g, street)
    for cx in sited_complexes:
        _draw_complex(cx, g, col, poly, stroke, bar, ground, water, dark_on, street, side)

    # 4. Water, with islands, piers and bridges.
    for w in g.waters:
        kind = w.get("kind", "river")
        if kind == "lake":
            lobes = w["lobes"]
            ctrl = [(w["x"] + w["r"] * l * math.cos(2 * math.pi * k / len(lobes)),
                     w["y"] + w["r"] * l * math.sin(2 * math.pi * k / len(lobes)))
                    for k, l in enumerate(lobes)]
            shore = _catmull(ctrl, 16, closed=True)
            poly(shore, water)
            if w.get("island"):
                ir = w["r"] * 0.28
                isle = [(w["x"] + ir * 1.3 * math.cos(a), w["y"] + ir * 0.8 * math.sin(a))
                        for a in (2 * math.pi * k / 24 for k in range(24))]
                poly(isle, col(DARK_GREY if g.ground != DARK_GREY else IVORY))
            if w.get("bridges"):
                # A causeway across the lake.
                a = frng.uniform(0, math.pi)
                d = (math.cos(a), math.sin(a))
                L = w["r"] * 2.8
                bar((w["x"], w["y"]), d, (-d[1], d[0]), L, street * 1.4, bridge_col)
            continue

        curve = _catmull(w["pts"], 40)
        if kind == "coast":
            sgn = w.get("side", 1)
            frames = _frames(curve)
            far = [(p[0] + n[0] * sgn * 3, p[1] + n[1] * sgn * 3) for p, _, n in frames]
            poly(curve + far[::-1], water)
            if w.get("piers"):
                prng = random.Random(w.get("pier_seed", 0))
                vis = [i for i, (p, _, _) in enumerate(frames) if _visible(p, -0.03)]
                if vis:
                    i0 = prng.choice(vis)
                    n_p = prng.randint(5, 14)
                    step = prng.randint(2, 4)
                    plen = prng.uniform(0.03, 0.08)
                    pw = street * prng.uniform(1.4, 2.8)
                    pc = col(prng.choice([BLACK, DARK_GREY, g.lead]))
                    if pc == water:
                        pc = bridge_col
                    for k in range(n_p):
                        i = i0 + (k - n_p // 2) * step
                        if 0 <= i < len(frames):
                            p, t, n = frames[i]
                            ln = plen * prng.uniform(0.6, 1.2)
                            c = (p[0] + n[0] * sgn * (ln / 2 - street),
                                 p[1] + n[1] * sgn * (ln / 2 - street))
                            bar(c, n, t, ln, pw, pc)
            continue

        # River.
        rw = w["width"]
        stroke(curve, rw, water)
        frames = _frames(curve)
        vis = [i for i, (p, _, _) in enumerate(frames) if _visible(p, -0.04)]
        if w.get("island") and vis:
            i = vis[len(vis) // 2 + frng.randint(-len(vis) // 5, len(vis) // 5)] \
                if len(vis) > 5 else vis[0]
            p, t, n = frames[i]
            L, W = rw * frng.uniform(2.0, 3.5), rw * 0.42
            isle = [(p[0] + t[0] * L / 2 * math.cos(a) + n[0] * W / 2 * math.sin(a),
                     p[1] + t[1] * L / 2 * math.cos(a) + n[1] * W / 2 * math.sin(a))
                    for a in (2 * math.pi * k / 28 for k in range(28))]
            poly(isle, col(frng.choice([DARK_GREY, WARM_GREY, BLACK])
                           if g.ground != BLACK else IVORY))
        nb = int(w.get("bridges", 0))
        if nb and vis:
            span = rw * 1.15 + street * 4
            deck = max(street * 2.0, 0.008)
            for k in range(nb):
                f = (k + 0.5) / nb + frng.uniform(-0.25, 0.25) / nb
                p, t, n = frames[vis[int(_clamp(f, 0, 0.999) * len(vis))]]
                if w.get("bridge_style", 0) == 1:
                    # Two thin girders with cross-ties: a truss in plan.
                    for o in (-deck / 2, deck / 2):
                        bar((p[0] + t[0] * o, p[1] + t[1] * o), n, t, span,
                            max(street * 0.35, 1.2 / side), bridge_col)
                    m = max(2, int(span / (deck * 0.8)))
                    for j in range(m + 1):
                        o = -span / 2 + span * j / m
                        bar((p[0] + n[0] * o, p[1] + n[1] * o), t, n, deck,
                            max(street * 0.25, 1.0 / side), bridge_col)
                else:
                    bar(p, n, t, span, deck, bridge_col)
                    for o in (-span / 2, span / 2):           # abutments
                        bar((p[0] + n[0] * o, p[1] + n[1] * o), n, t,
                            deck * 0.9, deck * 2.0, bridge_col)

    # 5. Plazas.
    for pz in sited_plazas:
        c, r = (pz["x"], pz["y"]), pz["r"]
        fill = pz.get("fill", "void")
        if pz.get("shape") == "square":
            a = math.radians(pz.get("angle", g.base_angle))
            t, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
            bar(c, t, n, 2 * (r + street), 2 * (r + street), ground)
            if fill in ("lead", "black"):
                bar(c, t, n, 2 * r, 2 * r, col(g.lead if fill == "lead" else dark_on))
            elif fill == "ring":
                bar(c, t, n, 2 * r * 0.7, 2 * r * 0.7, col(dark_on))
                bar(c, t, n, 2 * r * 0.7 - 2 * street, 2 * r * 0.7 - 2 * street, ground)
            continue
        cx, cy = px(c)
        for rr, fc in ((r + street, ground),
                       (r, col(g.lead) if fill == "lead" else col(dark_on) if fill == "black" else None),
                       ):
            if fc is not None:
                q = rr * side
                draw.ellipse([cx - q, cy - q, cx + q, cy + q], fill=fc)
        if fill == "ring":
            q = r * 0.75 * side
            draw.ellipse([cx - q, cy - q, cx + q, cy + q], outline=col(dark_on),
                         width=max(1, int(street * 0.8 * side)))
            q = r * 0.22 * side
            draw.ellipse([cx - q, cy - q, cx + q, cy + q], fill=col(g.lead))

    # 6. Railways: a cleared corridor, the line and its sleepers; they cross
    # water on their own bridge.
    for rl in g.rails:
        curve = _catmull(rl["pts"], 40)
        stroke(curve, street * 2.6, ground)
        line_w = max(street * 0.45, 1.5 / side)
        stroke(curve, line_w, col(dark_on), round_ends=False)
        frames = _frames(curve)
        acc, gap = 0.0, street * 2.2
        for i in range(1, len(frames)):
            p0, p1 = frames[i - 1][0], frames[i][0]
            acc += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if acc >= gap:
                acc = 0.0
                p, t, n = frames[i]
                bar(p, t, n, max(street * 0.3, 1.2 / side), street * 1.7, col(dark_on))

    img = _paper(img, ground, grain, rng)
    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)
    return img

def _ring_well(lot: Lot, fill, poly) -> None:
    p = lot.poly
    cx = sum(q[0] for q in p) / len(p)
    cy = sum(q[1] for q in p) / len(p)
    poly([(cx + (x - cx) * 0.5, cy + (y - cy) * 0.5) for x, y in p], fill)


def _ornament(lot: Lot, kind: str, rng: random.Random, g: CityGenome, col, poly,
              ground, dark_on: int, street: float, unit: float, side: int, draw, px) -> None:
    """Inner detail of one building: a light-well, a two-tone roof, roof
    ridges, a round skylight or a parapet frame."""
    p = lot.poly
    area = abs(_area(p))
    if area < (unit * 0.9) ** 2:
        return
    cx = sum(q[0] for q in p) / len(p)
    cy = sum(q[1] for q in p) / len(p)

    def shrink(f):
        return [(cx + (x - cx) * f, cy + (y - cy) * f) for x, y in p]

    # The lot's long axis.
    best, axis = 0.0, (1.0, 0.0)
    for i in range(len(p)):
        dx, dy = p[(i + 1) % len(p)][0] - p[i][0], p[(i + 1) % len(p)][1] - p[i][1]
        ln = math.hypot(dx, dy)
        if ln > best:
            best, axis = ln, (dx / ln, dy / ln)
    other = [i for i in range(N_COLORS) if g.weights[i] > 0 and i not in (lot.color, g.ground)]
    contrast = col(dark_on) if lot.color != dark_on else col(IVORY if g.ground != IVORY else PAPER)

    if kind == "court":
        poly(shrink(rng.uniform(0.35, 0.55)), ground)
    elif kind == "split":
        nx, ny = axis
        half = _clip_half(p, nx, ny, cx * nx + cy * ny + rng.uniform(-0.15, 0.15) * best)
        if len(half) >= 3:
            poly(half, col(rng.choice(other)) if other else contrast)
    elif kind == "stripe":
        nx, ny = axis
        lo, hi = _extent(p, nx, ny)
        pitch = max(street * 1.4, (hi - lo) / rng.randint(4, 9))
        w = pitch * 0.32
        x = lo + pitch * 0.5
        inner = _ccw(shrink(0.86))
        while x < hi - pitch * 0.3:
            band = _clip_half(_clip_half(inner, nx, ny, x), -nx, -ny, -(x + w))
            if len(band) >= 3:
                poly(band, contrast)
            x += pitch
    elif kind == "dot":
        r = math.sqrt(area) * rng.uniform(0.18, 0.3)
        X, Y = px((cx, cy))
        q = r * side
        draw.ellipse([X - q, Y - q, X + q, Y + q],
                     fill=col(g.lead) if lot.color != g.lead else contrast)
    elif kind == "frame":
        draw.polygon([px(q) for q in shrink(0.72)], outline=contrast,
                     width=max(1, int(street * 0.4 * side)))


def _draw_complex(c: dict, g: CityGenome, col, poly, stroke, bar, ground, water,
                  dark_on: int, street: float, side: int) -> None:
    """A big structure in plan: a site cleared out of the tiling and filled
    with a building complex that has its own sub-structure."""
    rng = random.Random(c.get("seed", 0))
    a = math.radians(c.get("angle", 0.0))
    t, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    x0, y0 = c["x"], c["y"]
    W, H = c["w"] * 0.5, c["h"] * 0.5                 # half sizes
    thin = max(street * 0.35, 1.3 / side)

    def L(u, v):
        return (x0 + t[0] * u + n[0] * v, y0 + t[1] * u + n[1] * v)

    def rect(u0, u1, v0, v1, fill):
        poly([L(u0, v0), L(u1, v0), L(u1, v1), L(u0, v1)], fill)

    def ell(u, v, ru, rv, fill, k=40):
        poly([L(u + ru * math.cos(2 * math.pi * i / k), v + rv * math.sin(2 * math.pi * i / k))
              for i in range(k)], fill)

    def ring_line(u, v, ru, rv, width, fill, k=48):
        stroke([L(u + ru * math.cos(2 * math.pi * i / k), v + rv * math.sin(2 * math.pi * i / k))
                for i in range(k + 1)], width, fill, False)

    def line(p0, p1, width, fill):
        stroke([L(*p0), L(*p1)], width, fill, False)

    weights = list(g.weights)
    weights[g.lead] += 3.0 * g.lead_weight
    weights[g.ground] = 0.0
    if sum(weights) <= 0:
        weights[dark_on] = 1.0

    def pick(*avoid):
        w = [0.0 if i in avoid else x for i, x in enumerate(weights)]
        if sum(w) <= 0:
            w = weights
        return rng.choices(range(N_COLORS), weights=w)[0]

    def green():
        for i in (VIRIDIAN, OLIVE, TEAL):
            if g.weights[i] > 0 and i != g.ground:
                return i
        return pick(g.lead, BLACK)

    dark = dark_on
    kind = c.get("kind", "courtyard")
    elab = int(c.get("elab", 0))
    if kind in ("circus", "gasworks", "planetarium"):
        H = W = max(W, H) if kind != "gasworks" else W
        if kind == "gasworks":
            rect(-W - street, W + street, -H - street, H + street, ground)
        else:
            ell(0, 0, W + street, H + street, ground)
    else:
        rect(-W - street, W + street, -H - street, H + street, ground)   # the site
    if elab >= 3:
        # A forecourt band and a frame of small pavilions around the site.
        k = max(4, int((W + H) / (street * 6)))
        pc = col(pick())
        for i in range(k):
            f = -W + 2 * W * (i + 0.5) / k
            rect(f - street * 0.8, f + street * 0.8, -H - street * 0.9, -H - street * 0.2, pc)

    if kind == "courtyard":
        A = pick()
        d = min(W, H) * rng.uniform(0.18, 0.32)
        rect(-W, W, -H, H, col(A))
        rect(-W + d, W - d, -H + d, H - d, ground)
        e = d * rng.uniform(0.8, 1.6)                       # gateways
        rect(-e / 2, e / 2, -H - street, -H + d, ground)
        if rng.random() < 0.6:
            rect(-W - street, -W + d, -e / 2, e / 2, ground)
        iu, iv = W - d - street * 1.5, H - d - street * 1.5
        mode = rng.choice(["rows", "grid", "centre", "rows"])
        if mode == "rows":
            th = d * rng.uniform(0.45, 0.7)
            pitch = th * rng.uniform(1.8, 2.6)
            v = -iv
            B = pick(A)
            while v + th < iv:
                rect(-iu * rng.uniform(0.7, 1.0), iu * rng.uniform(0.7, 1.0), v, v + th,
                     col(B if rng.random() < 0.7 else pick(A)))
                v += pitch
        elif mode == "grid":
            nu, nv = rng.randint(2, 4), rng.randint(2, 3)
            gu, gv = 2 * iu / nu, 2 * iv / nv
            for i in range(nu):
                for j in range(nv):
                    u0, v0 = -iu + i * gu, -iv + j * gv
                    rect(u0 + street, u0 + gu - street, v0 + street, v0 + gv - street,
                         col(pick()))
        else:
            r = min(iu, iv) * 0.55
            ell(0, 0, r, r, col(g.lead if A != g.lead else pick(A)))
            line((-iu, 0), (iu, 0), thin * 2, col(dark))
            line((0, -iv), (0, iv), thin * 2, col(dark))
        if elab >= 1:                                       # a court within the court
            r_u, r_v = iu * 0.45, iv * 0.45
            B = pick(A)
            rect(-r_u, r_u, -r_v, r_v, col(B))
            dd = min(r_u, r_v) * 0.35
            rect(-r_u + dd, r_u - dd, -r_v + dd, r_v - dd, ground)
            if elab >= 2:
                ell(0, 0, dd * 0.9, dd * 0.9, col(g.lead))
        if elab >= 2:                                       # window rhythm on the ring
            k = max(6, int(2 * W / (d * 0.6)))
            for i in range(1, k):
                u = -W + 2 * W * i / k
                rect(u - thin / 2, u + thin / 2, -H + d * 0.2, -H + d * 0.8, ground)
                rect(u - thin / 2, u + thin / 2, H - d * 0.8, H - d * 0.2, ground)
        if rng.random() < 0.5 or elab >= 1:                 # corner towers
            q = d * 0.75
            for su in (-1, 1):
                for sv in (-1, 1):
                    rect(su * W - q, su * W + q, sv * H - q, sv * H + q, col(g.lead))

    elif kind == "stadium":
        stands, tier = pick(), pick()
        track = next((i for i in (SALMON, SIENNA, VERMILION) if g.weights[i] > 0
                      or i == g.lead), SIENNA)
        ell(0, 0, W, H, col(stands))
        k = rng.randint(16, 30)                             # stand sections
        for i in range(k):
            ang = 2 * math.pi * i / k
            line((W * 0.86 * math.cos(ang), H * 0.86 * math.sin(ang)),
                 (W * math.cos(ang), H * math.sin(ang)), thin * 1.2, ground)
        ell(0, 0, W * 0.86, H * 0.86, col(tier if tier != stands else pick(stands)))
        ell(0, 0, W * 0.74, H * 0.74, col(track))
        fu, fv = W * 0.5, H * 0.46
        rect(-fu, fu, -fv, fv, col(green()))
        lc = col(IVORY if g.ground != IVORY else PAPER)
        for p0, p1 in (((-fu, -fv), (fu, -fv)), ((fu, -fv), (fu, fv)), ((fu, fv), (-fu, fv)),
                       ((-fu, fv), (-fu, -fv)), ((0, -fv), (0, fv))):
            line(p0, p1, thin, lc)
        ring_line(0, 0, fv * 0.3, fv * 0.3, thin, lc)
        if elab >= 1:                                       # running lanes
            for f in (0.78, 0.82):
                ring_line(0, 0, W * f, H * f, thin * 0.7, lc)
        if elab >= 2:                                       # floodlights
            for su in (-1, 1):
                for sv in (-1, 1):
                    ell(su * W * 0.78, sv * H * 0.78, thin * 3, thin * 3, col(g.lead))
        if elab >= 3:                                       # entrance gates
            for ang in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
                line((W * 0.86 * math.cos(ang), H * 0.86 * math.sin(ang)),
                     (W * 1.0 * math.cos(ang), H * 1.0 * math.sin(ang)), thin * 5, ground)

    elif kind == "factory":
        A = pick()
        B = pick(A)
        hall = W * rng.uniform(0.45, 0.7)
        rect(-W, hall, -H, H, col(A))
        nstripe = rng.randint(8, 22)                        # saw-tooth roof
        pitch = (hall + W) / nstripe
        sc = col(dark if A != dark else WARM_GREY)
        for i in range(nstripe):
            u = -W + i * pitch
            rect(u, u + pitch * rng.uniform(0.18, 0.32), -H + street, H - street, sc)
        rect(hall + street * 2, W, -H, -H * rng.uniform(-0.2, 0.3), col(B))
        nch = rng.randint(1, 3)
        r = min(W, H) * rng.uniform(0.06, 0.1)
        for i in range(nch):                                # chimneys
            u = hall + street * 2 + (W - hall) * (0.3 + 0.4 * (i % 2))
            v = H * (0.45 + 0.25 * i / max(1, nch - 1)) if nch > 1 else H * 0.55
            v = min(v, H - r)
            ell(u, v, r, r, col(g.lead))
            ell(u, v, r * 0.45, r * 0.45, col(dark))
        if elab >= 1:                                       # rail spur into the hall
            line((-W - street * 4, -H * 0.6), (hall, -H * 0.6), thin * 1.5, col(dark))
        if elab >= 2:                                       # a second, cross-ridged hall
            h0 = -H + (2 * H) * 0.08
            rect(-W * 0.9, -W * 0.2, h0, h0 + H * 0.5, col(B))
            k = rng.randint(4, 9)
            for i in range(k):
                v = h0 + H * 0.5 * (i + 0.3) / k
                rect(-W * 0.9, -W * 0.2, v, v + thin * 1.5, sc)
        if elab >= 3:                                       # gantry cranes
            for u in (hall * 0.3, hall * 0.7):
                line((u, -H - street * 3), (u, H + street * 3), thin * 1.2, col(g.lead))

    elif kind == "institute":
        th = H * rng.uniform(0.12, 0.2)
        rect(-W, W, -th / 2, th / 2, col(dark))
        cu = W * rng.uniform(-0.1, 0.4)
        rect(cu - th * 0.4, cu + th * 0.4, -H, H * rng.uniform(0.2, 0.6), col(pick(dark)))
        r = H * rng.uniform(0.35, 0.6)
        du = -W * rng.uniform(0.25, 0.6)
        ell(du, -r * 0.2, r, r, col(g.lead))
        ell(du, -r * 0.2, r * 0.25, r * 0.25, col(dark if g.lead != dark else IVORY))
        n_sq = rng.randint(4, 9)
        q = th * 0.7
        for i in range(n_sq):
            u = cu + th + i * q * 1.7
            if u + q > W:
                break
            rect(u, u + q, th * 0.9, th * 0.9 + q, col(pick()))
        line((du, -r * 0.2), (W, -H * 0.9), thin, col(dark))  # a stay / cable
        if elab >= 1:                                       # a second, smaller drum
            r2 = r * 0.4
            ell(W * 0.75, H * 0.55, r2, r2, col(pick(g.lead)))
            line((W * 0.75, H * 0.55), (cu, th / 2), thin, col(dark))
        if elab >= 2:                                       # radio mast rings
            for f in (1.25, 1.5, 1.75):
                ring_line(du, -r * 0.2, r * f, r * f, thin * 0.8, col(dark))
        if elab >= 3:                                       # a fan of cables
            for k in range(5):
                line((du, -r * 0.2), (-W + 2 * W * k / 4, H), thin * 0.7, col(dark))

    elif kind == "railyard":
        # Sidings running the length of the site, gathered at both ends by a
        # short throat into a lead track that joins the railway alongside.
        tc = col(dark)
        A = pick(dark)
        rail_v = c.get("rail_v")
        lead_v = (H if rail_v is not None and rail_v > 0 else -H) * 0.92
        ntr = max(5, min(16, int(2 * H / max(street * 1.6, thin * 4))))
        gap_v = 2 * H / ntr
        tracks = [-H + gap_v * (i + 0.5) for i in range(ntr)]
        throat = W * rng.uniform(0.18, 0.26)
        # A ladder at each end: one diagonal track from the lead corner, every
        # siding branching off it, so the yard is parallel lines between two
        # diagonals rather than a fan.
        far_v = -lead_v
        span_v = far_v - lead_v
        for v in tracks:
            f = (v - lead_v) / span_v                       # 0 at the lead side
            line((-W + throat * f, v), (W - throat * f, v), thin, tc)
        line((-W, lead_v), (-W + throat, far_v), thin * 1.3, tc)
        line((W, lead_v), (W - throat, far_v), thin * 1.3, tc)
        if rail_v is not None:                              # spurs onto the main line
            reach = abs(rail_v - lead_v) * 1.6
            line((-W, lead_v), (-W - reach, rail_v), thin * 1.4, tc)
            line((W, lead_v), (W + reach, rail_v), thin * 1.4, tc)
        # Long sheds over some sidings, platforms between others.
        sheds = rng.sample(range(ntr), k=max(1, ntr // 4))
        for i in sheds:
            u0 = rng.uniform(-W + throat, W * 0.1)
            u1 = min(W - throat, u0 + W * rng.uniform(0.6, 1.1))
            rect(u0, u1, tracks[i] - gap_v * 0.36, tracks[i] + gap_v * 0.36,
                 col(A if rng.random() < 0.75 else pick(dark)))
        if elab >= 2:
            for i in range(1, ntr - 1, 3):
                v = (tracks[i] + tracks[i + 1]) / 2
                rect(-W * 0.4, W * 0.4, v - gap_v * 0.12, v + gap_v * 0.12, col(g.lead))
        # Roundhouse: a turntable with stalls fanned round it, in the corner
        # away from the line.
        if elab >= 1 or rng.random() < 0.5:
            r = min(W * 0.18, H * 0.55)
            ru = W - throat - r * 1.6
            rv = -lead_v / 0.92 * (1 - 1.6 * r / H) if H > 1.6 * r else 0.0
            rect(ru - r * 1.75, ru + r * 1.75, rv - r * 1.75, rv + r * 1.75, ground)
            k = rng.randint(7, 11)
            away = -1 if lead_v > 0 else 1
            for i in range(k):
                a_ = math.pi * (i / (k - 1)) * (1 if away > 0 else -1)
                du, dv = math.cos(a_), math.sin(a_)
                cu, cv = ru + r * 1.35 * du, rv + r * 1.35 * dv
                bar(L(cu, cv), (t[0] * du + n[0] * dv, t[1] * du + n[1] * dv),
                    (-t[0] * dv + n[0] * du, -t[1] * dv + n[1] * du),
                    r * 0.7, r * 0.26, col(A))
            ell(ru, rv, r * 0.62, r * 0.62, col(g.lead))
            line((ru - r * 0.62, rv), (-W + throat, rv), thin, tc)

    elif kind == "circus":
        # Concentric rings of wedge-shaped buildings around a round square.
        rings = rng.randint(2, 3) + (1 if elab >= 2 else 0)
        r_in = W * rng.uniform(0.25, 0.38)
        dr = (W - r_in) / rings
        rays = rng.choice([4, 6, 8])
        for k in range(rings):
            r0, r1 = r_in + k * dr + street * 0.6, r_in + (k + 1) * dr - street * 0.6
            m = rays * max(1, int(round((r0 + r1) * math.pi / rays / (dr * 1.6))))
            for i in range(m):
                a0 = 2 * math.pi * i / m + street / max(r0, 1e-3)
                a1 = 2 * math.pi * (i + 1) / m - street / max(r0, 1e-3)
                if a1 <= a0:
                    continue
                arc = [(r1 * math.cos(a0 + (a1 - a0) * j / 6), r1 * math.sin(a0 + (a1 - a0) * j / 6))
                       for j in range(7)]
                arc += [(r0 * math.cos(a1 - (a1 - a0) * j / 6), r0 * math.sin(a1 - (a1 - a0) * j / 6))
                        for j in range(7)]
                poly([L(u, v) for u, v in arc], col(pick()))
        for i in range(rays):                               # radial streets
            a_ = 2 * math.pi * (i + 0.5) / rays
            line((r_in * 0.9 * math.cos(a_), r_in * 0.9 * math.sin(a_)),
                 (W * math.cos(a_), W * math.sin(a_)), street * 1.6, ground)
        ell(0, 0, r_in * 0.7, r_in * 0.7, col(g.lead))
        if elab >= 1:
            ell(0, 0, r_in * 0.3, r_in * 0.3, col(dark))
        if elab >= 3:
            ring_line(0, 0, r_in * 0.5, r_in * 0.5, thin, ground)

    elif kind == "gasworks":
        # Gas-holders: big round tanks with guide frames, joined by pipes.
        n_h = rng.randint(2, 4) + (1 if elab >= 2 else 0)
        A, B = pick(), pick()
        centres = []
        for i in range(n_h):
            r = min(W, H) * rng.uniform(0.35, 0.6) * (0.75 if n_h > 3 else 1.0)
            u = -W + r + (2 * W - 2 * r) * (i / max(1, n_h - 1)) if n_h > 1 else 0.0
            v = rng.uniform(-H + r, H - r) if H > r else 0.0
            centres.append((u, v, r))
        for (u0_, v0_), (u1_, v1_) in zip([c_[:2] for c_ in centres], [c_[:2] for c_ in centres[1:]]):
            line((u0_, v0_), (u1_, v1_), thin * 2.2, col(dark))
        for u, v, r in centres:
            ell(u, v, r, r, col(A))
            ring_line(u, v, r * 0.82, r * 0.82, thin * 1.2, ground)
            ell(u, v, r * 0.62, r * 0.62, col(B if B != A else g.lead))
            ell(u, v, r * 0.16, r * 0.16, col(g.lead))
            if elab >= 1:                                   # guide-frame columns
                for k in range(12):
                    a_ = 2 * math.pi * k / 12
                    ell(u + r * 1.06 * math.cos(a_), v + r * 1.06 * math.sin(a_),
                        thin * 2, thin * 2, col(dark), k=8)

    elif kind == "planetarium":
        # A painted dome: stacked discs stepping toward the light.
        r = W * 0.62
        base_c = col(g.lead)
        steps = 6
        for i in range(steps):
            f = 1 - i / (steps + 1)
            off = r * 0.07 * i
            ell(-off, -off, r * f, r * f, _mixc(base_c, (250, 246, 236), 0.11 * i))
        ring_line(0, 0, r * 1.12, r * 1.12, thin * 1.5, col(dark))
        th = r * 0.22
        rect(-W, W, r * 1.25, r * 1.25 + th, col(pick(g.lead)))       # a long wing
        if elab >= 1:
            ell(W * 0.72, -W * 0.7, r * 0.22, r * 0.22, col(dark))    # observatory
        if elab >= 2:
            for k in range(8):                                        # meridians
                a_ = math.pi * k / 8
                line((-r * math.cos(a_), -r * math.sin(a_)),
                     (r * math.cos(a_), r * math.sin(a_)), thin * 0.6,
                     _mixc(base_c, _WALL_DARK, 0.35))
        if elab >= 3:
            ring_line(0, 0, r * 1.3, r * 1.3, thin, col(g.lead))

    else:                                                   # park
        pc = green()
        rect(-W, W, -H, H, col(pc))
        pathc = ground
        line((-W, -H), (W, H), street * 1.2, pathc)
        line((-W, H), (W, -H), street * 1.2, pathc)
        line((-W, 0), (W, 0), street, pathc)
        r = min(W, H) * rng.uniform(0.25, 0.4)
        ell(0, 0, r + street, r + street, ground)
        pond = water if water != ground else col(SKY if g.ground != SKY else ULTRA)
        ell(0, 0, r, r, pond)
        tree = col(dark if pc != dark else WARM_GREY)
        step = max(street * 2.2, min(W, H) * 0.08)
        u = -W + step
        while u < W - step * 0.5:                          # allées
            for v in (-H + step * 0.6, H - step * 0.6):
                ell(u, v, step * 0.22, step * 0.22, tree, k=10)
            u += step
        if elab >= 1:                                       # a ring walk
            ring_line(0, 0, r * 1.5, r * 1.5, street * 0.8, ground)
        if elab >= 2:                                       # flower beds
            q = step * 0.4
            for su in (-1, 1):
                for sv in (-1, 1):
                    for k in range(3):
                        u, v = su * (r * 1.8 + k * q * 2.2), sv * r * 0.5
                        if abs(u) + q < W:
                            rect(u - q, u + q, v - q, v + q, col(pick(BLACK)))
        if elab >= 3:                                       # radial paths
            for k in range(8):
                ang = 2 * math.pi * k / 8
                line((r * 1.05 * math.cos(ang), r * 1.05 * math.sin(ang)),
                     (W * math.cos(ang), H * math.sin(ang)), street * 0.6, ground)
        for _ in range(rng.randint(1, 3) + elab):           # pavilions
            pu, pv = rng.uniform(-W * 0.8, W * 0.8), rng.uniform(-H * 0.7, H * 0.7)
            if math.hypot(pu, pv) > r * 1.3:
                q = step * 0.5
                rect(pu - q, pu + q, pv - q, pv + q, col(g.lead))


def _paper(img: Image.Image, ground: tuple[int, int, int], grain: float,
           rng: random.Random) -> Image.Image:
    """The sheet's own tooth: fine grain in the paint, faint mottling in the
    bare paper.  No gradients or vignettes."""
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    small = nrng.standard_normal((h // 2 + 1, w // 2 + 1)).astype(np.float32)
    tooth = np.asarray(Image.fromarray(((small + 4) * 32).clip(0, 255).astype(np.uint8))
                       .resize((w, h), Image.BILINEAR), dtype=np.float32) / 32 - 4
    arr += tooth[..., None] * (grain * 0.45)
    bare = (np.abs(arr - np.array(ground, np.float32)).sum(-1) < 30).astype(np.float32)
    blot = nrng.standard_normal((9, 9)).astype(np.float32)
    blot = np.asarray(Image.fromarray(((blot + 4) * 32).clip(0, 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC), dtype=np.float32) / 32 - 4
    arr += (blot * 5.0 * bare)[..., None] * np.array([1.0, 0.9, 0.7], np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")

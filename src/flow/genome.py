"""Flow-landscape genome.

A painting is a stack of landscape primitives, each of which owns a flow
field that its brushstrokes follow:

  bands     hill / field / water strips from the horizon down to the
            foreground; strokes run along their crest line, fields can rake
            toward the vanishing point (furrows), water lies flat and ripples
  orb       sun, moon or star with concentric halo rings (or radial rays)
  swirl     a vortex in the sky that bends the sky strokes around it
  cypress   a flame-shaped tree whose strokes climb and flicker
  tree      a round crown on a trunk, strokes curling around the crown
  road      a path or bridge sweeping from the foreground to the horizon

On top of that a small NEAT CPPN supplies turbulence (rotating the flow),
value variation and terrain wobble, and three style weights (Van Gogh,
Munch, Matisse) set the handling: short impasto dashes, long sinuous
ribbons, or flat fields with heavy contour.

``maturity`` is the convergence gene.  At 0 the picture is raw and
neo-expressionist -- turbulent flow, strokes crossing shapes, clashing
colour, bare ground, drips and scrawls.  Children inherit their parents'
maturity plus a drift, so a lineage settles toward a clean, coherent oil
painting as it evolves.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

from src.flow.pigments import ACCENTS, CHOICES
from src.neat.genome import Genome as CPPNGenome
from src.neat.genome import crossover as cppn_crossover
from src.neat.population import _dict_to_genome, _genome_to_dict

BAND_KINDS = ("hill", "field", "water")
FEATURE_KINDS = ("orb", "swirl", "cypress", "tree", "road")
MOTIFS = ("starry", "wheatfield", "fjord", "riviera", "orchard")

MAX_BANDS = 6
MAX_FEATURES = 9

# CPPN fields: inputs (x, y, d, bias) -> (turbulence, value, terrain).
CPPN_INPUTS = 4
CPPN_OUTPUTS = 3
CPPN_ACTIVATIONS = ["sin", "gaussian", "tanh", "sigmoid", "abs", "linear"]
CPPN_MUTATION = {
    "weight_perturb_rate": 0.8, "weight_perturb_power": 0.4,
    "weight_replace_rate": 0.05, "add_node_rate": 0.12,
    "add_connection_rate": 0.18, "activation_mutation_rate": 0.1,
    "toggle_enable_rate": 0.02, "activation_options": CPPN_ACTIVATIONS,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _accent_for(p: str) -> str:
    return random.choice(ACCENTS.get(p, ["zinc_white"]))


@dataclass
class Element:
    kind: str = "hill"
    x: float = 0.5          # horizontal anchor, fraction of the width
    y: float = 0.6          # crest (bands) or base / centre (features), height = 1
    size: float = 0.1       # orb/swirl radius, tree/cypress height, road width
    amp: float = 0.03       # crest undulation
    freq: float = 1.0       # crest undulation, cycles across the canvas
    phase: float = 0.0
    tilt: float = 0.0       # crest slope (bands) / road bend (road)
    twist: float = 0.5      # cypress sway, swirl tightness, tree lumpiness
    spin: int = 1           # swirl / halo rotation direction
    rays: float = 0.0       # orb: 0 concentric halo .. 1 radial rays
    halo: float = 2.0       # orb halo radius, multiple of the disc
    furrows: float = 0.0    # field: 0 strokes along the crest .. 1 toward the VP
    end_x: float = 0.5      # road: where it meets the horizon
    pigment: str = "viridian"
    accent: str = "emerald_green"


@dataclass
class FlowGenome:
    seed: int = 0
    motif: str = "starry"
    vp_x: float = 0.5

    sky_top: str = "ultramarine"
    sky_low: str = "cerulean"
    sky_accent: str = "zinc_white"
    sky_wave: float = 0.2     # Munch's banded, undulating sky
    sky_freq: float = 4.0

    bands: list[Element] = field(default_factory=list)
    features: list[Element] = field(default_factory=list)

    vg: float = 0.5           # style weights (normalised in repair)
    munch: float = 0.25
    matisse: float = 0.25

    maturity: float = 0.1     # 0 raw neo-expressionist .. 1 clean oil painting
    ground: str = "raw_sienna"
    impasto: float = 0.5
    contour: float = 0.4
    stroke_scale: float = 1.0
    turbulence: float = 0.5

    cppn: CPPNGenome | None = None

    # Interface shared with the shapes engines.
    fitness: float = 0.0
    engine_tag: str = "flow"
    comp: None = None

    def copy(self) -> "FlowGenome":
        g = copy.copy(self)
        g.bands = [copy.copy(e) for e in self.bands]
        g.features = [copy.copy(e) for e in self.features]
        g.cppn = self.cppn.copy() if self.cppn is not None else None
        g.fitness = 0.0
        return g

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("cppn", "bands", "features", "comp")}
        d["bands"] = [asdict(e) for e in self.bands]
        d["features"] = [asdict(e) for e in self.features]
        d["cppn"] = _genome_to_dict(self.cppn) if self.cppn is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FlowGenome":
        d = dict(d)
        bands = [Element(**e) for e in d.pop("bands", [])]
        feats = [Element(**e) for e in d.pop("features", [])]
        cp = d.pop("cppn", None)
        d.pop("comp", None)
        g = cls(**d)
        g.bands, g.features = bands, feats
        g.cppn = _dict_to_genome(cp) if cp else None
        return g

    @property
    def horizon(self) -> float:
        return min((b.y for b in self.bands), default=0.55)


# ---------------------------------------------------------------------------
# Creation from motif priors
# ---------------------------------------------------------------------------

def _new_cppn() -> CPPNGenome:
    g = CPPNGenome.create_minimal(0, CPPN_INPUTS, CPPN_OUTPUTS, "tanh", 2.0)
    for _ in range(random.randint(1, 3)):
        g.mutate_add_node(CPPN_ACTIVATIONS)
    for _ in range(2):
        g.mutate_add_connection()
    return g


def _pick(kind: str, avoid: set[str] = frozenset()) -> str:
    opts = [p for p in CHOICES[kind] if p not in avoid] or CHOICES[kind]
    return random.choice(opts)


def _band(kind: str, y: float, avoid: set[str], pig: str | None = None) -> Element:
    pig = pig or _pick(kind, avoid)
    if kind == "hill":
        amp, freq, tilt = random.uniform(0.02, 0.08), random.uniform(0.5, 2.2), random.uniform(-0.12, 0.12)
    elif kind == "field":
        amp, freq, tilt = random.uniform(0.004, 0.03), random.uniform(0.4, 1.5), random.uniform(-0.25, 0.25)
    else:
        amp, freq, tilt = random.uniform(0.0, 0.008), random.uniform(0.5, 2.0), random.uniform(-0.04, 0.04)
    return Element(kind=kind, y=y, amp=amp, freq=freq,
                   phase=random.uniform(0, 6.283), tilt=tilt,
                   furrows=random.uniform(0.3, 0.9) if kind == "field" and random.random() < 0.6 else 0.0,
                   pigment=pig, accent=_accent_for(pig))


def _orb(x: float, y: float, r: float, pig: str | None = None) -> Element:
    pig = pig or _pick("orb")
    return Element(kind="orb", x=x, y=y, size=r, spin=random.choice([-1, 1]),
                   rays=random.choices([0.0, random.uniform(0.5, 1.0)], [0.65, 0.35])[0],
                   halo=random.uniform(1.6, 3.2), pigment=pig, accent=_accent_for(pig))


def _swirl(x: float, y: float, r: float) -> Element:
    return Element(kind="swirl", x=x, y=y, size=r, spin=random.choice([-1, 1]),
                   twist=random.uniform(0.2, 0.9), pigment="zinc_white",
                   accent=random.choice(["lemon_yellow", "cerulean", "zinc_white"]))


def _cypress(x: float, base: float, h: float) -> Element:
    pig = _pick("cypress")
    return Element(kind="cypress", x=x, y=base, size=h, twist=random.uniform(0.2, 1.0),
                   phase=random.uniform(0, 6.283), amp=random.uniform(0.1, 0.16),
                   pigment=pig, accent=_accent_for(pig))


def _tree(x: float, base: float, h: float) -> Element:
    pig = _pick("tree")
    return Element(kind="tree", x=x, y=base, size=h, twist=random.uniform(0.05, 0.25),
                   phase=random.uniform(0, 6.283), spin=random.choice([-1, 1]),
                   pigment=pig, accent=_accent_for(pig))


def _road(x0: float, end_x: float, width: float, pig: str | None = None) -> Element:
    pig = pig or _pick("road")
    return Element(kind="road", x=x0, y=1.05, size=width, end_x=end_x,
                   tilt=random.uniform(-0.35, 0.35), pigment=pig, accent=_accent_for(pig))


def _bands_between(horizon: float, kinds: list[str]) -> list[Element]:
    n = len(kinds)
    ys = [horizon]
    for i in range(1, n):
        t = i / n
        ys.append(horizon + (0.97 - horizon) * (t ** 1.25) + random.uniform(-0.025, 0.025))
    out, used = [], set()
    for kind, y in zip(kinds, ys):
        b = _band(kind, y, used)
        used = {b.pigment}
        out.append(b)
    return out


def create_random(maturity: float | None = None) -> FlowGenome:
    motif = random.choice(MOTIFS)
    g = FlowGenome(seed=random.randrange(1 << 30), motif=motif,
                   vp_x=random.uniform(0.3, 0.7), cppn=_new_cppn())
    g.maturity = random.uniform(0.0, 0.25) if maturity is None else maturity
    g.impasto = random.uniform(0.3, 0.9)
    g.contour = random.uniform(0.1, 0.7)
    g.stroke_scale = random.uniform(0.8, 1.25)
    g.turbulence = random.uniform(0.3, 1.0)
    g.ground = _pick("ground")
    g.sky_freq = random.uniform(2.5, 7.0)

    if motif == "starry":
        hz = random.uniform(0.6, 0.74)
        g.vg, g.munch, g.matisse = 1.0, random.uniform(0, 0.3), random.uniform(0, 0.2)
        g.sky_top = random.choice(["ultramarine", "prussian_blue", "cobalt_blue"])
        g.sky_low = random.choice(["cerulean", "cobalt_blue", "zinc_white"])
        g.sky_accent = random.choice(["lemon_yellow", "zinc_white", "cerulean"])
        g.sky_wave = random.uniform(0.0, 0.3)
        g.bands = _bands_between(hz, random.choice([["hill", "hill"], ["hill", "hill", "field"],
                                                    ["hill", "field"]]))
        g.features.append(_orb(random.uniform(0.6, 0.9), random.uniform(0.1, 0.25),
                               random.uniform(0.04, 0.07), random.choice(["lemon_yellow", "chrome_yellow"])))
        for _ in range(random.randint(1, 3)):
            g.features.append(_swirl(random.uniform(0.2, 0.75), random.uniform(0.2, hz - 0.15),
                                     random.uniform(0.1, 0.22)))
        for _ in range(random.randint(2, 5)):
            g.features.append(_orb(random.uniform(0.05, 0.95), random.uniform(0.05, hz - 0.12),
                                   random.uniform(0.012, 0.025), random.choice(["lemon_yellow", "zinc_white"])))
        if random.random() < 0.85:
            g.features.append(_cypress(random.choice([random.uniform(0.08, 0.3), random.uniform(0.7, 0.92)]),
                                       1.0, random.uniform(0.55, 0.85)))
    elif motif == "wheatfield":
        hz = random.uniform(0.38, 0.55)
        g.vg, g.munch, g.matisse = 1.0, random.uniform(0, 0.2), random.uniform(0, 0.4)
        g.sky_top = random.choice(["cobalt_blue", "ultramarine", "cerulean"])
        g.sky_low = random.choice(["zinc_white", "cerulean", "lemon_yellow"])
        g.sky_accent = "zinc_white"
        g.sky_wave = random.uniform(0.0, 0.25)
        kinds = ["hill"] + ["field"] * random.randint(2, 3)
        g.bands = _bands_between(hz, kinds)
        for b in g.bands[1:]:
            b.pigment = random.choice(["chrome_yellow", "yellow_ochre", "lemon_yellow", "olive_green",
                                       "emerald_green"])
            b.accent = _accent_for(b.pigment)
        if random.random() < 0.7:
            g.features.append(_orb(random.uniform(0.15, 0.85), random.uniform(0.1, hz * 0.6),
                                   random.uniform(0.04, 0.08), random.choice(["lemon_yellow", "chrome_yellow"])))
        for _ in range(random.randint(0, 2)):
            g.features.append(_swirl(random.uniform(0.15, 0.85), random.uniform(0.08, hz * 0.7),
                                     random.uniform(0.08, 0.16)))
        for _ in range(random.randint(0, 2)):
            g.features.append(_cypress(random.uniform(0.1, 0.9), random.uniform(hz + 0.05, hz + 0.2),
                                       random.uniform(0.2, 0.4)))
        if random.random() < 0.4:
            g.features.append(_road(random.uniform(0.2, 0.8), g.vp_x, random.uniform(0.05, 0.12)))
    elif motif == "fjord":
        hz = random.uniform(0.35, 0.5)
        g.vg, g.munch, g.matisse = random.uniform(0, 0.3), 1.0, random.uniform(0, 0.3)
        g.sky_top = random.choice(["vermilion", "chrome_orange", "cadmium_red", "cobalt_blue"])
        g.sky_low = random.choice(["chrome_yellow", "chrome_orange", "lemon_yellow", "rose"])
        g.sky_accent = random.choice(["cobalt_blue", "lemon_yellow", "crimson_lake"])
        g.sky_wave = random.uniform(0.5, 1.0)
        kinds = random.choice([["hill", "water", "hill"], ["hill", "water", "field"],
                               ["water", "hill"], ["hill", "water"]])
        g.bands = _bands_between(hz, kinds)
        for b in g.bands:
            if b.kind == "hill":
                b.pigment = random.choice(["prussian_blue", "viridian", "olive_green", "ultramarine"])
                b.accent = _accent_for(b.pigment)
        if random.random() < 0.75:
            g.features.append(_road(random.choice([random.uniform(-0.1, 0.25), random.uniform(0.75, 1.1)]),
                                    random.uniform(0.3, 0.7), random.uniform(0.18, 0.3)))
        if random.random() < 0.4:
            g.features.append(_orb(random.uniform(0.2, 0.8), random.uniform(hz - 0.18, hz - 0.06),
                                   random.uniform(0.04, 0.08), random.choice(["chrome_yellow", "zinc_white"])))
    elif motif == "riviera":
        hz = random.uniform(0.4, 0.6)
        g.vg, g.munch, g.matisse = random.uniform(0, 0.25), random.uniform(0, 0.25), 1.0
        g.sky_top = random.choice(["cerulean", "rose", "cobalt_blue", "lemon_yellow"])
        g.sky_low = random.choice(["rose", "zinc_white", "lemon_yellow", "chrome_orange"])
        g.sky_accent = "zinc_white"
        g.sky_wave = random.uniform(0.0, 0.3)
        kinds = random.choice([["water", "hill", "field"], ["hill", "field", "field"],
                               ["water", "field"], ["hill", "water", "field"]])
        g.bands = _bands_between(hz, kinds)
        for b in g.bands:
            if b.kind != "water":
                b.pigment = random.choice(["rose", "cobalt_violet", "emerald_green", "vermilion",
                                           "chrome_orange", "lemon_yellow", "viridian"])
                b.accent = _accent_for(b.pigment)
        g.contour = random.uniform(0.5, 1.0)
        for _ in range(random.randint(1, 3)):
            g.features.append(_tree(random.uniform(0.1, 0.9), random.uniform(hz + 0.15, 1.0),
                                    random.uniform(0.2, 0.42)))
        if random.random() < 0.5:
            g.features.append(_orb(random.uniform(0.15, 0.85), random.uniform(0.08, hz * 0.6),
                                   random.uniform(0.05, 0.09), random.choice(["chrome_orange", "vermilion",
                                                                              "lemon_yellow"])))
    else:  # orchard
        hz = random.uniform(0.35, 0.5)
        g.vg, g.munch, g.matisse = 1.0, random.uniform(0, 0.3), random.uniform(0.2, 0.7)
        g.sky_top = random.choice(["cerulean", "cobalt_blue", "zinc_white"])
        g.sky_low = random.choice(["zinc_white", "lemon_yellow", "rose"])
        g.sky_accent = random.choice(["zinc_white", "rose"])
        g.sky_wave = random.uniform(0.0, 0.3)
        g.bands = _bands_between(hz, ["hill", "field", "field"])
        n = random.randint(2, 4)
        for i in range(n):
            base = hz + 0.15 + random.uniform(0, 0.4)
            g.features.append(_tree(0.12 + 0.76 * (i + random.uniform(0.2, 0.8)) / n, base,
                                    0.15 + 0.45 * (base - hz)))
        if random.random() < 0.5:
            g.features.append(_road(random.uniform(0.3, 0.7), g.vp_x, random.uniform(0.06, 0.14)))

    repair(g)
    return g


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def repair(g: FlowGenome) -> None:
    g.vp_x = _clamp(g.vp_x, 0.15, 0.85)
    g.sky_wave = _clamp(g.sky_wave, 0.0, 1.0)
    g.sky_freq = _clamp(g.sky_freq, 1.5, 9.0)
    for name in ("maturity", "impasto", "contour", "turbulence"):
        setattr(g, name, _clamp(getattr(g, name), 0.0, 1.0))
    g.stroke_scale = _clamp(g.stroke_scale, 0.6, 1.6)
    for name in ("vg", "munch", "matisse"):
        setattr(g, name, _clamp(getattr(g, name), 0.0, 1.0))
    total = g.vg + g.munch + g.matisse
    if total < 1e-3:
        g.vg, total = 1.0, 1.0
    g.vg, g.munch, g.matisse = g.vg / total, g.munch / total, g.matisse / total
    if g.sky_low == g.sky_top:
        g.sky_low = "zinc_white" if g.sky_top != "zinc_white" else "cerulean"

    if not g.bands:
        g.bands = [_band("hill", 0.55, set())]
    g.bands = sorted(g.bands, key=lambda b: b.y)[:MAX_BANDS]
    g.bands[0].y = _clamp(g.bands[0].y, 0.28, 0.78)
    for b in g.bands:
        if b.kind not in BAND_KINDS:
            b.kind = "hill"
        b.y = _clamp(b.y, 0.28, 0.97)
        b.amp = _clamp(b.amp, 0.0, 0.1)
        b.freq = _clamp(b.freq, 0.2, 3.0)
        b.tilt = _clamp(b.tilt, -0.3, 0.3)
        b.furrows = _clamp(b.furrows, 0.0, 1.0) if b.kind == "field" else 0.0
    for b1, b2 in zip(g.bands, g.bands[1:]):
        if b2.pigment == b1.pigment:
            b2.pigment = _pick(b2.kind, {b1.pigment})
            b2.accent = _accent_for(b2.pigment)
        if b2.y - b1.y < 0.04:
            b2.y = min(0.97, b1.y + 0.04)
    if g.bands[0].pigment == g.sky_low:
        g.bands[0].pigment = _pick(g.bands[0].kind, {g.sky_low})

    hz = g.horizon
    g.features = g.features[:MAX_FEATURES]
    for f in g.features:
        f.x = _clamp(f.x, -0.15, 1.15) if f.kind == "road" else _clamp(f.x, 0.03, 0.97)
        f.spin = 1 if f.spin >= 0 else -1
        if f.kind == "orb":
            f.size = _clamp(f.size, 0.008, 0.12)
            f.y = _clamp(f.y, 0.04, max(0.06, hz - 0.02))
            f.rays = _clamp(f.rays, 0.0, 1.0)
            f.halo = _clamp(f.halo, 1.2, 3.6)
        elif f.kind == "swirl":
            f.size = _clamp(f.size, 0.05, 0.28)
            f.y = _clamp(f.y, 0.05, max(0.08, hz - 0.05))
            f.twist = _clamp(f.twist, 0.0, 1.0)
        elif f.kind == "cypress":
            f.y = _clamp(f.y, hz + 0.02, 1.05)
            f.size = _clamp(f.size, 0.12, 0.95)
            f.amp = _clamp(f.amp, 0.06, 0.2)
            f.twist = _clamp(f.twist, 0.0, 1.2)
        elif f.kind == "tree":
            f.y = _clamp(f.y, hz + 0.03, 1.05)
            f.size = _clamp(f.size, 0.08, 0.6)
            f.twist = _clamp(f.twist, 0.0, 0.35)
        elif f.kind == "road":
            f.y = 1.05
            f.size = _clamp(f.size, 0.03, 0.35)
            f.end_x = _clamp(f.end_x, 0.1, 0.9)
            f.tilt = _clamp(f.tilt, -0.5, 0.5)
    # Paint order: sky features, then roads, then trees nearest-last.
    order = {"swirl": 0, "orb": 1, "road": 2, "cypress": 3, "tree": 3}
    g.features.sort(key=lambda f: (order[f.kind], f.y if order[f.kind] == 3 else 0))


# ---------------------------------------------------------------------------
# Variation
# ---------------------------------------------------------------------------

def _gauss(v: float, sigma: float, p: float, heat: float) -> float:
    return v + random.gauss(0, sigma * heat) if random.random() < p else v


def mutate(g: FlowGenome, strength: float = 1.0, temp: float = 1.0,
           drift: float = 0.0) -> None:
    """Perturb genes.  ``temp`` (1 explore .. 0.2 refine) scales how many
    genes change and by how much; ``drift`` is added to maturity so
    offspring converge toward the finished painting."""
    heat = max(0.05, strength) * temp
    p = min(0.9, 0.25 * strength * (0.5 + temp))
    rare = 0.12 * strength * temp

    g.maturity += drift + random.gauss(0, 0.03 * temp)
    g.vp_x = _gauss(g.vp_x, 0.06, p, heat)
    g.sky_wave = _gauss(g.sky_wave, 0.15, p, heat)
    g.sky_freq = _gauss(g.sky_freq, 0.8, p, heat)
    g.impasto = _gauss(g.impasto, 0.15, p, heat)
    g.contour = _gauss(g.contour, 0.15, p, heat)
    g.stroke_scale = _gauss(g.stroke_scale, 0.12, p, heat)
    g.turbulence = _gauss(g.turbulence, 0.15, p, heat)
    for name in ("vg", "munch", "matisse"):
        setattr(g, name, _gauss(getattr(g, name), 0.15, p, heat))

    if random.random() < rare:
        g.seed = random.randrange(1 << 30)
    if random.random() < rare:
        g.sky_top = _pick("sky")
    if random.random() < rare:
        g.sky_low = _pick("sky_low")
    if random.random() < rare:
        g.sky_accent = _accent_for(g.sky_top)
    if random.random() < rare * 0.5:
        g.ground = _pick("ground")

    for b in g.bands:
        b.y = _gauss(b.y, 0.03, p, heat)
        b.amp = _gauss(b.amp, 0.015, p, heat)
        b.freq = _gauss(b.freq, 0.3, p, heat)
        b.phase = _gauss(b.phase, 0.6, p, heat)
        b.tilt = _gauss(b.tilt, 0.06, p, heat)
        b.furrows = _gauss(b.furrows, 0.2, p, heat)
        if random.random() < rare:
            b.pigment = _pick(b.kind)
            b.accent = _accent_for(b.pigment)
        elif random.random() < rare:
            b.accent = _accent_for(b.pigment)
        if random.random() < rare * 0.3:
            b.kind = random.choice(BAND_KINDS)

    for f in g.features:
        f.x = _gauss(f.x, 0.05, p, heat)
        f.y = _gauss(f.y, 0.04, p, heat)
        f.size = _gauss(f.size, 0.12 * f.size + 0.005, p, heat)
        f.twist = _gauss(f.twist, 0.15, p, heat)
        f.phase = _gauss(f.phase, 0.5, p, heat)
        f.tilt = _gauss(f.tilt, 0.1, p, heat)
        f.rays = _gauss(f.rays, 0.2, p * 0.5, heat)
        f.halo = _gauss(f.halo, 0.3, p, heat)
        f.end_x = _gauss(f.end_x, 0.06, p, heat)
        if random.random() < rare * 0.5:
            f.spin = -f.spin
        if random.random() < rare and f.kind != "swirl":
            f.pigment = _pick(f.kind)
            f.accent = _accent_for(f.pigment)

    # Structural changes: add / remove a band or a feature.
    hz = g.horizon
    if random.random() < rare * 0.5 and len(g.bands) < MAX_BANDS:
        kind = random.choice(BAND_KINDS)
        g.bands.append(_band(kind, random.uniform(hz, 0.95), set()))
    if random.random() < rare * 0.4 and len(g.bands) > 1:
        g.bands.pop(random.randrange(1, len(g.bands)))
    if random.random() < rare * 0.7 and len(g.features) < MAX_FEATURES:
        k = random.choice(FEATURE_KINDS)
        if k == "orb":
            g.features.append(_orb(random.uniform(0.1, 0.9), random.uniform(0.05, hz - 0.05),
                                   random.uniform(0.015, 0.08)))
        elif k == "swirl":
            g.features.append(_swirl(random.uniform(0.1, 0.9), random.uniform(0.08, hz - 0.08),
                                     random.uniform(0.07, 0.2)))
        elif k == "cypress":
            base = random.uniform(hz + 0.05, 1.02)
            g.features.append(_cypress(random.uniform(0.08, 0.92), base, 0.15 + 0.9 * (base - hz)))
        elif k == "tree":
            base = random.uniform(hz + 0.08, 1.02)
            g.features.append(_tree(random.uniform(0.08, 0.92), base, 0.1 + 0.55 * (base - hz)))
        elif not any(f.kind == "road" for f in g.features):
            g.features.append(_road(random.uniform(0.0, 1.0), g.vp_x, random.uniform(0.05, 0.25)))
    if random.random() < rare * 0.5 and g.features:
        g.features.pop(random.randrange(len(g.features)))

    if g.cppn is not None and random.random() < 0.3 + 0.5 * temp:
        g.cppn.mutate(CPPN_MUTATION)

    repair(g)


_BLOCKS = [
    ("sky_top", "sky_low", "sky_accent", "sky_wave", "sky_freq"),
    ("vg", "munch", "matisse"),
    ("ground", "impasto", "contour", "stroke_scale", "turbulence"),
    ("seed", "vp_x"),
]


def crossover(a: FlowGenome, b: FlowGenome) -> FlowGenome:
    """Swap coherent gene blocks (sky, style, handling).  The land comes
    from one parent, the features from either, with a chance of borrowing a
    single feature across."""
    child = a.copy()
    for block in _BLOCKS:
        if random.random() < 0.5:
            for name in block:
                setattr(child, name, getattr(b, name))
    if random.random() < 0.5:
        child.bands = [copy.copy(e) for e in b.bands]
    if random.random() < 0.5:
        child.features = [copy.copy(e) for e in b.features]
    donor = b if random.random() < 0.5 else a
    if donor.features and random.random() < 0.35 and len(child.features) < MAX_FEATURES:
        child.features.append(copy.copy(random.choice(donor.features)))
    child.maturity = 0.5 * (a.maturity + b.maturity)
    child.motif = random.choice([a.motif, b.motif])
    if a.cppn is not None and b.cppn is not None:
        child.cppn = cppn_crossover(a.cppn, b.cppn, 0) if random.random() < 0.5 \
            else random.choice([a, b]).cppn.copy()
    repair(child)
    return child


# ---------------------------------------------------------------------------
# Prior score (used to pick among sibling candidates)
# ---------------------------------------------------------------------------

def composition_score(g: FlowGenome) -> float:
    """Cheap compositional prior: a horizon off the centre line, bands of
    varied depth, features balanced but not centred, and features that do
    not pile onto one another."""
    score = -3.0 * max(0.0, 0.06 - abs(g.horizon - 0.5))
    ys = [b.y for b in g.bands] + [1.0]
    gaps = [y2 - y1 for y1, y2 in zip(ys, ys[1:])]
    if len(gaps) > 1:
        mean = sum(gaps) / len(gaps)
        score -= 2.0 * (sum((x - mean) ** 2 for x in gaps) / len(gaps)) ** 0.5 / (mean + 1e-3) * 0.3
    solid = [f for f in g.features if f.kind in ("orb", "cypress", "tree")]
    if solid:
        w = [f.size for f in solid]
        cx = sum(f.x * wi for f, wi in zip(solid, w)) / sum(w)
        score -= 1.5 * max(0.0, abs(cx - 0.5) - 0.2)
        score -= 0.5 * max(0.0, 0.05 - abs(cx - 0.5)) * 10
    for i, f1 in enumerate(solid):
        for f2 in solid[i + 1:]:
            if abs(f1.x - f2.x) < 0.06 and abs(f1.y - f2.y) < 0.1:
                score -= 0.8
    if len(g.features) > 7:
        score -= 0.3 * (len(g.features) - 7)
    return score

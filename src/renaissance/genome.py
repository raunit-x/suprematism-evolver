"""Scene genome for Renaissance sacred painting.

A genome is a set of explicit, interpretable genes -- setting, one-point
perspective, light, sky, landscape, architecture, a list of draped figures
and the painting's finish -- plus a small CPPN (a NEAT genome) whose
outputs are smooth fields the renderer uses for cloud forms, drapery fold
drift and landscape variation.

Random genomes are not uniform noise: they are drawn from compositional
priors taken from 15th/16th-century practice (pyramidal grouping,
sacra conversazione, isocephalic friezes, the Virgin's blue mantle over a
red robe, hieratic scale, figures standing on the ground plane so their
heads sit near the horizon).
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

from src.neat.genome import Genome as CPPNGenome
from src.neat.genome import crossover as cppn_crossover
from src.neat.population import _dict_to_genome, _genome_to_dict
from src.renaissance.pigments import DRAPERY_PIGMENTS, GROUND_PIGMENTS, HAIR_PIGMENTS

SETTINGS = ["landscape", "loggia", "nave"]
COMPOSITIONS = ["pyramid", "frieze", "diagonal", "enthroned"]
POSES = ["standing", "kneeling", "seated"]
GESTURES = ["prayer", "open", "raised", "pointing", "cradle", "down"]

# CPPN fields: inputs (x, y, d, bias) -> (clouds, fold drift, terrain).
CPPN_INPUTS = 4
CPPN_OUTPUTS = 3
CPPN_ACTIVATIONS = ["sin", "gaussian", "tanh", "sigmoid", "abs", "linear"]
CPPN_MUTATION = {
    "weight_perturb_rate": 0.8, "weight_perturb_power": 0.4,
    "weight_replace_rate": 0.05, "add_node_rate": 0.15,
    "add_connection_rate": 0.2, "activation_mutation_rate": 0.1,
    "toggle_enable_rate": 0.02, "activation_options": CPPN_ACTIVATIONS,
}

MAX_FIGURES = 7


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Figure:
    x: float = 0.5            # horizontal position of the feet, canvas fraction
    depth: float = 0.0        # 0 = front plane .. 1 = near the horizon
    pose: str = "standing"
    facing: int = 1           # +1 turned right, -1 turned left
    lean: float = 0.0
    head_tilt: float = 0.0
    gesture: str = "down"
    robe: str = "vermilion"
    mantle: str | None = "ultramarine"
    veil: bool = False        # mantle drawn up over the head
    mantle_open: float = 0.05  # width of the robe left visible down the front
    halo: bool = True
    skin: float = 0.3
    hair: str = "burnt_umber"
    bulk: float = 1.0
    flare: float = 1.0
    folds: float = 7.0
    fold_depth: float = 0.5
    scale: float = 1.0        # hieratic scale on top of perspective


@dataclass
class SceneGenome:
    seed: int = 0
    setting: str = "landscape"
    composition: str = "pyramid"
    horizon: float = 0.45
    vp_x: float = 0.5

    light_x: float = -0.6     # -1 light from the left .. +1 from the right
    light_elev: float = 0.6
    chiaroscuro: float = 0.6  # shadow depth
    darkness: float = 0.0     # tenebrism: how far the setting sinks into shadow

    sky_top: str = "ultramarine"
    sky_low: str = "lead_white"
    sky_warmth: float = 0.2   # dawn warmth near the horizon
    clouds: float = 0.4
    gloria: float = 0.0       # banded golden glory in the sky

    hill_layers: int = 3
    hill_height: float = 0.12
    hill_rough: float = 0.5
    trees: int = 3

    room_width: float = 0.25  # back-wall half width
    room_top: float = 0.30    # back-wall top above horizon
    room_floor: float = 0.10  # back-wall bottom below horizon
    openings: int = 3
    tiles: int = 6
    floor_a: str = "lead_white"
    floor_b: str = "red_ochre"
    pilasters: int = 3

    throne: bool = False
    cloth_of_honor: str = "madder_lake"

    figures: list[Figure] = field(default_factory=list)

    ground: str = "yellow_ochre"  # imprimatura tone showing through thin paint
    glaze: float = 0.6
    brush: float = 0.5
    varnish: float = 0.3
    craquelure: float = 0.3

    cppn: CPPNGenome | None = None

    # Interface shared with the shapes engines.
    fitness: float = 0.0
    engine_tag: str = "renaissance"
    comp: None = None

    def copy(self) -> "SceneGenome":
        g = copy.copy(self)
        g.figures = [copy.copy(f) for f in self.figures]
        g.cppn = self.cppn.copy() if self.cppn is not None else None
        g.fitness = 0.0
        return g

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("cppn", "figures", "comp")}
        d["figures"] = [asdict(f) for f in self.figures]
        d["cppn"] = _genome_to_dict(self.cppn) if self.cppn is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SceneGenome":
        d = dict(d)
        figs = [Figure(**f) for f in d.pop("figures", [])]
        cp = d.pop("cppn", None)
        d.pop("comp", None)
        g = cls(**d)
        g.figures = figs
        g.cppn = _dict_to_genome(cp) if cp else None
        return g


# ---------------------------------------------------------------------------
# Creation from priors
# ---------------------------------------------------------------------------

def _new_cppn() -> CPPNGenome:
    g = CPPNGenome.create_minimal(0, CPPN_INPUTS, CPPN_OUTPUTS, "tanh", 2.0)
    for _ in range(random.randint(1, 3)):
        g.mutate_add_node(CPPN_ACTIVATIONS)
    for _ in range(2):
        g.mutate_add_connection()
    return g


def _saint(x: float, depth: float, avoid: set[str]) -> Figure:
    choices = [p for p in DRAPERY_PIGMENTS if p not in avoid] or DRAPERY_PIGMENTS
    robe = random.choice(choices)
    mantle = None
    if random.random() < 0.6:
        mantle = random.choice([p for p in DRAPERY_PIGMENTS if p != robe])
    return Figure(
        x=x, depth=depth,
        pose=random.choices(POSES, [0.7, 0.25, 0.05])[0],
        facing=1 if x < 0.5 else -1,
        lean=random.uniform(-0.06, 0.06),
        head_tilt=random.uniform(-0.25, 0.25),
        gesture=random.choices(GESTURES, [0.3, 0.15, 0.1, 0.15, 0.0, 0.3])[0],
        robe=robe, mantle=mantle,
        veil=random.random() < 0.15,
        mantle_open=random.uniform(0.02, 0.09),
        halo=random.random() < 0.6,
        skin=random.uniform(0.2, 0.8),
        hair=random.choice(HAIR_PIGMENTS),
        bulk=random.uniform(0.85, 1.15),
        flare=random.uniform(0.85, 1.25),
        folds=random.uniform(5, 11),
        fold_depth=random.uniform(0.3, 0.8),
    )


def _virgin(x: float, depth: float, pose: str) -> Figure:
    return Figure(
        x=x, depth=depth, pose=pose,
        facing=random.choice([-1, 1]),
        lean=random.uniform(-0.03, 0.03),
        head_tilt=random.uniform(-0.3, 0.3),
        gesture=random.choices(["cradle", "prayer", "open"], [0.6, 0.25, 0.15])[0],
        robe=random.choice(["vermilion", "madder_lake", "madder_lake"]),
        mantle=random.choice(["ultramarine", "ultramarine", "azurite"]),
        veil=True,
        mantle_open=random.uniform(0.03, 0.07),
        halo=True,
        skin=random.uniform(0.1, 0.35),
        hair=random.choice(["burnt_umber", "yellow_ochre"]),
        bulk=random.uniform(1.0, 1.2),
        flare=random.uniform(1.1, 1.35),
        folds=random.uniform(6, 10),
        fold_depth=random.uniform(0.4, 0.8),
        scale=random.uniform(1.0, 1.15),
    )


def _figures_for(composition: str) -> list[Figure]:
    figs: list[Figure] = []
    if composition == "pyramid":
        figs.append(_virgin(random.uniform(0.44, 0.56), 0.05,
                            random.choice(["seated", "standing"])))
        pairs = random.randint(1, 2)
        for k in range(pairs):
            dx = 0.2 + 0.13 * k + random.uniform(-0.03, 0.03)
            d = 0.15 + 0.1 * k
            used = {figs[0].robe, figs[0].mantle}
            left = _saint(0.5 - dx, d, used)
            right = _saint(0.5 + dx, d, used | {left.robe})
            if random.random() < 0.4:
                left.pose = right.pose = "kneeling"
            figs += [left, right]
    elif composition == "enthroned":
        figs.append(_virgin(0.5, 0.15, "seated"))
        figs[0].scale = random.uniform(1.1, 1.3)
        for k in range(random.randint(1, 2)):
            dx = 0.22 + 0.12 * k
            used = {figs[0].robe, figs[0].mantle}
            left = _saint(0.5 - dx, 0.05 + 0.05 * k, used)
            right = _saint(0.5 + dx, 0.05 + 0.05 * k, used | {left.robe})
            left.pose = right.pose = "standing"
            figs += [left, right]
    elif composition == "frieze":
        n = random.randint(3, 6)
        d = random.uniform(0.0, 0.2)
        xs = [0.12 + 0.76 * (i + 0.5) / n for i in range(n)]
        prev: set[str] = set()
        for i, x in enumerate(xs):
            f = _saint(x + random.uniform(-0.02, 0.02), d + random.uniform(0, 0.05), prev)
            prev = {f.robe}
            figs.append(f)
        if random.random() < 0.5:
            centre = figs[n // 2]
            figs[n // 2] = _virgin(centre.x, centre.depth, "standing")
    else:  # diagonal
        n = random.randint(3, 5)
        left_to_right = random.random() < 0.5
        prev = set()
        for i in range(n):
            t = i / max(1, n - 1)
            x = 0.15 + 0.7 * (t if left_to_right else 1 - t)
            f = _saint(x, 0.6 * t, prev)
            prev = {f.robe}
            figs.append(f)
        if random.random() < 0.5:
            figs[0] = _virgin(figs[0].x, figs[0].depth, "standing")
    return figs


def create_random() -> SceneGenome:
    composition = random.choices(COMPOSITIONS, [0.35, 0.2, 0.25, 0.2])[0]
    setting = random.choices(SETTINGS, [0.45, 0.35, 0.2])[0]
    if composition == "enthroned" and setting == "landscape" and random.random() < 0.6:
        setting = "loggia"
    g = SceneGenome(
        seed=random.randrange(1 << 30),
        setting=setting,
        composition=composition,
        horizon=random.uniform(0.38, 0.52),
        vp_x=random.uniform(0.44, 0.56),
        light_x=random.choice([-1, 1]) * random.uniform(0.35, 0.9),
        light_elev=random.uniform(0.4, 0.85),
        chiaroscuro=random.uniform(0.35, 0.9),
        darkness=random.choices([0.0, random.uniform(0.15, 0.55)], [0.8, 0.2])[0],
        sky_top=random.choice(["ultramarine", "ultramarine", "azurite"]),
        sky_low=random.choice(["lead_white", "lead_white", "lead_tin_yellow"]),
        sky_warmth=random.uniform(0.0, 0.6),
        clouds=random.uniform(0.0, 0.8),
        gloria=random.choices([0.0, random.uniform(0.4, 1.0)], [0.7, 0.3])[0],
        hill_layers=random.randint(2, 4),
        hill_height=random.uniform(0.06, 0.18),
        hill_rough=random.uniform(0.2, 0.9),
        trees=random.randint(0, 6),
        room_width=random.uniform(0.17, 0.32),
        room_top=random.uniform(0.22, 0.38),
        room_floor=random.uniform(0.06, 0.14),
        openings=random.randint(1, 3),
        tiles=random.randint(4, 10),
        floor_a=random.choice(["lead_white", "lead_white", "yellow_ochre"]),
        floor_b=random.choice(["red_ochre", "bone_black", "green_earth", "raw_umber"]),
        pilasters=random.randint(2, 5),
        throne=composition == "enthroned",
        cloth_of_honor=random.choice(["madder_lake", "verdigris", "ultramarine", "red_ochre"]),
        figures=_figures_for(composition),
        ground=random.choice(GROUND_PIGMENTS),
        glaze=random.uniform(0.4, 0.9),
        brush=random.uniform(0.3, 0.8),
        varnish=random.uniform(0.1, 0.6),
        craquelure=random.uniform(0.0, 0.6),
        cppn=_new_cppn(),
    )
    repair(g)
    return g


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def repair(g: SceneGenome) -> None:
    """Pull genes back inside the priors after mutation/crossover."""
    g.horizon = _clamp(g.horizon, 0.34, 0.56)
    g.vp_x = _clamp(g.vp_x, 0.35, 0.65)
    g.light_x = _clamp(g.light_x, -1.0, 1.0)
    if abs(g.light_x) < 0.2:
        g.light_x = 0.2 if g.light_x >= 0 else -0.2
    g.light_elev = _clamp(g.light_elev, 0.25, 0.95)
    for name in ("chiaroscuro", "glaze", "brush"):
        setattr(g, name, _clamp(getattr(g, name), 0.15, 1.0))
    for name in ("darkness", "sky_warmth", "clouds", "gloria", "varnish",
                 "craquelure", "hill_rough"):
        setattr(g, name, _clamp(getattr(g, name), 0.0, 1.0))
    g.hill_layers = int(_clamp(g.hill_layers, 1, 5))
    g.hill_height = _clamp(g.hill_height, 0.03, 0.22)
    g.trees = int(_clamp(g.trees, 0, 9))
    g.room_width = _clamp(g.room_width, 0.14, 0.36)
    g.room_top = _clamp(g.room_top, 0.18, 0.42)
    g.room_floor = _clamp(g.room_floor, 0.04, 0.18)
    g.openings = int(_clamp(g.openings, 0, 3))
    g.tiles = int(_clamp(g.tiles, 3, 14))
    g.pilasters = int(_clamp(g.pilasters, 0, 6))
    if g.floor_a == g.floor_b:
        g.floor_b = "red_ochre" if g.floor_a != "red_ochre" else "bone_black"

    if not g.figures:
        g.figures = [_saint(0.5, 0.1, set())]
    g.figures = g.figures[:MAX_FIGURES]
    for f in g.figures:
        f.x = _clamp(f.x, 0.08, 0.92)
        f.depth = _clamp(f.depth, 0.0, 0.8)
        f.lean = _clamp(f.lean, -0.12, 0.12)
        f.head_tilt = _clamp(f.head_tilt, -0.4, 0.4)
        f.mantle_open = _clamp(f.mantle_open, 0.0, 0.12)
        f.skin = _clamp(f.skin, 0.0, 1.0)
        f.bulk = _clamp(f.bulk, 0.75, 1.3)
        f.flare = _clamp(f.flare, 0.75, 1.45)
        f.folds = _clamp(f.folds, 3.0, 14.0)
        f.fold_depth = _clamp(f.fold_depth, 0.1, 1.0)
        f.scale = _clamp(f.scale, 0.85, 1.35)
        f.facing = 1 if f.facing >= 0 else -1
        if f.mantle == f.robe:
            f.mantle = None
        if f.mantle is None:
            f.veil = False
        if f.gesture == "cradle" and f.pose == "kneeling":
            f.gesture = "prayer"


# ---------------------------------------------------------------------------
# Variation
# ---------------------------------------------------------------------------

def _gauss(v: float, sigma: float, p: float, heat: float) -> float:
    return v + random.gauss(0, sigma * heat) if random.random() < p else v


def mutate(g: SceneGenome, strength: float = 1.0, temp: float = 1.0) -> None:
    """Perturb genes; ``temp`` (1 explore .. 0.2 refine) scales both how many
    genes change and by how much, and gates structural changes."""
    heat = max(0.05, strength) * temp
    p = min(0.9, 0.25 * strength * (0.5 + temp))

    g.horizon = _gauss(g.horizon, 0.04, p, heat)
    g.vp_x = _gauss(g.vp_x, 0.05, p, heat)
    g.light_x = _gauss(g.light_x, 0.3, p, heat)
    g.light_elev = _gauss(g.light_elev, 0.15, p, heat)
    g.chiaroscuro = _gauss(g.chiaroscuro, 0.15, p, heat)
    g.darkness = _gauss(g.darkness, 0.15, p * 0.5, heat)
    g.sky_warmth = _gauss(g.sky_warmth, 0.15, p, heat)
    g.clouds = _gauss(g.clouds, 0.2, p, heat)
    g.gloria = _gauss(g.gloria, 0.2, p * 0.4, heat)
    g.hill_height = _gauss(g.hill_height, 0.03, p, heat)
    g.hill_rough = _gauss(g.hill_rough, 0.2, p, heat)
    g.room_width = _gauss(g.room_width, 0.04, p, heat)
    g.room_top = _gauss(g.room_top, 0.05, p, heat)
    g.room_floor = _gauss(g.room_floor, 0.03, p, heat)
    g.glaze = _gauss(g.glaze, 0.15, p, heat)
    g.brush = _gauss(g.brush, 0.15, p, heat)
    g.varnish = _gauss(g.varnish, 0.15, p, heat)
    g.craquelure = _gauss(g.craquelure, 0.15, p, heat)

    rare = 0.12 * strength * temp
    if random.random() < rare:
        g.seed = random.randrange(1 << 30)
    if random.random() < rare * 0.5:
        g.setting = random.choice(SETTINGS)
    if random.random() < rare:
        g.hill_layers += random.choice([-1, 1])
    if random.random() < rare:
        g.trees += random.choice([-2, -1, 1, 2])
    if random.random() < rare:
        g.openings += random.choice([-1, 1])
    if random.random() < rare:
        g.tiles += random.choice([-2, -1, 1, 2])
    if random.random() < rare:
        g.pilasters += random.choice([-1, 1])
    if random.random() < rare:
        g.sky_top = random.choice(["ultramarine", "azurite"])
    if random.random() < rare:
        g.sky_low = random.choice(["lead_white", "lead_tin_yellow"])
    if random.random() < rare:
        g.floor_b = random.choice(["red_ochre", "bone_black", "green_earth", "raw_umber"])
    if random.random() < rare:
        g.ground = random.choice(GROUND_PIGMENTS)
    if random.random() < rare:
        g.cloth_of_honor = random.choice(["madder_lake", "verdigris", "ultramarine", "red_ochre"])
    if random.random() < rare * 0.5:
        g.throne = not g.throne

    for f in g.figures:
        _mutate_figure(f, p, heat, rare)

    # Structural: add or remove an attendant figure.
    if random.random() < rare * 0.6 and len(g.figures) < MAX_FIGURES:
        x = random.uniform(0.1, 0.9)
        g.figures.append(_saint(x, random.uniform(0.0, 0.6),
                                {f.robe for f in g.figures}))
    if random.random() < rare * 0.5 and len(g.figures) > 1:
        g.figures.pop(random.randrange(1, len(g.figures)))

    if g.cppn is not None and random.random() < 0.3 + 0.5 * temp:
        g.cppn.mutate(CPPN_MUTATION)

    repair(g)


def _mutate_figure(f: Figure, p: float, heat: float, rare: float) -> None:
    f.x = _gauss(f.x, 0.05, p, heat)
    f.depth = _gauss(f.depth, 0.08, p, heat)
    f.lean = _gauss(f.lean, 0.04, p, heat)
    f.head_tilt = _gauss(f.head_tilt, 0.12, p, heat)
    f.mantle_open = _gauss(f.mantle_open, 0.02, p, heat)
    f.skin = _gauss(f.skin, 0.1, p * 0.5, heat)
    f.bulk = _gauss(f.bulk, 0.06, p, heat)
    f.flare = _gauss(f.flare, 0.08, p, heat)
    f.folds = _gauss(f.folds, 1.5, p, heat)
    f.fold_depth = _gauss(f.fold_depth, 0.15, p, heat)
    f.scale = _gauss(f.scale, 0.05, p * 0.4, heat)
    if random.random() < rare:
        f.gesture = random.choice(GESTURES)
    if random.random() < rare * 0.5:
        f.pose = random.choice(POSES)
    if random.random() < rare * 0.5:
        f.facing = -f.facing
    if random.random() < rare:
        f.robe = random.choice(DRAPERY_PIGMENTS)
    if random.random() < rare:
        f.mantle = random.choice(DRAPERY_PIGMENTS + [None])
    if random.random() < rare * 0.5:
        f.veil = not f.veil
    if random.random() < rare * 0.4:
        f.halo = not f.halo
    if random.random() < rare * 0.5:
        f.hair = random.choice(HAIR_PIGMENTS)


_BLOCKS = [
    ("setting", "composition", "horizon", "vp_x"),
    ("light_x", "light_elev", "chiaroscuro", "darkness"),
    ("sky_top", "sky_low", "sky_warmth", "clouds", "gloria"),
    ("hill_layers", "hill_height", "hill_rough", "trees", "seed"),
    ("room_width", "room_top", "room_floor", "openings", "tiles",
     "floor_a", "floor_b", "pilasters", "throne", "cloth_of_honor"),
    ("ground", "glaze", "brush", "varnish", "craquelure"),
]


def crossover(a: SceneGenome, b: SceneGenome) -> SceneGenome:
    """Swap whole gene blocks (lighting, sky, room, finish, ...) so each
    block stays internally coherent.  Figures come from one parent with a
    chance of borrowing one figure from the other."""
    child = a.copy()
    for block in _BLOCKS:
        if random.random() < 0.5:
            for name in block:
                setattr(child, name, getattr(b, name))
    other = b
    if random.random() < 0.5:
        child.figures = [copy.copy(f) for f in b.figures]
        child.composition = b.composition
        other = a
    if other.figures and random.random() < 0.35:
        i = random.randrange(len(child.figures))
        donor = copy.copy(random.choice(other.figures))
        donor.x, donor.depth = child.figures[i].x, child.figures[i].depth
        child.figures[i] = donor
    if a.cppn is not None and b.cppn is not None:
        child.cppn = cppn_crossover(a.cppn, b.cppn, 0) if random.random() < 0.5 \
            else random.choice([a, b]).cppn.copy()
    repair(child)
    return child


# ---------------------------------------------------------------------------
# Prior score (used to pick among sibling candidates)
# ---------------------------------------------------------------------------

def composition_score(g: SceneGenome) -> float:
    """Cheap compositional prior: rewards balance about the centre, figures
    that do not pile on top of each other, and neighbouring figures in
    different colours."""
    figs = g.figures
    score = 0.0
    if figs:
        # Visual weight balance (front figures weigh more).
        w = [(1.2 - f.depth) * f.scale for f in figs]
        cx = sum(f.x * wi for f, wi in zip(figs, w)) / sum(w)
        score -= 4.0 * abs(cx - 0.5)
    by_x = sorted(figs, key=lambda f: f.x)
    for f1, f2 in zip(by_x, by_x[1:]):
        gap = abs(f1.x - f2.x)
        if gap < 0.08 and abs(f1.depth - f2.depth) < 0.15:
            score -= 1.0
        if f1.robe == f2.robe:
            score -= 0.5
    return score

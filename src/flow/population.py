"""Interactive population for the Flow engine.

Same interface as ShapePopulation (genomes, generation, temperature(),
evolve_with_selection, save_genome) so the server drives both alike.

Breeding follows the Renaissance engine (picks survive, open slots are
filled with the best of a few sibling candidates under a compositional
prior, newcomers keep the gene pool open while the lineage is hot).

Convergence: every child inherits its parents' maturity plus a drift that
grows as the lineage cools, so the population travels from raw,
neo-expressionist sketches toward a clean, resolved oil painting.  The
user's picks steer which painting it converges to; newcomers arrive a step
behind the population so they don't look out of place.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src import lineage
from src.flow.genome import (
    FlowGenome, composition_score, create_random, crossover, mutate,
)

CANDIDATES = 3
NEWCOMERS = 3            # at full heat; scaled by temperature
COOLING = 0.9
WARM_GENERATIONS = 2
MIN_TEMP = 0.2
DRIFT_HOT = 0.05         # maturity gained per generation while exploring
DRIFT_COOL = 0.12        # ... and once the lineage has cooled


class FlowPopulation:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.pop_size: int = cfg.get("pop_size", 24)
        self.elitism: int = cfg.get("elitism", 2)
        self.crossover_rate: float = cfg.get("crossover_rate", 0.7)
        self.mutation_strength: float = cfg.get("mutation_strength", 1.0)
        self.refine: float | None = cfg.get("refine")
        self.detail: float | None = cfg.get("detail")
        self.genomes: list[FlowGenome] = []
        self.generation: int = 0

    def initialize(self) -> None:
        self.genomes = [create_random() for _ in range(self.pop_size)]
        self.generation = 0

    def temperature(self) -> float:
        if self.refine is not None:
            return max(MIN_TEMP, 1.0 - max(0.0, min(1.0, self.refine)))
        return max(MIN_TEMP, COOLING ** max(0, self.generation - WARM_GENERATIONS))

    def _drift(self, temp: float) -> float:
        cool = (1.0 - temp) / (1.0 - MIN_TEMP)
        return DRIFT_HOT + (DRIFT_COOL - DRIFT_HOT) * cool

    def _child(self, parents: list[FlowGenome], temp: float) -> FlowGenome:
        if len(parents) >= 2 and random.random() < self.crossover_rate:
            src = random.sample(parents, 2)
            child = crossover(*src)
        else:
            src = [random.choice(parents)]
            child = src[0].copy()
        lineage.child(child, *src)
        mutate(child, self.mutation_strength, temp, self._drift(temp))
        return child

    def evolve_with_selection(self, selected: list[int]) -> None:
        sel = [i for i in selected if 0 <= i < len(self.genomes)]
        if not sel:
            sel = [random.randrange(len(self.genomes))]
        parents = [self.genomes[i] for i in sel]
        lineage.mark_parents(parents, sel)
        temp = self.temperature()
        keep = [lineage.survivor(p.copy(), p)
                for p in parents[: max(self.elitism, len(parents))]][: self.pop_size]
        # Survivors are repainted a little more carefully too, so keeping a
        # favourite never stalls the convergence.  Their strokes are seeded,
        # so the picture stays recognisably the same.
        for k in keep:
            k.maturity = min(1.0, k.maturity + 0.6 * self._drift(temp))
        n_new = min(int(round(NEWCOMERS * temp)), max(0, self.pop_size - len(keep) - 1))
        children = []
        while len(keep) + len(children) < self.pop_size - n_new:
            cands = [self._child(parents, temp) for _ in range(CANDIDATES)]
            children.append(max(cands, key=composition_score))
        mean = sum(p.maturity for p in parents) / len(parents)
        fresh = [create_random(maturity=max(0.0, mean - 0.15)) for _ in range(n_new)]
        self.genomes = (keep + children + fresh)[: self.pop_size]
        self.generation += 1

    def save_genome(self, index: int, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.genomes[index].to_dict(), indent=2))

    @staticmethod
    def load_genome(path: str | Path) -> FlowGenome:
        return FlowGenome.from_dict(json.loads(Path(path).read_text()))

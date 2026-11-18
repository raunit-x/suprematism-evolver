"""Population manager for shape-genome evolutionary art.

Simple generational EA with tournament selection, elitism,
crossover, and mutation.  No speciation — the genome space
is low-dimensional enough that it isn't needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src import lineage
from src.shapes import citymap
from src.shapes import composition as comp_ops
from src.shapes.genome import (
    ShapeGenome,
    architecton_harmony,
    create_random,
    create_map,
    crossover,
    mutate,
)


# Siblings bred per offspring slot for Hadid; the most harmonious is kept.
HARMONY_CANDIDATES = 4

# Composition (Hadid) breeding: candidate pool size per open slot, and fresh
# random newcomers injected every generation to prevent collapse.
POOL_PER_SLOT = 6
NEWCOMERS = 4             # at full heat; scaled by temperature
IMMIGRANT_CROSS = 0.2     # share of children that are a pick x newcomer cross
COOLING = 0.9             # auto: temperature = COOLING ** (generation - WARM_GENERATIONS)
WARM_GENERATIONS = 3      # auto: fully exploratory for the first generations
MIN_TEMP = 0.2


class ShapePopulation:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.pop_size: int = cfg.get("pop_size", 20)
        self.elitism: int = cfg.get("elitism", 4)
        self.crossover_rate: float = cfg.get("crossover_rate", 0.7)
        self.tournament_k: int = cfg.get("tournament_k", 3)
        self.num_palette_colors: int = cfg.get("num_palette_colors", 6)
        self.mutation_strength: float = cfg.get("mutation_strength", 1.0)
        # Explore <-> Refine: None = automatic (cools with generation),
        # otherwise 0.0 (explore) .. 1.0 (refine).
        self.refine: float | None = cfg.get("refine")
        # Detail: how much new structure children grow, 0..1; None = auto
        # (tied to Explore <-> Refine).  Independent of mutation strength.
        self.detail: float | None = cfg.get("detail")

        self.genomes: list[ShapeGenome] = []
        self.generation: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.genomes = [
            create_random(self.num_palette_colors)
            for _ in range(self.pop_size)
        ]
        self.generation = 0

    def initialize_architecton(self) -> None:
        """Initialize with composition genomes, picked for diversity across
        (spread, element count) from a larger random pool."""
        pool = [comp_ops.new_individual() for _ in range(self.pop_size * POOL_PER_SLOT)]
        self.genomes = comp_ops.select_diverse(pool, self.pop_size)
        self.generation = 0

    def initialize_maps(self) -> None:
        """Initialize population with abstract city-map genomes; city types
        and palettes are dealt round-robin so the first grid spans them all."""
        plans = list(citymap.PLANS)
        palettes = list(citymap.PALETTE_SETS)
        random.shuffle(plans)
        random.shuffle(palettes)
        self.genomes = [
            create_map(self.num_palette_colors, plans[i % len(plans)],
                       palettes[i % len(palettes)])
            for i in range(self.pop_size)
        ]
        self.generation = 0

    def branch_from(self, path: str | Path) -> None:
        """Fork a population from a saved genome JSON."""
        base = self.load_genome(path)
        self.genomes = []
        self.genomes.append(base.copy())
        for _ in range(self.pop_size - 1):
            child = base.copy()
            if child.city is not None:
                citymap.mutate_city(child.city, self.mutation_strength, 1.0)
            elif child.comp is not None:
                comp_ops.mutate_composition(child.comp, self.mutation_strength, 1.0)
                child = comp_ops.express(child.comp)
            else:
                mutate(child, self.num_palette_colors)
            self.genomes.append(child)
        self.generation = 0

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def set_fitness(self, scores: list[float]) -> None:
        for genome, score in zip(self.genomes, scores):
            genome.fitness = score

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self) -> ShapeGenome:
        candidates = random.sample(self.genomes, min(self.tournament_k, len(self.genomes)))
        return max(candidates, key=lambda g: g.fitness)

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self) -> None:
        elite = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        next_gen: list[ShapeGenome] = []

        for i in range(min(self.elitism, len(elite))):
            next_gen.append(elite[i].copy())

        while len(next_gen) < self.pop_size:
            if random.random() < self.crossover_rate:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                if p1.city is not None and p2.city is not None:
                    child = p1.copy()
                    child.city = citymap.crossover_city(p1.city, p2.city)
                else:
                    child = crossover(p1, p2)
            else:
                child = self._tournament_select().copy()

            if child.city is not None:
                citymap.mutate_city(child.city, self.mutation_strength, 1.0)
            else:
                mutate(child, self.num_palette_colors, strength=self.mutation_strength)
            next_gen.append(child)

        self.genomes = next_gen[: self.pop_size]
        self.generation += 1

    def evolve_with_selection(self, selected_indices: list[int]) -> None:
        """Breed next generation exclusively from the user-selected parents."""
        sel = [i for i in selected_indices if 0 <= i < len(self.genomes)]
        if not sel:
            sel = [random.randrange(len(self.genomes))]
        parents = [self.genomes[i] for i in sel]
        lineage.mark_parents(parents, sel)

        next_gen: list[ShapeGenome] = []

        for p in parents[:self.elitism]:
            next_gen.append(lineage.survivor(p.copy(), p))

        if all(p.city is not None for p in parents):
            self._evolve_cities(parents)
            return

        # Hadid architectons: breed a few siblings per slot and keep the most
        # harmonious.  Parents are still exactly the user's picks.
        if all(p.comp is not None for p in parents):
            self._evolve_compositions(parents)
            return

        n_tries = HARMONY_CANDIDATES if parents[0].engine_tag == "hadid" else 1
        while len(next_gen) < self.pop_size:
            candidates = [self._breed_child(parents) for _ in range(n_tries)]
            if n_tries > 1:
                child = max(candidates, key=architecton_harmony)
            else:
                child = candidates[0]
            next_gen.append(child)

        self.genomes = next_gen[: self.pop_size]
        self.generation += 1

    def temperature(self) -> float:
        """1.0 = exploring .. MIN_TEMP = refining."""
        if self.refine is not None:
            return max(MIN_TEMP, 1.0 - max(0.0, min(1.0, self.refine)))
        return max(MIN_TEMP, COOLING ** max(0, self.generation - WARM_GENERATIONS))

    def _evolve_compositions(self, parents: list[ShapeGenome]) -> None:
        """Hadid breeding: the picks survive unchanged and children of the
        picks fill the generation.  Early on (hot) children change a lot,
        some are crossed with newcomers and selection favours difference;
        as the lineage cools, changes get smaller, cleanup takes over,
        newcomers stop and selection favours harmony."""
        temp = self.temperature()
        refine = 1.0 - temp
        keep = [lineage.survivor(p.copy(), p) for p in parents[:self.elitism]]
        n_new = min(int(round(NEWCOMERS * temp)), max(0, self.pop_size - len(keep) - 1))
        n_children = self.pop_size - len(keep) - n_new

        pool = []
        for _ in range(n_children * POOL_PER_SLOT):
            roll = random.random()
            if roll < IMMIGRANT_CROSS * temp:
                # Cross a pick with a random newcomer: novelty with lineage.
                src = [random.choice(parents)]
                genes = comp_ops.crossover_composition(
                    src[0].comp, comp_ops.new_individual().comp)
            elif len(parents) >= 2 and roll < IMMIGRANT_CROSS * temp + self.crossover_rate * 0.8:
                src = random.sample(parents, 2)
                genes = comp_ops.crossover_composition(src[0].comp, src[1].comp)
            else:
                src = [random.choice(parents)]
                genes = src[0].comp.copy()
            for _ in range(1 + (random.random() < 0.5 * temp) + (random.random() < 0.25 * temp)):
                comp_ops.mutate_composition(genes, self.mutation_strength, temp, self.detail)
            pool.append(lineage.child(comp_ops.express(genes), *src))
        children = comp_ops.select_diverse(pool, n_children, refine, existing=keep)

        fresh = comp_ops.select_diverse(
            [comp_ops.new_individual() for _ in range(n_new * POOL_PER_SLOT)], n_new,
            existing=keep + children)

        self.genomes = (keep + children + fresh)[: self.pop_size]
        self.generation += 1

    def _evolve_cities(self, parents: list[ShapeGenome]) -> None:
        """Maps breeding: picks survive unchanged; children are mutated
        (often crossed) picks, with a few fresh cities while still hot."""
        temp = self.temperature()
        keep = [lineage.survivor(p.copy(), p) for p in parents[:self.elitism]]
        n_new = min(int(round(NEWCOMERS * temp * 0.5)),
                    max(0, self.pop_size - len(keep) - 1))
        children = []
        while len(keep) + len(children) < self.pop_size - n_new:
            if len(parents) >= 2 and random.random() < self.crossover_rate:
                src = random.sample(parents, 2)
                genes = citymap.crossover_city(src[0].city, src[1].city)
            else:
                src = [random.choice(parents)]
                genes = src[0].city.copy()
            citymap.mutate_city(genes, self.mutation_strength, temp)
            child = parents[0].copy()
            child.city = genes
            children.append(lineage.child(child, *src))
        fresh = [create_map(self.num_palette_colors) for _ in range(n_new)]
        self.genomes = (keep + children + fresh)[: self.pop_size]
        self.generation += 1

    def _breed_child(self, parents: list[ShapeGenome]) -> ShapeGenome:
        if len(parents) >= 2 and random.random() < self.crossover_rate:
            src = random.sample(parents, 2)
            child = crossover(*src)
        else:
            src = [random.choice(parents)]
            child = src[0].copy()
        mutate(child, self.num_palette_colors, strength=self.mutation_strength)
        return lineage.child(child, *src)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_genome(self, index: int, path: str | Path) -> None:
        path = Path(path)
        genome = self.genomes[index]
        path.write_text(json.dumps(genome.to_dict(), indent=2))

    @staticmethod
    def load_genome(path: str | Path) -> ShapeGenome:
        path = Path(path)
        data = json.loads(path.read_text())
        return ShapeGenome.from_dict(data)

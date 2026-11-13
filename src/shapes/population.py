"""Population manager for shape-genome evolutionary art.

Simple generational EA with tournament selection, elitism,
crossover, and mutation.  No speciation — the genome space
is low-dimensional enough that it isn't needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

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
POOL_PER_SLOT = 4
NEWCOMERS = 2
IMMIGRANT_CROSS = 0.2     # share of children that are a pick x newcomer cross


class ShapePopulation:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.pop_size: int = cfg.get("pop_size", 20)
        self.elitism: int = cfg.get("elitism", 4)
        self.crossover_rate: float = cfg.get("crossover_rate", 0.7)
        self.tournament_k: int = cfg.get("tournament_k", 3)
        self.num_palette_colors: int = cfg.get("num_palette_colors", 6)
        self.mutation_strength: float = cfg.get("mutation_strength", 1.0)

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
        """Initialize population with abstract city-plan genomes."""
        self.genomes = [
            create_map(self.num_palette_colors)
            for _ in range(self.pop_size)
        ]
        self.generation = 0

    def branch_from(self, path: str | Path) -> None:
        """Fork a population from a saved genome JSON."""
        base = self.load_genome(path)
        self.genomes = []
        self.genomes.append(base.copy())
        for _ in range(self.pop_size - 1):
            child = base.copy()
            if child.comp is not None:
                comp_ops.mutate_composition(child.comp, self.mutation_strength)
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
                child = crossover(p1, p2)
            else:
                child = self._tournament_select().copy()

            mutate(child, self.num_palette_colors, strength=self.mutation_strength)
            next_gen.append(child)

        self.genomes = next_gen[: self.pop_size]
        self.generation += 1

    def evolve_with_selection(self, selected_indices: list[int]) -> None:
        """Breed next generation exclusively from the user-selected parents."""
        parents = [self.genomes[i] for i in selected_indices
                   if 0 <= i < len(self.genomes)]
        if not parents:
            parents = [random.choice(self.genomes)]

        next_gen: list[ShapeGenome] = []

        for p in parents[:self.elitism]:
            next_gen.append(p.copy())

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

    def _evolve_compositions(self, parents: list[ShapeGenome]) -> None:
        """Hadid breeding: the picks survive unchanged, children of the picks
        fill most slots (pre-filtered and diversified), plus a couple of
        random newcomers."""
        keep = [p.copy() for p in parents[:self.elitism]]
        n_new = min(NEWCOMERS, max(0, self.pop_size - len(keep) - 1))
        n_children = self.pop_size - len(keep) - n_new

        pool = []
        for _ in range(n_children * POOL_PER_SLOT):
            roll = random.random()
            if roll < IMMIGRANT_CROSS:
                # Cross a pick with a random newcomer: novelty with lineage.
                genes = comp_ops.crossover_composition(
                    random.choice(parents).comp, comp_ops.new_individual().comp)
            elif len(parents) >= 2 and roll < IMMIGRANT_CROSS + self.crossover_rate * 0.8:
                a, b = random.sample(parents, 2)
                genes = comp_ops.crossover_composition(a.comp, b.comp)
            else:
                genes = random.choice(parents).comp.copy()
            for _ in range(1 + (random.random() < 0.5) + (random.random() < 0.25)):
                comp_ops.mutate_composition(genes, self.mutation_strength)
            pool.append(comp_ops.express(genes))
        children = comp_ops.select_diverse(pool, n_children)

        fresh = comp_ops.select_diverse(
            [comp_ops.new_individual() for _ in range(n_new * POOL_PER_SLOT)], n_new)

        self.genomes = (keep + children + fresh)[: self.pop_size]
        self.generation += 1

    def _breed_child(self, parents: list[ShapeGenome]) -> ShapeGenome:
        if len(parents) >= 2 and random.random() < self.crossover_rate:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
        else:
            child = random.choice(parents).copy()
        mutate(child, self.num_palette_colors, strength=self.mutation_strength)
        return child

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

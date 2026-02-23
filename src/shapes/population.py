"""Population manager for shape-genome evolutionary art.

Simple generational EA with tournament selection, elitism,
crossover, and mutation.  No speciation — the genome space
is low-dimensional enough that it isn't needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.shapes.genome import (
    ShapeGenome,
    create_random,
    crossover,
    mutate,
)


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

    def branch_from(self, path: str | Path) -> None:
        """Fork a population from a saved genome JSON."""
        base = self.load_genome(path)
        self.genomes = []
        self.genomes.append(base.copy())
        for _ in range(self.pop_size - 1):
            child = base.copy()
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

        while len(next_gen) < self.pop_size:
            if len(parents) >= 2 and random.random() < self.crossover_rate:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
            else:
                child = random.choice(parents).copy()

            mutate(child, self.num_palette_colors, strength=self.mutation_strength)
            next_gen.append(child)

        self.genomes = next_gen[: self.pop_size]
        self.generation += 1

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

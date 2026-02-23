"""NEAT evolution: selection, reproduction, and speciation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.neat.genome import Genome, crossover, compatibility_distance


@dataclass
class Species:
    key: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0

    @property
    def adjusted_fitness(self) -> float:
        """Sum of fitness-shared members."""
        if not self.members:
            return 0.0
        total = sum(m.fitness or 0.0 for m in self.members)
        return total / len(self.members)


class Reproducer:
    """Handles NEAT reproduction: speciation, selection, offspring creation."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._next_species_key = 0

    def speciate(
        self,
        genomes: list[Genome],
        existing_species: list[Species],
    ) -> list[Species]:
        """Assign genomes to species based on compatibility distance."""
        threshold = self.config.get("compatibility_threshold", 3.0)
        c1 = self.config.get("excess_coefficient", 1.0)
        c2 = self.config.get("disjoint_coefficient", 1.0)
        c3 = self.config.get("weight_diff_coefficient", 0.5)

        # Start with existing representatives
        new_species: list[Species] = []
        representatives = []
        for s in existing_species:
            ns = Species(key=s.key, representative=s.representative)
            new_species.append(ns)
            representatives.append((ns, s.representative))

        unassigned = list(genomes)

        for genome in unassigned:
            placed = False
            for species, rep in representatives:
                dist = compatibility_distance(genome, rep, c1, c2, c3)
                if dist < threshold:
                    species.members.append(genome)
                    placed = True
                    break

            if not placed:
                # Create new species
                ns = Species(
                    key=self._next_species_key,
                    representative=genome,
                    members=[genome],
                )
                self._next_species_key += 1
                new_species.append(ns)
                representatives.append((ns, genome))

        # Remove empty species, update representatives
        active = []
        for s in new_species:
            if s.members:
                s.representative = random.choice(s.members)
                active.append(s)

        return active

    def reproduce(
        self,
        species_list: list[Species],
        pop_size: int,
        mutation_config: dict | None = None,
        genome_counter: list[int] | None = None,
    ) -> list[Genome]:
        """Produce next generation via fitness-proportionate selection + NEAT reproduction."""
        if genome_counter is None:
            genome_counter = [0]

        if mutation_config is None:
            mutation_config = self.config

        # Calculate offspring allocation per species
        total_adjusted = sum(s.adjusted_fitness for s in species_list)
        if total_adjusted <= 0:
            # Equal allocation
            total_adjusted = len(species_list)
            for s in species_list:
                s._alloc = max(1, pop_size // len(species_list))
        else:
            for s in species_list:
                s._alloc = max(1, int(round(s.adjusted_fitness / total_adjusted * pop_size)))

        offspring = []
        elitism = self.config.get("elitism", 1)

        for species in species_list:
            n_offspring = min(s._alloc, pop_size - len(offspring))
            if n_offspring <= 0:
                continue

            # Sort by fitness (descending)
            members = sorted(
                species.members,
                key=lambda g: g.fitness or 0.0,
                reverse=True,
            )

            # Elitism: preserve top individual(s)
            for i in range(min(elitism, len(members), n_offspring)):
                elite = members[i].copy()
                elite.key = genome_counter[0]
                genome_counter[0] += 1
                offspring.append(elite)

            # Fill remaining with mutation and crossover
            remaining = n_offspring - min(elitism, len(members), n_offspring)
            for _ in range(remaining):
                if len(members) >= 2 and random.random() < self.config.get("crossover_rate", 0.5):
                    p1 = self._tournament_select(members)
                    p2 = self._tournament_select(members)
                    if (p1.fitness or 0) < (p2.fitness or 0):
                        p1, p2 = p2, p1
                    child = crossover(p1, p2, genome_counter[0])
                else:
                    parent = self._tournament_select(members)
                    child = parent.copy()
                    child.key = genome_counter[0]

                genome_counter[0] += 1
                child.mutate(mutation_config)
                offspring.append(child)

        # Trim or pad to exact pop_size
        while len(offspring) < pop_size:
            # Clone and mutate a random existing offspring
            parent = random.choice(offspring)
            child = parent.copy()
            child.key = genome_counter[0]
            genome_counter[0] += 1
            child.mutate(mutation_config)
            offspring.append(child)

        return offspring[:pop_size]

    def _tournament_select(self, members: list[Genome], k: int = 3) -> Genome:
        """Tournament selection: pick best of k random members."""
        candidates = random.sample(members, min(k, len(members)))
        return max(candidates, key=lambda g: g.fitness or 0.0)

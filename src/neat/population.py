"""Population manager: ties together genome creation, evolution, and speciation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.neat.genome import Genome
from src.neat.evolution import Reproducer, Species
from src.cppn.network import CPPNNetwork


# Default NEAT/CPPN configuration
DEFAULT_CONFIG = {
    # CPPN structure
    "num_inputs": 7,  # x, y, d, theta, bias, armature_d, armature_t
    "num_outputs": 3,  # R, G, B (or H, S, V)
    "output_activation": "tanh",

    # Population
    "pop_size": 20,

    # Mutation rates
    "weight_perturb_rate": 0.8,
    "weight_perturb_power": 0.5,
    "weight_replace_rate": 0.1,
    "add_node_rate": 0.03,
    "add_connection_rate": 0.05,
    "activation_mutation_rate": 0.1,
    "toggle_enable_rate": 0.01,

    # Crossover
    "crossover_rate": 0.5,

    # Speciation
    "compatibility_threshold": 3.0,
    "excess_coefficient": 1.0,
    "disjoint_coefficient": 1.0,
    "weight_diff_coefficient": 0.5,

    # Reproduction
    "elitism": 1,

    # Activation function options (None = all)
    "activation_options": None,
}


class Population:
    """Manages a population of CPPN genomes evolved via NEAT."""

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.genomes: list[Genome] = []
        self.species: list[Species] = []
        self.generation: int = 0
        self._genome_counter = [0]
        self._reproducer = Reproducer(self.config)

    def initialize(self):
        """Create initial population of minimal-topology genomes."""
        self.genomes = []
        for _ in range(self.config["pop_size"]):
            g = Genome.create_minimal(
                genome_key=self._genome_counter[0],
                num_inputs=self.config["num_inputs"],
                num_outputs=self.config["num_outputs"],
                output_activation=self.config["output_activation"],
            )
            self._genome_counter[0] += 1
            self.genomes.append(g)

        self.generation = 0

    def get_networks(self) -> list[CPPNNetwork]:
        """Build CPPN networks from all current genomes."""
        return [CPPNNetwork.from_genome(g) for g in self.genomes]

    def set_fitness(self, fitness_values: list[float]):
        """Set fitness for all genomes in current generation."""
        for g, f in zip(self.genomes, fitness_values):
            g.fitness = f

    def evolve(self):
        """Run one generation of NEAT evolution.

        Must call set_fitness() first.
        """
        # Speciate
        self.species = self._reproducer.speciate(self.genomes, self.species)

        # Reproduce
        self.genomes = self._reproducer.reproduce(
            self.species,
            self.config["pop_size"],
            self.config,
            self._genome_counter,
        )

        self.generation += 1

    def evolve_with_selection(self, selected_indices: list[int]):
        """Convenience: set fitness from selection, then evolve.

        Selected genomes get fitness 1.0, others get 0.0.
        """
        fitness = [0.0] * len(self.genomes)
        for idx in selected_indices:
            if 0 <= idx < len(self.genomes):
                fitness[idx] = 1.0
        self.set_fitness(fitness)
        self.evolve()

    # --- Persistence ---

    def save_genome(self, index: int, path: str | Path):
        """Save a single genome to JSON."""
        g = self.genomes[index]
        data = _genome_to_dict(g)
        Path(path).write_text(json.dumps(data, indent=2))

    def load_genome(self, path: str | Path) -> Genome:
        """Load a genome from JSON."""
        data = json.loads(Path(path).read_text())
        return _dict_to_genome(data)

    def branch_from(self, path: str | Path):
        """Reset population by branching from a saved genome.

        Creates pop_size mutated variants of the loaded genome.
        """
        parent = self.load_genome(path)
        self.genomes = []
        # First individual is the parent unchanged
        parent.key = self._genome_counter[0]
        self._genome_counter[0] += 1
        self.genomes.append(parent)

        # Rest are mutated copies
        for _ in range(self.config["pop_size"] - 1):
            child = parent.copy()
            child.key = self._genome_counter[0]
            self._genome_counter[0] += 1
            child.mutate(self.config)
            self.genomes.append(child)

        self.generation = 0
        self.species = []


def _genome_to_dict(g: Genome) -> dict:
    """Serialize a genome to a JSON-compatible dict."""
    return {
        "key": g.key,
        "input_keys": list(g.input_keys),
        "output_keys": list(g.output_keys),
        "next_node_key": g._next_node_key,
        "nodes": {
            str(k): {
                "key": v.key,
                "type": v.node_type,
                "activation": v.activation,
            }
            for k, v in g.nodes.items()
        },
        "connections": {
            f"{k[0]},{k[1]}": {
                "in": k[0],
                "out": k[1],
                "weight": v.weight,
                "enabled": v.enabled,
            }
            for k, v in g.connections.items()
        },
        "comp_focal_x": g.comp_focal_x,
        "comp_focal_y": g.comp_focal_y,
        "comp_armature_angle": g.comp_armature_angle,
    }


def _dict_to_genome(d: dict) -> Genome:
    """Deserialize a genome from a dict."""
    from src.neat.genome import NodeGene, ConnectionGene

    g = Genome(key=d["key"])
    g.input_keys = tuple(d["input_keys"])
    g.output_keys = tuple(d["output_keys"])
    g._next_node_key = d["next_node_key"]

    for _, node_data in d["nodes"].items():
        n = NodeGene(
            key=node_data["key"],
            node_type=node_data["type"],
            activation=node_data["activation"],
        )
        g.nodes[n.key] = n

    for _, conn_data in d["connections"].items():
        key = (conn_data["in"], conn_data["out"])
        c = ConnectionGene(
            key=key,
            weight=conn_data["weight"],
            enabled=conn_data["enabled"],
        )
        g.connections[key] = c

    g.comp_focal_x = d.get("comp_focal_x", 0.0)
    g.comp_focal_y = d.get("comp_focal_y", 0.0)
    g.comp_armature_angle = d.get("comp_armature_angle", 0.0)

    return g

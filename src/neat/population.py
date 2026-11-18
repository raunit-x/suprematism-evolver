"""Population manager: ties together genome creation, evolution, and speciation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src import lineage
from src.neat.genome import Genome, InnovationTracker, compatibility_distance, crossover
from src.neat.evolution import Reproducer, Species
from src.cppn.network import CPPNNetwork


# Default NEAT/CPPN configuration
DEFAULT_CONFIG = {
    # CPPN structure
    "num_inputs": 4,  # x, y, d, bias (see render_cppn for 5/7)
    "num_outputs": 3,  # R, G, B (or H, S, V)
    "output_activation": "tanh",

    # Population
    "pop_size": 20,

    # Mutation rates.  Interactive evolution only runs for tens of
    # generations, so structure has to grow fast enough to matter, while
    # weight noise stays low enough that a child still looks like its pick.
    "weight_perturb_rate": 0.3,
    "weight_perturb_power": 0.9,
    "weight_replace_rate": 0.05,
    "add_node_rate": 0.3,
    "add_connection_rate": 0.4,
    "activation_mutation_rate": 0.15,
    "toggle_enable_rate": 0.02,
    # Inputs are in [-1, 1]; a sin node needs weights well past 4 to fold
    # the plane more than once, so the ranges are wide.
    "weight_range": 8.0,
    "new_weight_range": 4.0,
    # Weight into a freshly split node (see Genome.mutate_add_node).
    "split_input_weight_range": 6.0,

    # Starting topology: hidden nodes / extra links grown into each
    # generation-0 genome, so the first grid isn't just linear gradients.
    "initial_hidden_nodes": (1, 3),
    "initial_extra_connections": (1, 3),

    # Crossover
    "crossover_rate": 0.3,

    # Interactive breeding: how many picks carry over unchanged.
    "max_survivors": 4,

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
        self.innovations = InnovationTracker()
        self.mutation_strength: float = self.config.get("mutation_strength", 1.0)

    def _next_key(self) -> int:
        k = self._genome_counter[0]
        self._genome_counter[0] += 1
        return k

    def initialize(self):
        """Create the first generation: minimal genomes, each grown a few
        hidden nodes so the starting grid already has some structure."""
        self.genomes = []
        n_in = self.config["num_inputs"]
        n_out = self.config["num_outputs"]
        self.innovations = InnovationTracker(n_in + n_out)
        lo_h, hi_h = self.config["initial_hidden_nodes"]
        lo_c, hi_c = self.config["initial_extra_connections"]
        for _ in range(self.config["pop_size"]):
            g = Genome.create_minimal(
                genome_key=self._next_key(),
                num_inputs=n_in,
                num_outputs=n_out,
                output_activation=self.config["output_activation"],
                weight_range=self.config.get("new_weight_range", 2.0),
            )
            for _ in range(random.randint(lo_h, hi_h)):
                g.mutate_add_node(self.config.get("activation_options"), self.innovations,
                                  self.config.get("split_input_weight_range"))
            for _ in range(random.randint(lo_c, hi_c)):
                g.mutate_add_connection(weight_range=self.config.get("new_weight_range", 2.0))
            self.genomes.append(g)

        self.generation = 0
        self.species = []

    def get_networks(self) -> list[CPPNNetwork]:
        """Build CPPN networks from all current genomes."""
        return [CPPNNetwork.from_genome(g) for g in self.genomes]

    def set_fitness(self, fitness_values: list[float]):
        """Set fitness for all genomes in current generation."""
        for g, f in zip(self.genomes, fitness_values):
            g.fitness = f

    def evolve(self):
        """Run one generation of fitness-driven NEAT evolution (speciated).

        Must call set_fitness() first.  For interactive picking use
        evolve_with_selection(), which breeds only from the picks.
        """
        self.innovations.sync(self.genomes)
        self.species = self._reproducer.speciate(self.genomes, self.species)
        self.genomes = self._reproducer.reproduce(
            self.species,
            self.config["pop_size"],
            self.config,
            self._genome_counter,
            innovations=self.innovations,
            strength=self.mutation_strength,
        )
        self.generation += 1

    def evolve_with_selection(
        self,
        selected_indices: list[int],
        bonus: list[float] | None = None,
    ):
        """Picbreeder-style breeding: only the picked genomes are parents.

        The picks carry over unchanged (up to ``max_survivors``) so nothing
        the user liked is lost; the rest of the grid is their offspring, by
        mutation or by crossover between two picks.  ``bonus`` (one value per
        current genome, e.g. a style score) gives a pick a larger share of the
        children; it never makes an unpicked genome a parent.

        The previous implementation gave unpicked genomes fitness 0 but left
        them in the breeding pool, so roughly 70% of parents were images the
        user had passed over and lineages could not accumulate structure.
        """
        n = len(self.genomes)
        sel = [i for i in dict.fromkeys(selected_indices) if 0 <= i < n]
        if not sel:
            sel = list(range(n))
        parents = [self.genomes[i] for i in sel]
        lineage.mark_parents(parents, sel)
        weights = [1.0 + max(0.0, (bonus[i] if bonus else 0.0)) for i in sel]
        for p, w in zip(parents, weights):
            p.fitness = w
        self.innovations.sync(self.genomes)

        pop_size = self.config["pop_size"]
        n_keep = min(len(parents), self.config.get("max_survivors", 4), pop_size)
        keep_order = sorted(range(len(parents)), key=lambda j: -weights[j])[:n_keep]
        next_gen: list[Genome] = []
        for j in keep_order:
            g = parents[j].copy()
            g.key = self._next_key()
            next_gen.append(lineage.survivor(g, parents[j]))

        # Spread the children evenly over the picks (weighted by bonus), so
        # every pick gets explored rather than one winning a random draw.
        n_children = pop_size - len(next_gen)
        total = sum(weights)
        quota = [w / total * n_children for w in weights]
        counts = [int(q) for q in quota]
        for j in sorted(range(len(parents)), key=lambda j: counts[j] - quota[j])[:n_children - sum(counts)]:
            counts[j] += 1

        for j, count in enumerate(counts):
            p = parents[j]
            for _ in range(count):
                mate = self._pick_mate(p, parents)
                if mate is not None and random.random() < self.config.get("crossover_rate", 0.3):
                    a, b = (p, mate) if (p.fitness or 0) >= (mate.fitness or 0) else (mate, p)
                    child = crossover(a, b, self._next_key())
                    strength = 0.5 * self.mutation_strength
                    lineage.child(child, p, mate)
                else:
                    child = p.copy()
                    child.key = self._next_key()
                    # Mix small and large steps across the grid.
                    strength = self.mutation_strength * random.choice((0.6, 1.0, 1.0, 1.6))
                    lineage.child(child, p)
                child.mutate(self.config, self.innovations, strength)
                next_gen.append(child)

        random.shuffle(next_gen)
        self.genomes = next_gen[:pop_size]
        self.species = []
        self.generation += 1

    def _pick_mate(self, genome: Genome, parents: list[Genome]) -> Genome | None:
        """A crossover partner among the other picks, preferring close relatives
        (crossing very different networks mostly yields noise)."""
        others = [q for q in parents if q is not genome]
        if not others:
            return None
        thr = self.config.get("compatibility_threshold", 3.0)
        close = [q for q in others if compatibility_distance(genome, q) < thr]
        return random.choice(close or others)

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
        self.innovations = InnovationTracker()
        self.innovations.sync([parent])
        # First individual is the parent unchanged
        parent.key = self._genome_counter[0]
        self._genome_counter[0] += 1
        self.genomes.append(parent)

        # Rest are mutated copies
        for _ in range(self.config["pop_size"] - 1):
            child = parent.copy()
            child.key = self._genome_counter[0]
            self._genome_counter[0] += 1
            child.mutate(self.config, self.innovations, self.mutation_strength)
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

"""NEAT genome encoding: node genes, connection genes, and structural operations."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from src.cppn.activations import ALL_ACTIVATIONS


@dataclass
class NodeGene:
    key: int
    node_type: str  # "input", "hidden", "output"
    activation: str = "tanh"


@dataclass
class ConnectionGene:
    key: tuple[int, int]  # (in_node, out_node)
    weight: float = 0.0
    enabled: bool = True


@dataclass
class Genome:
    key: int
    nodes: dict[int, NodeGene] = field(default_factory=dict)
    connections: dict[tuple[int, int], ConnectionGene] = field(default_factory=dict)
    fitness: float | None = None

    # Set during creation, not stored per-connection
    input_keys: tuple[int, ...] = ()
    output_keys: tuple[int, ...] = ()
    _next_node_key: int = 0

    # Compositional parameters (per-genome, evolvable)
    comp_focal_x: float = 0.0
    comp_focal_y: float = 0.0
    comp_armature_angle: float = 0.0

    def copy(self) -> Genome:
        """Deep copy this genome with a new key."""
        g = Genome(key=self.key)
        g.nodes = {k: copy.copy(v) for k, v in self.nodes.items()}
        g.connections = {k: copy.copy(v) for k, v in self.connections.items()}
        g.input_keys = self.input_keys
        g.output_keys = self.output_keys
        g._next_node_key = self._next_node_key
        g.fitness = None
        g.comp_focal_x = self.comp_focal_x
        g.comp_focal_y = self.comp_focal_y
        g.comp_armature_angle = self.comp_armature_angle
        return g

    def allocate_node_key(self) -> int:
        key = self._next_node_key
        self._next_node_key += 1
        return key

    @classmethod
    def create_minimal(
        cls,
        genome_key: int,
        num_inputs: int,
        num_outputs: int,
        output_activation: str = "sigmoid",
        weight_range: float = 2.0,
    ) -> Genome:
        """Create a minimal-topology genome: direct input->output connections.

        Input node keys: 0 .. num_inputs-1
        Output node keys: num_inputs .. num_inputs+num_outputs-1
        """
        g = Genome(key=genome_key)

        input_keys = []
        for i in range(num_inputs):
            key = i
            g.nodes[key] = NodeGene(key=key, node_type="input", activation="linear")
            input_keys.append(key)

        output_keys = []
        for i in range(num_outputs):
            key = num_inputs + i
            g.nodes[key] = NodeGene(
                key=key, node_type="output", activation=output_activation
            )
            output_keys.append(key)

        g.input_keys = tuple(input_keys)
        g.output_keys = tuple(output_keys)
        g._next_node_key = num_inputs + num_outputs

        for i_key in input_keys:
            for o_key in output_keys:
                conn_key = (i_key, o_key)
                w = random.uniform(-weight_range, weight_range)
                g.connections[conn_key] = ConnectionGene(key=conn_key, weight=w)

        g.comp_focal_x = random.uniform(-0.6, 0.6)
        g.comp_focal_y = random.uniform(-0.6, 0.6)
        g.comp_armature_angle = random.uniform(-3.14159, 3.14159)

        return g

    # --- Mutation operators ---

    def mutate_weights(
        self,
        perturb_rate: float = 0.8,
        perturb_power: float = 0.5,
        replace_rate: float = 0.1,
        weight_range: float = 4.0,
    ):
        """Perturb or replace connection weights."""
        for conn in self.connections.values():
            r = random.random()
            if r < replace_rate:
                conn.weight = random.uniform(-weight_range, weight_range)
            elif r < replace_rate + perturb_rate:
                conn.weight += random.gauss(0, perturb_power)
                conn.weight = max(-weight_range, min(weight_range, conn.weight))

    def mutate_add_node(self, activation_options: list[str] | None = None):
        """Split a random enabled connection by inserting a new hidden node."""
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return

        conn = random.choice(enabled)
        conn.enabled = False

        if activation_options is None:
            activation_options = ALL_ACTIVATIONS

        new_key = self.allocate_node_key()
        act = random.choice(activation_options)
        self.nodes[new_key] = NodeGene(key=new_key, node_type="hidden", activation=act)

        # Old connection: A -> B (weight w, now disabled)
        # New: A -> new_node (weight 1.0), new_node -> B (weight w)
        src, dst = conn.key
        key1 = (src, new_key)
        key2 = (new_key, dst)
        self.connections[key1] = ConnectionGene(key=key1, weight=1.0)
        self.connections[key2] = ConnectionGene(key=key2, weight=conn.weight)

    def mutate_add_connection(self, weight_range: float = 2.0, max_attempts: int = 20):
        """Add a new connection between two previously unconnected nodes."""
        node_keys = list(self.nodes.keys())
        input_set = set(self.input_keys)
        output_set = set(self.output_keys)

        for _ in range(max_attempts):
            src = random.choice(node_keys)
            dst = random.choice(node_keys)

            # Can't connect to inputs or from outputs (feedforward constraint)
            if dst in input_set:
                continue
            if src in output_set:
                continue
            if src == dst:
                continue

            conn_key = (src, dst)
            if conn_key in self.connections:
                # Re-enable if disabled
                if not self.connections[conn_key].enabled:
                    self.connections[conn_key].enabled = True
                continue

            # Check for cycles: dst must not be an ancestor of src
            if self._creates_cycle(src, dst):
                continue

            w = random.uniform(-weight_range, weight_range)
            self.connections[conn_key] = ConnectionGene(key=conn_key, weight=w)
            return

    def _creates_cycle(self, src: int, dst: int) -> bool:
        """Check if adding dst->... can reach src (making src->dst a cycle)."""
        # BFS from dst through existing connections
        visited = set()
        queue = [dst]
        while queue:
            node = queue.pop(0)
            if node == src:
                return True
            if node in visited:
                continue
            visited.add(node)
            for (s, d), conn in self.connections.items():
                if s == node and conn.enabled and d not in visited:
                    queue.append(d)
        return False

    def mutate_activation(self, activation_options: list[str] | None = None):
        """Change a random hidden node's activation function."""
        hidden = [n for n in self.nodes.values() if n.node_type == "hidden"]
        if not hidden:
            return
        if activation_options is None:
            activation_options = ALL_ACTIVATIONS
        node = random.choice(hidden)
        node.activation = random.choice(activation_options)

    def mutate_toggle_enable(self):
        """Toggle a random connection's enabled state."""
        if not self.connections:
            return
        conn = random.choice(list(self.connections.values()))
        conn.enabled = not conn.enabled

    def mutate(self, config: dict | None = None):
        """Apply all mutation operators according to rates."""
        if config is None:
            config = {}

        self.mutate_weights(
            perturb_rate=config.get("weight_perturb_rate", 0.8),
            perturb_power=config.get("weight_perturb_power", 0.5),
            replace_rate=config.get("weight_replace_rate", 0.1),
        )

        if random.random() < config.get("add_node_rate", 0.03):
            self.mutate_add_node(config.get("activation_options"))

        if random.random() < config.get("add_connection_rate", 0.05):
            self.mutate_add_connection()

        if random.random() < config.get("activation_mutation_rate", 0.1):
            self.mutate_activation(config.get("activation_options"))

        if random.random() < config.get("toggle_enable_rate", 0.01):
            self.mutate_toggle_enable()

        if random.random() < 0.08:
            self.comp_focal_x = max(-0.8, min(0.8,
                self.comp_focal_x + random.gauss(0, 0.15)))
            self.comp_focal_y = max(-0.8, min(0.8,
                self.comp_focal_y + random.gauss(0, 0.15)))
        if random.random() < 0.05:
            self.comp_armature_angle += random.gauss(0, 0.2)


def crossover(parent1: Genome, parent2: Genome, child_key: int) -> Genome:
    """NEAT crossover: align by connection key, prefer fitter parent.

    parent1 should be the fitter (or equal) parent.
    """
    child = Genome(key=child_key)
    child.input_keys = parent1.input_keys
    child.output_keys = parent1.output_keys
    child._next_node_key = max(parent1._next_node_key, parent2._next_node_key)

    # Inherit nodes from fitter parent, add any extra from parent2
    child.nodes = {k: copy.copy(v) for k, v in parent1.nodes.items()}
    for k, v in parent2.nodes.items():
        if k not in child.nodes:
            child.nodes[k] = copy.copy(v)

    # Align connections by key
    all_conn_keys = set(parent1.connections.keys()) | set(parent2.connections.keys())
    for conn_key in all_conn_keys:
        in_p1 = conn_key in parent1.connections
        in_p2 = conn_key in parent2.connections

        if in_p1 and in_p2:
            # Matching gene: inherit randomly
            if random.random() < 0.5:
                child.connections[conn_key] = copy.copy(parent1.connections[conn_key])
            else:
                child.connections[conn_key] = copy.copy(parent2.connections[conn_key])
        elif in_p1:
            # Disjoint/excess from fitter parent
            child.connections[conn_key] = copy.copy(parent1.connections[conn_key])
        elif parent1.fitness == parent2.fitness:
            child.connections[conn_key] = copy.copy(parent2.connections[conn_key])

    p = parent1 if random.random() < 0.5 else parent2
    child.comp_focal_x = p.comp_focal_x
    child.comp_focal_y = p.comp_focal_y
    child.comp_armature_angle = p.comp_armature_angle

    return child


def compatibility_distance(
    g1: Genome,
    g2: Genome,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.5,
) -> float:
    """Compute NEAT compatibility distance between two genomes."""
    keys1 = set(g1.connections.keys())
    keys2 = set(g2.connections.keys())

    matching = keys1 & keys2
    disjoint_excess = (keys1 - keys2) | (keys2 - keys1)

    n = max(len(keys1), len(keys2), 1)

    # Average weight difference of matching genes
    if matching:
        w_diff = sum(
            abs(g1.connections[k].weight - g2.connections[k].weight) for k in matching
        ) / len(matching)
    else:
        w_diff = 0.0

    return (c1 * len(disjoint_excess) / n) + (c3 * w_diff)

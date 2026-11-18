"""NEAT genome encoding: node genes, connection genes, and structural operations."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from src.cppn.activations import ALL_ACTIVATIONS


class InnovationTracker:
    """Population-wide registry of structural innovations.

    NEAT's crossover and speciation align genes by id, which only works if the
    same structural mutation gets the same id in every genome.  Without this,
    each genome numbered its own new nodes, so node 12 in one genome and node
    12 in another were unrelated, and crossover spliced together structure that
    had nothing in common.  Splitting connection (a, b) now always yields the
    same hidden-node id across the population.
    """

    def __init__(self, next_node_key: int = 0):
        self.next_node_key = next_node_key
        self._splits: dict[tuple[int, int], int] = {}

    def sync(self, genomes) -> None:
        """Make sure fresh ids never collide with ids already in ``genomes``."""
        for g in genomes:
            top = max(g.nodes, default=-1) + 1
            self.next_node_key = max(self.next_node_key, g._next_node_key, top)

    def _fresh(self, genome: "Genome") -> int:
        key = max(self.next_node_key, genome._next_node_key)
        self.next_node_key = key + 1
        return key

    def node_for_split(self, genome: "Genome", conn_key: tuple[int, int]) -> int:
        key = self._splits.get(conn_key)
        if key is None:
            key = self._fresh(genome)
            self._splits[conn_key] = key
        elif key in genome.nodes:
            # This genome already split that connection once; a second split
            # is a different innovation.
            key = self._fresh(genome)
        genome._next_node_key = max(genome._next_node_key, key + 1)
        return key


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

    def mutate_add_node(
        self,
        activation_options: list[str] | None = None,
        innovations: InnovationTracker | None = None,
        input_weight_range: float | None = None,
    ):
        """Split a random enabled connection by inserting a new hidden node.

        Classic NEAT feeds the new node at weight 1.0 so behaviour barely
        changes.  For a CPPN that keeps a periodic/gaussian node in the
        near-linear part of its curve (inputs are only in [-1, 1]), so it adds
        no pattern; ``input_weight_range`` draws the weight at random instead.
        """
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return

        conn = random.choice(enabled)
        conn.enabled = False

        if activation_options is None:
            activation_options = ALL_ACTIVATIONS

        if innovations is not None:
            new_key = innovations.node_for_split(self, conn.key)
        else:
            new_key = self.allocate_node_key()
        act = random.choice(activation_options)
        self.nodes[new_key] = NodeGene(key=new_key, node_type="hidden", activation=act)

        # Old connection: A -> B (weight w, now disabled)
        # New: A -> new_node (weight 1.0), new_node -> B (weight w)
        src, dst = conn.key
        key1 = (src, new_key)
        key2 = (new_key, dst)
        w_in = 1.0
        if input_weight_range:
            w_in = random.choice((-1, 1)) * random.uniform(0.5, input_weight_range)
        self.connections[key1] = ConnectionGene(key=key1, weight=w_in)
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
                    return
                continue

            # Check for cycles: dst must not be an ancestor of src
            if self._creates_cycle(src, dst):
                continue

            w = random.uniform(-weight_range, weight_range)
            self.connections[conn_key] = ConnectionGene(key=conn_key, weight=w)
            return

    def _creates_cycle(self, src: int, dst: int) -> bool:
        """Would adding src->dst close a loop?

        Disabled connections count too: they can be re-enabled later (toggle
        mutation, add-connection re-enabling), and a loop that appears then
        silently drops every node on it from the network.
        """
        return _reaches(self.connections, dst, src)

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
        """Toggle a random connection's enabled state.

        Never disables a node's last enabled input, which would cut it (and
        often a whole output) off and flatten the image.
        """
        if not self.connections:
            return
        conn = random.choice(list(self.connections.values()))
        if conn.enabled:
            dst = conn.key[1]
            live_in = sum(1 for c in self.connections.values()
                          if c.enabled and c.key[1] == dst)
            if live_in <= 1:
                return
        conn.enabled = not conn.enabled

    def mutate(
        self,
        config: dict | None = None,
        innovations: InnovationTracker | None = None,
        strength: float = 1.0,
    ):
        """Apply all mutation operators according to rates.

        ``strength`` scales every rate and the weight step (1.0 = as
        configured), so a UI mutation slider can drive it.
        """
        if config is None:
            config = {}
        k = max(0.0, strength)

        def chance(rate: float) -> bool:
            return random.random() < min(1.0, rate * k)

        self.mutate_weights(
            perturb_rate=min(1.0, config.get("weight_perturb_rate", 0.8) * k),
            perturb_power=config.get("weight_perturb_power", 0.5) * k,
            replace_rate=min(1.0, config.get("weight_replace_rate", 0.1) * k),
            weight_range=config.get("weight_range", 4.0),
        )

        if chance(config.get("add_node_rate", 0.03)):
            self.mutate_add_node(config.get("activation_options"), innovations,
                                 config.get("split_input_weight_range"))

        if chance(config.get("add_connection_rate", 0.05)):
            self.mutate_add_connection(weight_range=config.get("new_weight_range", 2.0))

        if chance(config.get("activation_mutation_rate", 0.1)):
            self.mutate_activation(config.get("activation_options"))

        if chance(config.get("toggle_enable_rate", 0.01)):
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

    # Align connections by key.  The fitter parent's genes go in first; a gene
    # taken from the other parent is skipped if it would close a loop (each
    # parent is acyclic, but their union need not be).
    inputs, outputs = set(child.input_keys), set(child.output_keys)
    for conn_key in sorted(parent1.connections, key=lambda k: k not in parent2.connections):
        src = parent1.connections[conn_key]
        if conn_key in parent2.connections and random.random() < 0.5:
            src = parent2.connections[conn_key]
        child.connections[conn_key] = copy.copy(src)
        if conn_key in parent2.connections:
            # NEAT: a gene disabled in either parent is usually disabled.
            either_off = not (parent1.connections[conn_key].enabled
                              and parent2.connections[conn_key].enabled)
            child.connections[conn_key].enabled = not (either_off and random.random() < 0.75)
    if parent1.fitness == parent2.fitness:
        for conn_key, conn in parent2.connections.items():
            if conn_key in child.connections:
                continue
            s, d = conn_key
            if s not in inputs and d not in outputs and _reaches(child.connections, d, s):
                continue
            child.connections[conn_key] = copy.copy(conn)

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


def _reaches(connections: dict, start: int, target: int) -> bool:
    """Is ``target`` reachable from ``start`` over any connection (enabled or not)?"""
    if start == target:
        return True
    adj: dict[int, list[int]] = {}
    for s, d in connections:
        adj.setdefault(s, []).append(d)
    seen = {start}
    stack = [start]
    while stack:
        for nxt in adj.get(stack.pop(), ()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False

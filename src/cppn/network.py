"""CPPN network: phenotype built from a NEAT genome.

Evaluates the network for a batch of input coordinates using topological sort
and vectorized numpy operations. Each node can have a different activation function.
"""

from __future__ import annotations

import numpy as np
from src.cppn.activations import get_activation


class CPPNNetwork:
    """A CPPN built from a genome, ready for batch evaluation."""

    def __init__(
        self,
        input_keys: list[int],
        output_keys: list[int],
        node_activations: dict[int, str],
        connections: list[tuple[int, int, float]],
    ):
        """
        Args:
            input_keys: Node IDs for inputs (x, y, d, bias, ...).
            output_keys: Node IDs for outputs (R, G, B or H, S, V).
            node_activations: {node_id: activation_name} for hidden + output nodes.
            connections: List of (in_node, out_node, weight) for enabled connections.
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.node_activations = node_activations
        self.connections = connections

        # Build adjacency and compute evaluation order
        self._eval_order = self._topological_sort()
        # Pre-group connections by destination for fast lookup
        self._incoming: dict[int, list[tuple[int, float]]] = {}
        for src, dst, w in connections:
            self._incoming.setdefault(dst, []).append((src, w))

    def _topological_sort(self) -> list[int]:
        """Return hidden + output node IDs in topological order."""
        # Gather all non-input nodes that need evaluation
        all_nodes = set()
        graph: dict[int, set[int]] = {}  # node -> set of nodes it depends on
        input_set = set(self.input_keys)

        for src, dst, _ in self.connections:
            all_nodes.add(dst)
            if dst not in graph:
                graph[dst] = set()
            if src not in input_set:
                all_nodes.add(src)
                graph[dst].add(src)
                if src not in graph:
                    graph[src] = set()

        # Kahn's algorithm
        in_degree = {n: len(deps) for n, deps in graph.items()}
        queue = [n for n in graph if in_degree[n] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for other, deps in graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        # Filter to only nodes we need (connected to outputs)
        needed = set(self.output_keys)
        # Walk backwards to find all nodes that feed into outputs
        changed = True
        while changed:
            changed = False
            for src, dst, _ in self.connections:
                if dst in needed and src not in needed and src not in input_set:
                    needed.add(src)
                    changed = True

        return [n for n in order if n in needed]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Batch evaluate the CPPN.

        Args:
            inputs: (N, num_inputs) array of input values.

        Returns:
            (N, num_outputs) array of output values.
        """
        n = inputs.shape[0]
        values: dict[int, np.ndarray] = {}

        # Set input values
        for i, key in enumerate(self.input_keys):
            values[key] = inputs[:, i]

        # Evaluate in topological order
        for node_id in self._eval_order:
            incoming = self._incoming.get(node_id, [])
            if not incoming:
                values[node_id] = np.zeros(n)
                continue

            total = np.zeros(n)
            for src, weight in incoming:
                if src in values:
                    total += values[src] * weight

            act_name = self.node_activations.get(node_id, "tanh")
            act_fn = get_activation(act_name)
            values[node_id] = act_fn(total)

        # Collect outputs
        output = np.column_stack(
            [values.get(key, np.zeros(n)) for key in self.output_keys]
        )
        return output

    @classmethod
    def from_genome(cls, genome) -> CPPNNetwork:
        """Build a CPPN from a Genome object."""
        node_activations = {}
        for key, node in genome.nodes.items():
            if node.node_type != "input":
                node_activations[key] = node.activation

        connections = []
        for conn in genome.connections.values():
            if conn.enabled:
                connections.append((conn.key[0], conn.key[1], conn.weight))

        return cls(
            input_keys=list(genome.input_keys),
            output_keys=list(genome.output_keys),
            node_activations=node_activations,
            connections=connections,
        )

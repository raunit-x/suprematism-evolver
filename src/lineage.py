"""Parentage tags for the lineage tree.

Breeding code tags every genome it puts into a new generation with where it
came from, as plain attributes on the genome object:

  ``_lk``  kind: "survivor" (a pick carried over), "child" (bred from picks)
           or absent (a newcomer with no parents)
  ``_lp``  parent indices into the previous generation

Before breeding, ``mark_parents`` stamps each pick with its index (``_li``)
so children can name their parents without the breeding code tracking
indices.  The server reads the tags with ``read`` right after a generation
is bred and then removes them with ``clear``, so they never go stale through
later copies.
"""

from __future__ import annotations

_ATTRS = ("_lk", "_lp", "_li")


def mark_parents(parents: list, indices: list[int]) -> None:
    for p, i in zip(parents, indices):
        p._li = i


def _indices(parents) -> tuple:
    seen = []
    for p in parents:
        i = getattr(p, "_li", None)
        if i is not None and i not in seen:
            seen.append(i)
    return tuple(seen)


def survivor(genome, parent):
    genome._lk = "survivor"
    genome._lp = _indices([parent])
    return genome


def child(genome, *parents):
    genome._lk = "child"
    genome._lp = _indices(parents)
    return genome


def read(genomes: list) -> list[tuple[str, list[int]]]:
    """``(kind, parent indices)`` per genome; untagged genomes are "new"."""
    return [(getattr(g, "_lk", "new"), list(getattr(g, "_lp", ()))) for g in genomes]


def clear(genomes: list) -> None:
    for g in genomes:
        for a in _ATTRS:
            g.__dict__.pop(a, None)

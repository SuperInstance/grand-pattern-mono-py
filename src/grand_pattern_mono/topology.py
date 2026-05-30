"""Topology generators for CellGraph."""

from __future__ import annotations

import random
from typing import List

from .graph import CellGraph
from .room import Room


def _make_rooms(n: int, alpha: float = 0.3) -> List[Room]:
    return [Room(id=f"r{i}", vibe=0.0, alpha=alpha) for i in range(n)]


def chain(n: int, alpha: float = 0.3) -> CellGraph:
    """Linear chain of *n* rooms."""
    g = CellGraph()
    rooms = _make_rooms(n, alpha)
    for r in rooms:
        g.add_room(r)
    for i in range(n - 1):
        g.add_edge(f"r{i}", f"r{i + 1}")
    return g


def ring(n: int, alpha: float = 0.3) -> CellGraph:
    """Ring of *n* rooms (chain with wrap-around)."""
    g = chain(n, alpha)
    if n > 2:
        g.add_edge(f"r0", f"r{n - 1}")
    return g


def star(n: int, alpha: float = 0.3) -> CellGraph:
    """Star: room 0 connected to rooms 1..n-1."""
    g = CellGraph()
    rooms = _make_rooms(n, alpha)
    for r in rooms:
        g.add_room(r)
    for i in range(1, n):
        g.add_edge("r0", f"r{i}")
    return g


def mesh(n: int, alpha: float = 0.3) -> CellGraph:
    """Fully connected mesh: every room connected to every other."""
    g = CellGraph()
    rooms = _make_rooms(n, alpha)
    for r in rooms:
        g.add_room(r)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(f"r{i}", f"r{j}")
    return g


def small_world(n: int, k: int = 4, p: float = 0.1, alpha: float = 0.3) -> CellGraph:
    """Watts-Strogatz small-world graph.

    Parameters
    ----------
    n : int
        Number of rooms.
    k : int
        Each room connects to k nearest neighbours (must be even).
    p : float
        Rewiring probability.
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    g = CellGraph()
    rooms = _make_rooms(n, alpha)
    for r in rooms:
        g.add_room(r)
    # Build ring lattice
    edge_set: set = set()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            a, b = i, (i + j) % n
            edge_set.add((min(a, b), max(a, b)))
    # Rewire
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if random.random() < p:
                a, b = i, (i + j) % n
                edge_set.discard((min(a, b), max(a, b)))
                new_b = random.choice([x for x in range(n) if x != i])
                edge_set.add((min(i, new_b), max(i, new_b)))
    for a, b in edge_set:
        g.add_edge(f"r{a}", f"r{b}")
    return g


def scale_free(n: int, m: int = 2, alpha: float = 0.3) -> CellGraph:
    """Barabási-Albert scale-free graph.

    Parameters
    ----------
    n : int
        Total rooms.
    m : int
        Edges added per new room.
    """
    if m < 1 or m >= n:
        raise ValueError("m must be in [1, n-1]")
    g = CellGraph()
    rooms = _make_rooms(n, alpha)
    for r in rooms:
        g.add_room(r)
    # Preferential attachment
    repeated: List[int] = []
    # Start with a small complete graph of m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            g.add_edge(f"r{i}", f"r{j}")
            repeated.extend([i, j])
    # Add remaining nodes
    for new in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            targets.add(random.choice(repeated))
        for t in targets:
            g.add_edge(f"r{new}", f"r{t}")
            repeated.extend([new, t])
    return g

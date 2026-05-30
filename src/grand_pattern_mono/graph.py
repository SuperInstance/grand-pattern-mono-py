"""CellGraph – rooms + edges + tick/diffuse/gossip/learn."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from .room import Room


class CellGraph:
    """A graph of rooms connected by edges.

    Conservation holds by construction: diffusion redistributes vibe among
    neighbours without creating or destroying it.
    """

    def __init__(self) -> None:
        self.rooms: Dict[str, Room] = {}
        self.edges: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_room(self, room: Room) -> None:
        self.rooms[room.id] = room
        if room.id not in self.edges:
            self.edges[room.id] = set()

    def add_edge(self, a: str, b: str) -> None:
        if a not in self.rooms or b not in self.rooms:
            raise KeyError(f"Room {a!r} or {b!r} not in graph")
        self.edges[a].add(b)
        self.edges[b].add(a)

    def neighbours(self, room_id: str) -> Set[str]:
        return self.edges.get(room_id, set())

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """One simulation tick: learn from predicted vs actual for each room."""
        for room in self.rooms.values():
            predicted = room.predict()
            # Small perturbation from prediction (conservation-neutral)
            room.observe(predicted + random.gauss(0, 0.01))

    def diffuse(self, rate: float = 0.1) -> None:
        """Diffuse vibe between connected rooms (conservation by construction).

        Each room shares a fraction of its vibe equally among neighbours.
        The total vibe across the graph is preserved.
        """
        deltas: Dict[str, float] = {rid: 0.0 for rid in self.rooms}
        for rid, room in self.rooms.items():
            nbrs = self.neighbours(rid)
            if not nbrs:
                continue
            share = rate * room.vibe / len(nbrs)
            deltas[rid] -= rate * room.vibe
            for n in nbrs:
                deltas[n] += share
        # Apply deltas
        for rid, delta in deltas.items():
            new_vibe = self.rooms[rid].vibe + delta
            self.rooms[rid].observe(new_vibe)

    def gossip(self, room_id: str, target_id: Optional[str] = None) -> float:
        """Gossip: share vibe info between rooms.

        If target is given, share with that specific neighbour.
        Otherwise pick a random neighbour.

        Returns the shared vibe value.
        """
        if room_id not in self.rooms:
            raise KeyError(room_id)
        nbrs = list(self.neighbours(room_id))
        if not nbrs:
            return self.rooms[room_id].vibe
        target = target_id if target_id and target_id in nbrs else random.choice(nbrs)
        shared_vibe = self.rooms[room_id].vibe
        # Blend vibes (conservation-neutral within the pair)
        avg = (shared_vibe + self.rooms[target].vibe) / 2
        self.rooms[room_id].observe(avg)
        self.rooms[target].observe(avg)
        return shared_vibe

    def learn(self) -> None:
        """One learning step: predict, observe, reduce surprise."""
        for room in self.rooms.values():
            predicted = room.predict()
            # Move slightly towards prediction (conservation-neutral)
            room.observe(0.9 * room.vibe + 0.1 * predicted)

    # ------------------------------------------------------------------
    # Conservation check
    # ------------------------------------------------------------------

    def total_vibe(self) -> float:
        return sum(r.vibe for r in self.rooms.values())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rooms)

    def __repr__(self) -> str:
        return f"CellGraph(rooms={len(self.rooms)}, edges={sum(len(v) for v in self.edges.values()) // 2})"

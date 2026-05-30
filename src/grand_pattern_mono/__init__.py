"""grand-pattern-mono – mono-vibe corrected architecture.

Vibe = float. JEPA = weighted history predictor.
Conservation holds by construction.
"""

from .jepa import Jepa
from .room import Room
from .graph import CellGraph
from .topology import (
    chain,
    ring,
    star,
    mesh,
    small_world,
    scale_free,
)

__all__ = [
    "Jepa",
    "Room",
    "CellGraph",
    "chain",
    "ring",
    "star",
    "mesh",
    "small_world",
    "scale_free",
]

__version__ = "0.1.0"

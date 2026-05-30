"""Room – a cell in the mono-vibe architecture."""

from __future__ import annotations

from .jepa import Jepa


class Room:
    """A room / cell with an id, a mono vibe, and a JEPA predictor.

    Parameters
    ----------
    id : str
        Unique identifier.
    vibe : float
        Initial vibe value.
    alpha : float
        JEPA smoothing factor.
    """

    def __init__(self, id: str, vibe: float = 0.0, alpha: float = 0.3) -> None:
        self.id = id
        self.vibe: float = vibe
        self.jepa = Jepa(alpha=alpha)
        self.last_surprise: float = 0.0
        # Seed JEPA with initial vibe
        self.jepa.observe(vibe)

    def observe(self, new_vibe: float) -> None:
        """Observe a new vibe, update JEPA, and record surprise."""
        self.last_surprise = self.jepa.surprise(new_vibe)
        self.vibe = new_vibe
        self.jepa.observe(new_vibe)

    def predict(self) -> float:
        return self.jepa.predict()

    def __repr__(self) -> str:
        return f"Room(id={self.id!r}, vibe={self.vibe:.4f}, surprise={self.last_surprise:.4f})"

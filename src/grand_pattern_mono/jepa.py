"""JEPA – weighted history predictor.

The JEPA (Joint-Embedding Predictive Architecture) reads a room's weighted
history and predicts the next vibe.  It is *not* a neural net here – it is a
simple exponentially-weighted moving average that learns to weight prior
readings.  Conservation holds by construction: the prediction is always a
convex combination of past observations.
"""

from __future__ import annotations

import math
from typing import List


class Jepa:
    """Weighted-history predictor for a mono vibe stream.

    Parameters
    ----------
    alpha : float
        Exponential smoothing factor in (0, 1].  Higher → more weight on
        recent observations.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._history: List[float] = []
        self._weighted_sum: float = 0.0
        self._weight_total: float = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, vibe: float) -> None:
        """Record a new vibe observation."""
        self._history.append(vibe)
        # Exponential weights: w_k = alpha * (1 - alpha)^(n - 1 - k)
        n = len(self._history)
        self._weight_total = 1.0  # weights sum to 1 by EWM property
        self._weighted_sum = self._recompute_weighted_sum()

    def predict(self) -> float:
        """Predict the next vibe (conservation-safe: convex combo of past)."""
        if not self._history:
            return 0.0
        return self._weighted_sum

    def surprise(self, observed: float) -> float:
        """Return surprise = |observed − predicted|."""
        return abs(observed - self.predict())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[float]:
        return list(self._history)

    @property
    def n_observations(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute_weighted_sum(self) -> float:
        n = len(self._history)
        total = 0.0
        weight_sum = 0.0
        for i, v in enumerate(self._history):
            age = n - 1 - i  # 0 for most recent
            w = self.alpha * ((1 - self.alpha) ** age)
            total += w * v
            weight_sum += w
        if weight_sum == 0.0:
            return 0.0
        return total / weight_sum

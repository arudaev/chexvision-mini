"""Loss functions with hand-derived gradients."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (no overflow for large |x|)."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class BCEWithLogitsLoss:
    """Binary cross-entropy computed on raw logits (fuses sigmoid + BCE).

    Fusing the sigmoid into the loss is done for two reasons:

    1. **Numerical stability** — the loss is evaluated in the log-sum-exp form
       ``max(z, 0) - z*y + log(1 + exp(-|z|))`` which never overflows.
    2. **A clean gradient** — ``dL/dz = (sigmoid(z) - y) / N``, which we derive
       by hand in the report and is what :meth:`backward` returns.

    Optional label smoothing mirrors the parent CheXVision project: positive
    targets become ``1 - eps`` and negative targets ``eps / 2``, regularising
    against noisy labels.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = label_smoothing
        self._probs: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._n: int = 0

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        targets = targets.astype(np.float64)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + (1.0 - targets) * (eps / 2.0)
        z = logits
        loss = np.maximum(z, 0) - z * targets + np.log1p(np.exp(-np.abs(z)))
        self._probs = sigmoid(z)
        self._targets = targets
        self._n = logits.shape[0]
        return float(loss.sum() / self._n)

    def backward(self) -> np.ndarray:
        return (self._probs - self._targets) / self._n

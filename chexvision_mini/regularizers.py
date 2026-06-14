"""Regularisation: inverted dropout and an L2 weight penalty.

The course explicitly lists L1/L2 regularisation and dropout, so both are
implemented from scratch here rather than pulled from a framework.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer


class Dropout(Layer):
    """Inverted dropout.

    During training each activation is kept with probability ``1 - p`` and the
    survivors are scaled up by ``1 / (1 - p)`` so the expected activation is
    unchanged — this is why no rescaling is needed at inference time, where the
    layer is a no-op (``training=False``).
    """

    def __init__(self, p: float = 0.5, *, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return dout
        return dout * self._mask


def l2_penalty(model, weight_decay: float) -> float:
    """Return the L2 penalty ``0.5 * wd * sum(W**2)`` over weight matrices only.

    Biases are excluded by convention — penalising them does not curb model
    complexity and can hurt fit. This matches the weights-only regularisation in
    :mod:`chexvision_mini.optim`. ``model`` exposes ``params_and_grads()``.
    """
    if weight_decay == 0.0:
        return 0.0
    total = 0.0
    for (_, name), param, _ in model.params_and_grads():
        if name == "W":
            total += float((param**2).sum())
    return 0.5 * weight_decay * total


def l1_penalty(model, l1: float) -> float:
    """Return the L1 penalty ``l1 * sum(|W|)`` over weight matrices only.

    Gradient ``l1 * sign(W)`` is applied in :mod:`chexvision_mini.optim`.
    """
    if l1 == 0.0:
        return 0.0
    total = 0.0
    for (_, name), param, _ in model.params_and_grads():
        if name == "W":
            total += float(np.abs(param).sum())
    return l1 * total

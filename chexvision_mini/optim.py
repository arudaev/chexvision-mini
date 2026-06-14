"""Hand-coded optimizers: SGD (with momentum), RMSProp, and Adam.

Each implements its update rule directly so the report can explain exactly how a
step modifies the parameters. Regularisation is applied **to weight matrices
only** (not biases), matching :func:`chexvision_mini.regularizers.l2_penalty`:

- ``weight_decay`` adds the L2 gradient ``wd * W``.
- ``l1`` adds the L1 gradient ``l1 * sign(W)``.
"""

from __future__ import annotations

import numpy as np

from .network import Sequential


def _reg_grad(name: str, param: np.ndarray, grad: np.ndarray, weight_decay: float, l1: float) -> np.ndarray:
    """Add L2/L1 regularisation gradients to ``grad`` for weight matrices only."""
    if name != "W":
        return grad
    g = grad
    if weight_decay:
        g = g + weight_decay * param
    if l1:
        g = g + l1 * np.sign(param)
    return g


class SGD:
    """Stochastic gradient descent with optional (heavy-ball) momentum."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        l1: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.l1 = l1
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for (idx, name), param, grad in self.model.params_and_grads():
            if grad is None:
                continue
            g = _reg_grad(name, param, grad, self.weight_decay, self.l1)
            if self.momentum:
                v = self.velocity.get((idx, name))
                if v is None:
                    v = np.zeros_like(param)
                v = self.momentum * v - self.lr * g
                self.velocity[(idx, name)] = v
                param += v
            else:
                param -= self.lr * g


class RMSProp:
    """RMSProp — divides the step by a moving-average RMS of recent gradients."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 1e-3,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        l1: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.l1 = l1
        self.cache: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for (idx, name), param, grad in self.model.params_and_grads():
            if grad is None:
                continue
            g = _reg_grad(name, param, grad, self.weight_decay, self.l1)
            cache = self.rho * self.cache.get((idx, name), np.zeros_like(param)) + (1 - self.rho) * (g * g)
            self.cache[(idx, name)] = cache
            param -= self.lr * g / (np.sqrt(cache) + self.eps)


class Adam:
    """Adam optimizer (Kingma & Ba, 2015) with bias-corrected moments."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        l1: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.l1 = l1
        self.m: dict[tuple[int, str], np.ndarray] = {}
        self.v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for (idx, name), param, grad in self.model.params_and_grads():
            if grad is None:
                continue
            g = _reg_grad(name, param, grad, self.weight_decay, self.l1)
            m = self.beta1 * self.m.get((idx, name), np.zeros_like(param)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get((idx, name), np.zeros_like(param)) + (1 - self.beta2) * (g * g)
            self.m[(idx, name)] = m
            self.v[(idx, name)] = v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

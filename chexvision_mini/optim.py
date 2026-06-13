"""Hand-coded optimizers: SGD (with momentum) and Adam.

Both implement the update rules directly so the report can explain exactly how
each step modifies the parameters. ``weight_decay`` adds the gradient of the L2
penalty (``wd * param``) before the update — decoupled bias terms are included
here for simplicity but excluded from the reported L2 penalty value.
"""

from __future__ import annotations

import numpy as np

from .network import Sequential


class SGD:
    """Stochastic gradient descent with optional (heavy-ball) momentum."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for key, param, grad in self.model.params_and_grads():
            if grad is None:
                continue
            g = grad + self.weight_decay * param if self.weight_decay else grad
            if self.momentum:
                v = self.velocity.get(key)
                if v is None:
                    v = np.zeros_like(param)
                v = self.momentum * v - self.lr * g
                self.velocity[key] = v
                param += v
            else:
                param -= self.lr * g


class Adam:
    """Adam optimizer (Kingma & Ba, 2015) with bias-corrected moments."""

    def __init__(
        self,
        model: Sequential,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[tuple[int, str], np.ndarray] = {}
        self.v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for key, param, grad in self.model.params_and_grads():
            if grad is None:
                continue
            g = grad + self.weight_decay * param if self.weight_decay else grad
            m = self.beta1 * self.m.get(key, np.zeros_like(param)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(key, np.zeros_like(param)) + (1 - self.beta2) * (g * g)
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

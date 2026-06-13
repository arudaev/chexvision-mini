"""From-scratch neural-network layers (pure NumPy).

Every layer implements an explicit ``forward()`` and ``backward()`` so the chain
rule is visible and auditable — there is no autograd anywhere in this package.
This is the pedagogical core of the project: gradients are derived and coded by
hand, then verified numerically in :mod:`chexvision_mini.gradcheck`.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base layer.

    Subclasses populate ``params`` (learnable arrays) and, during ``backward``,
    ``grads`` (the gradient of the loss w.r.t. each parameter). Layers with no
    parameters (ReLU, Sigmoid) leave both dicts empty.
    """

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Layer):
    """Fully-connected layer: ``y = x @ W + b``.

    He initialisation (std = ``sqrt(2 / fan_in)``) is used because the network
    pairs these layers with ReLU activations. He init keeps the forward-pass
    activation variance roughly constant across depth, which avoids the
    vanishing/exploding activations that plague naive Gaussian init.
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / in_features)
        self.params["W"] = (rng.standard_normal((in_features, out_features)) * std).astype(np.float64)
        self.params["b"] = np.zeros(out_features, dtype=np.float64)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dL/dW = x^T @ dout ; dL/db = sum over batch ; dL/dx = dout @ W^T
        x = self._x
        self.grads["W"] = x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class ReLU(Layer):
    """Rectified linear unit: ``max(0, x)``.

    The local gradient is 1 where the input was positive and 0 elsewhere, so the
    backward pass just masks the upstream gradient.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class Sigmoid(Layer):
    """Logistic sigmoid, used at inference to turn logits into probabilities.

    Training uses :class:`chexvision_mini.losses.BCEWithLogitsLoss`, which fuses the
    sigmoid into the loss for numerical stability, so this standalone layer is
    only needed when producing probabilities at evaluation time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._out = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._out * (1.0 - self._out)

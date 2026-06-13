"""Sequential container that chains layer forward/backward passes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .layers import Layer


class Sequential:
    """A linear stack of layers.

    ``forward`` runs the layers in order; ``backward`` walks them in reverse,
    threading the upstream gradient through each layer's ``backward`` — this is
    backpropagation made explicit.
    """

    def __init__(self, *layers: Layer) -> None:
        self.layers: list[Layer] = list(layers)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params_and_grads(
        self,
    ) -> Iterator[tuple[tuple[int, str], np.ndarray, np.ndarray | None]]:
        """Yield ``((layer_index, param_name), param, grad)`` for every parameter.

        The ``(layer_index, param_name)`` key is stable across steps, so
        optimizers use it to key their per-parameter state (momentum, Adam
        moments).
        """
        for i, layer in enumerate(self.layers):
            for name in layer.params:
                yield (i, name), layer.params[name], layer.grads.get(name)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Flatten all parameters into a ``{"<idx>.<name>": array}`` dict."""
        state: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for name, val in layer.params.items():
                state[f"{i}.{name}"] = val
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            for name in layer.params:
                key = f"{i}.{name}"
                if key in state:
                    layer.params[name] = state[key]

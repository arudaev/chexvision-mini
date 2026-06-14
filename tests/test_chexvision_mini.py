"""Tests for the from-scratch NumPy network.

The headline test is :func:`test_gradient_check`: it proves the hand-derived
backward passes match numerical gradients. The remaining tests guard layer
shapes and that the whole stack can actually learn (overfit a tiny batch).
"""

from __future__ import annotations

import numpy as np

from chexvision_mini import (
    SGD,
    Adam,
    BCEWithLogitsLoss,
    Dropout,
    Linear,
    ReLU,
    RMSProp,
    Sequential,
    Sigmoid,
    best_threshold,
    gradient_check,
)


def _make_mlp(in_features: int = 8, hidden: int = 16) -> Sequential:
    return Sequential(
        Linear(in_features, hidden, seed=1),
        ReLU(),
        Linear(hidden, hidden, seed=2),
        ReLU(),
        Linear(hidden, 1, seed=3),
    )


def test_gradient_check_passes() -> None:
    """Analytic gradients must match central-difference estimates."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 8))
    y = rng.integers(0, 2, size=(5, 1)).astype(np.float64)

    model = _make_mlp()
    loss_fn = BCEWithLogitsLoss()
    max_rel, _ = gradient_check(model, loss_fn, x, y, num_checks=30)

    assert max_rel < 1e-6, f"gradient check failed: max relative error {max_rel:.2e}"


def test_linear_forward_shape() -> None:
    layer = Linear(4, 3, seed=0)
    out = layer.forward(np.ones((6, 4)))
    assert out.shape == (6, 3)


def test_relu_masks_negatives() -> None:
    relu = ReLU()
    out = relu.forward(np.array([[-1.0, 2.0, -3.0]]))
    np.testing.assert_array_equal(out, [[0.0, 2.0, 0.0]])
    # gradient flows only where the input was positive
    grad = relu.backward(np.array([[1.0, 1.0, 1.0]]))
    np.testing.assert_array_equal(grad, [[0.0, 1.0, 0.0]])


def test_sigmoid_layer_matches_definition() -> None:
    out = Sigmoid().forward(np.array([0.0]))
    np.testing.assert_allclose(out, [0.5])


def test_dropout_is_noop_at_inference() -> None:
    drop = Dropout(p=0.5, seed=0)
    x = np.ones((10, 10))
    np.testing.assert_array_equal(drop.forward(x, training=False), x)


def test_dropout_scales_in_training() -> None:
    drop = Dropout(p=0.5, seed=0)
    x = np.ones((1000, 1))
    out = drop.forward(x, training=True)
    # inverted dropout keeps the expected value ~1.0
    assert abs(out.mean() - 1.0) < 0.1


def _train_until(optimizer_cls, **kwargs) -> float:
    """Overfit a tiny linearly-separable batch and return the final loss."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 8))
    w_true = rng.standard_normal((8, 1))
    y = (x @ w_true > 0).astype(np.float64)

    model = _make_mlp()
    loss_fn = BCEWithLogitsLoss()
    opt = optimizer_cls(model, **kwargs)

    loss = float("inf")
    for _ in range(300):
        logits = model.forward(x, training=True)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        opt.step()
    return loss


def test_sgd_overfits_tiny_batch() -> None:
    assert _train_until(SGD, lr=0.1, momentum=0.9) < 0.1


def test_adam_overfits_tiny_batch() -> None:
    assert _train_until(Adam, lr=0.05) < 0.1


def test_rmsprop_overfits_tiny_batch() -> None:
    assert _train_until(RMSProp, lr=0.01) < 0.1


def test_best_threshold_in_range_and_separates() -> None:
    rng = np.random.default_rng(0)
    # well-separated scores: positives high, negatives low
    probs = np.concatenate([rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 50)]).reshape(-1, 1)
    targets = np.concatenate([np.ones(50), np.zeros(50)]).reshape(-1, 1)
    thr = best_threshold(probs, targets)
    assert 0.0 <= thr <= 1.0
    # on perfectly separable scores the Youden threshold separates the classes
    preds = (probs >= thr).astype(float)
    assert float((preds == targets).mean()) == 1.0

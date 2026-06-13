"""Finite-difference gradient checking — the correctness proof for backprop.

For a handful of randomly chosen parameter entries, this compares the analytic
gradient produced by the hand-written ``backward`` passes against a numerical
estimate from central differences::

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

In float64 a correct implementation agrees to a relative error below 1e-6
(with the default ``eps``; the textbook "gradients are correct" bound is the
looser 1e-5). This is the single most important test: it shows
the chain-rule code is right, independent of whether the model trains well.
"""

from __future__ import annotations

import numpy as np

from .network import Sequential


def gradient_check(
    model: Sequential,
    loss_fn,
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-5,
    num_checks: int = 20,
    seed: int = 0,
) -> tuple[float, list[tuple[tuple[int, str], tuple[int, ...], float, float, float]]]:
    """Return ``(max_relative_error, per_check_records)``.

    Dropout/randomised layers are evaluated with ``training=False`` so the
    function under test is deterministic.
    """
    # 1. Analytic gradients via one forward + backward pass.
    logits = model.forward(x, training=False)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    rng = np.random.default_rng(seed)
    max_rel = 0.0
    records: list[tuple[tuple[int, str], tuple[int, ...], float, float, float]] = []

    for key, param, grad in model.params_and_grads():
        if grad is None:
            continue
        n_entries = min(num_checks, param.size)
        for _ in range(n_entries):
            idx = tuple(int(rng.integers(0, s)) for s in param.shape)
            original = param[idx]

            param[idx] = original + eps
            loss_plus = loss_fn.forward(model.forward(x, training=False), y)
            param[idx] = original - eps
            loss_minus = loss_fn.forward(model.forward(x, training=False), y)
            param[idx] = original  # restore

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = float(grad[idx])
            denom = max(1e-12, abs(numeric) + abs(analytic))
            rel = abs(numeric - analytic) / denom
            max_rel = max(max_rel, rel)
            records.append((key, idx, analytic, numeric, rel))

    return max_rel, records

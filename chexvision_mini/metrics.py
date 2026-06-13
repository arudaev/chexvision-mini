"""Evaluation metrics.

Per the project rules, scikit-learn is allowed for *evaluation* metrics (it is
not part of the network's forward/backward maths). A dependency-free AUC
fallback is provided so the package still works if scikit-learn is absent.
"""

from __future__ import annotations

import numpy as np


def accuracy(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of correct binary predictions at the given probability threshold."""
    preds = (probs >= threshold).astype(np.float64)
    return float((preds == targets).mean())


def roc_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    """Area under the ROC curve. Uses scikit-learn when available."""
    y_true = targets.reshape(-1)
    y_score = probs.reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")  # AUC undefined with a single class present
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        return _auc_fallback(y_score, y_true)


def _auc_fallback(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC, used when scikit-learn is unavailable."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

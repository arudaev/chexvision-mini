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


def best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """Operating threshold that maximises Youden's J (= TPR − FPR) on these data.

    Chosen on validation only; the 0.5 default is heavily skewed toward the
    majority class when prevalence is not 50/50.
    """
    y = targets.reshape(-1)
    p = probs.reshape(-1)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0.0, 1.0)
    return float(thr[int(np.argmax(tpr - fpr))])


def _downsample(values: np.ndarray, max_points: int = 200) -> list[float]:
    """Evenly subsample a curve to at most ``max_points`` points for compact JSON."""
    if len(values) <= max_points:
        return [float(v) for v in values]
    idx = np.linspace(0, len(values) - 1, max_points).astype(int)
    return [float(v) for v in values[idx]]


def evaluation_report(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    """Full evaluation bundle for the report/deck figures.

    Returns scalar metrics (AUC, accuracy, precision, recall, specificity, F1),
    the confusion matrix at ``threshold``, and downsampled ROC and PR curves —
    everything needed to rebuild clean figures later without re-running the model.
    """
    y = targets.reshape(-1).astype(int)
    p = probs.reshape(-1)
    preds = (p >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    roc: dict[str, list[float]] = {"fpr": [], "tpr": []}
    pr: dict[str, list[float]] = {"precision": [], "recall": []}
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        roc = {"fpr": _downsample(fpr), "tpr": _downsample(tpr)}
        pr = {"precision": _downsample(prec), "recall": _downsample(rec)}
    except ImportError:
        pass

    return {
        "threshold": threshold,
        "auc": roc_auc(probs, targets),
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_curve": roc,
        "pr_curve": pr,
        "n_val": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }

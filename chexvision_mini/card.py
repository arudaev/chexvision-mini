"""Render the Hugging Face model card (README.md) from training results.

The card is written into the artifacts directory and uploaded to the HF repo, so
the model page shows the architecture, results, and how to use the checkpoint.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.4f}"  # NaN-safe


def render_model_card(meta: dict[str, Any], config: dict[str, Any], best_epoch: int, epochs: int) -> str:
    """Return the Markdown model card (with HF frontmatter).

    ``meta`` is the full metrics bundle written to ``metrics.json`` — it has
    ``test`` (final, headline), ``validation`` (model-selection), ``threshold``,
    and ``splits``.
    """
    test = meta["test"]
    val = meta["validation"]
    cm = test["confusion_matrix"]
    thr = meta["threshold"]
    splits = meta["splits"]
    hidden = " → ".join(str(h) for h in config["model"]["hidden_dims"])
    image_size = config["image_size"]
    inputs = image_size * image_size

    return f"""---
license: mit
library_name: numpy
tags:
- chest-xray
- medical-imaging
- from-scratch
- numpy
- education
pipeline_tag: image-classification
---

# CheXVision-mini — from-scratch NumPy neural network

A pure-**NumPy** multilayer perceptron (no autograd, no deep-learning framework),
with every forward and backward pass derived and coded by hand, trained for
binary chest X-ray screening (**normal vs abnormal**) on NIH ChestX-ray14.

Companion to [CheXVision](https://github.com/arudaev/chexvision) (PyTorch: a
custom CNN + a DenseNet-121 transfer model). This model demonstrates the
**fundamentals** — hand-written backprop verified by finite-difference gradient
checking. It is intentionally a fundamentals demo: the headline performance
belongs to the PyTorch models (DenseNet binary AUC ≈ 0.787), not to this MLP.

## Results — held-out test set (final)

Metrics on an **untouched test split**, at an operating threshold chosen on the
validation set only (Youden's J = {thr:.3f}). ROC-AUC is threshold-independent.

| Metric | Test | Validation |
|---|---|---|
| ROC-AUC | **{_fmt(test['auc'])}** | {_fmt(val['auc'])} |
| Accuracy | {_fmt(test['accuracy'])} | {_fmt(val['accuracy'])} |
| Balanced accuracy | {_fmt(test['balanced_accuracy'])} | {_fmt(val['balanced_accuracy'])} |
| Precision | {_fmt(test['precision'])} | {_fmt(val['precision'])} |
| Recall (sensitivity) | {_fmt(test['recall'])} | {_fmt(val['recall'])} |
| Specificity | {_fmt(test['specificity'])} | {_fmt(val['specificity'])} |
| F1 | {_fmt(test['f1'])} | {_fmt(val['f1'])} |

Checkpoint selected by best validation AUC (epoch {best_epoch}/{epochs}).
Samples — train {splits['train']['n']}, val {splits['val']['n']}, test {splits['test']['n']}
(test positive rate {_fmt(test['positive_rate'])}).
Test confusion matrix @ {thr:.3f}: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.

> **Note on the test split:** NIH ChestX-ray14's official `test` split is more
> positive-heavy ({_fmt(test['positive_rate'])}) than train/validation
> ({_fmt(val['positive_rate'])}). Because of that base-rate shift, plain accuracy
> can mislead — **ROC-AUC (threshold-independent) and balanced accuracy are the
> metrics to trust** for comparison.

## Architecture

MLP on {image_size}×{image_size} grayscale images: **{inputs} → {hidden} → 1** logit,
ReLU activations, dropout {config['model']['dropout']}, He initialisation.
Loss: BCE-with-logits (+ label smoothing {config['training']['label_smoothing']}).
Optimizer: {config['training']['optimizer']} with cosine LR decay; L2 weight decay
(weights only). Per-feature standardisation; augmentation: h-flip / noise / brightness.

## Files

- `model.npz` — best weights + normalisation stats (`_norm_mean`, `_norm_std`).
- `metrics.json` — test & validation metrics, ROC/PR curves, confusion matrices, config.
- `history.json` — per-epoch train/reg/val loss, val accuracy/AUC, learning rate.
- `val_scores.npy` / `val_labels.npy`, `test_scores.npy` / `test_labels.npy` — raw scores + labels.
- `loss_curve.png` — training curves + val AUC.

## Usage

```python
from chexvision_mini.inference import load_checkpoint, preprocess_image, predict_label
model, mean, std, threshold = load_checkpoint("artifacts")
x = preprocess_image("xray.png", image_size={image_size}, mean=mean, std=std)
prob, label = predict_label(model, x, threshold)   # P(abnormal), "normal"/"abnormal"
```

Or from the CLI: `python -m chexvision_mini predict --checkpoint artifacts --image xray.png`.

## Links

- Code: https://github.com/arudaev/chexvision-mini
- Parent project: https://github.com/arudaev/chexvision
"""

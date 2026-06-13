"""Render the Hugging Face model card (README.md) from training results.

The card is written into the artifacts directory and uploaded to the HF repo, so
the model page shows the architecture, results, and how to use the checkpoint.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.4f}"  # NaN-safe


def render_model_card(metrics: dict[str, Any], config: dict[str, Any], best_epoch: int, epochs: int) -> str:
    """Return the Markdown model card (with HF frontmatter)."""
    cm = metrics["confusion_matrix"]
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

## Results (validation)

| Metric | Value |
|---|---|
| ROC-AUC | **{_fmt(metrics['auc'])}** |
| Accuracy | {_fmt(metrics['accuracy'])} |
| Precision | {_fmt(metrics['precision'])} |
| Recall (sensitivity) | {_fmt(metrics['recall'])} |
| Specificity | {_fmt(metrics['specificity'])} |
| F1 | {_fmt(metrics['f1'])} |

Selected by best validation AUC (epoch {best_epoch}/{epochs}); validation
n={metrics['n_val']}, positive rate {_fmt(metrics['positive_rate'])}.

Confusion matrix @ threshold {metrics['threshold']}: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.

## Architecture

MLP on {image_size}×{image_size} grayscale images: **{inputs} → {hidden} → 1** logit,
ReLU activations, dropout {config['model']['dropout']}, He initialisation.
Loss: BCE-with-logits (+ label smoothing {config['training']['label_smoothing']}).
Optimizer: {config['training']['optimizer']} with cosine LR decay.
Per-feature standardisation; augmentation: horizontal flip / noise / brightness.

## Files

- `model.npz` — best weights + normalisation stats (`_norm_mean`, `_norm_std`).
- `history.json` — per-epoch train/val loss, val accuracy/AUC, learning rate.
- `metrics.json` — ROC & PR curves, confusion matrix, scalar metrics (for figures).
- `val_scores.npy` / `val_labels.npy` — raw validation scores + labels.
- `loss_curve.png` — training curves + val AUC.

## Usage

```python
import numpy as np
ckpt = np.load("model.npz")
mean, std = ckpt["_norm_mean"], ckpt["_norm_std"]
# rebuild the MLP from chexvision_mini, load the layer weights, standardise
# inputs with (x - mean) / std, then forward(...). See the repo for details.
```

## Links

- Code: https://github.com/arudaev/chexvision-mini
- Parent project: https://github.com/arudaev/chexvision
"""

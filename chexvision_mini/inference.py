"""Checkpoint loading and single-image prediction (forward-only).

Completes the book's final chapters (model loading + prediction): reconstruct the
MLP from a saved artifacts directory, reapply the exact training preprocessing,
and turn a raw chest X-ray into a normal/abnormal decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .losses import sigmoid
from .network import Sequential
from .train import build_mlp


def load_checkpoint(output_dir: Path | str) -> tuple[Sequential, np.ndarray, np.ndarray, float]:
    """Rebuild the model from ``output_dir`` and return (model, mean, std, threshold)."""
    output_dir = Path(output_dir)
    ckpt = np.load(output_dir / "model.npz")
    meta = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    input_dim = cfg["image_size"] * cfg["image_size"]
    # Same dropout as training so layer indices line up with the saved state_dict
    # (dropout is a no-op at inference, training=False).
    model = build_mlp(input_dim, cfg["hidden_dims"], cfg["dropout"], seed=0)
    model.load_state_dict({k: ckpt[k] for k in ckpt.files if not k.startswith("_norm")})
    return model, ckpt["_norm_mean"], ckpt["_norm_std"], float(meta.get("threshold", 0.5))


def preprocess_image(
    path: Path | str, image_size: int, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> np.ndarray:
    """Load an image and apply the exact training preprocessing → shape (1, image_size**2)."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((image_size, image_size))
    x = np.asarray(img, dtype=np.float64).reshape(1, -1) / 255.0
    if mean is not None and std is not None:
        x = (x - mean) / std
    return x


def predict_proba(model: Sequential, x: np.ndarray) -> float:
    """Return P(abnormal) for a single preprocessed sample."""
    return float(sigmoid(model.forward(x, training=False)).reshape(-1)[0])


def predict_label(model: Sequential, x: np.ndarray, threshold: float = 0.5) -> tuple[float, str]:
    """Return (probability, "normal"|"abnormal") at the given threshold."""
    prob = predict_proba(model, x)
    return prob, ("abnormal" if prob >= threshold else "normal")


def predict_cli(argv: list[str] | None = None) -> tuple[float, str]:
    parser = argparse.ArgumentParser(
        prog="chexvision_mini predict", description="Predict normal/abnormal for one chest X-ray."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved operating threshold.")
    args = parser.parse_args(argv)

    model, mean, std, threshold = load_checkpoint(args.checkpoint)
    if args.threshold is not None:
        threshold = args.threshold
    meta = json.loads((Path(args.checkpoint) / "metrics.json").read_text(encoding="utf-8"))
    x = preprocess_image(args.image, meta["config"]["image_size"], mean, std)
    prob, label = predict_label(model, x, threshold)
    print(f"P(abnormal)={prob:.4f}  ->  {label}  (threshold {threshold:.3f})")
    return prob, label

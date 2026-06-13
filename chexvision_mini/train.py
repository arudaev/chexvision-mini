"""Training pipeline for the from-scratch NumPy net.

Runs the whole process — data prep, augmentation, training, and checkpointing —
on a laptop CPU. The run ``mode`` selects how much data to stream and how many
epochs to run. Artifacts (``model.npz``, ``history.json``, ``loss_curve.png``)
are written to ``--output-dir``; uploading them anywhere is the caller's job
(the parent CheXVision repo handles cloud dispatch + HF upload).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .augment import augment_batch
from .data import load_xray_subset
from .layers import Linear, ReLU
from .losses import BCEWithLogitsLoss, sigmoid
from .metrics import accuracy, roc_auc
from .network import Sequential
from .optim import SGD, Adam
from .regularizers import Dropout

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float, seed: int) -> Sequential:
    """Assemble ``[Linear -> ReLU -> Dropout] * k -> Linear(1)`` producing a logit."""
    layers: list[Any] = []
    prev = input_dim
    for i, width in enumerate(hidden_dims):
        layers.append(Linear(prev, width, seed=seed + i))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(dropout, seed=seed + 100 + i))
        prev = width
    layers.append(Linear(prev, 1, seed=seed + 999))
    return Sequential(*layers)


def _iterate_minibatches(
    x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
):
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def evaluate(model: Sequential, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = sigmoid(model.forward(x, training=False))
    return {"acc": accuracy(probs, y), "auc": roc_auc(probs, y)}


def train(config: dict[str, Any], mode: str, output_dir: Path) -> dict[str, Any]:
    """Run training end to end and persist artifacts to ``output_dir``."""
    set_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    image_size = config["image_size"]
    mode_cfg = config["data"]["modes"][mode]
    tcfg = config["training"]
    epochs = mode_cfg["epochs"]

    print(f"[train] mode={mode} image_size={image_size} epochs={epochs}")
    x_train, y_train = load_xray_subset(
        mode, mode_cfg["n_train"], image_size=image_size, split="train",
        seed=config["seed"], dataset_repo=config["data"]["dataset_repo"],
    )
    x_val, y_val = load_xray_subset(
        mode, mode_cfg["n_val"], image_size=image_size,
        split="validation" if mode != "synthetic" else "train",
        seed=config["seed"] + 1, dataset_repo=config["data"]["dataset_repo"],
    )
    print(f"[train] train={x_train.shape} val={x_val.shape} positives={float(y_train.mean()):.3f}")

    model = build_mlp(x_train.shape[1], config["model"]["hidden_dims"], config["model"]["dropout"], config["seed"])
    loss_fn = BCEWithLogitsLoss(label_smoothing=tcfg["label_smoothing"])
    opt_name = tcfg["optimizer"].lower()
    if opt_name == "adam":
        optimizer: SGD | Adam = Adam(model, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    elif opt_name == "sgd":
        optimizer = SGD(model, lr=tcfg["lr"], momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for xb, yb in _iterate_minibatches(x_train, y_train, tcfg["batch_size"], rng):
            if tcfg["augment"]:
                xb = augment_batch(xb, image_size, rng)
            logits = model.forward(xb, training=True)
            epoch_losses.append(loss_fn.forward(logits, yb))
            model.backward(loss_fn.backward())
            optimizer.step()

        train_loss = float(np.mean(epoch_losses))
        val_logits = model.forward(x_val, training=False)
        val_loss = loss_fn.forward(val_logits, y_val)
        val_metrics = evaluate(model, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])
        print(
            f"[train] epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_metrics['acc']:.3f}  val_auc={val_metrics['auc']:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "model.npz", **model.state_dict())
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_loss_curve(history, output_dir / "loss_curve.png")
    print(f"[train] artifacts written to {output_dir}")
    return history


def _save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train] matplotlib unavailable — skipping loss curve.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("From-scratch NumPy net — training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train the from-scratch NumPy net.")
    parser.add_argument("--mode", choices=["synthetic", "local", "kaggle"], default="synthetic")
    parser.add_argument("--epochs", type=int, default=None, help="Override the per-mode epoch count.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    config = load_config()
    if args.epochs is not None:
        config["data"]["modes"][args.mode]["epochs"] = args.epochs

    return train(config, args.mode, args.output_dir)


if __name__ == "__main__":
    main()

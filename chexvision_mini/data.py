"""Two-mode data loading for the from-scratch net.

The full CheXVision dataset (``arudaev/chest-xray-14-320``) is ~7.97 GB of
320x320 parquet shards — far too large to pull onto a laptop. This module never
downloads the whole thing:

* ``local``     — HF *streaming* pulls only a few hundred images on the fly,
                  downsamples them to a small grayscale square, and flattens
                  them. Runs in seconds on CPU with no full download.
* ``kaggle``    — same streaming path, larger subset + more epochs, run inside
                  the Kaggle kernel where bandwidth is free.
* ``synthetic`` — offline, deterministic, network-free. A learnable random
                  problem used by CI and quick smoke tests.

The task is binary **normal vs abnormal**: label 0 when the pipe-delimited
``labels`` string is ``"No Finding"``, else 1.
"""

from __future__ import annotations

import numpy as np

# Pinned to match the parent project's reproducible dataset revision.
DATASET_REPO = "arudaev/chest-xray-14-320"
DATASET_REVISION = "44443e6ee968b3c6094b63f14a27698c40b50680"


def binary_label(labels_str: object) -> float:
    """0.0 for a normal study ("No Finding"/empty), 1.0 for any abnormal finding."""
    if not isinstance(labels_str, str):
        return 0.0
    return 0.0 if labels_str.strip() in ("", "No Finding") else 1.0


# Fixed "concept" seed so the synthetic decision boundary is identical across
# the train and validation splits (only the samples differ) — otherwise the
# model could never generalise from one split to the other.
_SYNTH_CONCEPT_SEED = 12345


def _synthetic_subset(n_samples: int, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A learnable, offline classification problem (no network needed).

    Features are uniform in [0, 1] (image-like). Labels come from a logistic
    model whose weight "image" is **horizontally symmetric**, so the horizontal
    flip in :func:`chexvision_mini.augment.augment_batch` is genuinely label-
    preserving. The logit is standardised to a fixed spread so the difficulty is
    independent of ``image_size``.
    """
    dim = image_size * image_size
    concept_rng = np.random.default_rng(_SYNTH_CONCEPT_SEED)
    w_img = concept_rng.standard_normal((image_size, image_size))
    w_sym = (w_img + w_img[:, ::-1]).reshape(-1, 1)  # symmetric -> flip-invariant

    sample_rng = np.random.default_rng(seed)
    x = sample_rng.random((n_samples, dim))
    logit = (x - 0.5) @ w_sym
    logit = 2.5 * logit / (logit.std() + 1e-9)
    probs = 1.0 / (1.0 + np.exp(-logit))
    y = (sample_rng.random((n_samples, 1)) < probs).astype(np.float64)
    return x.astype(np.float64), y


def _streamed_subset(
    n_samples: int,
    image_size: int,
    split: str,
    dataset_repo: str,
    revision: str,
    seed: int = 0,
    shuffle_buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream ``n_samples`` examples from the HF dataset and vectorise them.

    When ``shuffle_buffer > 0`` the stream is shuffled (rolling buffer) **before**
    ``take``, so the subset is a representative random sample of the split rather
    than its first N rows. The parquet shards are label-ordered, so an unshuffled
    take skews the class balance (this caused the ~64% positive rate seen in the
    unshuffled test subset vs ~42% in train/val).
    """
    from datasets import load_dataset

    from .hub import load_hf_token

    load_hf_token()  # populate HF_TOKEN env so a gated/private dataset still loads
    stream = load_dataset(dataset_repo, split=split, streaming=True, revision=revision)
    if shuffle_buffer > 0:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    images: list[np.ndarray] = []
    labels: list[float] = []
    for sample in stream.take(n_samples):
        gray = sample["image"].convert("L").resize((image_size, image_size))
        images.append(np.asarray(gray, dtype=np.float64).reshape(-1) / 255.0)
        labels.append(binary_label(sample.get("labels")))

    x = np.stack(images)
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)
    return x, y


def load_xray_subset(
    mode: str,
    n_samples: int,
    *,
    image_size: int = 32,
    split: str = "train",
    seed: int = 0,
    dataset_repo: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
    shuffle_buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` with ``x`` of shape ``(n, image_size**2)`` in [0, 1].

    ``mode`` is ``"synthetic"`` (offline), ``"local"`` or ``"kaggle"`` (both
    stream from HF; they differ only in how many samples/epochs the caller asks
    for). ``shuffle_buffer`` (streamed modes) randomises the stream before ``take``.
    """
    if mode == "synthetic":
        return _synthetic_subset(n_samples, image_size, seed)
    if mode in ("local", "kaggle"):
        return _streamed_subset(
            n_samples, image_size, split, dataset_repo, revision,
            seed=seed, shuffle_buffer=shuffle_buffer,
        )
    raise ValueError(f"Unknown data mode: {mode!r} (expected synthetic|local|kaggle)")

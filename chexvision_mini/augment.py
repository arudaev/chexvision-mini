"""From-scratch data augmentation for flattened grayscale images.

All transforms operate on a batch of flattened images ``(n, image_size**2)``,
reshaping to ``(n, H, W)`` internally. Kept deliberately simple (flip / noise /
brightness) because they are implemented by hand — naive rotation needs
interpolation, which is left out to keep the maths transparent.
"""

from __future__ import annotations

import numpy as np


def augment_batch(
    x: np.ndarray,
    image_size: int,
    rng: np.random.Generator,
    *,
    flip_prob: float = 0.5,
    noise_std: float = 0.03,
    brightness: float = 0.1,
) -> np.ndarray:
    """Return an augmented copy of ``x`` (chest X-rays are left-right symmetric,
    so horizontal flips are label-preserving; mild noise/brightness improve
    robustness to acquisition differences)."""
    n = x.shape[0]
    imgs = x.reshape(n, image_size, image_size).copy()

    # Per-sample horizontal flip.
    flip_mask = rng.random(n) < flip_prob
    imgs[flip_mask] = imgs[flip_mask, :, ::-1]

    # Additive Gaussian noise.
    if noise_std > 0:
        imgs += rng.normal(0.0, noise_std, size=imgs.shape)

    # Per-sample brightness shift.
    if brightness > 0:
        imgs += rng.uniform(-brightness, brightness, size=(n, 1, 1))

    return np.clip(imgs, 0.0, 1.0).reshape(n, -1)

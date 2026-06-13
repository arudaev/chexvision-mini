"""chexvision-mini — a true from-scratch neural network in pure NumPy.

Companion to the CheXVision project. Where the parent repo uses PyTorch autograd
and a pretrained DenseNet, this package implements every forward and backward
pass by hand to demonstrate the underlying mathematics (the goal of the
"Deep Learning and Big Data" course). It mounts into CheXVision as the
``src/numpy_net`` git submodule and also runs as a standalone project.
"""

from __future__ import annotations

from .gradcheck import gradient_check
from .layers import Linear, ReLU, Sigmoid
from .losses import BCEWithLogitsLoss, sigmoid
from .network import Sequential
from .optim import SGD, Adam
from .regularizers import Dropout, l2_penalty

__all__ = [
    "SGD",
    "Adam",
    "BCEWithLogitsLoss",
    "Dropout",
    "Linear",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "gradient_check",
    "l2_penalty",
    "sigmoid",
]

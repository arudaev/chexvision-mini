"""Entry point.

- ``python -m chexvision_mini [--mode synthetic|local|kaggle] ...`` — train.
- ``python -m chexvision_mini predict --checkpoint <dir> --image <path>`` — infer.
"""

from __future__ import annotations

import sys

from .train import main as train_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        from .inference import predict_cli

        predict_cli(sys.argv[2:])
    else:
        train_main()

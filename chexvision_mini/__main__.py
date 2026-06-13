"""Entry point: ``python -m chexvision_mini [--mode synthetic|local|kaggle] ...``."""

from __future__ import annotations

from .train import main

if __name__ == "__main__":
    main()

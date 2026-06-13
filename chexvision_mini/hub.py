"""Minimal Hugging Face Hub helpers (standalone copy).

``chexvision-mini`` is a standalone repo and must not import the parent
project's ``src.utils.hub``. This re-implements just the two things the mini
needs: resolving an HF token (env / .env / Kaggle dataset mount / Kaggle
secret, in that order — matching the parent) and uploading a results folder.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_MODEL_REPO = "arudaev/chexvision-mini"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def load_hf_token() -> str | None:
    """Resolve an HF token and mirror it into the canonical env vars.

    Resolution order (first non-empty wins): environment variables, then the
    ``hlexnc/chexvision-secrets`` Kaggle dataset mount, then the interactive
    Kaggle ``UserSecretsClient``.
    """
    _load_dotenv_if_available()

    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            return token

    for token_file in (
        Path("/kaggle/input/datasets/hlexnc/chexvision-secrets/hf_token.txt"),
        Path("/kaggle/input/chexvision-secrets/hf_token.txt"),
    ):
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                return token

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            return token
    except Exception:
        pass

    return None


def upload_results(local_dir: Path | str, repo_id: str = HF_MODEL_REPO) -> str | None:
    """Upload everything in ``local_dir`` to an HF model repo (created if needed).

    Returns the repo URL on success, or ``None`` when no token is available
    (e.g. a local run with uploads disabled).
    """
    token = load_hf_token()
    if not token:
        print("[hub] No HF token found — skipping upload.")
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir))
    url = f"https://huggingface.co/{repo_id}"
    print(f"[hub] Uploaded {local_dir} -> {url}")
    return url

"""Load `.env` into os.environ without printing secrets."""

from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT


def load_env() -> bool:
    path = ROOT / ".env"
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    # Common alias used by huggingface_hub
    if not os.getenv("HUGGINGFACE_HUB_TOKEN") and os.getenv("HF_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    return bool(os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"))

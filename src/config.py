"""Pemuatan konfigurasi dan utilitas umum (seed, path)."""

from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Baca ``configs/config.yaml`` dan gabungkan dengan ``best_params.yaml`` bila ada.

    Hasil tuning (``make tune``) menimpa parameter default model sehingga
    ``make train`` otomatis memakai hyperparameter terbaik.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    best_path = ROOT / cfg["paths"]["best_params"]
    if best_path.exists():
        with open(best_path, encoding="utf-8") as f:
            best = yaml.safe_load(f) or {}
        for model_name, params in best.items():
            cfg["models"].setdefault(model_name, {}).update(params or {})
    return cfg


def path(key: str) -> Path:
    """Path absolut dari entri ``paths`` di config."""
    return ROOT / load_config()["paths"][key]


def set_seed(seed: int | None = None) -> int:
    """Set seed untuk random, numpy, dan torch (bila terpasang)."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass
    return seed


def get_logger(name: str) -> logging.Logger:
    """Logger dengan format seragam."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

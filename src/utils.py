"""Zajedničke pomoćne funkcije i konstante projekta."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRICE_CACHE_DIR = RAW_DATA_DIR / "price_cache"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Prozor projekta (odjeljak 4.3 specifikacije).
PROJECT_START = "2005-01"
PROJECT_END = "2025-12"

# Konfiguracija kliznog prozora (odjeljak 4.4).
TRAIN_LOOKBACK_MONTHS = 60
TEST_HORIZON_MONTHS = 12
REFIT_STEP_MONTHS = 12
MIN_TRAINING_MONTHS = 60

# Ograničenja portfelja (odjeljak 6).
W_MAX = 0.02
GROUP_CAP = 0.15

# Konfiguracija bootstrapa (odjeljak 7).
N_BOOTSTRAP_RETURNS = 1000
N_BOOTSTRAP_CLUSTERS = 500
BLOCK_SIZE_MONTHS = 12

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Postavi sjeme generatorima slučajnih brojeva Pythona i NumPyja."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Stvori sve izlazne direktorije projekta koji možda još ne postoje."""
    for directory in (
        RAW_DATA_DIR,
        PRICE_CACHE_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

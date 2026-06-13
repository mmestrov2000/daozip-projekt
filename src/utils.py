"""Zajedničke pomoćne funkcije i konstante projekta."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.config import CONFIG


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

# Sve konstante vežu se na config.yaml (jedino mjesto za parametre).

# Prozor projekta (odjeljak 4.3 specifikacije).
PROJECT_START = CONFIG["project_start"]
PROJECT_END = CONFIG["project_end"]

# Konfiguracija kliznog prozora (odjeljak 4.4).
TRAIN_LOOKBACK_MONTHS = CONFIG["train_lookback_months"]
TEST_HORIZON_MONTHS = CONFIG["test_horizon_months"]
REFIT_STEP_MONTHS = CONFIG["refit_step_months"]
MIN_TRAINING_MONTHS = CONFIG["min_training_months"]

# Ograničenja portfelja (odjeljak 6).
W_MAX = CONFIG["w_max"]
GROUP_CAP = CONFIG["group_cap"]

# Sanitacija mjesečnih prinosa (granice fizički mogućeg; izvan -> NaN).
MONTHLY_RETURN_CLIP_LOW = CONFIG["monthly_return_clip_low"]
MONTHLY_RETURN_CLIP_HIGH = CONFIG["monthly_return_clip_high"]

# Overlay faktorske neutralnosti: ε-mreža.
EPSILON_GRID = CONFIG["epsilon_grid"]

# Transakcijski troškovi (baznih bodova po jednostranom obrtaju).
TC_BPS = CONFIG["tc_bps"]

# Konfiguracija bootstrapa (odjeljak 7).
N_BOOTSTRAP_RETURNS = CONFIG["n_bootstrap_returns"]
N_BOOTSTRAP_CLUSTERS = CONFIG["n_bootstrap_clusters"]
BLOCK_SIZE_MONTHS = CONFIG["block_size_months"]

# Model Confidence Set.
MCS_ALPHA = CONFIG["mcs_alpha"]
MCS_LOSS = CONFIG["mcs_loss"]

RANDOM_SEED = CONFIG["random_seed"]


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

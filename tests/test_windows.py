"""Validacija pomičnih prozora za razdoblje 2000–2025 (F0.10)."""

import pandas as pd

from src.backtest import generate_rolling_windows


def test_windows_2000_2025_count_and_bounds():
    """60/12/12 na 2000-01..2025-12 daje 21 prozor; prvi label 2004-12,
    zadnji test završava 2025-12."""
    windows = generate_rolling_windows("2000-01", "2025-12", 60, 12, 12)

    assert len(windows) == 21
    assert windows[0].label == "2004-12"
    assert windows[0].test_start == pd.Period("2005-01", freq="M").to_timestamp(
        how="end"
    ).normalize()
    assert windows[-1].test_end == pd.Period("2025-12", freq="M").to_timestamp(
        how="end"
    ).normalize()

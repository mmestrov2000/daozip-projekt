"""Spot-provjere point-in-time članstva S&P 500 (F0.5)."""

import pandas as pd
import pytest

from src.backtest import generate_rolling_windows
from src.data import (
    SP500_MEMBERSHIP_PATH,
    fetch_sp500_membership,
    membership_on,
    universe_for_window,
)


@pytest.fixture(scope="module")
def membership() -> pd.DataFrame:
    """Tablica članstva s diska; preuzmi jednom ako ne postoji."""
    if SP500_MEMBERSHIP_PATH.exists():
        return pd.read_csv(SP500_MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"])
    return fetch_sp500_membership()


def test_membership_schema(membership):
    """Duga tablica intervala ima propisane stupce."""
    assert list(membership.columns) == ["ticker", "name", "start_date", "end_date", "source"]
    assert len(membership) > 0


def test_tsla_enters_2020_12(membership):
    """TSLA ulazi u indeks u prosincu 2020."""
    assert "TSLA" not in membership_on("2020-11-30", membership)
    assert "TSLA" in membership_on("2020-12-31", membership)


def test_gm_exits_2009_06(membership):
    """GM (stari, bankrot) izlazi iz indeksa u lipnju 2009."""
    assert "GM" in membership_on("2009-05-31", membership)
    assert "GM" not in membership_on("2009-06-30", membership)


def test_leh_exits_2008_09(membership):
    """LEH (Lehman Brothers) izlazi iz indeksa u rujnu 2008."""
    assert "LEH" in membership_on("2008-08-31", membership)
    assert "LEH" not in membership_on("2008-09-30", membership)


def test_aig_remains_member_after_2008(membership):
    """AIG ostaje član indeksa i nakon krize 2008."""
    assert "AIG" in membership_on("2009-12-31", membership)
    assert "AIG" in membership_on("2015-12-31", membership)


def test_universe_size_per_window(membership):
    """Point-in-time univerzum (prije filtara kvalitete) je u [490, 510] za svaki od 21 prozora."""
    windows = generate_rolling_windows("2000-01", "2025-12", 60, 12, 12)
    assert len(windows) == 21
    for window in windows:
        universe = universe_for_window(window, membership)
        assert 490 <= len(universe) <= 510, (
            f"prozor {window.label}: {len(universe)} članova izvan [490, 510]"
        )

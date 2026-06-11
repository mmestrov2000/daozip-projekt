"""Testovi sloja transakcijskih troškova (F0.9)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import apply_costs, sharpe_ratio, turnover_per_window


def _weights_panel() -> pd.DataFrame:
    """Dva prozora, tri imovine, poznat obrtaj."""
    rows = [
        # prozor 2020-12 (testna godina 2021): puna izgradnja
        {"train_window": "2020-12", "portfolio": "p", "ticker": "A", "weight": 0.5},
        {"train_window": "2020-12", "portfolio": "p", "ticker": "B", "weight": 0.3},
        {"train_window": "2020-12", "portfolio": "p", "ticker": "C", "weight": 0.2},
        # prozor 2021-12 (testna godina 2022)
        {"train_window": "2021-12", "portfolio": "p", "ticker": "A", "weight": 0.2},
        {"train_window": "2021-12", "portfolio": "p", "ticker": "B", "weight": 0.5},
        {"train_window": "2021-12", "portfolio": "p", "ticker": "C", "weight": 0.3},
    ]
    return pd.DataFrame(rows)


def test_turnover_first_window_is_full_build():
    """Prvi prozor svakog portfelja ima obrtaj 1.0."""
    turnover = turnover_per_window(_weights_panel())
    first = turnover.loc[turnover["train_window"] == "2020-12", "turnover"].iloc[0]
    assert first == 1.0


def test_turnover_known_value():
    """Jednostrani obrtaj drugog prozora = 0.5·(0.3+0.2+0.1) = 0.3."""
    turnover = turnover_per_window(_weights_panel())
    second = turnover.loc[turnover["train_window"] == "2021-12", "turnover"].iloc[0]
    assert second == pytest.approx(0.3)


def test_apply_costs_subtracts_at_first_test_month():
    """Trošak na 10 bps skida se s prvog mjeseca testne godine."""
    turnover = turnover_per_window(_weights_panel())
    index = pd.period_range("2021-01", "2022-12", freq="M").to_timestamp(how="end")
    gross = pd.DataFrame({"p": np.full(len(index), 0.01)}, index=index)

    net = apply_costs(gross, turnover, tc_bps=10)

    # prvi testni mjesec prozora 2020-12 = 2021-01, trošak = 1.0 * 0.001
    assert net["p"].iloc[0] == pytest.approx(0.01 - 0.001)
    # prvi testni mjesec prozora 2021-12 = 2022-01, trošak = 0.3 * 0.001
    pos_2022_01 = np.flatnonzero(
        pd.PeriodIndex(net.index, freq="M") == pd.Period("2022-01", freq="M")
    )[0]
    assert net["p"].iloc[pos_2022_01] == pytest.approx(0.01 - 0.3 * 0.001)
    # mjeseci bez refita ostaju bruto
    assert net["p"].iloc[1] == pytest.approx(0.01)


def test_sharpe_ratio_annualizes():
    """Sharpe = mean/std · √12; rf skalar i Series daju isti rezultat kad je rf=0."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.015])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(12)
    assert sharpe_ratio(returns) == pytest.approx(expected)
    assert sharpe_ratio(returns, rf=pd.Series(0.0, index=returns.index)) == pytest.approx(
        expected
    )

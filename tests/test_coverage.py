"""Jedinični testovi za uniju univerzuma i izvještaj pokrivenosti (F0.7/F0.8)."""

import numpy as np
import pandas as pd

from src.data import (
    _membership_bounds,
    membership_coverage_report,
    union_universe,
)


def _toy_membership() -> pd.DataFrame:
    """Tri tickera: A trajni član, B s dva disjunktna intervala, C izvan razdoblja."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "B", "C"],
            "name": ["Alpha", "Beta", "Beta", "Gamma"],
            "start_date": pd.to_datetime(
                ["2000-01-31", "2000-06-30", "2010-01-31", "1990-01-31"]
            ),
            "end_date": pd.to_datetime(
                ["2005-12-31", "2003-12-31", pd.NaT, "1995-12-31"]
            ),
            "source": ["t"] * 4,
        }
    )


def test_union_universe_overlap_only():
    """Uniju čine samo tickeri čiji se interval preklapa s razdobljem projekta."""
    membership = _toy_membership()
    union = union_universe("2000-01", "2025-12", membership)
    assert union == ["A", "B"]  # C je član samo 1990.–1995.


def test_membership_bounds_open_interval_is_nat():
    """member_to je NaT kad je bilo koji interval još otvoren; inače najkasniji izlazak."""
    bounds = _membership_bounds(_toy_membership())
    assert bounds.loc["A", "member_from"] == pd.Timestamp("2000-01-31")
    assert bounds.loc["A", "member_to"] == pd.Timestamp("2005-12-31")
    assert bounds.loc["B", "member_from"] == pd.Timestamp("2000-06-30")
    assert pd.isna(bounds.loc["B", "member_to"])  # otvoreni interval od 2010.


def test_coverage_report_schema_and_counts():
    """Tablica pokrivenosti ima propisane stupce i točne brojeve na sintetičkom panelu."""
    # Dvije imovine, jedan prozor; A ima 60 valjanih mjeseci, B samo 3.
    index = pd.date_range("2000-01-31", periods=60, freq="ME")
    returns = pd.DataFrame(
        {
            "A": np.full(60, 0.01),
            "B": [0.01, 0.01, 0.01] + [np.nan] * 57,
        },
        index=index,
    )

    class _Window:
        train_start = index[0]
        train_end = index[-1]
        label = "2004-12"

    membership = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Alpha", "Beta"],
            "start_date": pd.to_datetime(["2000-01-31", "2000-01-31"]),
            "end_date": pd.to_datetime([pd.NaT, pd.NaT]),
            "source": ["t", "t"],
        }
    )

    coverage = membership_coverage_report(
        returns,
        windows=[_Window()],
        membership=membership,
        min_training_months=60,
        output_path=None,
    )
    assert list(coverage.columns) == [
        "train_window",
        "n_members",
        "n_with_prices",
        "n_with_60m",
        "pct_prices",
        "pct_60m",
    ]
    row = coverage.iloc[0]
    assert row["n_members"] == 2
    assert row["n_with_prices"] == 2  # oba imaju bar jedan valjani mjesec
    assert row["n_with_60m"] == 1  # samo A prolazi filtar ≥60 mjeseci
    assert row["pct_60m"] == 50.0

"""Testovi za Model Confidence Set (F1.7, Hansen–Lunde–Nason 2011)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import model_confidence_set


def _month_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2005-01-31", periods=n, freq="ME")


def test_dominated_variant_drops_out_of_mcs():
    """Tri niza: dva jednaka (ista distribucija) + jedan dominiran s 2× varijancom.

    Pod gubitkom ``(r − r̄)²`` dominirani niz ima sustavno veći gubitak pa ga
    iterativna eliminacija po max-t izbacuje iz MCS-a na α = 0.10, a dva jednaka
    ostaju.
    """
    rng = np.random.default_rng(0)
    n = 240
    idx = _month_index(n)
    sigma = 0.04
    panel = pd.DataFrame(
        {
            "equal_a": rng.normal(0.0, sigma, n),
            "equal_b": rng.normal(0.0, sigma, n),
            "dominated": rng.normal(0.0, sigma * np.sqrt(2.0), n),
        },
        index=idx,
    )

    result = model_confidence_set(
        panel, alpha=0.10, n_bootstraps=1000, block_size=12, seed=42
    )

    assert set(result["portfolio"]) == {"equal_a", "equal_b", "dominated"}
    in_mcs = result.set_index("portfolio")["in_mcs"]
    assert bool(in_mcs["equal_a"]) is True
    assert bool(in_mcs["equal_b"]) is True
    assert bool(in_mcs["dominated"]) is False

    # Dominirani niz mora imati najgori rang (najveći prosječni gubitak).
    rank = result.set_index("portfolio")["rank"]
    assert rank["dominated"] == 3

    # P-vrijednost izbačenog modela < α; preostalih ≥ α.
    pval = result.set_index("portfolio")["p_value"]
    assert pval["dominated"] < 0.10
    assert pval["equal_a"] >= 0.10 and pval["equal_b"] >= 0.10


def test_unbalanced_panel_excludes_undercovered_variant():
    """Varijanta s > 20 % nedostajućih mjeseci isključuje se (K3), ostali ulaze.

    Tri gusto pokrivene varijante (240 mj.) + jedna rijetka (180 mj., 25 %
    nedostaje). Rijetka se označava ``excluded`` i ne ulazi u MCS, ali presjek i
    brojevi ispuštenih mjeseci moraju biti točni; MCS se računa na presjeku
    gusto pokrivenih varijanti (240 mj.).
    """
    rng = np.random.default_rng(7)
    n = 240
    idx = _month_index(n)
    sigma = 0.04
    sparse = rng.normal(0.0, sigma, n)
    sparse[:60] = np.nan  # prvih 60 mjeseci nedostaje → 180 valjanih (25 % manjka)
    panel = pd.DataFrame(
        {
            "dense_a": rng.normal(0.0, sigma, n),
            "dense_b": rng.normal(0.0, sigma, n),
            "dense_c": rng.normal(0.0, sigma, n),
            "sparse": sparse,
        },
        index=idx,
    )

    result = model_confidence_set(
        panel, alpha=0.10, n_bootstraps=500, block_size=12, seed=42
    )
    by_pf = result.set_index("portfolio")

    # Rijetka varijanta isključena iz MCS-a, ostaje zabilježena.
    assert by_pf.loc["sparse", "status"] == "excluded"
    assert bool(by_pf.loc["sparse", "in_mcs"]) is False
    assert np.isnan(by_pf.loc["sparse", "p_value"])
    assert by_pf.loc["sparse", "n_months_used"] == 180
    assert by_pf.loc["sparse", "n_months_dropped"] == 60

    # Gusto pokrivene varijante ulaze u MCS na presjeku od 240 mjeseci.
    for col in ["dense_a", "dense_b", "dense_c"]:
        assert by_pf.loc[col, "status"] == "included"
        assert by_pf.loc[col, "n_months_used"] == 240
        assert by_pf.loc[col, "n_months_dropped"] == 0


def test_output_has_required_columns():
    rng = np.random.default_rng(1)
    n = 120
    panel = pd.DataFrame(
        {"a": rng.normal(0, 0.03, n), "b": rng.normal(0, 0.03, n)},
        index=_month_index(n),
    )
    result = model_confidence_set(
        panel, alpha=0.10, n_bootstraps=300, block_size=12, seed=42
    )
    for column in [
        "portfolio", "in_mcs", "p_value", "rank",
        "n_months_used", "n_months_dropped",
    ]:
        assert column in result.columns
    # MCS nikad nije prazan: barem jedna varijanta ostaje.
    assert bool(result["in_mcs"].any())

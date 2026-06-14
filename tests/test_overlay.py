"""Testovi projekcijskog QP overlaya (F4.1 project_factor_neutral)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    OVERLAY_BASE_PORTFOLIOS,
    generate_rolling_windows,
    run_overlay_sweep,
)
from src.portfolio import project_factor_neutral


TICKERS = ["A", "B", "C", "D"]


def _pd_covariance(seed: int = 7) -> pd.DataFrame:
    """Pozitivno definitna Σ iz sintetičkih prinosa (fiksno sjeme)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(scale=0.04, size=(150, len(TICKERS)))
    cov = np.cov(returns, rowvar=False)
    return pd.DataFrame(cov, index=TICKERS, columns=TICKERS)


def _correlated_covariance() -> pd.DataFrame:
    """Σ s izraženim izvandijagonalnim korelacijama (norme moraju projicirati različito)."""
    sigma = np.array(
        [
            [0.040, 0.030, 0.005, 0.000],
            [0.030, 0.050, 0.000, 0.004],
            [0.005, 0.000, 0.060, 0.035],
            [0.000, 0.004, 0.035, 0.070],
        ]
    )
    return pd.DataFrame(sigma, index=TICKERS, columns=TICKERS)


def _zero_sum_betas() -> pd.DataFrame:
    """Bete kod kojih jednake težine daju w'β = 0 → izvediv skup je neprazan."""
    return pd.DataFrame(
        {
            "SMB": [1.0, -1.0, 0.5, -0.5],
            "HML": [0.8, 0.2, -0.5, -0.5],
        },
        index=TICKERS,
    )


def _factor_exposure(weights: pd.Series, betas: pd.DataFrame) -> np.ndarray:
    aligned = betas.loc[weights.index]
    return weights.to_numpy() @ aligned.to_numpy()


# ---------------------------------------------------------------------------
# Kriterij 1 — w0 koji već zadovoljava ograničenja vraća se nepromijenjen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["sigma", "euclidean"])
def test_feasible_w0_returned_unchanged(norm):
    """Jednake težine zadovoljavaju sva ograničenja → projekcija vraća w0 (Δw ≤ 1e-6)."""
    sigma = _pd_covariance()
    betas = _zero_sum_betas()
    w0 = pd.Series(np.repeat(0.25, 4), index=TICKERS)

    result = project_factor_neutral(
        w0, sigma, betas, w_max=1.0, epsilon=0.0, norm=norm
    )

    assert isinstance(result, pd.Series)
    assert np.max(np.abs(result.loc[w0.index].to_numpy() - w0.to_numpy())) <= 1e-6


# ---------------------------------------------------------------------------
# Kriterij 2 — izlaz uvijek zadovoljava sve uvjete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["sigma", "euclidean"])
@pytest.mark.parametrize("epsilon", [0.0, 0.05])
def test_output_satisfies_all_constraints(norm, epsilon):
    """Za neizvediv w0 izlaz je dugi, potpuno investiran, ≤ w_max i |w'β| ≤ epsilon."""
    sigma = _correlated_covariance()
    betas = _zero_sum_betas()
    w0 = pd.Series([0.5, 0.2, 0.2, 0.1], index=TICKERS)
    w_max = 0.6

    result = project_factor_neutral(
        w0, sigma, betas, w_max=w_max, epsilon=epsilon, norm=norm
    )

    assert isinstance(result, pd.Series)
    assert result.sum() == pytest.approx(1.0, abs=1e-8)
    assert (result >= -1e-9).all()
    assert (result <= w_max + 1e-6).all()

    exposure = _factor_exposure(result, betas)
    assert np.all(np.abs(exposure) <= epsilon + 1e-6)


# ---------------------------------------------------------------------------
# Kriterij 3 — Σ-norma i euklidska norma daju različita rješenja
# ---------------------------------------------------------------------------


def test_sigma_and_euclidean_norms_differ():
    """Uz izvandijagonalne korelacije projekcija u Σ-metrici ≠ euklidska projekcija."""
    sigma = _correlated_covariance()
    betas = _zero_sum_betas()
    w0 = pd.Series([0.5, 0.2, 0.2, 0.1], index=TICKERS)

    w_sigma = project_factor_neutral(
        w0, sigma, betas, w_max=1.0, epsilon=0.0, norm="sigma"
    )
    w_eucl = project_factor_neutral(
        w0, sigma, betas, w_max=1.0, epsilon=0.0, norm="euclidean"
    )

    assert np.max(np.abs(w_sigma.to_numpy() - w_eucl.to_numpy())) > 1e-3


# ---------------------------------------------------------------------------
# Pomoćni: ndarray ulaz (bez oznaka) i validacija argumenata
# ---------------------------------------------------------------------------


def test_ndarray_input_returns_array():
    """Bez oznaka (ndarray ulazi) rezultat je ndarray koji zadovoljava uvjete."""
    sigma = _correlated_covariance().to_numpy()
    betas = _zero_sum_betas().reset_index(drop=True)
    w0 = np.array([0.5, 0.2, 0.2, 0.1])

    result = project_factor_neutral(w0, sigma, betas, w_max=0.6, epsilon=0.0)

    assert isinstance(result, np.ndarray)
    assert result.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.abs(result @ betas.to_numpy()) <= 1e-6)


def test_invalid_norm_raises():
    sigma = _pd_covariance()
    betas = _zero_sum_betas()
    w0 = pd.Series(np.repeat(0.25, 4), index=TICKERS)
    with pytest.raises(ValueError, match="norm"):
        project_factor_neutral(w0, sigma, betas, w_max=1.0, epsilon=0.0, norm="l1")


def test_negative_epsilon_raises():
    sigma = _pd_covariance()
    betas = _zero_sum_betas()
    w0 = pd.Series(np.repeat(0.25, 4), index=TICKERS)
    with pytest.raises(ValueError, match="epsilon"):
        project_factor_neutral(w0, sigma, betas, w_max=1.0, epsilon=-0.01)


# ---------------------------------------------------------------------------
# F4.2/F4.3 — smoke test pokretača run_overlay_sweep (struktura izlaza)
# (izvan popisa datoteka taska, podupire kriterij — presedan F1.5)
# ---------------------------------------------------------------------------


def _synthetic_overlay_inputs():
    """Mali sintetički ulaz: 8 dionica, 2 prozora, bete sa sumom nula po stupcu.

    Bete sa sumom nula znače da jednake težine (bazni w0) zadovoljavaju ε=0 →
    sve overlay varijante izvedive bez obzira na ε.
    """
    tickers = [f"T{i}" for i in range(8)]
    index = (
        pd.period_range("2018-01", "2019-12", freq="M")
        .to_timestamp(how="end")
        .normalize()
    )
    rng = np.random.default_rng(0)
    monthly_returns = pd.DataFrame(
        rng.normal(scale=0.04, size=(len(index), len(tickers))),
        index=index,
        columns=tickers,
    )

    windows = generate_rolling_windows("2018-01", "2019-12", 12, 6, 6)

    beta_values = rng.normal(size=(len(tickers), 4))
    beta_values = beta_values - beta_values.mean(axis=0)  # svaki stupac → suma 0
    style_cols = ["beta_smb", "beta_hml", "beta_rmw", "beta_cma"]

    weight_rows, beta_rows = [], []
    equal = 1.0 / len(tickers)
    for window in windows:
        for ticker_idx, ticker in enumerate(tickers):
            beta_rows.append(
                {
                    "train_window": window.label,
                    "ticker": ticker,
                    **dict(zip(style_cols, beta_values[ticker_idx])),
                }
            )
            for base_name, _allocator, _space in OVERLAY_BASE_PORTFOLIOS:
                weight_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": base_name,
                        "ticker": ticker,
                        "weight": equal,
                    }
                )

    base_weights_panel = pd.DataFrame(weight_rows)
    factor_exposures = pd.DataFrame(beta_rows)
    return base_weights_panel, monthly_returns, factor_exposures, windows


@pytest.mark.parametrize("norm", ["sigma", "euclidean"])
def test_run_overlay_sweep_structure(norm):
    """24 varijante (6 baza × 4 ε), čist status, valjane težine i bete ≤ ε."""
    base_weights_panel, monthly_returns, factor_exposures, windows = (
        _synthetic_overlay_inputs()
    )
    epsilons = (0.0, 0.05, 0.10, 0.15)

    result = run_overlay_sweep(
        base_weights_panel,
        monthly_returns,
        factor_exposures,
        windows,
        epsilons=epsilons,
        norm=norm,
        w_max=0.5,
        min_training_months=12,
    )

    panel = result["port_returns_panel"]
    status = result["portfolio_status"]
    weights = result["weights_panel"]

    # 6 baznih portfelja × 4 ε = 24 overlay varijante.
    expected_names = {
        f"{allocator}_{space}_ov_e{eps:g}"
        for _, allocator, space in OVERLAY_BASE_PORTFOLIOS
        for eps in epsilons
    }
    assert len(expected_names) == 24
    assert set(panel.columns) == expected_names
    assert set(status["portfolio"]) == expected_names

    # Bete su zero-sum → jednake težine izvedive za sve ε.
    assert (status["status"] == "ok").all()
    assert {"allocator", "space", "epsilon", "norm"}.issubset(status.columns)

    # Težine po (overlay portfelj, prozor): dugo, potpuno investirano, ≤ w_max.
    grouped = weights.groupby(["portfolio", "train_window"])["weight"]
    assert np.allclose(grouped.sum().to_numpy(), 1.0, atol=1e-6)
    assert (weights["weight"] >= -1e-9).all()
    assert (weights["weight"] <= 0.5 + 1e-6).all()


def test_run_overlay_sweep_rejects_bad_norm():
    base_weights_panel, monthly_returns, factor_exposures, windows = (
        _synthetic_overlay_inputs()
    )
    with pytest.raises(ValueError, match="norm"):
        run_overlay_sweep(
            base_weights_panel,
            monthly_returns,
            factor_exposures,
            windows,
            norm="l1",
        )

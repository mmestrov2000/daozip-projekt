"""Testovi za Deflated Sharpe Ratio (F4.4, Bailey–López de Prado 2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from src.evaluation import _deflated_sharpe_from_moments, deflated_sharpe_ratio


def test_dsr_matches_hand_computed_example():
    """Ručno izračunat primjer: ŜR=0.20, γ₃=−0.5, γ₄=4.0, T=120, N=40.

    Korak po korak (Bailey–López de Prado 2014):
      σ_ŜR = sqrt((1 − γ₃·ŜR + ((γ₄−1)/4)·ŜR²)/(T−1))
           = sqrt((1 + 0.5·0.20 + 0.75·0.04)/119)
           = sqrt(1.13/119)               = 0.0974463869
      Z⁻¹(1 − 1/40)      = Z⁻¹(0.975)      = 1.9599639845
      Z⁻¹(1 − 1/(40·e))  = Z⁻¹(0.99080300) = 2.3575904818
      SR₀ = σ_ŜR·[(1−γ)·1.9599639845 + γ·2.3575904818],  γ = 0.5772156649
          = 0.0974463869·2.1895062…        = 0.2133569374
      DSR = Φ((0.20 − 0.2133569374)/0.0974463869)
          = Φ(−0.1370093…)                 = 0.4454878903
    → zaokruženo na 4 decimale: 0.4455.
    """
    dsr = _deflated_sharpe_from_moments(
        sr=0.20, gamma3=-0.5, gamma4=4.0, n_obs=120, n_trials=40
    )
    assert round(dsr, 4) == 0.4455


def test_dsr_from_moments_matches_closed_form():
    """Neovisna rekonstrukcija zatvorene formule mora se poklopiti (≥ 10 dec.)."""
    sr, g3, g4, T, N = 0.15, 0.3, 5.0, 200, 25
    sr_std = np.sqrt((1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2) / (T - 1.0))
    em = 0.5772156649015329
    sr0 = sr_std * (
        (1.0 - em) * norm.ppf(1.0 - 1.0 / N) + em * norm.ppf(1.0 - 1.0 / (N * np.e))
    )
    expected = float(norm.cdf((sr - sr0) / sr_std))
    got = _deflated_sharpe_from_moments(sr, g3, g4, T, N)
    assert abs(got - expected) < 1e-10


def test_more_trials_lowers_dsr():
    """Više isprobanih varijanti (veći N) → veći benchmark SR₀ → niži DSR."""
    common = dict(sr=0.20, gamma3=0.0, gamma4=3.0, n_obs=156)
    dsr_few = _deflated_sharpe_from_moments(**common, n_trials=5)
    dsr_many = _deflated_sharpe_from_moments(**common, n_trials=50)
    assert dsr_few > dsr_many


def test_single_trial_equals_probabilistic_sharpe():
    """N=1: nema deflacije (SR₀=0), DSR = PSR(0) = Φ(ŜR/σ_ŜR)."""
    sr, g3, g4, T = 0.18, -0.2, 4.0, 150
    sr_std = np.sqrt((1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2) / (T - 1.0))
    expected = float(norm.cdf(sr / sr_std))
    got = _deflated_sharpe_from_moments(sr, g3, g4, T, n_trials=1)
    assert abs(got - expected) < 1e-12


def test_public_wrapper_extracts_moments_from_series():
    """Javni omotač na nizu daje isti DSR kao izravan poziv s momentima niza."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2013-01-31", periods=156, freq="ME")
    returns = pd.Series(rng.normal(0.012, 0.04, 156), index=idx)
    n_trials = 37

    excess = returns.to_numpy()  # rf = 0
    sr = excess.mean() / excess.std(ddof=1)
    gamma3 = float(skew(excess, bias=True))
    gamma4 = float(kurtosis(excess, fisher=False, bias=True))
    expected = _deflated_sharpe_from_moments(sr, gamma3, gamma4, 156, n_trials)

    got = deflated_sharpe_ratio(returns, n_trials=n_trials, rf=0.0)
    assert abs(got - expected) < 1e-12
    assert 0.0 <= got <= 1.0


def test_rf_series_reduces_excess_and_dsr():
    """``rf`` kao Series poravnava se i smanjuje višak prinosa → niži DSR."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2013-01-31", periods=156, freq="ME")
    returns = pd.Series(rng.normal(0.012, 0.045, 156), index=idx)
    rf = pd.Series(np.full(156, 0.002), index=idx)

    dsr_zero = deflated_sharpe_ratio(returns, n_trials=37, rf=0.0)
    dsr_rf = deflated_sharpe_ratio(returns, n_trials=37, rf=rf)
    assert 0.0 <= dsr_rf <= 1.0
    assert dsr_rf < dsr_zero


def test_too_few_observations_returns_nan():
    """Manje od tri opservacije → ŜR/momenti nedefinirani → nan."""
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2013-01-31", periods=2, freq="ME"))
    assert np.isnan(deflated_sharpe_ratio(returns, n_trials=37))


def test_invalid_n_trials_raises():
    """``n_trials`` mora biti ≥ 1."""
    returns = pd.Series(
        np.random.default_rng(0).normal(0.01, 0.04, 50),
        index=pd.date_range("2013-01-31", periods=50, freq="ME"),
    )
    try:
        deflated_sharpe_ratio(returns, n_trials=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Očekivan ValueError za n_trials=0.")

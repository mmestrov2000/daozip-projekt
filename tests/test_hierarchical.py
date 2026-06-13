"""Testovi hijerarhijskih alokatora (F1.1 HRP, F1.1b stabla, F1.2 HERC, F1.3 NCO)."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from src.backtest import (
    HIERARCHICAL_PORTFOLIO_NAMES,
    generate_rolling_windows,
    run_hierarchical_walk_forward,
)
from src.hierarchical import (
    apply_w_max,
    build_correlation_tree,
    correlation_distance,
    herc_weights,
    hrp_weights,
    nco_weights,
    quasi_diagonal_order,
)
from src.portfolio import min_variance


def _random_covariance(n_assets: int, seed: int = 42) -> np.ndarray:
    """Pozitivno definitna Σ iz sintetičkih prinosa (fiksno sjeme)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(scale=0.05, size=(120, n_assets))
    return np.cov(returns, rowvar=False)


def _block_diagonal_sigma() -> pd.DataFrame:
    """Dva 2×2 bloka; analitičko min-var rješenje je strogo pozitivno."""
    sigma = np.array(
        [
            [0.04, 0.01, 0.00, 0.00],
            [0.01, 0.09, 0.00, 0.00],
            [0.00, 0.00, 0.16, 0.02],
            [0.00, 0.00, 0.02, 0.25],
        ]
    )
    tickers = ["A", "B", "C", "D"]
    return pd.DataFrame(sigma, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# F1.1 — HRP
# ---------------------------------------------------------------------------


def test_quasi_diagonal_order_is_permutation():
    """Poredak listova je permutacija 0..n-1."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(8, 3))
    linkage_matrix = scipy_linkage(features, method="ward")
    order = quasi_diagonal_order(linkage_matrix)
    assert np.array_equal(np.sort(order), np.arange(8))


def test_hrp_weights_valid():
    """Težine ≥ 0 i Σw = 1; DataFrame Σ vraća Series s istim labelama."""
    sigma = _random_covariance(6)
    tickers = list("ABCDEF")
    sigma_frame = pd.DataFrame(sigma, index=tickers, columns=tickers)
    order = [3, 0, 5, 1, 4, 2]

    weights = hrp_weights(sigma_frame, order)

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == tickers
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)


def test_hrp_diagonal_sigma_equals_inverse_variance():
    """Na dijagonalnoj Σ HRP = inverzno-varijančne težine (analitički)."""
    variances = np.array([0.01, 0.02, 0.04, 0.08, 0.05])
    sigma = np.diag(variances)
    expected = (1.0 / variances) / (1.0 / variances).sum()

    weights = hrp_weights(sigma, order=[2, 4, 0, 3, 1])

    assert weights == pytest.approx(expected, abs=1e-12)


def test_apply_w_max_caps_and_renormalizes():
    """Cap grize: max ≤ w_max, Σw = 1, poznate konačne težine i udio na capu."""
    weights, share = apply_w_max(np.array([0.5, 0.2, 0.2, 0.1]), w_max=0.4)

    assert weights == pytest.approx([0.4, 0.24, 0.24, 0.12])
    assert weights.max() <= 0.4 + 1e-12
    assert weights.sum() == pytest.approx(1.0)
    assert share == pytest.approx(0.4)


def test_apply_w_max_share_zero_when_cap_not_binding():
    """Cap ne grize: težine netaknute, capped_weight_share = 0."""
    weights, share = apply_w_max(np.array([0.25, 0.25, 0.25, 0.25]), w_max=0.5)

    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert share == 0.0


def test_apply_w_max_infeasible_raises():
    """n·w_max < 1 → potpuno investiranje neizvedivo."""
    with pytest.raises(ValueError):
        apply_w_max(np.array([0.5, 0.3, 0.2]), w_max=0.3)


# ---------------------------------------------------------------------------
# F1.1b — korelacijska stabla (single + ward)
# ---------------------------------------------------------------------------


def _factor_model_correlation(n_assets: int = 10, seed: int = 0) -> np.ndarray:
    """Korelacijska matrica iz trofaktorskog modela (fiksno sjeme)."""
    rng = np.random.default_rng(seed)
    betas = rng.normal(size=(n_assets, 3))
    idio = rng.uniform(0.5, 2.0, size=n_assets)
    cov = betas @ betas.T + np.diag(idio)
    vol = np.sqrt(np.diag(cov))
    return cov / np.outer(vol, vol)


def test_correlation_distance_formula():
    """d = √(½(1−ρ)): ρ=0.5 → 0.5; dijagonala 0; DataFrame čuva labele."""
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    distance = correlation_distance(corr)

    assert isinstance(distance, pd.DataFrame)
    assert list(distance.index) == ["A", "B"]
    assert distance.loc["A", "B"] == pytest.approx(0.5)
    assert distance.loc["A", "A"] == 0.0


def test_build_correlation_tree_valid_linkage_matrices():
    """Obje veze vraćaju linkage oblika (n−1, 4); ward monotone udaljenosti."""
    corr = _factor_model_correlation()
    n = corr.shape[0]

    for method in ("single", "ward"):
        linkage_matrix = build_correlation_tree(corr, linkage=method)
        assert linkage_matrix.shape == (n - 1, 4)
    ward_matrix = build_correlation_tree(corr, linkage="ward")
    assert np.all(np.diff(ward_matrix[:, 2]) >= 0)


def test_single_and_ward_leaf_orders_differ():
    """Na konstruiranom primjeru single i ward daju različit poredak listova."""
    corr = _factor_model_correlation()
    order_single = quasi_diagonal_order(build_correlation_tree(corr, linkage="single"))
    order_ward = quasi_diagonal_order(build_correlation_tree(corr, linkage="ward"))
    assert not np.array_equal(order_single, order_ward)


def test_build_correlation_tree_from_returns_matches_corr():
    """Panel prinosa i njegova korelacijska matrica daju isto stablo."""
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(scale=0.05, size=(120, 6)), columns=list("ABCDEF"))
    from_returns = build_correlation_tree(returns, linkage="ward")
    from_corr = build_correlation_tree(returns.corr(), linkage="ward")
    assert from_returns == pytest.approx(from_corr)


def test_build_correlation_tree_rejects_unknown_linkage():
    """Dopuštene veze su samo single i ward."""
    corr = _factor_model_correlation()
    with pytest.raises(ValueError):
        build_correlation_tree(corr, linkage="complete")


# ---------------------------------------------------------------------------
# F1.2 — HERC
# ---------------------------------------------------------------------------


def _equal_blocks_sigma_and_tree() -> tuple[pd.DataFrame, np.ndarray]:
    """Dva identična 3×3 bloka (ρ=0.6 unutar, 0 između) + ward stablo."""
    vols = np.array([0.2, 0.3, 0.4, 0.2, 0.3, 0.4])
    corr = np.eye(6)
    for block in (range(0, 3), range(3, 6)):
        for i in block:
            for j in block:
                if i != j:
                    corr[i, j] = 0.6
    sigma = corr * np.outer(vols, vols)
    tickers = list("ABCDEF")
    sigma_frame = pd.DataFrame(sigma, index=tickers, columns=tickers)
    tree = build_correlation_tree(
        pd.DataFrame(corr, index=tickers, columns=tickers), linkage="ward"
    )
    return sigma_frame, tree


def test_herc_equal_blocks_split_capital_50_50():
    """k=2 i blok-dijagonalna Σ s dva jednaka bloka → podjela kapitala 50/50."""
    sigma_frame, tree = _equal_blocks_sigma_and_tree()

    weights, share = herc_weights(sigma_frame, tree, k=2, w_max=0.5)

    block_1 = weights.loc[["A", "B", "C"]].sum()
    block_2 = weights.loc[["D", "E", "F"]].sum()
    assert block_1 == pytest.approx(0.5)
    assert block_2 == pytest.approx(0.5)
    # naivni risk parity unutar bloka: w ∝ 1/σ
    inverse_vol = 1.0 / np.array([0.2, 0.3, 0.4])
    expected_block = 0.5 * inverse_vol / inverse_vol.sum()
    assert weights.loc[["A", "B", "C"]].to_numpy() == pytest.approx(expected_block)
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 0.5 + 1e-12
    assert share == 0.0


def test_herc_cap_binds():
    """Vezujući cap: težine ≤ w_max uz Σw = 1, capped_weight_share > 0."""
    sigma_frame, tree = _equal_blocks_sigma_and_tree()

    weights, share = herc_weights(sigma_frame, tree, k=2, w_max=0.2)

    assert weights.max() <= 0.2 + 1e-12
    assert weights.sum() == pytest.approx(1.0)
    assert share > 0.0


# ---------------------------------------------------------------------------
# F1.3 — NCO
# ---------------------------------------------------------------------------


def test_nco_k1_unbinding_cap_equals_min_variance():
    """K=1 i nevezujući cap (w_max=1) → NCO = min_variance na punoj Σ."""
    sigma = _random_covariance(5)
    expected = min_variance(sigma, w_max=1.0)

    weights, share = nco_weights(sigma, labels=np.ones(5, dtype=int), w_max=1.0)

    assert np.max(np.abs(weights - expected)) <= 1e-8
    assert share == 0.0


def test_nco_block_diagonal_matches_analytic_solution():
    """Blok-dijagonalna Σ: kombinirane težine = analitičko Σ⁻¹1 rješenje."""
    sigma_frame = _block_diagonal_sigma()
    labels = pd.Series([1, 1, 2, 2], index=sigma_frame.index)
    raw = np.linalg.solve(sigma_frame.to_numpy(), np.ones(4))
    expected = raw / raw.sum()
    assert (expected > 0).all()  # long-only ograničenje ne grize

    weights, _ = nco_weights(sigma_frame, labels, w_max=1.0)

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == list(sigma_frame.index)
    assert weights.to_numpy() == pytest.approx(expected, abs=1e-5)


def test_nco_cap_binds():
    """Vezujući cap: sve težine ≤ w_max uz Σw = 1, capped_weight_share > 0."""
    sigma_frame = _block_diagonal_sigma()
    labels = pd.Series([1, 1, 2, 2], index=sigma_frame.index)

    weights, share = nco_weights(sigma_frame, labels, w_max=0.3)

    assert weights.max() <= 0.3 + 1e-12
    assert weights.sum() == pytest.approx(1.0)
    assert share > 0.0


# ---------------------------------------------------------------------------
# F1.5 — integracija u walk-forward
# ---------------------------------------------------------------------------


def _synthetic_walk_forward_inputs(n_assets: int = 8, seed: int = 7):
    """Mali sintetički ulazi (2 prozora) za smoke test runnera."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_assets)]
    n_months = 84  # 2000-01 .. 2006-12
    index = (
        pd.period_range("2000-01", periods=n_months, freq="M")
        .to_timestamp(how="end")
        .normalize()
    )
    common = rng.normal(scale=0.04, size=(n_months, 1))
    loadings = rng.uniform(0.5, 1.5, size=(1, n_assets))
    idio = rng.normal(scale=0.03, size=(n_months, n_assets))
    monthly = pd.DataFrame(common * loadings + idio, index=index, columns=tickers)

    windows = generate_rolling_windows("2000-01", "2006-12", 60, 12, 12)
    labels = [w.label for w in windows]
    factor_exposures = pd.DataFrame(
        [{"train_window": lab, "ticker": t} for lab in labels for t in tickers]
    )
    factor_clusters = pd.DataFrame(
        [
            {"train_window": lab, "ticker": t, "factor_cluster": 1 + i % 3}
            for lab in labels
            for i, t in enumerate(tickers)
        ]
    )
    correlation_clusters = pd.DataFrame(
        [
            {"train_window": lab, "ticker": t, "correlation_cluster": 1 + i % 3}
            for lab in labels
            for i, t in enumerate(tickers)
        ]
    )
    metadata = pd.DataFrame(
        {"ticker": tickers, "sector": [f"S{i % 2}" for i in range(n_assets)]}
    )
    return monthly, factor_exposures, factor_clusters, correlation_clusters, metadata, windows


def test_run_hierarchical_walk_forward_structure():
    """Runner vraća 4 stupca prinosa, čist status i valjane težine."""
    monthly, fexp, fclust, cclust, meta, windows = _synthetic_walk_forward_inputs()

    result = run_hierarchical_walk_forward(
        monthly, fexp, fclust, cclust, meta, windows, k=3, w_max=0.5
    )

    panel = result["port_returns_panel"]
    status = result["portfolio_status"]
    assert list(panel.columns) == list(HIERARCHICAL_PORTFOLIO_NAMES)
    assert len(panel) == 24  # 2 prozora × 12 testnih mjeseci
    assert (status["status"] == "ok").all()
    assert "capped_weight_share" in status.columns
    assert status["capped_weight_share"].notna().all()

    weight_sums = (
        result["weights_panel"].groupby(["train_window", "portfolio"])["weight"].sum()
    )
    assert weight_sums.to_numpy() == pytest.approx(1.0)
    assert result["weights_panel"]["weight"].max() <= 0.5 + 1e-9


def test_run_hierarchical_walk_forward_factor_space_not_implemented():
    """tree_space='factor' je rezerviran za Fazu 3 (F3.1)."""
    monthly, fexp, fclust, cclust, meta, windows = _synthetic_walk_forward_inputs()
    with pytest.raises(NotImplementedError):
        run_hierarchical_walk_forward(
            monthly, fexp, fclust, cclust, meta, windows, k=3, tree_space="factor"
        )


def test_run_hierarchical_walk_forward_membership_filter():
    """Point-in-time presjek reže univerzum na članove na train_end (§3.1)."""
    monthly, fexp, fclust, cclust, meta, windows = _synthetic_walk_forward_inputs()
    tickers = list(monthly.columns)
    members, nonmembers = tickers[:5], tickers[5:]
    membership = pd.DataFrame(
        [{"ticker": t, "start_date": "1999-01-31", "end_date": pd.NaT} for t in members]
        + [
            {"ticker": t, "start_date": "1999-01-31", "end_date": "2001-12-31"}
            for t in nonmembers
        ]
    )

    result = run_hierarchical_walk_forward(
        monthly, fexp, fclust, cclust, meta, windows, k=3, w_max=0.5, membership=membership
    )

    used = set(result["weights_panel"]["ticker"])
    assert used <= set(members)
    assert not used & set(nonmembers)

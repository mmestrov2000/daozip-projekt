"""Metrike evaluacije izvan uzorka za fazu 5.

Pomoćnici za rizik, diversifikaciju i faktorsku atribuciju rade na mjesečnim
prinosima i proizvode anualizirane brojeve (`* sqrt(12)` za volatilnost,
`* 12` za varijancu). Cijeli skup koristi
``notebooks/04_evaluation.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


MONTHS_PER_YEAR = 12
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BETA_OUTPUT_NAMES = {
    "Mkt-RF": "beta_mkt",
    "SMB": "beta_smb",
    "HML": "beta_hml",
    "RMW": "beta_rmw",
    "CMA": "beta_cma",
}
STYLE_FACTORS = ("beta_smb", "beta_hml", "beta_rmw", "beta_cma")


def _as_weight_series(weights: pd.Series | dict | np.ndarray) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.copy()
    elif isinstance(weights, dict):
        series = pd.Series(weights)
    else:
        series = pd.Series(np.asarray(weights, dtype=float))

    if series.empty:
        raise ValueError("weights ne smije biti prazan.")
    if series.index.has_duplicates:
        raise ValueError("indeks weights mora biti jedinstven.")

    series = series.dropna()
    if series.empty:
        raise ValueError("weights su sve nedostajuće nakon izbacivanja NaN.")
    return series.astype(float)


def _as_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns mora biti pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns ne smije biti prazan.")
    if returns.columns.has_duplicates:
        raise ValueError("stupci returns moraju biti jedinstveni.")
    return returns.apply(pd.to_numeric, errors="raise")


def portfolio_returns(
    weights: pd.Series,
    test_returns: pd.DataFrame,
) -> pd.Series:
    """Vrati mjesečne prinose portfelja uz fiksne (mjesečno rebalansirane) težine.

    Prinos svakog mjeseca je ``Σ w_i r_{i,t}``, izračunat na presjeku
    oznaka dionica prisutnih i u ``weights`` i u ``test_returns``. Težine se
    ne renormaliziraju; funkcija umjesto toga provjerava da sve težinske oznake
    imaju podatke i inače diže iznimku.
    """
    weight_series = _as_weight_series(weights)
    returns_frame = _as_return_frame(test_returns)

    missing = weight_series.index.difference(returns_frame.columns)
    if len(missing) > 0:
        raise ValueError(
            f"Nedostaju prinosi za težinske oznake dionica: {missing.tolist()}"
        )

    aligned = returns_frame.loc[:, weight_series.index]
    if aligned.isna().any().any():
        raise ValueError("test_returns sadrži NaN za težinske oznake dionica.")

    return aligned.dot(weight_series).rename("portfolio_return")


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def _historical_var_cvar(returns: pd.Series, alpha: float) -> tuple[float, float]:
    if returns.empty:
        return float("nan"), float("nan")
    sorted_returns = np.sort(returns.to_numpy(dtype=float))
    var = float(np.quantile(sorted_returns, alpha))
    tail = sorted_returns[sorted_returns <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def risk_metrics(port_returns: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Anualizirana volatilnost, maksimalni pad, povijesni VaR/CVaR pri ``alpha``.

    ``alpha=0.05`` glavna je razina repa u projektu. VaR/CVaR izvještavaju se
    kao gubici na mjesečnoj osnovi (iste jedinice kao ulazni prinosi);
    negativan broj znači gubitak.
    """
    if not isinstance(port_returns, pd.Series):
        raise TypeError("port_returns mora biti pandas Series.")
    series = port_returns.dropna().astype(float)
    if series.empty:
        raise ValueError("port_returns nema opservacija nakon izbacivanja NaN.")

    ann_vol = float(series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    var, cvar = _historical_var_cvar(series, alpha)
    metrics = pd.Series(
        {
            "ann_return": float((1.0 + series).prod() ** (MONTHS_PER_YEAR / len(series)) - 1.0),
            "ann_vol": ann_vol,
            "max_drawdown": _max_drawdown(series),
            "var_5": var,
            "cvar_5": cvar,
            "n_months": int(len(series)),
        }
    )
    return metrics


def diversification_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """Statistika koncentracije i diversifikacije za pojedini portfelj.

    Vraća HHI (Σ w²), efektivni N (1/HHI), omjer diversifikacije
    ``(Σ w_i σ_i) / σ_p`` izračunat na danom prozoru prinosa i
    težinski prosjek korelacije po parovima na istom prozoru.
    Prozor prinosa trebao bi biti prozor treniranja ako je cilj sažeti
    ulaze konstrukcije portfelja, ili testni prozor ako je cilj
    sažeti ostvarenu diversifikaciju.
    """
    weight_series = _as_weight_series(weights)
    returns_frame = _as_return_frame(returns)

    missing = weight_series.index.difference(returns_frame.columns)
    if len(missing) > 0:
        raise ValueError(
            f"Nedostaju prinosi za težinske oznake dionica: {missing.tolist()}"
        )
    aligned = returns_frame.loc[:, weight_series.index].dropna(axis=0, how="any")
    if len(aligned) < 2:
        raise ValueError("Potrebne su barem dvije potpune opservacije prinosa.")

    w = weight_series.to_numpy(dtype=float)
    sigma_assets = aligned.std(ddof=1).to_numpy(dtype=float) * np.sqrt(MONTHS_PER_YEAR)
    cov = aligned.cov().to_numpy(dtype=float) * MONTHS_PER_YEAR
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else float("nan")

    weighted_sigma = float(np.dot(w, sigma_assets))
    diversification_ratio = (
        weighted_sigma / port_vol if port_vol and np.isfinite(port_vol) else float("nan")
    )

    hhi = float(np.sum(w**2))
    n_eff = float(1.0 / hhi) if hhi > 0 else float("nan")

    correlation = aligned.corr().to_numpy(dtype=float)
    pair_weights = np.outer(w, w)
    np.fill_diagonal(pair_weights, 0.0)
    total_pair_weight = pair_weights.sum()
    avg_pair_corr = (
        float(np.sum(pair_weights * correlation) / total_pair_weight)
        if total_pair_weight > 0
        else float("nan")
    )

    return pd.Series(
        {
            "hhi": hhi,
            "n_eff": n_eff,
            "diversification_ratio": diversification_ratio,
            "avg_pairwise_corr": avg_pair_corr,
            "weighted_avg_vol": weighted_sigma,
            "portfolio_vol": port_vol,
        }
    )


def factor_attribution(
    port_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.Series:
    """Regresiraj viškove prinosa portfelja na pet Fama-French faktora.

    ``factors`` mora sadržavati ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``
    i ``RF``. Višak prinosa portfelja je ``r_p - RF`` kroz iste
    mjesece. Vraća alfu, pet beta, R² i glavnu projektovu
    ``style_concentration = |β_SMB|+|β_HML|+|β_RMW|+|β_CMA|``.
    """
    if not isinstance(port_returns, pd.Series):
        raise TypeError("port_returns mora biti pandas Series.")

    required = set(FACTOR_COLUMNS + ["RF"])
    missing = required.difference(factors.columns)
    if missing:
        raise ValueError(f"factors nedostaju stupci: {sorted(missing)}")

    aligned = (
        pd.concat(
            [port_returns.rename("portfolio_return"), factors],
            axis=1,
            join="inner",
        )
        .dropna()
    )
    min_observations = len(FACTOR_COLUMNS) + 2
    if len(aligned) < min_observations:
        raise ValueError(
            f"Potrebno je barem {min_observations} poravnatih opservacija; dobiveno {len(aligned)}."
        )

    y = aligned["portfolio_return"] - aligned["RF"]
    x = sm.add_constant(aligned[FACTOR_COLUMNS], has_constant="add")
    model = sm.OLS(y, x).fit()

    row: dict[str, float] = {"alpha": float(model.params["const"])}
    for factor_name, output_name in BETA_OUTPUT_NAMES.items():
        row[output_name] = float(model.params[factor_name])
    row["r_squared"] = float(model.rsquared)
    row["style_concentration"] = float(
        sum(abs(row[name]) for name in STYLE_FACTORS)
    )
    row["n_months"] = int(len(aligned))
    return pd.Series(row)


# ---------------------------------------------------------------------------
# Evaluacija po testnoj godini i na punom uzorku
# ---------------------------------------------------------------------------


def _attribution_row(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
) -> dict[str, float]:
    aligned = (
        pd.concat([portfolio_returns.rename("y"), factors], axis=1, join="inner")
        .dropna()
    )
    y = aligned["y"] - aligned["RF"]
    x = sm.add_constant(aligned[FACTOR_COLUMNS], has_constant="add")
    model = sm.OLS(y, x).fit()
    row: dict[str, float] = {"alpha": float(model.params["const"])}
    for factor_name, output_name in BETA_OUTPUT_NAMES.items():
        row[output_name] = float(model.params[factor_name])
    row["r_squared"] = float(model.rsquared)
    row["style_concentration"] = float(
        sum(abs(row[name]) for name in STYLE_FACTORS)
    )
    row["n_months"] = int(len(aligned))
    return row


def factor_attribution_per_window(
    port_returns_panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Regresiraj prinose svakog portfelja po testnoj godini na 5 faktora.

    Vraća jedan redak po (test_year, portfolio) s alfom, pet beta,
    R² i ``style_concentration = |β_SMB|+|β_HML|+|β_RMW|+|β_CMA|``.
    „Testna godina” identificira se kalendarskom godinom mjesečnog indeksa.
    """
    if not isinstance(port_returns_panel, pd.DataFrame):
        raise TypeError("port_returns_panel mora biti DataFrame.")
    if port_returns_panel.empty:
        return pd.DataFrame(
            columns=[
                "test_year", "portfolio", "alpha",
                "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
                "r_squared", "style_concentration", "n_months",
            ]
        )

    panel = port_returns_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    rows: list[dict[str, float | int | str]] = []
    for portfolio in panel.columns:
        series = panel[portfolio].dropna()
        if series.empty:
            continue
        for year, group in series.groupby(series.index.year):
            if len(group) < len(FACTOR_COLUMNS) + 2:
                continue
            try:
                row = _attribution_row(group, factors)
            except Exception:
                continue
            row.update({"test_year": int(year), "portfolio": str(portfolio)})
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    column_order = [
        "test_year", "portfolio", "alpha",
        "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
        "r_squared", "style_concentration", "n_months",
    ]
    return result[column_order].sort_values(["test_year", "portfolio"]).reset_index(drop=True)


def factor_attribution_full_sample(
    port_returns_panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Jedan redak po portfelju s petfaktorskom atribucijom na punom uzorku.

    Iste metrike kao :func:`factor_attribution_per_window`, ali regresirane na
    cijelom spojenom nizu prinosa testnog razdoblja (tj. glavni
    brojevi na punom uzorku).
    """
    if not isinstance(port_returns_panel, pd.DataFrame):
        raise TypeError("port_returns_panel mora biti DataFrame.")
    panel = port_returns_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    rows: list[dict[str, float | int | str]] = []
    for portfolio in panel.columns:
        series = panel[portfolio].dropna()
        if len(series) < len(FACTOR_COLUMNS) + 2:
            continue
        row = _attribution_row(series, factors)
        row["portfolio"] = str(portfolio)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    column_order = [
        "portfolio", "alpha",
        "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
        "r_squared", "style_concentration", "n_months",
    ]
    return result[column_order].sort_values("portfolio").reset_index(drop=True)


def risk_decomposition_per_window(
    port_returns_panel: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Razloži anualiziranu varijancu portfelja po testnoj godini i portfelju.

    Dekompozicija: ukupna varijanca = faktorska varijanca + idiosinkratska varijanca,
    gdje je ``faktorska varijanca = β' Σ_f β`` (β = pet faktorskih beta
    portfelja u toj testnoj godini, Σ_f = kovarijanca prinosa pet FF
    faktora kroz tu testnu godinu), a ``idiosinkratska varijanca = rezidualna
    varijanca iz OLS regresije``. Svi brojevi anualizirani s × 12.
    """
    if not isinstance(port_returns_panel, pd.DataFrame):
        raise TypeError("port_returns_panel mora biti DataFrame.")
    panel = port_returns_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    rows: list[dict[str, float | int | str]] = []
    for portfolio in panel.columns:
        series = panel[portfolio].dropna()
        for year, group in series.groupby(series.index.year):
            if len(group) < len(FACTOR_COLUMNS) + 2:
                continue
            aligned = (
                pd.concat(
                    [group.rename("y"), factors], axis=1, join="inner"
                ).dropna()
            )
            if len(aligned) < len(FACTOR_COLUMNS) + 2:
                continue
            y = aligned["y"] - aligned["RF"]
            x = sm.add_constant(aligned[FACTOR_COLUMNS], has_constant="add")
            model = sm.OLS(y, x).fit()
            betas = np.array(
                [model.params[factor_name] for factor_name in FACTOR_COLUMNS]
            )
            factor_cov = aligned[FACTOR_COLUMNS].cov().to_numpy()
            factor_var = float(betas @ factor_cov @ betas)
            idio_var = float(np.var(model.resid, ddof=1))
            total_var = factor_var + idio_var
            rows.append(
                {
                    "test_year": int(year),
                    "portfolio": str(portfolio),
                    "factor_var_ann": MONTHS_PER_YEAR * factor_var,
                    "idio_var_ann": MONTHS_PER_YEAR * idio_var,
                    "total_var_ann": MONTHS_PER_YEAR * total_var,
                    "factor_share": (
                        float(factor_var / total_var) if total_var > 0 else float("nan")
                    ),
                    "r_squared": float(model.rsquared),
                    "n_months": int(len(aligned)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["test_year", "portfolio"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Blok-bootstrap za razlike metrika portfelja
# ---------------------------------------------------------------------------


def _block_indices(
    n_months: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vrati vektor duljine ``n_months`` uzorkovan u susjednim blokovima
    (preklapajući; kružni). To je standardna konstrukcija ``bootstrapa pomičnim
    blokovima`` (Künsch 1989.) — uzorkovanje blok po blok čuva
    serijsku ovisnost kratkog horizonta koja nam je bitna u mjesečnim prinosima
    portfelja.
    """
    if block_size < 1 or block_size > n_months:
        raise ValueError("block_size mora biti u [1, n_months].")
    n_blocks = int(np.ceil(n_months / block_size))
    starts = rng.integers(0, n_months, size=n_blocks)
    indices = np.concatenate(
        [(start + np.arange(block_size)) % n_months for start in starts]
    )
    return indices[:n_months]


def style_concentration(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
) -> float:
    """Koncentracija stila na punom uzorku ``|β_SMB|+|β_HML|+|β_RMW|+|β_CMA|``."""
    aligned = (
        pd.concat([portfolio_returns.rename("y"), factors], axis=1, join="inner")
        .dropna()
    )
    if len(aligned) < len(FACTOR_COLUMNS) + 2:
        return float("nan")
    y = aligned["y"] - aligned["RF"]
    x = sm.add_constant(aligned[FACTOR_COLUMNS], has_constant="add")
    model = sm.OLS(y, x).fit()
    return float(sum(abs(model.params[name]) for name in FACTOR_COLUMNS if name != "Mkt-RF"))


def annualized_vol(portfolio_returns: pd.Series) -> float:
    series = portfolio_returns.dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(portfolio_returns: pd.Series) -> float:
    series = portfolio_returns.dropna().astype(float)
    if series.empty:
        return float("nan")
    cumulative = (1.0 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def block_bootstrap_diff(
    portfolio_a: pd.Series,
    portfolio_b: pd.Series,
    metric_fn,
    metric_kwargs: dict | None = None,
    factors: pd.DataFrame | None = None,
    n_bootstraps: int = 1000,
    block_size: int = 12,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """Blok-bootstrap CI za razliku metrika ``metric(a) - metric(b)``.

    Dva niza prinosa poravnavaju se na zajedničkom (sortiranom) indeksu; isti
    ponovno uzorkovani blok mjeseci primjenjuje se na oba, pa bootstrap
    čuva zajedničku distribuciju između portfelja.

    Ako ``metric_fn`` treba panel faktora (npr. koncentracija stila, koja
    interno pokreće OLS), proslijedi ``factors`` i funkcija će zajednički
    ponovno uzorkovati retke faktora kako bi unutarnja regresija vidjela iste mjesece
    kao prinosi portfelja. ``metric_fn`` se tada poziva kao
    ``metric_fn(series, factors_subset, **metric_kwargs)``. Inače se
    poziva kao ``metric_fn(series, **metric_kwargs)``.

    Vraća rječnik s točkastom procjenom na punom uzorku, bootstrap
    distribucijom uzorkovanja, 95 % percentilnim CI-jem i dvostranom
    p-vrijednošću iz percentila za ``H0: diff == 0``.
    """
    rng = np.random.default_rng(seed)
    metric_kwargs = metric_kwargs or {}

    frames = [portfolio_a, portfolio_b]
    if factors is not None:
        frames.append(factors)
    aligned = pd.concat(frames, axis=1, join="inner").dropna()
    if aligned.shape[1] < 2:
        raise ValueError("portfolio_a i portfolio_b nemaju preklapanje.")

    aligned_a = aligned.iloc[:, 0]
    aligned_b = aligned.iloc[:, 1]
    aligned_factors = aligned.iloc[:, 2:] if factors is not None else None

    if aligned_factors is None:
        point = metric_fn(aligned_a, **metric_kwargs) - metric_fn(
            aligned_b, **metric_kwargs
        )
    else:
        point = metric_fn(
            aligned_a, aligned_factors, **metric_kwargs
        ) - metric_fn(aligned_b, aligned_factors, **metric_kwargs)

    diffs = np.zeros(n_bootstraps, dtype=float)
    n = len(aligned)
    placeholder_index = aligned.index[: n]
    for i in range(n_bootstraps):
        idx = _block_indices(n, block_size, rng)
        # Ponovno uzorkuj zajednički. Dodijeli čist sekvencijalni indeks kako bi se
        # unutarnja regresijska ``concat([series, factors], join="inner")`` poravnala
        # na iste ponovno uzorkovane mjesece na objema stranama.
        sample_a = aligned_a.iloc[idx].reset_index(drop=True)
        sample_a.index = placeholder_index
        sample_b = aligned_b.iloc[idx].reset_index(drop=True)
        sample_b.index = placeholder_index
        if aligned_factors is None:
            diffs[i] = metric_fn(sample_a, **metric_kwargs) - metric_fn(
                sample_b, **metric_kwargs
            )
        else:
            sample_fac = aligned_factors.iloc[idx].reset_index(drop=True)
            sample_fac.index = placeholder_index
            diffs[i] = metric_fn(
                sample_a, sample_fac, **metric_kwargs
            ) - metric_fn(sample_b, sample_fac, **metric_kwargs)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "point": float(point),
        "boot_mean": float(np.mean(diffs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": p_two_sided,
        "diffs": diffs,
    }


def block_bootstrap_metric(
    portfolio_returns_with_factors: pd.DataFrame,
    metric_fn,
    n_bootstraps: int = 1000,
    block_size: int = 12,
    seed: int = 42,
) -> dict[str, float | np.ndarray]:
    """Blok-bootstrap CI za metriku pojedinog portfelja koja treba faktore.

    Očekuje se da ``portfolio_returns_with_factors`` ima stupce ``y``
    (prinos portfelja) te ``Mkt-RF, SMB, HML, RMW, CMA, RF``.
    Ponovno uzorkovani blok primjenjuje se zajednički na stupce portfelja i faktora.
    ``metric_fn`` se poziva kao ``metric_fn(panel)`` i mora vratiti
    skalar iz panela.
    """
    rng = np.random.default_rng(seed)
    panel = portfolio_returns_with_factors.dropna()
    if panel.empty:
        raise ValueError("portfolio_returns_with_factors nema upotrebljivih redaka.")
    point = metric_fn(panel)
    samples = np.zeros(n_bootstraps, dtype=float)
    n = len(panel)
    for i in range(n_bootstraps):
        idx = _block_indices(n, block_size, rng)
        sample = panel.iloc[idx].reset_index(drop=True)
        sample.index = panel.index[: len(sample)]
        samples[i] = metric_fn(sample)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return {
        "point": float(point),
        "boot_mean": float(np.mean(samples)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "samples": samples,
    }

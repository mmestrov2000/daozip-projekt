"""Orkestracija kliznog prozora i unaprijedni backtest portfelja."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from src.data import membership_on
from src.hierarchical import (
    apply_w_max,
    build_correlation_tree,
    herc_weights,
    hrp_weights,
    nco_weights,
    quasi_diagonal_order,
)
from src.portfolio import (
    equal_weight,
    ledoit_wolf_cov,
    min_var_factor_neutral,
    min_var_group_and_factor_neutral,
    min_var_group_constrained,
    min_variance,
    sample_cov,
)
from src.utils import (
    GROUP_CAP,
    MIN_TRAINING_MONTHS,
    PROJECT_END,
    PROJECT_START,
    REFIT_STEP_MONTHS,
    TEST_HORIZON_MONTHS,
    TRAIN_LOOKBACK_MONTHS,
    W_MAX,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollingWindow:
    """Jedan par treniranje/testiranje s granicama na kraju mjeseca.

    ``label`` je uključivi završni mjesec treniranja (``YYYY-MM``); to je
    kanonski identifikator korišten u faktorskim izloženostima, tablicama klasteriranja i
    panelima težina.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str

    def as_tuple(self) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        return (self.train_start, self.train_end, self.test_start, self.test_end)


def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp(how="end").normalize()


def generate_rolling_windows(
    project_start: str | pd.Timestamp = PROJECT_START,
    project_end: str | pd.Timestamp = PROJECT_END,
    train_lookback: int = TRAIN_LOOKBACK_MONTHS,
    test_horizon: int = TEST_HORIZON_MONTHS,
    refit_step: int = REFIT_STEP_MONTHS,
) -> list[RollingWindow]:
    """Daje prozore treniranje/testiranje koji hodaju unaprijed kroz prozor projekta.

    Konvencije (usklađene sa specifikacijom §4.4):
    - ``train_lookback`` mjeseci povijesti završava na ``train_end`` uključivo.
    - Testni prozor neposredno slijedi: ``test_start = train_end + 1 mjesec``
      i traje ``test_horizon`` mjeseci.
    - Razdoblje treniranja prvog prozora počinje na ``project_start``.
    - Testno razdoblje posljednjeg prozora završava na ili prije ``project_end``.
    - Ponovno procjenjivanje događa se svakih ``refit_step`` mjeseci.
    """
    if train_lookback <= 0 or test_horizon <= 0 or refit_step <= 0:
        raise ValueError("Lookback, horizont i korak moraju svi biti pozitivni.")

    project_start_period = pd.Period(project_start, freq="M")
    project_end_period = pd.Period(project_end, freq="M")
    if project_start_period > project_end_period:
        raise ValueError("project_start mora biti na ili prije project_end.")

    windows: list[RollingWindow] = []
    train_end_period = project_start_period + (train_lookback - 1)
    while True:
        test_start_period = train_end_period + 1
        test_end_period = test_start_period + (test_horizon - 1)
        if test_end_period > project_end_period:
            break
        train_start_period = train_end_period - (train_lookback - 1)
        if train_start_period < project_start_period:
            train_end_period += refit_step
            continue
        windows.append(
            RollingWindow(
                train_start=train_start_period.to_timestamp(how="end").normalize(),
                train_end=train_end_period.to_timestamp(how="end").normalize(),
                test_start=test_start_period.to_timestamp(how="end").normalize(),
                test_end=test_end_period.to_timestamp(how="end").normalize(),
                label=str(train_end_period),
            )
        )
        train_end_period += refit_step

    return windows


# ---------------------------------------------------------------------------
# Pokretač unaprijednog hoda portfelja
# ---------------------------------------------------------------------------


PORTFOLIO_NAMES = (
    "equal_weight",
    "min_var",
    "min_var_sector",
    "min_var_corr_cluster",
    "min_var_factor_cluster",
)


def _resolve_cov_estimator(
    cov_estimator: str | Callable[[pd.DataFrame], pd.DataFrame],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if callable(cov_estimator):
        return cov_estimator
    name = str(cov_estimator).lower()
    if name in {"ledoit_wolf", "ledoit-wolf", "lw"}:
        return ledoit_wolf_cov
    if name in {"sample", "cov", "empirical"}:
        return sample_cov
    raise ValueError(f"Nepoznat procjenitelj kovarijance: {cov_estimator!r}")


def _prepare_membership(membership: pd.DataFrame | None) -> pd.DataFrame | None:
    """Pripremi tablicu članstva za point-in-time presjek (datumski stupci).

    Vraća ``None`` ako članstvo nije proslijeđeno (filtar se preskače), inače
    kopiju s ``start_date``/``end_date`` kao datetime — neovisno o tome kako ih je
    pozivatelj učitao.
    """
    if membership is None:
        return None
    membership = membership.copy()
    membership["start_date"] = pd.to_datetime(membership["start_date"])
    membership["end_date"] = pd.to_datetime(membership["end_date"])
    return membership


def _restrict_to_members(
    universe: pd.Index,
    membership: pd.DataFrame | None,
    train_end: pd.Timestamp,
) -> pd.Index:
    """Reži univerzum prozora na point-in-time članove S&P 500 na ``train_end``.

    Provedba zaključane metodologije (PROJECT_SPEC §3.1: „po prozoru se reže na
    članstvo na datum train_end”) — bez ovog presjeka delistani/ponovo
    iskorišteni tickeri ulaze u univerzum s artefaktnim prinosima. Kad
    ``membership`` nije proslijeđen, presjek se preskače (npr. sintetički testovi).
    """
    if membership is None:
        return universe
    members = pd.Index(membership_on(train_end, membership))
    return universe.intersection(members)


def _lookup_groups(
    cluster_table: pd.DataFrame,
    window_label: str,
    tickers: Sequence[str],
    cluster_column: str,
) -> pd.Series:
    subset = cluster_table.loc[
        cluster_table["train_window"] == window_label, ["ticker", cluster_column]
    ]
    if subset.empty:
        raise ValueError(
            f"Nije pronađen nijedan redak {cluster_column} za prozor {window_label!r}."
        )
    grouped = subset.set_index("ticker")[cluster_column].reindex(tickers)
    if grouped.isna().any():
        missing = grouped.index[grouped.isna()].tolist()
        raise ValueError(
            f"Nedostaju oznake {cluster_column} za {len(missing)} oznaka dionica u "
            f"prozoru {window_label!r}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return grouped.astype(str)


def _solve_safely(
    name: str,
    window_label: str,
    solver: Callable[[], pd.Series],
) -> tuple[pd.Series | None, str]:
    try:
        weights = solver()
        return weights, "ok"
    except Exception as error:
        LOGGER.warning("Portfelj %s nije uspio u prozoru %s: %s", name, window_label, error)
        return None, f"failed: {type(error).__name__}: {error}"


def run_walk_forward(
    monthly_returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    factor_clusters: pd.DataFrame,
    correlation_clusters: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: Iterable[RollingWindow],
    cov_estimator: str | Callable[[pd.DataFrame], pd.DataFrame] = "ledoit_wolf",
    w_max: float = W_MAX,
    group_cap: float = GROUP_CAP,
    min_training_months: int = MIN_TRAINING_MONTHS,
    portfolios: Sequence[str] | None = None,
    membership: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Pokreni backtest s kliznim prozorom kroz odabrane varijante portfelja.

    ``portfolios`` ograničava izbor na podskup ``PORTFOLIO_NAMES``; default
    (``None``) pokreće svih pet varijanti (povijesno ponašanje).

    ``membership`` (tablica intervala ``ticker, start_date, end_date``) uključuje
    point-in-time presjek univerzuma na članove S&P 500 na ``train_end``
    (PROJECT_SPEC §3.1); kad je ``None``, presjek se preskače.

    Po prozoru, univerzum za treniranje presjek je:
    - oznaka dionica s barem ``min_training_months`` nenedostajućih prinosa
      treniranja,
    - oznaka s faktorskim izloženostima procijenjenima za taj prozor,
    - oznaka s dodjelama faktorskih i korelacijskih klastera za taj prozor.

    Svaki portfelj konstruira se na prozoru treniranja i primjenjuje
    fiksnim težinama na mjesečne prinose sljedećeg testnog prozora. Neuspjesi (npr.
    neizvediva rješenja) bilježe se u ``portfolio_status`` i ne prekidaju
    pokretanje.

    Vraća rječnik s:
      ``weights_panel``        — dugi DataFrame (prozor, portfelj, oznaka dionice, težina)
      ``port_returns_panel``   — široki DataFrame (mjesec × portfelj)
      ``portfolio_status``     — dnevnik izvedivosti po prozoru i portfelju
    """
    cov_fn = _resolve_cov_estimator(cov_estimator)
    membership = _prepare_membership(membership)
    requested = PORTFOLIO_NAMES if portfolios is None else tuple(portfolios)
    unknown = set(requested) - set(PORTFOLIO_NAMES)
    if unknown:
        raise ValueError(f"Nepoznati portfelji: {sorted(unknown)}")
    sector_map = metadata.set_index("ticker")["sector"]

    weights_rows: list[dict[str, object]] = []
    returns_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for window in windows:
        train_returns = monthly_returns.loc[window.train_start : window.train_end]
        valid_counts = train_returns.notna().sum(axis=0)
        eligible_tickers = valid_counts[valid_counts >= min_training_months].index

        factor_tickers = pd.Index(
            factor_exposures.loc[
                factor_exposures["train_window"] == window.label, "ticker"
            ]
        )
        factor_cluster_tickers = pd.Index(
            factor_clusters.loc[
                factor_clusters["train_window"] == window.label, "ticker"
            ]
        )
        correlation_cluster_tickers = pd.Index(
            correlation_clusters.loc[
                correlation_clusters["train_window"] == window.label, "ticker"
            ]
        )

        universe = (
            pd.Index(eligible_tickers)
            .intersection(factor_tickers)
            .intersection(factor_cluster_tickers)
            .intersection(correlation_cluster_tickers)
            .intersection(sector_map.dropna().index)
        )
        universe = _restrict_to_members(universe, membership, window.train_end)
        if len(universe) < 5:
            LOGGER.warning(
                "Prozor %s ima samo %d prihvatljivih oznaka dionica; preskačem.",
                window.label,
                len(universe),
            )
            continue

        train_panel = train_returns.loc[:, universe].dropna(axis=0, how="any")
        if len(train_panel) < min_training_months:
            LOGGER.warning(
                "Prozor %s ima %d potpunih redaka treniranja nakon presjeka; "
                "preskačem.",
                window.label,
                len(train_panel),
            )
            continue

        try:
            sigma = cov_fn(train_panel)
        except Exception as error:
            LOGGER.error(
                "Procjenitelj kovarijance nije uspio u prozoru %s: %s",
                window.label,
                error,
            )
            continue

        sigma = sigma.loc[universe, universe]

        test_returns = monthly_returns.loc[window.test_start : window.test_end, universe]
        # Oznake dionica s potpuno NaN testnim recima izbacuju se iz univerzuma samo za taj
        # testni prozor — ne mogu doprinijeti ostvarenom prinosu.
        usable_in_test = test_returns.columns[test_returns.notna().any(axis=0)]
        if len(usable_in_test) < len(universe):
            LOGGER.warning(
                "Prozor %s izbacuje %d oznaka dionica iz testnog univerzuma (nema testnih podataka).",
                window.label,
                len(universe) - len(usable_in_test),
            )
        # Za težine i dalje koristimo puni univerzum treniranja (praznine u testnim mjesecima
        # popunjavaju se nultim doprinosom umjesto ponovnog rješavanja kvadratnog programa).
        sector_groups = sector_map.loc[universe].astype(str)
        factor_groups = _lookup_groups(
            factor_clusters, window.label, universe, "factor_cluster"
        )
        portfolio_solvers: dict[str, Callable[[], pd.Series]] = {
            "equal_weight": lambda: equal_weight(len(universe), index=universe),
            "min_var": lambda: min_variance(sigma, w_max=w_max),
            "min_var_sector": lambda: min_var_group_constrained(
                sigma, sector_groups, w_max=w_max, group_cap=group_cap
            ),
            "min_var_factor_cluster": lambda: min_var_group_constrained(
                sigma, factor_groups, w_max=w_max, group_cap=group_cap
            ),
        }

        if "min_var_corr_cluster" in requested:
            try:
                corr_groups = _lookup_groups(
                    correlation_clusters, window.label, universe, "correlation_cluster"
                )

                def _corr_solver(sigma=sigma, corr_groups=corr_groups):
                    return min_var_group_constrained(
                        sigma, corr_groups, w_max=w_max, group_cap=group_cap
                    )

                portfolio_solvers["min_var_corr_cluster"] = _corr_solver
            except ValueError as error:
                LOGGER.warning(
                    "Prozor %s: oznake korelacijskih klastera nedostupne (%s); "
                    "preskačem min_var_corr_cluster.",
                    window.label,
                    error,
                )
                status_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": "min_var_corr_cluster",
                        "n_assets": len(universe),
                        "status": f"skipped: {error}",
                    }
                )

        portfolio_solvers = {
            name: solver
            for name, solver in portfolio_solvers.items()
            if name in requested
        }

        for portfolio_name, solver in portfolio_solvers.items():
            weights, status = _solve_safely(portfolio_name, window.label, solver)
            status_rows.append(
                {
                    "train_window": window.label,
                    "portfolio": portfolio_name,
                    "n_assets": len(universe),
                    "status": status,
                }
            )
            if weights is None:
                continue

            weights = weights.reindex(universe).fillna(0.0)
            for ticker, weight in weights.items():
                weights_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": portfolio_name,
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )

            test_slice = test_returns.loc[:, weights.index].fillna(0.0)
            port_return = test_slice.dot(weights)
            returns_rows.append(
                port_return.to_frame(portfolio_name)
                .assign(train_window=window.label)
                .reset_index()
            )

    weights_panel = pd.DataFrame(
        weights_rows,
        columns=["train_window", "portfolio", "ticker", "weight"],
    )
    portfolio_status = pd.DataFrame(
        status_rows,
        columns=["train_window", "portfolio", "n_assets", "status"],
    )

    if returns_rows:
        long_returns = pd.concat(returns_rows, ignore_index=True)
        date_column = (
            "date"
            if "date" in long_returns.columns
            else long_returns.columns[0]
        )
        long_returns = long_returns.rename(columns={date_column: "date"})
        return_long_frames: list[pd.DataFrame] = []
        for portfolio_name in PORTFOLIO_NAMES:
            if portfolio_name not in long_returns.columns:
                continue
            piece = (
                long_returns[["date", "train_window", portfolio_name]]
                .dropna(subset=[portfolio_name])
                .rename(columns={portfolio_name: "monthly_return"})
                .assign(portfolio=portfolio_name)
            )
            return_long_frames.append(piece)
        long_panel = pd.concat(return_long_frames, ignore_index=True)
        port_returns_panel = (
            long_panel.pivot_table(
                index="date",
                columns="portfolio",
                values="monthly_return",
                aggfunc="first",
            )
            .sort_index()
        )
        port_returns_panel.columns.name = "portfolio"
        port_returns_panel = port_returns_panel.reindex(
            columns=[name for name in PORTFOLIO_NAMES if name in port_returns_panel.columns]
        )
    else:
        port_returns_panel = pd.DataFrame()

    return {
        "weights_panel": weights_panel,
        "port_returns_panel": port_returns_panel,
        "portfolio_status": portfolio_status,
    }


__all__ = [
    "PORTFOLIO_NAMES",
    "HIERARCHICAL_PORTFOLIO_NAMES",
    "RollingWindow",
    "generate_rolling_windows",
    "run_walk_forward",
    "run_hierarchical_walk_forward",
    "run_factor_neutral_sweep",
]


# ---------------------------------------------------------------------------
# Unaprijedni prelet faktorske neutralnosti
# ---------------------------------------------------------------------------


STYLE_FACTOR_COLUMNS = ("beta_smb", "beta_hml", "beta_rmw", "beta_cma")


def run_factor_neutral_sweep(
    monthly_returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    factor_clusters: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: Iterable[RollingWindow],
    epsilons: Sequence[float] = (0.0, 0.05, 0.10, 0.15),
    style_factor_columns: Sequence[str] = STYLE_FACTOR_COLUMNS,
    include_hybrid: bool = True,
    cov_estimator: str | Callable[[pd.DataFrame], pd.DataFrame] = "ledoit_wolf",
    w_max: float = W_MAX,
    group_cap: float = GROUP_CAP,
    min_training_months: int = MIN_TRAINING_MONTHS,
    membership: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Pokreni dodatne unaprijedne portfelje s izravnom faktorskom neutralnošću.

    ``membership`` uključuje point-in-time presjek univerzuma na članove S&P 500
    na ``train_end`` (PROJECT_SPEC §3.1); kad je ``None``, presjek se preskače.

    Za svaki ``ε`` u ``epsilons``:

    - ``factor_neutral_e{ε}`` — min. varijanca s ``|w'β_f| ≤ ε`` za svaki
      ``f`` u ``style_factor_columns``.
    - ``factor_cluster_neutral_e{ε}`` (ako je ``include_hybrid``) — dodaje
      ograničenje težine po klasteru povrh ograničenja faktorske neutralnosti.

    Univerzum treniranja po prozoru presjek je:
      - oznaka dionica s barem ``min_training_months`` nenedostajućih prinosa
        treniranja;
      - oznaka s petfaktorskim izloženostima procijenjenima za taj prozor;
      - oznaka s oznakama faktorskih klastera za taj prozor (za hibrid).

    Težine svakog portfelja primjenjuju se uz mjesečno rebalansiranje na tih
    12 prinosa testnog razdoblja prozora. Panel prinosa je širok:
    ``(date × portfelj)``.
    """
    cov_fn = _resolve_cov_estimator(cov_estimator)
    membership = _prepare_membership(membership)
    sector_map = metadata.set_index("ticker")["sector"]

    weights_rows: list[dict[str, object]] = []
    long_return_pieces: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for window in windows:
        train_returns = monthly_returns.loc[window.train_start : window.train_end]
        valid_counts = train_returns.notna().sum(axis=0)
        eligible_tickers = valid_counts[valid_counts >= min_training_months].index

        window_betas = factor_exposures.loc[
            factor_exposures["train_window"] == window.label
        ].set_index("ticker")
        beta_tickers = pd.Index(window_betas.index)
        cluster_subset = factor_clusters.loc[
            factor_clusters["train_window"] == window.label
        ].set_index("ticker")["factor_cluster"]
        cluster_tickers = pd.Index(cluster_subset.index)

        universe = (
            pd.Index(eligible_tickers)
            .intersection(beta_tickers)
            .intersection(cluster_tickers)
            .intersection(sector_map.dropna().index)
        )
        universe = _restrict_to_members(universe, membership, window.train_end)
        if len(universe) < 5:
            LOGGER.warning(
                "Prozor %s: samo %d prihvatljivih oznaka dionica; preskačem prelet faktorske neutralnosti.",
                window.label,
                len(universe),
            )
            continue

        train_panel = train_returns.loc[:, universe].dropna(axis=0, how="any")
        if len(train_panel) < min_training_months:
            LOGGER.warning(
                "Prozor %s: %d potpunih redaka treniranja nakon presjeka; preskačem.",
                window.label,
                len(train_panel),
            )
            continue

        try:
            sigma = cov_fn(train_panel)
        except Exception as error:
            LOGGER.error(
                "Kovarijanca nije uspjela u prozoru %s: %s", window.label, error
            )
            continue
        sigma = sigma.loc[universe, universe]

        style_betas = window_betas.loc[universe, list(style_factor_columns)]
        cluster_groups = cluster_subset.loc[universe].astype(str)

        test_returns = monthly_returns.loc[
            window.test_start : window.test_end, universe
        ]

        for eps in epsilons:
            name = f"factor_neutral_e{eps:g}"
            try:
                weights = min_var_factor_neutral(
                    sigma, style_betas, w_max=w_max, max_abs_beta=eps
                )
                status = "ok"
            except Exception as error:
                LOGGER.warning(
                    "Portfolio %s failed in window %s: %s", name, window.label, error
                )
                weights, status = None, f"failed: {type(error).__name__}: {error}"
            status_rows.append(
                {
                    "train_window": window.label,
                    "portfolio": name,
                    "epsilon": float(eps),
                    "kind": "factor_neutral",
                    "n_assets": len(universe),
                    "status": status,
                }
            )
            if weights is None:
                continue
            weights = weights.reindex(universe).fillna(0.0)
            for ticker, weight in weights.items():
                weights_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": name,
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )
            test_slice = test_returns.loc[:, weights.index].fillna(0.0)
            long_return_pieces.append(
                test_slice.dot(weights).to_frame(name).reset_index()
            )

            if include_hybrid:
                hybrid_name = f"factor_cluster_neutral_e{eps:g}"
                try:
                    hybrid_weights = min_var_group_and_factor_neutral(
                        sigma,
                        cluster_groups,
                        style_betas,
                        w_max=w_max,
                        group_cap=group_cap,
                        max_abs_beta=eps,
                    )
                    hybrid_status = "ok"
                except Exception as error:
                    LOGGER.warning(
                        "Portfolio %s failed in window %s: %s",
                        hybrid_name,
                        window.label,
                        error,
                    )
                    hybrid_weights, hybrid_status = (
                        None,
                        f"failed: {type(error).__name__}: {error}",
                    )
                status_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": hybrid_name,
                        "epsilon": float(eps),
                        "kind": "factor_cluster_neutral",
                        "n_assets": len(universe),
                        "status": hybrid_status,
                    }
                )
                if hybrid_weights is None:
                    continue
                hybrid_weights = hybrid_weights.reindex(universe).fillna(0.0)
                for ticker, weight in hybrid_weights.items():
                    weights_rows.append(
                        {
                            "train_window": window.label,
                            "portfolio": hybrid_name,
                            "ticker": ticker,
                            "weight": float(weight),
                        }
                    )
                test_slice_h = test_returns.loc[:, hybrid_weights.index].fillna(0.0)
                long_return_pieces.append(
                    test_slice_h.dot(hybrid_weights)
                    .to_frame(hybrid_name)
                    .reset_index()
                )

    weights_panel = pd.DataFrame(
        weights_rows, columns=["train_window", "portfolio", "ticker", "weight"]
    )
    status_panel = pd.DataFrame(status_rows)

    if long_return_pieces:
        wide_pieces: dict[str, pd.Series] = {}
        for piece in long_return_pieces:
            piece = piece.copy()
            date_column = (
                "date" if "date" in piece.columns else piece.columns[0]
            )
            piece = piece.rename(columns={date_column: "date"}).set_index("date")
            value_column = piece.columns[0]
            existing = wide_pieces.get(value_column)
            new_series = piece[value_column].dropna()
            if existing is None:
                wide_pieces[value_column] = new_series
            else:
                wide_pieces[value_column] = pd.concat(
                    [existing, new_series]
                ).sort_index()
        port_returns_panel = pd.DataFrame(wide_pieces).sort_index()
        port_returns_panel.index.name = "date"
    else:
        port_returns_panel = pd.DataFrame()

    return {
        "weights_panel": weights_panel,
        "port_returns_panel": port_returns_panel,
        "portfolio_status": status_panel,
    }


# ---------------------------------------------------------------------------
# Unaprijedni hod hijerarhijskih alokatora (F1.5)
# ---------------------------------------------------------------------------


HIERARCHICAL_PORTFOLIO_NAMES = (
    "hrp_corr_single",
    "hrp_corr_ward",
    "herc_corr",
    "nco_corr",
)


def run_hierarchical_walk_forward(
    monthly_returns: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    factor_clusters: pd.DataFrame,
    correlation_clusters: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: Iterable[RollingWindow],
    k: int,
    tree_space: str = "correlation",
    cov_estimator: str | Callable[[pd.DataFrame], pd.DataFrame] = "ledoit_wolf",
    w_max: float = W_MAX,
    min_training_months: int = MIN_TRAINING_MONTHS,
    membership: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Pokreni backtest s kliznim prozorom za hijerarhijske alokatore (F1.5).

    Po uzoru na :func:`run_walk_forward`: **isti presjek univerzuma**, **ista
    Ledoit–Wolf Σ** (računa se jednom po prozoru i dijeli među alokatorima) i
    **isti format izlaza** (``weights_panel``, ``port_returns_panel``,
    ``portfolio_status``). Stablo/particiju gradi :mod:`src.hierarchical` iz
    prozora treniranja, pa su jedina razlika prema benchmarcima same težine.

    ``tree_space="correlation"`` (Faza 1) gradi korelacijska stabla iz F1.1b i
    pokreće četiri alokatora: ``hrp_corr_single`` (vjerna replikacija López de
    Prada 2016, jednostruka veza), ``hrp_corr_ward`` (krak kontrolirane
    usporedbe, K1, Wardova veza), ``herc_corr`` (Wardova veza, rez na ``k``) i
    ``nco_corr`` (Wardova veza, particija rezom na ``k``). ``tree_space="factor"``
    je predviđen za Fazu 3 (F3.1) i ovdje još nije implementiran.

    ``k`` je broj klastera za HERC i NCO iz primarne procedure odabira K (F1.0).
    Status tablica dobiva stupac ``capped_weight_share`` po (portfelj, prozor)
    (riješeno pitanje 1; ``NaN`` za neuspjele prozore).
    """
    if tree_space not in {"correlation", "factor"}:
        raise ValueError(
            f"tree_space mora biti 'correlation' ili 'factor', dobiveno {tree_space!r}."
        )
    if tree_space == "factor":
        raise NotImplementedError(
            "tree_space='factor' implementira se u Fazi 3 (F3.1); F1.5 pokriva "
            "samo korelacijski prostor."
        )
    if not isinstance(k, int) or k < 1:
        raise ValueError("k mora biti pozitivan cijeli broj.")

    cov_fn = _resolve_cov_estimator(cov_estimator)
    membership = _prepare_membership(membership)
    sector_map = metadata.set_index("ticker")["sector"]

    weights_rows: list[dict[str, object]] = []
    returns_pieces: dict[str, list[pd.Series]] = {}
    status_rows: list[dict[str, object]] = []

    for window in windows:
        train_returns = monthly_returns.loc[window.train_start : window.train_end]
        valid_counts = train_returns.notna().sum(axis=0)
        eligible_tickers = valid_counts[valid_counts >= min_training_months].index

        factor_tickers = pd.Index(
            factor_exposures.loc[
                factor_exposures["train_window"] == window.label, "ticker"
            ]
        )
        factor_cluster_tickers = pd.Index(
            factor_clusters.loc[
                factor_clusters["train_window"] == window.label, "ticker"
            ]
        )
        correlation_cluster_tickers = pd.Index(
            correlation_clusters.loc[
                correlation_clusters["train_window"] == window.label, "ticker"
            ]
        )

        universe = (
            pd.Index(eligible_tickers)
            .intersection(factor_tickers)
            .intersection(factor_cluster_tickers)
            .intersection(correlation_cluster_tickers)
            .intersection(sector_map.dropna().index)
        )
        universe = _restrict_to_members(universe, membership, window.train_end)
        if len(universe) < 5:
            LOGGER.warning(
                "Prozor %s ima samo %d prihvatljivih oznaka dionica; preskačem.",
                window.label,
                len(universe),
            )
            continue

        train_panel = train_returns.loc[:, universe].dropna(axis=0, how="any")
        if len(train_panel) < min_training_months:
            LOGGER.warning(
                "Prozor %s ima %d potpunih redaka treniranja nakon presjeka; "
                "preskačem.",
                window.label,
                len(train_panel),
            )
            continue

        try:
            sigma = cov_fn(train_panel)
        except Exception as error:
            LOGGER.error(
                "Procjenitelj kovarijance nije uspio u prozoru %s: %s",
                window.label,
                error,
            )
            continue
        sigma = sigma.loc[universe, universe]

        if k > len(universe):
            LOGGER.warning(
                "Prozor %s: k=%d > broja imovina %d; preskačem.",
                window.label,
                k,
                len(universe),
            )
            continue

        # Stabla se grade jednom po prozoru i dijele među alokatorima.
        train_corr = train_panel.corr()
        single_linkage = build_correlation_tree(train_corr, linkage="single")
        ward_linkage = build_correlation_tree(train_corr, linkage="ward")

        def _hrp_solver(linkage_matrix):
            order = quasi_diagonal_order(linkage_matrix)
            return apply_w_max(hrp_weights(sigma, order), w_max)

        portfolio_solvers: dict[str, Callable[[], tuple[pd.Series, float]]] = {
            "hrp_corr_single": lambda: _hrp_solver(single_linkage),
            "hrp_corr_ward": lambda: _hrp_solver(ward_linkage),
            "herc_corr": lambda: herc_weights(sigma, ward_linkage, k, w_max=w_max),
            "nco_corr": lambda: nco_weights(
                sigma,
                fcluster(ward_linkage, t=k, criterion="maxclust"),
                w_max=w_max,
            ),
        }

        test_returns = monthly_returns.loc[
            window.test_start : window.test_end, universe
        ]

        for portfolio_name, solver in portfolio_solvers.items():
            try:
                weights, capped_share = solver()
                status = "ok"
            except Exception as error:
                LOGGER.warning(
                    "Portfelj %s nije uspio u prozoru %s: %s",
                    portfolio_name,
                    window.label,
                    error,
                )
                weights, capped_share, status = (
                    None,
                    float("nan"),
                    f"failed: {type(error).__name__}: {error}",
                )
            status_rows.append(
                {
                    "train_window": window.label,
                    "portfolio": portfolio_name,
                    "n_assets": len(universe),
                    "status": status,
                    "capped_weight_share": float(capped_share),
                }
            )
            if weights is None:
                continue

            weights = weights.reindex(universe).fillna(0.0)
            for ticker, weight in weights.items():
                weights_rows.append(
                    {
                        "train_window": window.label,
                        "portfolio": portfolio_name,
                        "ticker": ticker,
                        "weight": float(weight),
                    }
                )

            test_slice = test_returns.loc[:, weights.index].fillna(0.0)
            returns_pieces.setdefault(portfolio_name, []).append(
                test_slice.dot(weights)
            )

    weights_panel = pd.DataFrame(
        weights_rows,
        columns=["train_window", "portfolio", "ticker", "weight"],
    )
    portfolio_status = pd.DataFrame(
        status_rows,
        columns=[
            "train_window",
            "portfolio",
            "n_assets",
            "status",
            "capped_weight_share",
        ],
    )

    if returns_pieces:
        columns: dict[str, pd.Series] = {}
        for name in HIERARCHICAL_PORTFOLIO_NAMES:
            if name in returns_pieces:
                columns[name] = pd.concat(returns_pieces[name]).sort_index()
        port_returns_panel = pd.DataFrame(columns).sort_index()
        port_returns_panel.index.name = "date"
        port_returns_panel.columns.name = "portfolio"
    else:
        port_returns_panel = pd.DataFrame()

    return {
        "weights_panel": weights_panel,
        "port_returns_panel": port_returns_panel,
        "portfolio_status": portfolio_status,
    }

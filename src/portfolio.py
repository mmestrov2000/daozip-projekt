"""Pomoćnici za konstrukciju portfelja za fazu 4."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.utils import GROUP_CAP, W_MAX


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
SOLVER_CANDIDATES = ("CLARABEL", "OSQP", "SCS")


def _as_returns_frame(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        frame = pd.DataFrame(returns)

    if frame.empty:
        raise ValueError("returns ne smije biti prazan.")
    if frame.columns.has_duplicates:
        raise ValueError("stupci returns moraju biti jedinstveni.")
    if frame.shape[0] < 2:
        raise ValueError("Potrebne su barem dvije opservacije prinosa.")

    frame = frame.apply(pd.to_numeric, errors="raise")
    frame = frame.dropna(axis=0, how="any")
    if frame.shape[0] < 2:
        raise ValueError("Potrebne su barem dvije potpune opservacije prinosa.")
    return frame


def _as_covariance_matrix(
    sigma: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(sigma, pd.DataFrame):
        if sigma.empty:
            raise ValueError("Sigma ne smije biti prazna.")
        if sigma.index.has_duplicates or sigma.columns.has_duplicates:
            raise ValueError("Indeks i stupci Sigme moraju biti jedinstveni.")
        if not sigma.index.equals(sigma.columns):
            raise ValueError("Indeks i stupci Sigma DataFramea moraju se podudarati.")
        labels: pd.Index | None = sigma.index
        matrix = sigma.to_numpy(dtype=float)
    else:
        labels = None
        matrix = np.asarray(sigma, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Sigma mora biti kvadratna kovarijacijska matrica.")
    if matrix.shape[0] == 0:
        raise ValueError("Sigma mora sadržavati barem jednu imovinu.")
    if not np.isfinite(matrix).all():
        raise ValueError("Sigma sadrži nekonačne vrijednosti.")

    matrix = (matrix + matrix.T) / 2.0
    return matrix, labels


def _validate_weight_bounds(n_assets: int, w_max: float) -> None:
    if not np.isfinite(w_max) or w_max <= 0:
        raise ValueError("w_max mora biti pozitivna konačna vrijednost.")
    if w_max > 1:
        raise ValueError("w_max ne smije premašiti 1 za potpuno investiran portfelj.")
    if n_assets * w_max < 1 - 1e-10:
        raise ValueError(
            f"Potpuno investiranje sa samo dugim pozicijama je neizvedivo: {n_assets} imovina * "
            f"w_max={w_max:.6f} manje je od 1."
        )


def _weights_result(
    values: np.ndarray,
    labels: pd.Index | None,
) -> pd.Series | np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = clipped.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Optimizator je vratio nevaljane težine.")
    normalized = clipped / total

    if labels is None:
        return normalized
    return pd.Series(normalized, index=labels, name="weight")


def _solve_min_variance(
    sigma: np.ndarray,
    w_max: float,
    constraint_builder: Callable[[cp.Variable], Sequence[cp.Constraint]] | None = None,
) -> np.ndarray:
    n_assets = sigma.shape[0]
    _validate_weight_bounds(n_assets, w_max)

    weights = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(sigma)))
    constraints: list[cp.Constraint] = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= w_max,
    ]
    if constraint_builder is not None:
        constraints.extend(constraint_builder(weights))
    problem = cp.Problem(objective, constraints)

    installed_solvers = set(cp.installed_solvers())
    last_error: Exception | None = None
    for solver in SOLVER_CANDIDATES:
        if solver not in installed_solvers:
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError as error:
            last_error = error
            continue
        if problem.status in OPTIMAL_STATUSES and weights.value is not None:
            return np.asarray(weights.value, dtype=float)

    status = problem.status
    if last_error is not None and status is None:
        raise RuntimeError(
            "Nijedan cvxpy rješavač nije mogao riješiti problem minimalne varijance."
        ) from last_error
    raise RuntimeError(f"Optimizacija minimalne varijance nije uspjela sa statusom {status!r}.")


def sample_cov(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Vrati anualiziranu uzoračku kovarijancu iz mjesečnih prinosa.

    Projekt koristi mjesečne prinose, pa se uzoračka kovarijanca množi s
    12 radi anualizacije.
    """
    frame = _as_returns_frame(returns)
    return frame.cov() * 12.0


def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Anualizirana Ledoit-Wolf sažeta kovarijanca iz mjesečnih prinosa.

    Omata ``sklearn.covariance.LedoitWolf`` (koji cilja skaliranu jediničnu
    metu sažimanja). Rezultat je simetričan, pozitivno semidefinitan i
    anualiziran s ``× 12``. Vraća DataFrame indeksiran i stupčan po
    oznaci dionice kad je ``returns`` DataFrame.
    """
    frame = _as_returns_frame(returns)
    estimator = LedoitWolf().fit(frame.to_numpy(dtype=float))
    cov = estimator.covariance_ * 12.0
    cov = (cov + cov.T) / 2.0
    return pd.DataFrame(cov, index=frame.columns, columns=frame.columns)


def equal_weight(
    n: int,
    index: Sequence[object] | pd.Index | None = None,
) -> pd.Series | np.ndarray:
    """Vrati potpuno investiran portfelj jednakih težina."""
    if not isinstance(n, int):
        raise TypeError("n mora biti cijeli broj.")
    if n <= 0:
        raise ValueError("n mora biti pozitivan.")

    weights = np.repeat(1.0 / n, n)
    if index is None:
        return weights

    index = pd.Index(index)
    if len(index) != n:
        raise ValueError("duljina index mora odgovarati n.")
    return pd.Series(weights, index=index, name="weight")


def min_variance(
    Sigma: pd.DataFrame | np.ndarray,
    w_max: float = W_MAX,
) -> pd.Series | np.ndarray:
    """Riješi potpuno investiran portfelj minimalne varijance sa samo dugim pozicijama."""
    sigma, labels = _as_covariance_matrix(Sigma)
    weights = _solve_min_variance(sigma=sigma, w_max=w_max)
    return _weights_result(weights, labels)


def min_var_group_constrained(
    Sigma: pd.DataFrame | np.ndarray,
    groups: Sequence[object] | pd.Series | np.ndarray,
    w_max: float = W_MAX,
    group_cap: float = GROUP_CAP,
) -> pd.Series | np.ndarray:
    """Riješi portfelj minimalne varijance s ograničenjima težine po grupi.

    ``groups`` mora sadržavati jednu grupnu oznaku po imovini, istim redoslijedom kao
    ``Sigma``. Oznake mogu biti cijeli brojevi, nizovi ili bilo koje s pandasom kompatibilne vrijednosti.
    """
    sigma, labels = _as_covariance_matrix(Sigma)
    n_assets = sigma.shape[0]
    _validate_weight_bounds(n_assets, w_max)

    if not np.isfinite(group_cap) or group_cap <= 0:
        raise ValueError("group_cap mora biti pozitivna konačna vrijednost.")
    if group_cap > 1:
        raise ValueError("group_cap ne smije premašiti 1 za potpuno investiran portfelj.")

    if isinstance(groups, pd.Series):
        group_series = groups.copy()
        if labels is not None:
            missing = labels.difference(group_series.index)
            if len(missing) > 0:
                raise ValueError(f"Nedostaju grupne oznake za imovine: {missing.tolist()}")
            group_series = group_series.loc[labels]
        elif len(group_series) != n_assets:
            raise ValueError("duljina groups mora odgovarati dimenzijama Sigme.")
    else:
        if len(groups) != n_assets:
            raise ValueError("duljina groups mora odgovarati dimenzijama Sigme.")
        group_series = pd.Series(groups, index=labels if labels is not None else None)

    if group_series.isna().any():
        raise ValueError("groups sadrži nedostajuće oznake.")

    group_capacity = group_series.groupby(group_series, sort=False).size().mul(w_max)
    effective_capacity = np.minimum(
        group_capacity.to_numpy(dtype=float),
        group_cap,
    ).sum()
    if effective_capacity < 1 - 1e-10:
        raise ValueError(
            "Grupna ograničenja su neizvediva: kombinirani ograničeni kapacitet grupa "
            f"je {effective_capacity:.6f}, ispod 1."
        )

    def group_constraint_builder(variable: cp.Variable) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for group in pd.unique(group_series):
            member_mask = (group_series.to_numpy() == group).astype(float)
            constraints.append(member_mask @ variable <= group_cap)
        return constraints

    weights = _solve_min_variance(
        sigma=sigma,
        w_max=w_max,
        constraint_builder=group_constraint_builder,
    )
    return _weights_result(weights, labels)


def _align_factor_betas(
    factor_betas: pd.DataFrame,
    labels: pd.Index | None,
    n_assets: int,
) -> tuple[np.ndarray, list[str]]:
    """Poravnaj DataFrame beta (ticker × faktor) s poretkom Sigme.

    Vraća ``(beta_matrix, factor_names)`` gdje ``beta_matrix`` ima oblik
    ``(n_assets, n_factors)`` poredan da odgovara ``Sigmi``.
    """
    if not isinstance(factor_betas, pd.DataFrame):
        raise TypeError("factor_betas mora biti pandas DataFrame.")
    if factor_betas.empty:
        raise ValueError("factor_betas ne smije biti prazan.")
    if factor_betas.isna().any().any():
        raise ValueError("factor_betas sadrži nedostajuće vrijednosti.")
    if labels is not None:
        missing = labels.difference(factor_betas.index)
        if len(missing) > 0:
            raise ValueError(
                f"Nedostaju faktorske bete za imovine: {missing.tolist()}"
            )
        ordered = factor_betas.loc[labels]
    else:
        if len(factor_betas) != n_assets:
            raise ValueError(
                "duljina factor_betas mora odgovarati dimenzijama Sigme kad nema oznaka."
            )
        ordered = factor_betas
    return ordered.to_numpy(dtype=float), list(ordered.columns)


def min_var_factor_neutral(
    Sigma: pd.DataFrame | np.ndarray,
    factor_betas: pd.DataFrame,
    w_max: float = W_MAX,
    max_abs_beta: float = 0.10,
) -> pd.Series | np.ndarray:
    """Min. varijanca sa samo dugim pozicijama uz faktorske bete portfelja ograničene blizu nule.

    Dodaje linearna ograničenja ``|w' β_f| ≤ max_abs_beta`` za svaki faktorski
    stupac u ``factor_betas``. Koristi ovo za nametanje *izravne* faktorske
    neutralnosti na odabranom skupu faktora (tipično četiri Fama-French
    stilska faktora SMB, HML, RMW, CMA).

    ``factor_betas`` je DataFrame indeksiran po oznaci dionice s jednim stupcem po
    faktoru čiju neutralnost želiš nametnuti. Reci se poravnavaju sa
    ``Sigmom`` prije rješavanja.

    ``max_abs_beta`` je dopuštena veličina bete portfelja prema
    svakom faktoru. ``max_abs_beta = 0`` nameće točnu neutralnost; postavljanje
    blago pozitivne vrijednosti (npr. 0.05, 0.10) daje zazor i pomaže
    kvadratnom programu da ostane izvediv kad su bete treniranja šumovite.
    """
    sigma, labels = _as_covariance_matrix(Sigma)
    n_assets = sigma.shape[0]
    _validate_weight_bounds(n_assets, w_max)
    if not np.isfinite(max_abs_beta) or max_abs_beta < 0:
        raise ValueError("max_abs_beta mora biti nenegativna konačna vrijednost.")

    beta_matrix, _ = _align_factor_betas(factor_betas, labels, n_assets)

    def factor_neutrality_constraints(variable: cp.Variable) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for f_idx in range(beta_matrix.shape[1]):
            beta_col = beta_matrix[:, f_idx]
            constraints.append(beta_col @ variable <= max_abs_beta)
            constraints.append(beta_col @ variable >= -max_abs_beta)
        return constraints

    weights = _solve_min_variance(
        sigma=sigma,
        w_max=w_max,
        constraint_builder=factor_neutrality_constraints,
    )
    return _weights_result(weights, labels)


def min_var_group_and_factor_neutral(
    Sigma: pd.DataFrame | np.ndarray,
    groups: Sequence[object] | pd.Series | np.ndarray,
    factor_betas: pd.DataFrame,
    w_max: float = W_MAX,
    group_cap: float = GROUP_CAP,
    max_abs_beta: float = 0.10,
) -> pd.Series | np.ndarray:
    """Min. varijanca s ograničenjem klastera/sektora i faktorskom neutralnošću.

    Spaja :func:`min_var_group_constrained` i
    :func:`min_var_factor_neutral`: ograničenja težine po grupi i ograničenja
    bete portfelja po faktoru nameću se istodobno. Korisno za testiranje
    nadopunjuju li se dvije obitelji ograničenja informacijama.
    """
    sigma, labels = _as_covariance_matrix(Sigma)
    n_assets = sigma.shape[0]
    _validate_weight_bounds(n_assets, w_max)

    if not np.isfinite(group_cap) or group_cap <= 0:
        raise ValueError("group_cap mora biti pozitivna konačna vrijednost.")
    if group_cap > 1:
        raise ValueError("group_cap ne smije premašiti 1 za potpuno investiran portfelj.")
    if not np.isfinite(max_abs_beta) or max_abs_beta < 0:
        raise ValueError("max_abs_beta mora biti nenegativna konačna vrijednost.")

    if isinstance(groups, pd.Series):
        group_series = groups.copy()
        if labels is not None:
            missing = labels.difference(group_series.index)
            if len(missing) > 0:
                raise ValueError(f"Nedostaju grupne oznake za imovine: {missing.tolist()}")
            group_series = group_series.loc[labels]
        elif len(group_series) != n_assets:
            raise ValueError("duljina groups mora odgovarati dimenzijama Sigme.")
    else:
        if len(groups) != n_assets:
            raise ValueError("duljina groups mora odgovarati dimenzijama Sigme.")
        group_series = pd.Series(groups, index=labels if labels is not None else None)
    if group_series.isna().any():
        raise ValueError("groups sadrži nedostajuće oznake.")

    group_capacity = group_series.groupby(group_series, sort=False).size().mul(w_max)
    effective_capacity = np.minimum(
        group_capacity.to_numpy(dtype=float),
        group_cap,
    ).sum()
    if effective_capacity < 1 - 1e-10:
        raise ValueError(
            "Grupna ograničenja su neizvediva: kombinirani ograničeni kapacitet grupa "
            f"je {effective_capacity:.6f}, ispod 1."
        )

    beta_matrix, _ = _align_factor_betas(factor_betas, labels, n_assets)

    def constraints_builder(variable: cp.Variable) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for group in pd.unique(group_series):
            member_mask = (group_series.to_numpy() == group).astype(float)
            constraints.append(member_mask @ variable <= group_cap)
        for f_idx in range(beta_matrix.shape[1]):
            beta_col = beta_matrix[:, f_idx]
            constraints.append(beta_col @ variable <= max_abs_beta)
            constraints.append(beta_col @ variable >= -max_abs_beta)
        return constraints

    weights = _solve_min_variance(
        sigma=sigma,
        w_max=w_max,
        constraint_builder=constraints_builder,
    )
    return _weights_result(weights, labels)

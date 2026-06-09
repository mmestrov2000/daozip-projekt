"""Procjena Fama-French faktorske izloženosti, po prozoru treniranja."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils import MIN_TRAINING_MONTHS


FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
BETA_COLUMN_MAP = {
    "Mkt-RF": "beta_mkt",
    "SMB": "beta_smb",
    "HML": "beta_hml",
    "RMW": "beta_rmw",
    "CMA": "beta_cma",
}
OUTPUT_COLUMNS = [
    "train_window",
    "ticker",
    "alpha",
    "beta_mkt",
    "beta_smb",
    "beta_hml",
    "beta_rmw",
    "beta_cma",
    "residual_vol",
    "r_squared",
    "n_obs",
]


def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp(how="end").normalize()


def _window_label(train_end: pd.Timestamp) -> str:
    """Označi prozor treniranja njegovim uključivim završnim mjesecom, npr. ``2009-12``."""
    return pd.Timestamp(train_end).strftime("%Y-%m")


def estimate_betas_window(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    min_months: int = MIN_TRAINING_MONTHS,
    max_abs_beta: float = 5.0,
    max_residual_vol: float = 0.5,
    min_r_squared: float = 0.05,
) -> pd.DataFrame:
    """Procijeni petfaktorske izloženosti po dionici na jednom prozoru treniranja.

    Oznaka dionice uključuje se samo ako ima barem ``min_months`` zajednički
    nenedostajućih opažanja viškova prinosa i faktora unutar
    uključivog prozora ``[train_start, train_end]``. Vraća jedan redak po
    kvalificiranoj oznaci s alfom, pet beta, rezidualnom volatilnošću,
    R² i brojem opažanja.

    Filtri kvalitete podataka (primijenjeni nakon prilagodbe):
    - ``max_abs_beta`` — izbaci oznaku ako bilo koja faktorska beta po
      apsolutnoj vrijednosti prelazi taj prag. Stvarne faktorske bete žive otprilike
      u ``[-3, 3]``; sve izvan ±5 pretežno je znak da
      oznaka ima lošu povijest prinosa (prije IPO-a, ponovna uporaba oznake nakon spajanja,
      ustajali Yahoo podaci), a njezino uključivanje proizvodi stršeće vrijednosti u klasteriranju.
    - ``max_residual_vol`` — izbaci oznake čija mjesečna rezidualna standardna
      devijacija prelazi tu razinu (50 % mjesečno = po konstrukciji apsurdno).
    - ``min_r_squared`` — izbaci oznake čiji faktorski model objašnjava manje
      od tog udjela varijance; bete tih oznaka pod su
      dominacijom šuma i štetne za klasteriranje.

    Proslijedi bilo koji od parametara filtra kao ``float('inf')`` (ili 0 za
    ``min_r_squared``) kako bi onemogućio taj filtar.
    """
    missing_factors = [col for col in FACTOR_COLUMNS if col not in factors.columns]
    if missing_factors:
        raise ValueError(f"Nedostaju faktorski stupci: {missing_factors}")

    start, end = _month_end(train_start), _month_end(train_end)
    if start > end:
        raise ValueError("train_start mora biti na ili prije train_end.")

    returns_window = excess_returns.sort_index().loc[start:end]
    factors_window = factors[FACTOR_COLUMNS].sort_index().loc[start:end]
    if returns_window.empty or factors_window.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    label = _window_label(end)
    rows: list[dict[str, float | str]] = []
    for ticker in returns_window.columns:
        regression_data = pd.concat(
            [returns_window[ticker].rename("y"), factors_window],
            axis=1,
            join="inner",
        ).dropna()
        if len(regression_data) < min_months:
            continue
        y = regression_data["y"]
        x = sm.add_constant(regression_data[FACTOR_COLUMNS], has_constant="add")
        model = sm.OLS(y, x).fit()
        residual_vol = float(np.sqrt(model.mse_resid))
        r_squared = float(model.rsquared)
        betas = {
            output_column: float(model.params[factor_column])
            for factor_column, output_column in BETA_COLUMN_MAP.items()
        }
        if any(abs(value) > max_abs_beta for value in betas.values()):
            continue
        if residual_vol > max_residual_vol:
            continue
        if r_squared < min_r_squared:
            continue
        row: dict[str, float | str] = {
            "train_window": label,
            "ticker": ticker,
            "alpha": float(model.params["const"]),
            "residual_vol": residual_vol,
            "r_squared": r_squared,
            "n_obs": int(len(regression_data)),
        }
        row.update(betas)
        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def estimate_betas_all_windows(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    windows: Iterable[Sequence[str | pd.Timestamp]],
    min_months: int = MIN_TRAINING_MONTHS,
) -> pd.DataFrame:
    """Procijeni faktorske izloženosti za svaki klizni prozor treniranja.

    ``windows`` daje n-torke ``(train_start, train_end, test_start, test_end)``
    (dodatne vrijednosti se ignoriraju). Vraća spoj
    DataFrameova po prozoru s identifikatorom ``train_window``.
    """
    frames: list[pd.DataFrame] = []
    for window in windows:
        train_start, train_end = window[0], window[1]
        frame = estimate_betas_window(
            excess_returns=excess_returns,
            factors=factors,
            train_start=train_start,
            train_end=train_end,
            min_months=min_months,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, axis=0, ignore_index=True)


# Unatrag kompatibilan naziv koji koriste prijašnji notebookovi.
def estimate_betas(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: Sequence[str | pd.Timestamp] | None = None,
    min_months: int = MIN_TRAINING_MONTHS,
) -> pd.DataFrame:
    """Procijeni faktorske izloženosti na jednom prozoru (zastarjela ulazna točka)."""
    if window is None:
        raise ValueError(
            "Mora se proslijediti prozor (train_start, train_end); "
            "prijašnji zadani prozor treniranja obuhvaćao je cijeli projekt i "
            "uklonjen je u preradi s kliznim prozorom."
        )
    return estimate_betas_window(
        excess_returns=excess_returns,
        factors=factors,
        train_start=window[0],
        train_end=window[1],
        min_months=min_months,
    )

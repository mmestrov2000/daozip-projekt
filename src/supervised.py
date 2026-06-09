"""Nadzirano KDD proširenje — predviđanje ostvarenog pada (predavanja 3-4).

Ovo implementira opcionalno proširenje skicirano u ``PROJECT_SPEC.md`` §9 i
preslikava projekt na nadzirano gradivo kolegija:

- **Stabla odlučivanja** (predavanje 3): kriteriji podjele *entropija / porast
  informacije* u stilu C4.5 i *Gini* u stilu CART-a, putem ``DecisionTreeClassifier``.
- **Ansambli** (predavanje 4): bagging (``RandomForestClassifier``) i boosting
  (``GradientBoostingClassifier``).

Zadatak: za svaku dionicu i svaku kliznu testnu godinu označi je ``1`` („visok
pad”) ako je njezin ostvareni 12-mjesečni pad gori od medijana presjeka te
godine, inače ``0``. Predvidi oznaku iz značajki dionice po prozoru
i ispitaj pomaže li dodavanje **oznake faktorskog klastera** (notebook 02) ili
**GICS sektora** nad sirovim faktorskim betama. Evaluacija je
``GroupKFold`` po testnoj godini, tako da nijedna godina ne curi između treniranja i validacije.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.utils import RANDOM_SEED

BETA_FEATURES = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "residual_vol"]


def _ticker_drawdown(returns: pd.Series, min_months: int = 6) -> float:
    """Maksimalni pad niza mjesečnih prinosa kroz testni prozor."""
    series = returns.dropna().astype(float)
    if len(series) < min_months:
        return float("nan")
    cumulative = (1.0 + series).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    return float(drawdown.min())


def build_drawdown_dataset(
    factor_exposures: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    factor_clusters: pd.DataFrame,
    metadata: pd.DataFrame,
    windows: Iterable,
) -> pd.DataFrame:
    """Sastavi nadziranu tablicu po (testna godina, dionica).

    Vraća jedan redak po (prozor, ticker) sa šest faktorskih značajki, oznakama GICS
    ``sector`` i ``factor_cluster`` (obje procijenjene na prozoru *treniranja*
    — bez gledanja unaprijed), ostvarenim ``drawdown`` kroz testni prozor i
    binarnim ``target`` (1 = gore od medijana presjeka te godine).
    """
    monthly_returns = monthly_returns.copy()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    sector_map = metadata.set_index("ticker")["sector"]

    frames: list[pd.DataFrame] = []
    for window in windows:
        label = window.label
        feats = (
            factor_exposures.loc[factor_exposures["train_window"] == label]
            .drop_duplicates("ticker")
            .set_index("ticker")[BETA_FEATURES]
        )
        clusters = (
            factor_clusters.loc[factor_clusters["train_window"] == label]
            .drop_duplicates("ticker")
            .set_index("ticker")["factor_cluster"]
        )
        if feats.empty:
            continue
        test_slice = monthly_returns.loc[window.test_start : window.test_end]
        drawdowns = {t: _ticker_drawdown(test_slice[t]) for t in feats.index if t in test_slice.columns}
        dd = pd.Series(drawdowns, name="drawdown").dropna()
        if dd.empty:
            continue
        rows = feats.loc[dd.index].copy()
        rows["sector"] = sector_map.reindex(dd.index).to_numpy()
        rows["factor_cluster"] = clusters.reindex(dd.index).to_numpy()
        rows["drawdown"] = dd
        rows["test_year"] = int(pd.Timestamp(window.test_start).year)
        rows["ticker"] = dd.index
        # Binariziraj na medijanu presjeka OVE testne godine.
        rows["target"] = (rows["drawdown"] < rows["drawdown"].median()).astype(int)
        frames.append(rows.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    return data.dropna(subset=["sector", "factor_cluster"]).reset_index(drop=True)


def _design_matrix(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Izgradi X za jedan od {'betas', 'betas+sector', 'betas+cluster'}."""
    X = data[BETA_FEATURES].copy()
    if feature_set == "betas":
        return X
    if feature_set == "betas+sector":
        return pd.concat([X, pd.get_dummies(data["sector"], prefix="sec")], axis=1)
    if feature_set == "betas+cluster":
        return pd.concat([X, pd.get_dummies(data["factor_cluster"].astype(int), prefix="clu")], axis=1)
    raise ValueError(f"Nepoznat feature_set {feature_set!r}")


def evaluate_feature_sets(
    data: pd.DataFrame,
    feature_sets: Sequence[str] = ("betas", "betas+sector", "betas+cluster"),
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """GroupKFold (po testnoj godini) CV točnost i ROC-AUC po modelu × skupu značajki.

    Modeli: stablo odlučivanja (entropija i Gini, ``max_depth=3``), slučajna šuma
    (bagging), gradijentni boosting. Vraća jedan redak po (model, feature_set) sa
    srednjom ± std točnošću i ROC-AUC kroz preklope.
    """
    groups = data["test_year"].to_numpy()
    y = data["target"].to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)

    models = {
        "tree_entropy(d3)": DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=seed),
        "tree_gini(d3)": DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1),
        "grad_boosting": GradientBoostingClassifier(random_state=seed),
    }

    rows: list[dict[str, float | str]] = []
    for fs in feature_sets:
        X = _design_matrix(data, fs).to_numpy(dtype=float)
        for name, model in models.items():
            acc = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="accuracy")
            auc = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="roc_auc")
            rows.append({
                "model": name, "feature_set": fs, "n_features": X.shape[1],
                "acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
                "auc_mean": float(auc.mean()), "auc_std": float(auc.std()),
            })
    return pd.DataFrame(rows)


def fit_interpretable_tree(
    data: pd.DataFrame,
    feature_set: str = "betas+cluster",
    max_depth: int = 3,
    criterion: str = "entropy",
    seed: int = RANDOM_SEED,
) -> tuple[DecisionTreeClassifier, list[str]]:
    """Prilagodi jedno plitko, čitljivo stablo na punom uzorku radi pregleda."""
    X = _design_matrix(data, feature_set)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=seed)
    model.fit(X.to_numpy(dtype=float), data["target"].to_numpy())
    return model, list(X.columns)


def feature_importances(
    data: pd.DataFrame,
    feature_set: str = "betas+cluster",
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """Važnosti po nečistoći iz slučajne šume (bagging) za jedan skup značajki."""
    X = _design_matrix(data, feature_set)
    model = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    model.fit(X.to_numpy(dtype=float), data["target"].to_numpy())
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)


__all__ = [
    "BETA_FEATURES",
    "build_drawdown_dataset",
    "evaluate_feature_sets",
    "fit_interpretable_tree",
    "feature_importances",
]

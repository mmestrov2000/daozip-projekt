"""Pomoćnici za hijerarhijsko klasteriranje i metrike usporedbe."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import MIN_TRAINING_MONTHS, N_BOOTSTRAP_CLUSTERS, RANDOM_SEED


LOGGER = logging.getLogger(__name__)

FACTOR_FEATURE_COLUMNS = [
    "beta_mkt",
    "beta_smb",
    "beta_hml",
    "beta_rmw",
    "beta_cma",
    "residual_vol",
]
SECTOR_LABEL_NAMES = {"sector", "sectors", "gics", "gics_sector", "gics sectors"}


def _as_numeric_frame(
    data: pd.DataFrame | np.ndarray,
    name: str,
    allow_na: bool = False,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    if frame.empty:
        raise ValueError(f"{name} ne smije biti prazan.")
    if frame.index.has_duplicates:
        raise ValueError(f"{name} indeks mora biti jedinstven.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} stupci moraju biti jedinstveni.")

    frame = frame.apply(pd.to_numeric, errors="raise")
    if not allow_na and frame.isna().any().any():
        raise ValueError(f"{name} sadrži nedostajuće vrijednosti.")
    return frame


def _validate_cluster_count(k: int, n_items: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k mora biti cijeli broj.")
    if k < 1:
        raise ValueError("k mora biti barem 1.")
    if k > n_items:
        raise ValueError(f"k={k} ne smije premašiti broj stavki ({n_items}).")
    if n_items < 2:
        raise ValueError("Za hijerarhijsko klasteriranje potrebne su barem dvije stavke.")


def _standardize(frame: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(frame.to_numpy(dtype=float))


def factor_cluster(
    features: pd.DataFrame | np.ndarray,
    k: int,
    linkage: str = "ward",
) -> tuple[pd.Series, np.ndarray]:
    """Klasteriraj dionice prema standardiziranim značajkama faktorske izloženosti.

    Parametri
    ----------
    features:
        Matrica s jednim retkom po dionici i značajkama faktorske izloženosti kao stupcima.
        Indeks DataFramea čuva se na vraćenom Series oznaka.
    k:
        Broj klastera koje treba odrezati iz hijerarhije.
    linkage:
        SciPy metoda povezivanja. Zadana u projektu je Wardova veza s
        euklidskom udaljenošću.

    Vraća
    -------
    tuple[pandas.Series, numpy.ndarray]
        Oznake klastera nazvane ``factor_cluster`` i SciPy matricu povezivanja.
    """
    feature_frame = _as_numeric_frame(features, "features")
    _validate_cluster_count(k, len(feature_frame))

    standardized = _standardize(feature_frame)
    linkage_matrix = scipy_linkage(
        standardized,
        method=linkage,
        metric="euclidean",
        optimal_ordering=True,
    )
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust").astype(int)
    return (
        pd.Series(labels, index=feature_frame.index, name="factor_cluster"),
        linkage_matrix,
    )


def correlation_cluster(
    returns: pd.DataFrame | np.ndarray,
    k: int,
    linkage: str = "complete",
) -> tuple[pd.Series, np.ndarray]:
    """Klasteriraj dionice prema korelacijama prinosa u razdoblju treniranja.

    Korelacijska udaljenost je ``sqrt(2 * (1 - rho))``. Zadana u projektu je
    potpuna veza: jednostruka/prosječna/centroidna veza lančaju se na prinosima dionica
    (dominantni tržišni mod proizvodi jedan golemi klaster i singletone),
    čineći dobivene skupine neupotrebljivima kao ograničenje potpuno
    investiranog portfelja. Potpuna veza valjana je na unaprijed izračunatoj matrici udaljenosti i
    proizvodi uravnotežene klastere. Ward se odbacuje jer zahtijeva euklidske
    koordinate umjesto unaprijed izračunate matrice udaljenosti.
    """
    if linkage.lower() == "ward":
        raise ValueError(
            "Wardova veza nije valjana za unaprijed izračunate korelacijske udaljenosti; "
            "koristi potpunu vezu za osnovicu projekta."
        )

    returns_frame = _as_numeric_frame(returns, "returns", allow_na=True)
    _validate_cluster_count(k, returns_frame.shape[1])

    usable_returns = returns_frame.dropna(axis=0, how="any")
    if len(usable_returns) < 2:
        raise ValueError("Potrebne su barem dvije potpune opservacije prinosa.")

    correlation = usable_returns.corr()
    if correlation.isna().any().any():
        raise ValueError("Korelacijska matrica prinosa sadrži nedostajuće vrijednosti.")

    correlation = correlation.clip(lower=-1.0, upper=1.0)
    distance = np.sqrt(2.0 * (1.0 - correlation))
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)

    condensed_distance = squareform(distance_values, checks=False)
    linkage_matrix = scipy_linkage(
        condensed_distance,
        method=linkage,
        optimal_ordering=True,
    )
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust").astype(int)
    return (
        pd.Series(labels, index=correlation.index, name="correlation_cluster"),
        linkage_matrix,
    )


def _coerce_labels(
    labels: pd.Series | Sequence[object] | np.ndarray,
    index: pd.Index,
    name: str,
) -> pd.Series:
    if isinstance(labels, pd.Series):
        series = labels.copy()
    else:
        if len(labels) != len(index):
            raise ValueError(
                f"Skup oznaka {name!r} ima {len(labels)} oznaka, ali očekuje se "
                f"{len(index)} stavki."
            )
        series = pd.Series(labels, index=index, name=name)

    if series.index.has_duplicates:
        raise ValueError(f"Indeks skupa oznaka {name!r} mora biti jedinstven.")
    return series.rename(name).dropna()


def _sector_reference_name(label_sets: Mapping[str, pd.Series]) -> str | None:
    for name in label_sets:
        if name.lower() in SECTOR_LABEL_NAMES:
            return name
    return None


def _purity(cluster_labels: pd.Series, class_labels: pd.Series) -> float:
    counts = (
        pd.DataFrame({"cluster": cluster_labels, "class": class_labels})
        .groupby(["cluster", "class"], observed=True)
        .size()
    )
    if counts.empty:
        return np.nan
    return float(counts.groupby(level=0).max().sum() / len(cluster_labels))


def _mean_within_cluster_correlation(
    labels: pd.Series,
    correlation: pd.DataFrame,
) -> float:
    cluster_means: list[float] = []
    cluster_pair_counts: list[int] = []

    for _, members in labels.groupby(labels, observed=True):
        tickers = members.index
        if len(tickers) < 2:
            continue

        sub_corr = correlation.loc[tickers, tickers].to_numpy()
        upper_triangle = sub_corr[np.triu_indices_from(sub_corr, k=1)]
        finite_values = upper_triangle[np.isfinite(upper_triangle)]
        if finite_values.size == 0:
            continue

        cluster_means.append(float(finite_values.mean()))
        cluster_pair_counts.append(int(finite_values.size))

    if not cluster_means:
        return np.nan
    return float(np.average(cluster_means, weights=cluster_pair_counts))


def compare_clusterings(
    label_sets: Mapping[str, pd.Series | Sequence[object] | np.ndarray],
    factor_matrix: pd.DataFrame | np.ndarray,
    returns: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Usporedi označavanja klastera ARI-jem, čistoćom, siluetom i korelacijom.

    ``label_sets`` treba sadržavati jedan unos po označavanju, primjerice
    ``{"factor": factor_labels, "correlation": corr_labels, "sector": sectors}``.
    Ako je prisutan skup oznaka nalik sektoru, sektorska čistoća izvještava se za
    ostale skupove oznaka.
    """
    if not label_sets:
        raise ValueError("Potreban je barem jedan skup oznaka.")

    factor_frame = _as_numeric_frame(factor_matrix, "factor_matrix")
    returns_frame = _as_numeric_frame(returns, "returns", allow_na=True)

    base_index = factor_frame.index.intersection(returns_frame.columns)
    if base_index.empty:
        raise ValueError("Nema zajedničkih oznaka između factor_matrix i returns.")

    coerced_labels: dict[str, pd.Series] = {}
    for name, labels in label_sets.items():
        series = _coerce_labels(labels, factor_frame.index, name)
        coerced_labels[name] = series
        base_index = base_index.intersection(series.index)

    ordered_index = pd.Index(
        [ticker for ticker in factor_frame.index if ticker in base_index]
    )
    if len(ordered_index) < 2:
        raise ValueError("Potrebne su barem dvije zajedničke označene oznake dionica.")

    aligned_factors = factor_frame.loc[ordered_index]
    aligned_returns = returns_frame.loc[:, ordered_index].dropna(axis=0, how="any")
    if len(aligned_returns) < 2:
        raise ValueError("Potrebne su barem dvije potpune opservacije prinosa.")
    aligned_labels = {
        name: series.loc[ordered_index] for name, series in coerced_labels.items()
    }

    rows: list[dict[str, float | int | str | None]] = []
    n_items = len(ordered_index)

    for left, right in combinations(aligned_labels, 2):
        rows.append(
            {
                "metric": "adjusted_rand_index",
                "label_set": left,
                "compared_to": right,
                "value": float(
                    adjusted_rand_score(aligned_labels[left], aligned_labels[right])
                ),
                "n_items": n_items,
                "n_clusters": np.nan,
            }
        )

    sector_name = _sector_reference_name(aligned_labels)
    if sector_name is not None:
        sector_labels = aligned_labels[sector_name]
        for name, labels in aligned_labels.items():
            if name == sector_name:
                continue
            rows.append(
                {
                    "metric": "sector_purity",
                    "label_set": name,
                    "compared_to": sector_name,
                    "value": _purity(labels, sector_labels),
                    "n_items": n_items,
                    "n_clusters": int(labels.nunique()),
                }
            )

    standardized_factors = _standardize(aligned_factors)
    return_correlation = aligned_returns.corr().clip(lower=-1.0, upper=1.0)

    for name, labels in aligned_labels.items():
        n_clusters = int(labels.nunique())
        silhouette = (
            float(silhouette_score(standardized_factors, labels))
            if 1 < n_clusters < n_items
            else np.nan
        )
        rows.extend(
            [
                {
                    "metric": "silhouette_factor_space",
                    "label_set": name,
                    "compared_to": None,
                    "value": silhouette,
                    "n_items": n_items,
                    "n_clusters": n_clusters,
                },
                {
                    "metric": "mean_within_cluster_return_correlation",
                    "label_set": name,
                    "compared_to": None,
                    "value": _mean_within_cluster_correlation(
                        labels, return_correlation
                    ),
                    "n_items": n_items,
                    "n_clusters": n_clusters,
                },
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "label_set",
            "compared_to",
            "value",
            "n_items",
            "n_clusters",
        ],
    )


# ---------------------------------------------------------------------------
# Orkestracija po prozoru
# ---------------------------------------------------------------------------


def _factor_features_for_window(
    factor_exposures: pd.DataFrame,
    train_window: str,
) -> pd.DataFrame:
    subset = factor_exposures.loc[
        factor_exposures["train_window"] == train_window
    ].copy()
    if subset.empty:
        raise ValueError(f"Nema faktorskih izloženosti za prozor {train_window!r}.")
    return subset.set_index("ticker")[FACTOR_FEATURE_COLUMNS]


def _training_returns_for_window(
    monthly_returns: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    tickers: Sequence[str],
    min_months: int,
) -> pd.DataFrame:
    panel = monthly_returns.loc[train_start:train_end, list(tickers)]
    eligible = panel.notna().sum(axis=0) >= min_months
    panel = panel.loc[:, eligible[eligible].index]
    return panel.dropna(axis=0, how="any")


def cluster_all_windows(
    factor_exposures: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    windows: Iterable,
    k: int,
    metadata: pd.DataFrame | None = None,
    min_training_months: int = MIN_TRAINING_MONTHS,
) -> dict[str, pd.DataFrame]:
    """Klasteriraj svaki prozor treniranja faktorskom i korelacijskom vezom.

    ``windows`` treba davati :class:`RollingWindow` objekte (ili bilo koji objekt
    s atributima ``train_start``, ``train_end``, ``label``). Vraća DataFrameove
    u dugom formatu ključane po ``train_window, ticker`` kako bi ih daljnji kod
    mogao spojiti s panelima težina i metapodacima.
    """
    sector_map = (
        metadata.set_index("ticker")["sector"]
        if metadata is not None and "sector" in metadata.columns
        else None
    )

    factor_rows: list[pd.DataFrame] = []
    correlation_rows: list[pd.DataFrame] = []

    for window in windows:
        label = window.label
        try:
            features = _factor_features_for_window(factor_exposures, label)
        except ValueError as error:
            LOGGER.warning("Prozor %s preskočen: %s", label, error)
            continue

        factor_labels, _ = factor_cluster(features, k=k)
        factor_frame = (
            factor_labels.rename("factor_cluster")
            .reset_index()
            .rename(columns={"index": "ticker"})
        )
        if "ticker" not in factor_frame.columns and "index" in factor_frame.columns:
            factor_frame = factor_frame.rename(columns={"index": "ticker"})
        factor_frame["train_window"] = label
        if sector_map is not None:
            factor_frame["sector"] = factor_frame["ticker"].map(sector_map)
        factor_rows.append(factor_frame)

        training_returns = _training_returns_for_window(
            monthly_returns,
            train_start=window.train_start,
            train_end=window.train_end,
            tickers=features.index,
            min_months=min_training_months,
        )
        if training_returns.shape[1] < k:
            LOGGER.warning(
                "Prozor %s: samo %d oznaka s punim prinosima treniranja; "
                "preskačem korelacijsko klasteriranje.",
                label,
                training_returns.shape[1],
            )
            continue

        correlation_labels, _ = correlation_cluster(training_returns, k=k)
        correlation_frame = (
            correlation_labels.rename("correlation_cluster")
            .reset_index()
            .rename(columns={"index": "ticker"})
        )
        if "ticker" not in correlation_frame.columns and "index" in correlation_frame.columns:
            correlation_frame = correlation_frame.rename(columns={"index": "ticker"})
        correlation_frame["train_window"] = label
        if sector_map is not None:
            correlation_frame["sector"] = correlation_frame["ticker"].map(sector_map)
        correlation_rows.append(correlation_frame)

    factor_table = (
        pd.concat(factor_rows, ignore_index=True)
        if factor_rows
        else pd.DataFrame(columns=["train_window", "ticker", "factor_cluster", "sector"])
    )
    correlation_table = (
        pd.concat(correlation_rows, ignore_index=True)
        if correlation_rows
        else pd.DataFrame(
            columns=["train_window", "ticker", "correlation_cluster", "sector"]
        )
    )

    column_order = ["train_window", "ticker", "factor_cluster"] + (
        ["sector"] if sector_map is not None else []
    )
    factor_table = factor_table[column_order]
    column_order = ["train_window", "ticker", "correlation_cluster"] + (
        ["sector"] if sector_map is not None else []
    )
    correlation_table = correlation_table[column_order]

    return {
        "factor_clusters": factor_table,
        "correlation_clusters": correlation_table,
    }


# ---------------------------------------------------------------------------
# Bootstrap stabilnost klastera
# ---------------------------------------------------------------------------


def _comembership_pairs(labels: pd.Series) -> set[tuple[str, str]]:
    """Vrati skup neuređenih parova oznaka dionica koji dijele klaster."""
    pairs: set[tuple[str, str]] = set()
    for _, members in labels.groupby(labels):
        members_sorted = sorted(members.index)
        for i in range(len(members_sorted)):
            for j in range(i + 1, len(members_sorted)):
                pairs.add((members_sorted[i], members_sorted[j]))
    return pairs


def _jaccard_pairs(reference: set, candidate: set) -> float:
    if not reference and not candidate:
        return 1.0
    union = reference | candidate
    return float(len(reference & candidate) / len(union)) if union else 1.0


def _comembership_matrix(
    labels: pd.Series,
    ticker_to_index: dict[str, int],
    n_tickers: int,
) -> np.ndarray:
    """Vrati ``{0,1}`` matricu zajedničke pripadnosti dimenzija ``n_tickers × n_tickers``.

    Vektorizirano vanjskim produktima indikatorskih vektora po klasteru —
    O(N × K) umjesto O(N²) parova u čistom Pythonu.
    """
    mat = np.zeros((n_tickers, n_tickers), dtype=np.uint16)
    for _, members in labels.groupby(labels):
        idx = np.fromiter(
            (ticker_to_index[t] for t in members.index if t in ticker_to_index),
            dtype=np.int64,
        )
        if idx.size == 0:
            continue
        mat[np.ix_(idx, idx)] = 1
    return mat


def cluster_stability(
    data: pd.DataFrame,
    k: int,
    mode: str = "factor",
    n_bootstraps: int = N_BOOTSTRAP_CLUSTERS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Bootstrap ponovno uzorkuj univerzum i izmjeri reproducibilnost klastera.

    Parametri
    ----------
    data:
        Standardizirane faktorske značajke (reci = oznake dionica) kad je ``mode='factor'``;
        panel prinosa razdoblja treniranja (reci = mjeseci, stupci = oznake dionica)
        kad je ``mode='correlation'``.
    k:
        Broj klastera, isti kao kod prilagodbe na punom uzorku.
    mode:
        ``'factor'`` ili ``'correlation'`` — koju konfiguraciju povezivanja koristiti.
    n_bootstraps:
        Broj ponovnih uzoraka (s ponavljanjem).
    seed:
        Sjeme generatora slučajnih brojeva.

    Vraća
    -------
    rječnik s ključevima:
        - ``mean_jaccard`` — srednji Jaccard zajedničke pripadnosti po parovima kroz
          bootstrapove naspram klasteriranja na punom uzorku.
        - ``co_membership`` — DataFrame indeksiran/stupčan po oznaci dionice s
          udjelom bootstrapova u kojima je svaki par završio u istom klasteru.
        - ``mean_within_cluster_comembership`` — prosječna vjerojatnost para unutar klastera
          pod klasteriranjem na punom uzorku (jedan skalar).
        - ``n_bootstraps_used`` — stvaran broj nakon preskočenih iteracija.
        - ``full_labels`` — oznake iz klasteriranja na punom uzorku.
    """
    rng = np.random.default_rng(seed)
    if mode == "factor":
        tickers = data.index.tolist()
        full_labels, _ = factor_cluster(data, k=k)

        def cluster_subset(sample: list[str]) -> pd.Series:
            subset = data.loc[sample]
            labels, _ = factor_cluster(subset, k=k)
            return labels
    elif mode == "correlation":
        tickers = data.columns.tolist()
        full_labels, _ = correlation_cluster(data, k=k)

        def cluster_subset(sample: list[str]) -> pd.Series:
            subset = data.loc[:, sample]
            labels, _ = correlation_cluster(subset, k=k)
            return labels
    else:
        raise ValueError("mode mora biti 'factor' ili 'correlation'.")

    n_tickers = len(tickers)
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(tickers)}
    full_co = _comembership_matrix(full_labels, ticker_to_index, n_tickers)
    full_pair_count = (np.triu(full_co, k=1)).sum()

    co_membership_counts = np.zeros((n_tickers, n_tickers), dtype=np.int32)
    appearance_counts = np.zeros((n_tickers, n_tickers), dtype=np.int32)
    jaccards: list[float] = []

    for _ in range(n_bootstraps):
        sample_indices = rng.integers(0, n_tickers, size=n_tickers)
        unique_idx = np.unique(sample_indices)
        if unique_idx.size < k:
            continue
        unique_tickers = [tickers[i] for i in unique_idx]
        try:
            labels = cluster_subset(unique_tickers)
        except Exception as error:
            LOGGER.debug("Bootstrap iteracija preskočena (%s)", error)
            continue

        sample_co = _comembership_matrix(labels, ticker_to_index, n_tickers)
        # Ograniči usporedbu na skup parova pokriven objema klasteriranjima:
        # presjek parova (i, j) gdje se obje oznake dionica pojavljuju u
        # bootstrap uzorku.
        appearance_mask = np.zeros((n_tickers, n_tickers), dtype=bool)
        appearance_mask[np.ix_(unique_idx, unique_idx)] = True
        appearance_mask[np.arange(n_tickers), np.arange(n_tickers)] = False

        union = ((full_co | sample_co) & appearance_mask)
        intersection = ((full_co & sample_co) & appearance_mask)
        n_union = np.triu(union, k=1).sum()
        if n_union == 0:
            jaccards.append(1.0)
        else:
            jaccards.append(float(np.triu(intersection, k=1).sum() / n_union))

        co_membership_counts += sample_co & appearance_mask
        appearance_counts += appearance_mask.astype(np.int32)

    with np.errstate(invalid="ignore", divide="ignore"):
        co_membership_array = np.where(
            appearance_counts > 0,
            co_membership_counts / np.maximum(appearance_counts, 1),
            0.0,
        )
    co_membership = pd.DataFrame(co_membership_array, index=tickers, columns=tickers)

    if full_pair_count > 0:
        triu_mask = np.triu(full_co.astype(bool), k=1)
        within_values = co_membership_array[triu_mask]
        within_cluster_avg = float(within_values.mean()) if within_values.size else float("nan")
    else:
        within_cluster_avg = float("nan")

    return {
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "co_membership": co_membership,
        "mean_within_cluster_comembership": within_cluster_avg,
        "n_bootstraps_used": len(jaccards),
        "full_labels": full_labels,
    }

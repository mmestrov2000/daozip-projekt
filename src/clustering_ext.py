"""Dodatni algoritmi klasteriranja i indeksi interne validacije.

Ovaj modul nadopunjuje :mod:`src.clustering` (koji nosi projektovo
hijerarhijsko klasteriranje Wardovom / potpunom vezom) drugim paradigmama
klasteriranja koje se uče na kolegiju „Dubinska analiza podataka”:

- **Particijsko** (predavanje 9b): K-means i kompaktan k-medoid (PAM).
- **Bazirano na gustoći** (predavanje 10): DBSCAN, čija oznaka ``-1`` označava dionice
  bez jasnog stilskog arhetipa (šum).
- **Probabilističko** (predavanje 10): meko klasteriranje Gaussovom mješavinom / EM-om.
- **Interna validacija / odabir K** (predavanje 9b): silueta,
  Calinski-Harabasz, Davies-Bouldin i Tibshiranijeva Gap statistika.

Sve radi na istih šest standardiziranih značajki faktorske izloženosti koje koristi
hijerarhijsko klasteriranje, pa su metode izravno usporedive
(prilagođeni Randov indeks, silueta u zajedničkom prostoru značajki).

Funkcije su namjerno aditivne — ne mijenjaju validirani
protok :mod:`src.clustering`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.clustering import factor_cluster
from src.utils import RANDOM_SEED


def _standardize(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Z-standardiziraj matricu značajki (nulta sredina, jedinična varijanca po stupcu)."""
    frame = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(
        features, dtype=float
    )
    return StandardScaler().fit_transform(frame)


def _as_labels(values: np.ndarray, index: pd.Index | None, name: str) -> pd.Series | np.ndarray:
    if index is None:
        return values
    return pd.Series(values, index=index, name=name)


# ---------------------------------------------------------------------------
# Particijsko klasteriranje (predavanje 9b)
# ---------------------------------------------------------------------------


def kmeans_cluster(
    features: pd.DataFrame | np.ndarray,
    k: int,
    seed: int = RANDOM_SEED,
) -> pd.Series | np.ndarray:
    """K-means na standardiziranim faktorskim značajkama (predavanje 9b, cilj SKP).

    Oznake su indeksirane od 1 radi usklađenosti s konvencijom hijerarhijskog
    klasteriranja korištenom drugdje u projektu.
    """
    index = features.index if isinstance(features, pd.DataFrame) else None
    X = _standardize(features)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(X).astype(int) + 1
    return _as_labels(labels, index, "kmeans_cluster")


def kmedoids_cluster(
    features: pd.DataFrame | np.ndarray,
    k: int,
    metric: str = "euclidean",
    max_iter: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.Series | np.ndarray:
    """Kompaktan k-medoid u stilu PAM-a (varijanta K-meansa iz predavanja 9b, otporan na stršeće vrijednosti).

    Implementiran NumPy/scikit-learn udaljenostima po parovima pa ne treba dodatnu
    ovisnost. Medoidi su stvarne podatkovne točke (član koji minimizira udaljenost
    unutar klastera), što metodu čini manje osjetljivom na
    ekstremne stršeće vrijednosti faktorske izloženosti od K-meansa baziranog na sredini.
    """
    index = features.index if isinstance(features, pd.DataFrame) else None
    X = _standardize(features)
    n = X.shape[0]
    if k < 1 or k > n:
        raise ValueError("k mora biti u [1, n_samples].")

    distances = pairwise_distances(X, metric=metric)
    rng = np.random.default_rng(seed)
    medoids = rng.choice(n, size=k, replace=False)
    labels = distances[:, medoids].argmin(axis=1)

    for _ in range(max_iter):
        new_medoids = medoids.copy()
        for cluster in range(k):
            members = np.where(labels == cluster)[0]
            if members.size == 0:
                continue
            within = distances[np.ix_(members, members)].sum(axis=1)
            new_medoids[cluster] = members[within.argmin()]
        new_labels = distances[:, new_medoids].argmin(axis=1)
        if np.array_equal(new_labels, labels) and np.array_equal(new_medoids, medoids):
            break
        medoids, labels = new_medoids, new_labels

    return _as_labels(labels.astype(int) + 1, index, "kmedoids_cluster")


# ---------------------------------------------------------------------------
# Klasteriranje bazirano na gustoći (predavanje 10)
# ---------------------------------------------------------------------------


def dbscan_cluster(
    features: pd.DataFrame | np.ndarray,
    eps: float = 1.5,
    min_samples: int = 5,
) -> pd.Series | np.ndarray:
    """DBSCAN na standardiziranim faktorskim značajkama (predavanje 10).

    Vraća cjelobrojne oznake gdje ``-1`` označava *šum* — dionice koje ne leže ni u jednoj
    gustoj regiji prostora faktorske izloženosti, tj. dionice bez jasnog stilskog
    arhetipa. ``eps`` je radijus udaljenosti u standardiziranom 6-D prostoru i
    obično ga treba ugoditi po univerzumu; zadana vrijednost razumno je polazište
    za z-standardizirane značajke.
    """
    index = features.index if isinstance(features, pd.DataFrame) else None
    X = _standardize(features)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X).astype(int)
    return _as_labels(labels, index, "dbscan_cluster")


def suggest_dbscan_eps(
    features: pd.DataFrame | np.ndarray,
    min_samples: int = 5,
    quantile: float = 0.90,
) -> float:
    """Heuristički ``eps`` iz grafa k-udaljenosti (k = ``min_samples``).

    Vraća ``quantile`` udaljenosti svake točke do njezina
    ``min_samples``-tog najbližeg susjeda — klasična heuristika „koljena” za
    izbor DBSCAN radijusa.
    """
    X = _standardize(features)
    distances = pairwise_distances(X)
    distances.sort(axis=1)
    kth = distances[:, min(min_samples, distances.shape[1] - 1)]
    return float(np.quantile(kth, quantile))


# ---------------------------------------------------------------------------
# Probabilističko klasteriranje (predavanje 10 — GMM / EM)
# ---------------------------------------------------------------------------


def gmm_cluster(
    features: pd.DataFrame | np.ndarray,
    k: int,
    seed: int = RANDOM_SEED,
) -> tuple[pd.Series | np.ndarray, np.ndarray]:
    """Meko klasteriranje Gaussovom mješavinom (EM) (predavanje 10).

    Vraća ``(hard_labels, responsibilities)`` gdje je ``responsibilities``
    matrica ``n × k`` aposteriornih vjerojatnosti pripadnosti ``γ(z_nk)``.
    Tvrde oznake indeksirane su od 1.
    """
    index = features.index if isinstance(features, pd.DataFrame) else None
    X = _standardize(features)
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    model.fit(X)
    responsibilities = model.predict_proba(X)
    labels = responsibilities.argmax(axis=1).astype(int) + 1
    return _as_labels(labels, index, "gmm_cluster"), responsibilities


# ---------------------------------------------------------------------------
# Interna validacija i odabir K (predavanje 9b)
# ---------------------------------------------------------------------------


def internal_validation_scores(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Silueta, Calinski-Harabasz i Davies-Bouldin za jedno označavanje.

    Sva tri računaju se u standardiziranom prostoru faktorske izloženosti, zanemarujući
    DBSCAN točke šuma (oznaka ``-1``). Viša silueta / Calinski-Harabasz
    i niži Davies-Bouldin upućuju na bolje razdvojene klastere.
    """
    X = _standardize(features)
    label_array = np.asarray(labels)
    mask = label_array != -1
    X_eval, y_eval = X[mask], label_array[mask]
    n_clusters = len(np.unique(y_eval))
    if not (1 < n_clusters < len(y_eval)):
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_clusters": int(n_clusters),
            "n_noise": int((label_array == -1).sum()),
        }
    return {
        "silhouette": float(silhouette_score(X_eval, y_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, y_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, y_eval)),
        "n_clusters": int(n_clusters),
        "n_noise": int((label_array == -1).sum()),
    }


def gap_statistic(
    features: pd.DataFrame | np.ndarray,
    k_range: Sequence[int],
    n_refs: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Tibshiranijeva Gap statistika preko ``k_range`` za K-means (predavanje 9b).

    Za svaki ``k`` disperzija unutar klastera ``log(W_k)`` na stvarnim podacima
    uspoređuje se s njezinim očekivanjem pod ``n_refs`` uniformnih referentnih uzoraka
    izvučenih iz graničnog okvira (standardiziranih) podataka:

        Gap(k) = mean_b log(W_k^ref(b)) - log(W_k)

    Najmanji ``k`` s ``Gap(k) >= Gap(k+1) - s_{k+1}`` predloženi je
    broj klastera (``s`` je referentna standardna pogreška, skalirana s
    ``sqrt(1 + 1/n_refs)``). Vraća jedan redak po ``k`` s ``gap`` i ``s_k``.
    """
    X = _standardize(features)
    rng = np.random.default_rng(seed)
    mins, maxs = X.min(axis=0), X.max(axis=0)

    def dispersion(data: np.ndarray, k: int) -> float:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(data)
        return float(model.inertia_)  # suma kvadrata unutar klastera = W_k

    rows: list[dict[str, float]] = []
    for k in k_range:
        log_wk = np.log(dispersion(X, k))
        ref_logs = np.empty(n_refs)
        for b in range(n_refs):
            ref = rng.uniform(mins, maxs, size=X.shape)
            ref_logs[b] = np.log(dispersion(ref, k))
        gap = float(ref_logs.mean() - log_wk)
        sk = float(ref_logs.std(ddof=0) * np.sqrt(1.0 + 1.0 / n_refs))
        rows.append({"k": int(k), "gap": gap, "s_k": sk, "log_wk": float(log_wk)})

    table = pd.DataFrame(rows)
    # Predloženi K putem Tibshiranijeva pravila 1SE.
    suggested = np.nan
    for i in range(len(table) - 1):
        if table.loc[i, "gap"] >= table.loc[i + 1, "gap"] - table.loc[i + 1, "s_k"]:
            suggested = int(table.loc[i, "k"])
            break
    table.attrs["suggested_k"] = suggested
    return table


def kselection_table(
    features: pd.DataFrame | np.ndarray,
    k_range: Sequence[int],
    methods: Sequence[str] = ("ward", "kmeans"),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Silueta / Calinski-Harabasz / Davies-Bouldin kroz ``k`` i metode.

    Proizvodi dugu tablicu (jedan redak po ``method × k``) iz koje se može iščitati
    „lakat”/vrh svakog indeksa. Nadopunjuje Gap statistiku i projektov
    postojeći prelet prosječne siluete.
    """
    rows: list[dict[str, float]] = []
    for k in k_range:
        for method in methods:
            if method == "ward":
                labels, _ = factor_cluster(features, k=k)
                labels = labels.to_numpy() if isinstance(labels, pd.Series) else labels
            elif method == "kmeans":
                series = kmeans_cluster(features, k=k, seed=seed)
                labels = series.to_numpy() if isinstance(series, pd.Series) else series
            elif method == "kmedoids":
                series = kmedoids_cluster(features, k=k, seed=seed)
                labels = series.to_numpy() if isinstance(series, pd.Series) else series
            else:
                raise ValueError(f"Nepoznata metoda {method!r}.")
            scores = internal_validation_scores(features, labels)
            rows.append({"method": method, "k": int(k), **scores})
    return pd.DataFrame(rows)


def compare_methods(
    features: pd.DataFrame | np.ndarray,
    k: int,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    include_gmm: bool = True,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Prilagodi Ward / K-means / k-medoid / GMM / DBSCAN i usporedi ih.

    Vraća rječnik s:
      - ``labels`` — rječnik metoda → Series oznaka,
      - ``scores`` — DataFrame indeksa interne validacije po metodi,
      - ``ari``    — DataFrame prilagođenog Randova indeksa po parovima između metoda.

    DBSCAN baziran na gustoći ne uzima ``k``; ako je ``dbscan_eps`` ``None``,
    postavlja se iz :func:`suggest_dbscan_eps`.
    """
    index = features.index if isinstance(features, pd.DataFrame) else pd.RangeIndex(len(features))

    ward, _ = factor_cluster(features, k=k)
    labels: dict[str, pd.Series] = {
        "ward": ward if isinstance(ward, pd.Series) else pd.Series(ward, index=index),
        "kmeans": kmeans_cluster(features, k=k, seed=seed),
        "kmedoids": kmedoids_cluster(features, k=k, seed=seed),
    }
    if include_gmm:
        gmm_labels, _ = gmm_cluster(features, k=k, seed=seed)
        labels["gmm"] = gmm_labels
    eps = suggest_dbscan_eps(features, min_samples=dbscan_min_samples) if dbscan_eps is None else dbscan_eps
    labels["dbscan"] = dbscan_cluster(features, eps=eps, min_samples=dbscan_min_samples)

    labels = {name: (s if isinstance(s, pd.Series) else pd.Series(s, index=index)) for name, s in labels.items()}

    score_rows = []
    for name, series in labels.items():
        scores = internal_validation_scores(features, series.to_numpy())
        score_rows.append({"method": name, **scores})
    scores_df = pd.DataFrame(score_rows)

    names = list(labels)
    ari = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            ari.loc[a, b] = adjusted_rand_score(labels[a], labels[b])

    return {"labels": labels, "scores": scores_df, "ari": ari, "dbscan_eps": float(eps)}


__all__ = [
    "kmeans_cluster",
    "kmedoids_cluster",
    "dbscan_cluster",
    "suggest_dbscan_eps",
    "gmm_cluster",
    "internal_validation_scores",
    "gap_statistic",
    "kselection_table",
    "compare_methods",
]

"""Hijerarhijski alokatori za fazu 1: HRP, HERC i NCO + korelacijska stabla.

Ulazno stablo/particija uvijek je vanjski parametar (linkage matrica za
HRP/HERC, oznake klastera za NCO), pa iste funkcije služe korelacijskom i
faktorskom kraku usporedbe. Rizik se u svim koracima čita iz proslijeđene Σ
(u produkciji Ledoit–Wolf; zaključana odluka 6).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, to_tree
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

from src.portfolio import _as_covariance_matrix, _validate_weight_bounds, min_variance
from src.utils import W_MAX


CORRELATION_TREE_LINKAGES = {"single", "ward"}


_CAP_TOL = 1e-12


def quasi_diagonal_order(linkage_matrix: np.ndarray) -> np.ndarray:
    """Vrati kvazidijagonalni poredak listova iz SciPy linkage matrice.

    Ekvivalent getQuasiDiag koraka iz López de Prada (2016): listovi se
    poredaju tako da se srodne imovine nađu jedna uz drugu, čime korelacijska
    matrica postaje kvazidijagonalna.
    """
    matrix = np.asarray(linkage_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("linkage_matrix mora biti SciPy linkage oblika (n-1, 4).")
    return np.asarray(leaves_list(matrix), dtype=int)


def _cluster_variance(sigma: np.ndarray, indices: Sequence[int]) -> float:
    """Varijanca grane s inverzno-varijančnim težinama iz pod-bloka Σ."""
    sub = sigma[np.ix_(indices, indices)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(
    Sigma: pd.DataFrame | np.ndarray,
    order: Sequence[int] | np.ndarray,
) -> pd.Series | np.ndarray:
    """HRP težine rekurzivnom bisekcijom (López de Prado 2016).

    ``order`` je kvazidijagonalni poredak listova (pozicijski indeksi u
    ``Sigmu``), tipično iz :func:`quasi_diagonal_order`. Poredak dolazi
    izvana, pa ista funkcija služi svim krakovima usporedbe. Rizik grana
    računa se isključivo iz proslijeđene ``Sigme``. Težine se vraćaju u
    izvornom poretku ``Sigme``; cap je zaseban post-korak
    (:func:`apply_w_max`).
    """
    sigma, labels = _as_covariance_matrix(Sigma)
    n = sigma.shape[0]
    if (np.diag(sigma) <= 0).any():
        raise ValueError("Dijagonala Sigme mora biti strogo pozitivna.")

    order_array = np.asarray(order, dtype=int)
    if not np.array_equal(np.sort(order_array), np.arange(n)):
        raise ValueError("order mora biti permutacija 0..n-1 za danu Sigmu.")

    weights = np.ones(n)
    clusters: list[list[int]] = [order_array.tolist()]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_left = _cluster_variance(sigma, left)
            var_right = _cluster_variance(sigma, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights /= weights.sum()
    if labels is None:
        return weights
    return pd.Series(weights, index=labels, name="weight")


def apply_w_max(
    weights: pd.Series | np.ndarray,
    w_max: float = W_MAX,
) -> tuple[pd.Series | np.ndarray, float]:
    """Iterativno odsijeci težine na ``w_max`` uz renormalizaciju ostatka.

    Mehanizam capa zajednički svim alokatorima (riješeno pitanje 1): težine
    iznad ``w_max`` postavljaju se na ``w_max``, a neukapirani ostatak se
    renormalizira da ukupna težina ostane 1; postupak se ponavlja dok nijedna
    težina ne prelazi ``w_max``.

    Vraća ``(weights, capped_weight_share)`` gdje je ``capped_weight_share``
    udio ukupne težine koji u konačnim težinama leži na capu (0.0 kad cap ne
    grize) — runner ga bilježi u status tablicu po (portfelj, prozor).
    """
    if isinstance(weights, pd.Series):
        if weights.index.has_duplicates:
            raise ValueError("Indeks weights mora biti jedinstven.")
        labels: pd.Index | None = weights.index
        values = weights.to_numpy(dtype=float)
    else:
        labels = None
        values = np.asarray(weights, dtype=float).copy()

    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights mora biti neprazan jednodimenzionalan vektor.")
    if not np.isfinite(values).all():
        raise ValueError("weights sadrži nekonačne vrijednosti.")
    if (values < 0).any():
        raise ValueError("weights mora biti nenegativan.")
    total = values.sum()
    if total <= 0:
        raise ValueError("weights mora imati pozitivan zbroj.")
    values = values / total

    _validate_weight_bounds(values.size, w_max)

    capped = np.zeros(values.size, dtype=bool)
    while True:
        over = values > w_max + _CAP_TOL
        if not over.any():
            break
        capped |= over
        values[capped] = w_max
        residual = 1.0 - w_max * capped.sum()
        free = ~capped
        free_total = values[free].sum()
        if free_total <= _CAP_TOL:
            break
        values[free] *= residual / free_total

    capped_weight_share = float(values[values >= w_max - _CAP_TOL].sum())

    if labels is None:
        return values, capped_weight_share
    return pd.Series(values, index=labels, name="weight"), capped_weight_share


def correlation_distance(
    corr: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """Korelacijska udaljenost d = √(½(1−ρ)) (López de Prado 2016).

    Vrijednosti se odsijecaju na [−1, 1], dijagonala se postavlja na 0.
    Napomena: postojeći ``correlation_cluster`` koristi skaliranje
    √(2(1−ρ)); obje varijante daju istu topologiju stabla (konstantan
    faktor), ovdje se drži zapis iz izvornika.
    """
    if isinstance(corr, pd.DataFrame):
        if not corr.index.equals(corr.columns):
            raise ValueError("Indeks i stupci korelacijske matrice moraju se podudarati.")
        labels: pd.Index | None = corr.index
        matrix = corr.to_numpy(dtype=float)
    else:
        labels = None
        matrix = np.asarray(corr, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("corr mora biti kvadratna korelacijska matrica.")
    if not np.isfinite(matrix).all():
        raise ValueError("corr sadrži nekonačne vrijednosti.")

    clipped = np.clip(matrix, -1.0, 1.0)
    distance = np.sqrt(0.5 * (1.0 - clipped))
    np.fill_diagonal(distance, 0.0)

    if labels is None:
        return distance
    return pd.DataFrame(distance, index=labels, columns=labels)


def build_correlation_tree(
    returns_or_corr: pd.DataFrame | np.ndarray,
    linkage: str = "single",
) -> np.ndarray:
    """Izgradi korelacijsko stablo (SciPy linkage) za hijerarhijske alokatore.

    ``returns_or_corr`` je ili panel prinosa (reci = mjeseci, stupci =
    imovine) ili već izračunata korelacijska matrica (prepoznaje se kao
    kvadratna simetrična matrica s jedinicama na dijagonali). Udaljenost je
    d = √(½(1−ρ)) iz :func:`correlation_distance`; poredak listova = poredak
    stupaca ulaza.

    ``linkage="single"`` je vjerna replikacija López de Prada (2016) — samo
    za ``hrp_corr_single``. ``linkage="ward"`` je krak kontrolirane usporedbe
    prostora (korekcija K1: ista veza kao faktorsko stablo, pa Faza 3
    manipulira samo prostorom).

    Metodološka napomena (Lance–Williams): SciPy Ward na unaprijed
    izračunatoj kondenziranoj udaljenosti provodi Lance–Williamsova
    ažuriranja, što je standardna praksa u HERC literaturi (Raffinot 2018)
    iako Ward formalno pretpostavlja euklidske koordinate. Postojeći
    ``correlation_cluster`` (``src/clustering.py``) takav poziv namjerno
    odbija i **ne mijenja se** — guard ostaje; stabla za alokatore žive
    isključivo ovdje. Za razliku od ``factor_cluster``, ne koristi se
    ``optimal_ordering`` (vjernost izvornoj kvazidijagonalizaciji).
    """
    if linkage not in CORRELATION_TREE_LINKAGES:
        raise ValueError(
            f"linkage mora biti jedan od {sorted(CORRELATION_TREE_LINKAGES)}, "
            f"dobiveno {linkage!r}."
        )

    if isinstance(returns_or_corr, pd.DataFrame):
        frame = returns_or_corr.copy()
    else:
        frame = pd.DataFrame(np.asarray(returns_or_corr, dtype=float))
    if frame.empty:
        raise ValueError("returns_or_corr ne smije biti prazan.")
    if frame.columns.has_duplicates:
        raise ValueError("Stupci returns_or_corr moraju biti jedinstveni.")

    values = frame.to_numpy(dtype=float)
    is_correlation = (
        values.shape[0] == values.shape[1]
        and np.isfinite(values).all()
        and np.allclose(values, values.T, atol=1e-10)
        and np.allclose(np.diag(values), 1.0, atol=1e-10)
    )
    if is_correlation:
        correlation = frame
    else:
        usable = frame.dropna(axis=0, how="any")
        if len(usable) < 2:
            raise ValueError("Potrebne su barem dvije potpune opservacije prinosa.")
        correlation = usable.corr()
        if correlation.isna().any().any():
            raise ValueError("Korelacijska matrica prinosa sadrži nedostajuće vrijednosti.")

    if correlation.shape[0] < 2:
        raise ValueError("Za izgradnju stabla potrebne su barem dvije imovine.")

    distance = correlation_distance(correlation)
    condensed = squareform(np.asarray(distance, dtype=float), checks=False)
    return scipy_linkage(condensed, method=linkage)


def _validate_linkage_for_sigma(linkage_matrix: np.ndarray, n_assets: int) -> np.ndarray:
    matrix = np.asarray(linkage_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or matrix.shape[0] != n_assets - 1:
        raise ValueError(
            "linkage_matrix mora biti SciPy linkage oblika (n-1, 4) usklađen sa Sigmom."
        )
    return matrix


def herc_weights(
    Sigma: pd.DataFrame | np.ndarray,
    linkage_matrix: np.ndarray,
    k: int,
    w_max: float = W_MAX,
) -> tuple[pd.Series | np.ndarray, float]:
    """HERC težine (Raffinot 2018) na proslijeđenom stablu.

    Rez stabla na ``k`` klastera (``fcluster``, ``criterion="maxclust"`` —
    ista konvencija kao ``src/clustering.py``), zatim top-down podjela
    kapitala **niz stvarni dendrogram**: na svakom čvoru iznad reza kapital
    se dijeli među granama obrnuto proporcionalno riziku grane (zbroj rizika
    klastera u grani; rizik klastera = varijanca Σ pod-bloka s naivnim
    inverzno-varijančnim težinama — isti izračun kao kod HRP grana). Unutar
    klastera naivni risk parity (1/σ_i). Na kraju isti :func:`apply_w_max`
    post-korak; vraća ``(weights, capped_weight_share)``.
    """
    sigma, asset_labels = _as_covariance_matrix(Sigma)
    n_assets = sigma.shape[0]
    if (np.diag(sigma) <= 0).any():
        raise ValueError("Dijagonala Sigme mora biti strogo pozitivna.")
    matrix = _validate_linkage_for_sigma(linkage_matrix, n_assets)
    if not isinstance(k, int):
        raise TypeError("k mora biti cijeli broj.")
    if k < 1 or k > n_assets:
        raise ValueError(f"k={k} mora biti u [1, {n_assets}].")

    cluster_labels = fcluster(matrix, t=k, criterion="maxclust").astype(int)
    members_by_cluster = {
        cluster_id: np.flatnonzero(cluster_labels == cluster_id)
        for cluster_id in np.unique(cluster_labels)
    }
    cluster_var = {
        cluster_id: _cluster_variance(sigma, members)
        for cluster_id, members in members_by_cluster.items()
    }

    # Top-down niz dendrogram: rez na maxclust jamči da je svaki klaster
    # povezano podstablo, pa čvor iznad reza dijeli klastere disjunktno.
    cluster_capital = {cluster_id: 0.0 for cluster_id in members_by_cluster}
    stack: list[tuple[object, float]] = [(to_tree(matrix), 1.0)]
    while stack:
        node, capital = stack.pop()
        node_clusters = set(cluster_labels[node.pre_order()])
        if len(node_clusters) == 1:
            cluster_capital[node_clusters.pop()] += capital
            continue
        left, right = node.get_left(), node.get_right()
        risk_left = sum(
            cluster_var[cluster_id]
            for cluster_id in set(cluster_labels[left.pre_order()])
        )
        risk_right = sum(
            cluster_var[cluster_id]
            for cluster_id in set(cluster_labels[right.pre_order()])
        )
        alpha = risk_right / (risk_left + risk_right)
        stack.append((left, capital * alpha))
        stack.append((right, capital * (1.0 - alpha)))

    weights = np.zeros(n_assets)
    volatilities = np.sqrt(np.diag(sigma))
    for cluster_id, members in members_by_cluster.items():
        inverse_vol = 1.0 / volatilities[members]
        weights[members] = cluster_capital[cluster_id] * inverse_vol / inverse_vol.sum()

    if asset_labels is not None:
        return apply_w_max(pd.Series(weights, index=asset_labels, name="weight"), w_max)
    return apply_w_max(weights, w_max)


def _align_cluster_labels(
    labels: Sequence[object] | pd.Series | np.ndarray,
    asset_labels: pd.Index | None,
    n_assets: int,
) -> pd.Series:
    """Poravnaj oznake klastera s poretkom Sigme (konvencija kao groups)."""
    if isinstance(labels, pd.Series):
        label_series = labels.copy()
        if asset_labels is not None:
            missing = asset_labels.difference(label_series.index)
            if len(missing) > 0:
                raise ValueError(f"Nedostaju oznake klastera za imovine: {missing.tolist()}")
            label_series = label_series.loc[asset_labels]
        elif len(label_series) != n_assets:
            raise ValueError("duljina labels mora odgovarati dimenzijama Sigme.")
    else:
        if len(labels) != n_assets:
            raise ValueError("duljina labels mora odgovarati dimenzijama Sigme.")
        label_series = pd.Series(
            labels, index=asset_labels if asset_labels is not None else None
        )
    if label_series.isna().any():
        raise ValueError("labels sadrži nedostajuće oznake.")
    return label_series


def nco_weights(
    Sigma: pd.DataFrame | np.ndarray,
    labels: Sequence[object] | pd.Series | np.ndarray,
    w_max: float = W_MAX,
) -> tuple[pd.Series | np.ndarray, float]:
    """NCO težine (López de Prado 2019) na proslijeđenoj particiji.

    ``labels`` su oznake klastera po imovini (particija = rez hijerarhijskog
    stabla na K; riješeno pitanje 7 — rez radi pozivatelj). Koraci:
    min-varijanca **unutar** svakog klastera na Σ pod-bloku bez capa
    (``w_max=1.0``; korekcija K2 — unutar-klasterski cap se ne izvodi iz
    težine klastera), reducirana kovarijanca ``Σ_red = W' Σ W``, min-varijanca
    **među** klasterima na ``Σ_red``, pa isti :func:`apply_w_max` post-korak
    na kombinirane konačne težine kao kod HRP/HERC.

    Vraća ``(weights, capped_weight_share)``.
    """
    sigma, asset_labels = _as_covariance_matrix(Sigma)
    n_assets = sigma.shape[0]
    label_series = _align_cluster_labels(labels, asset_labels, n_assets)

    cluster_ids = pd.unique(label_series)
    n_clusters = len(cluster_ids)
    intra = np.zeros((n_assets, n_clusters))
    for j, cluster_id in enumerate(cluster_ids):
        members = np.flatnonzero(label_series.to_numpy() == cluster_id)
        if members.size == 1:
            intra[members[0], j] = 1.0
            continue
        sub_sigma = sigma[np.ix_(members, members)]
        intra[members, j] = min_variance(sub_sigma, w_max=1.0)

    if n_clusters == 1:
        inter = np.array([1.0])
    else:
        sigma_reduced = intra.T @ sigma @ intra
        inter = min_variance(sigma_reduced, w_max=1.0)

    combined = intra @ inter
    if asset_labels is not None:
        combined = pd.Series(combined, index=asset_labels, name="weight")
    return apply_w_max(combined, w_max)

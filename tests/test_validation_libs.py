"""F1.4 — validacija HRP/HERC/NCO protiv riskfolio-lib i skfolio.

Sintetički primjer (15 imovina, fiksno sjeme, Σ iz faktorskog modela): usporedba
vlastitih implementacija (`src/hierarchical.py`) s referentnim bibliotekama.
Testovi se preskaču ako biblioteka nije instalirana (`@pytest.mark.skipif`) —
biblioteke su validacijska, a ne runtime ovisnost i žive u zasebnom venv-u zbog
pinova `numpy 2.4 / pandas 3.0` (registar rizika u `TASKS.md`). Kad je biblioteka
prisutna, testovi zapisuju usporednu tablicu
`outputs/tables/09_validation_vs_libraries.csv`.

Izvori razlika (kvantificirano u CSV-u, dokumentirano i u `PROJECT_SPEC.md` §4.6):

* **HRP single ↔ riskfolio-lib** — identičan algoritam López de Prado (2016):
  d = √(½(1−ρ)), single linkage, `leaves_list` poredak, IVP rekurzivna bisekcija
  dijeljenjem *seriiranog poretka napola*. Slaganje na razini strojne
  preciznosti (tolerancija ≤ 1e-6).
* **HRP single ↔ skfolio** — ista udaljenost i veza, ali skfolio bisektira po
  *strukturi dendrograma* (svaki čvor → njegove dvije grane) umjesto dijeljenjem
  seriiranog poretka napola kako propisuje izvornik → sustavna razlika težina
  (nije greška, druga varijanta algoritma).
* **HERC ward ↔ skfolio** — broj klastera poravnat na skfoliov izbor pa razlika
  nije u K. Preostala razlika je pravilo alokacije: naš HERC (Raffinot 2018)
  dijeli kapital niz dendrogram po zbroju varijanci klastera i koristi naivni
  risk parity (1/σ) unutar klastera, dok skfolio rekurzivno izjednačava doprinos
  varijanci. (riskfolio-lib 7.3.0 HERC put je neispravan — upstream bug u potpisu
  `_hierarchical_recursive_bisection` — pa nije referenca za HERC.)
* **NCO ward ↔ riskfolio-lib** — identičan algoritam López de Prado (2019):
  min-var unutar klastera, reducirana kovarijanca, min-var među klasterima na
  istom rezu stabla (K, ward). Preostala razlika je isključivo tolerancija QP
  rješavača (CLARABEL).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster

from src.hierarchical import (
    build_correlation_tree,
    herc_weights,
    hrp_weights,
    nco_weights,
    quasi_diagonal_order,
)

try:  # validacijske biblioteke nisu u glavnom venv-u (skipif niže)
    import riskfolio as rp

    HAS_RISKFOLIO = True
except ImportError:  # pragma: no cover - ovisi o okruženju
    HAS_RISKFOLIO = False

try:
    from skfolio import RiskMeasure
    from skfolio.cluster import HierarchicalClustering, LinkageMethod
    from skfolio.distance import PearsonDistance
    from skfolio.optimization import (
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
    )

    HAS_SKFOLIO = True
except ImportError:  # pragma: no cover - ovisi o okruženju
    HAS_SKFOLIO = False


# Sintetički problem: 15 imovina, Σ = uzoračka kovarijanca prinosa iz
# faktorskog modela (fiksno sjeme). Σ i poretci listova dijele se s bibliotekama
# tako da obje strane vide *isti* korelacijski i kovarijančni ulaz.
SEED = 20260613
N_PERIODS = 240
N_ASSETS = 15
N_FACTORS = 4
NCO_K = 4  # fiksan rez stabla za NCO usporedbu (oba kraka isti K)
HRP_TOL = 1e-6  # stroga tolerancija — isti algoritam
NCO_TOL = 1e-4  # samo tolerancija QP rješavača
HERC_BOUND = 0.10  # labava granica — dokumentirana razlika pravila alokacije

_CSV_RELATIVE = Path("outputs/tables/09_validation_vs_libraries.csv")


@lru_cache(maxsize=1)
def _synthetic_problem() -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Prinosi (T×N), uzoračka Σ i imena imovina iz faktorskog modela."""
    rng = np.random.default_rng(SEED)
    betas = rng.normal(size=(N_ASSETS, N_FACTORS))
    factor = rng.normal(scale=0.04, size=(N_PERIODS, N_FACTORS))
    idio = rng.normal(scale=0.02, size=(N_PERIODS, N_ASSETS)) * rng.uniform(
        0.5, 1.5, N_ASSETS
    )
    returns = factor @ betas.T + idio
    tickers = tuple(f"A{i:02d}" for i in range(N_ASSETS))
    returns_df = pd.DataFrame(returns, columns=list(tickers))
    sigma = returns_df.cov()  # uzoračka (ddof=1) — identično method_cov="hist"
    return returns_df, sigma, tickers


def _max_abs_diff(left: pd.Series, right: pd.Series, tickers: tuple[str, ...]) -> float:
    aligned = pd.Index(tickers)
    return float((left.reindex(aligned) - right.reindex(aligned)).abs().max())


# ---------------------------------------------------------------------------
# Naše implementacije na sintetičkom problemu
# ---------------------------------------------------------------------------


def _our_hrp_single() -> pd.Series:
    returns_df, sigma, _ = _synthetic_problem()
    tree = build_correlation_tree(returns_df.corr(), linkage="single")
    return hrp_weights(sigma, quasi_diagonal_order(tree))


def _our_herc_ward(k: int) -> pd.Series:
    returns_df, sigma, _ = _synthetic_problem()
    tree = build_correlation_tree(returns_df.corr(), linkage="ward")
    weights, _ = herc_weights(sigma, tree, k=k, w_max=1.0)
    return weights


def _our_nco_ward(k: int) -> pd.Series:
    returns_df, sigma, tickers = _synthetic_problem()
    tree = build_correlation_tree(returns_df.corr(), linkage="ward")
    labels = pd.Series(fcluster(tree, t=k, criterion="maxclust"), index=list(tickers))
    weights, _ = nco_weights(sigma, labels, w_max=1.0)
    return weights


# ---------------------------------------------------------------------------
# Bibliotečne referente (samo kad je biblioteka instalirana)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _riskfolio_hrp_single() -> pd.Series:
    returns_df, _, tickers = _synthetic_problem()
    port = rp.HCPortfolio(returns=returns_df)
    weights = port.optimization(
        model="HRP",
        codependence="pearson",
        rm="MV",
        linkage="single",
        leaf_order=False,  # bez optimal_ordering — kao build_correlation_tree
        obj="MinRisk",
    )["weights"]
    return weights.reindex(list(tickers))


@lru_cache(maxsize=1)
def _riskfolio_nco_ward() -> pd.Series:
    returns_df, _, tickers = _synthetic_problem()
    port = rp.HCPortfolio(returns=returns_df)
    weights = port.optimization(
        model="NCO",
        codependence="pearson",
        rm="MV",
        linkage="ward",
        k=NCO_K,
        leaf_order=False,
        obj="MinRisk",
    )["weights"]
    return weights.reindex(list(tickers))


@lru_cache(maxsize=1)
def _skfolio_herc_ward() -> tuple[pd.Series, int]:
    returns_df, _, tickers = _synthetic_problem()
    model = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.VARIANCE,
        distance_estimator=PearsonDistance(),
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=LinkageMethod.WARD
        ),
    )
    model.fit(returns_df)
    n_clusters = int(model.hierarchical_clustering_estimator_.n_clusters_)
    return pd.Series(model.weights_, index=list(tickers)), n_clusters


@lru_cache(maxsize=1)
def _skfolio_hrp_single() -> pd.Series:
    returns_df, _, tickers = _synthetic_problem()
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.VARIANCE,
        distance_estimator=PearsonDistance(),
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=LinkageMethod.SINGLE
        ),
    )
    model.fit(returns_df)
    return pd.Series(model.weights_, index=list(tickers))


# ---------------------------------------------------------------------------
# Usporedbe — slažu CSV i napajaju tvrdnje testova
# ---------------------------------------------------------------------------


def _collect_records() -> list[dict[str, object]]:
    """Izračunaj sve dostupne usporedbe (ovisno o instaliranim bibliotekama)."""
    _, _, tickers = _synthetic_problem()
    records: list[dict[str, object]] = []

    if HAS_RISKFOLIO:
        records.append(
            {
                "method": "HRP",
                "linkage": "single",
                "library": f"riskfolio-lib {rp.__version__}",
                "max_abs_weight_diff": _max_abs_diff(
                    _our_hrp_single(), _riskfolio_hrp_single(), tickers
                ),
                "tolerance": HRP_TOL,
                "explanation": (
                    "Identičan algoritam López de Prado (2016): d=√(½(1−ρ)), "
                    "single linkage, leaves_list poredak, IVP rekurzivna bisekcija "
                    "dijeljenjem seriiranog poretka napola; slaganje na razini "
                    "strojne preciznosti."
                ),
            }
        )
        records.append(
            {
                "method": "NCO",
                "linkage": "ward",
                "library": f"riskfolio-lib {rp.__version__}",
                "max_abs_weight_diff": _max_abs_diff(
                    _our_nco_ward(NCO_K), _riskfolio_nco_ward(), tickers
                ),
                "tolerance": NCO_TOL,
                "explanation": (
                    "Identičan algoritam López de Prado (2019): min-var unutar "
                    f"klastera, reducirana kovarijanca, min-var među klasterima na "
                    f"istom rezu stabla (K={NCO_K}, ward); preostala razlika je "
                    "isključivo tolerancija QP rješavača (CLARABEL)."
                ),
            }
        )

    if HAS_SKFOLIO:
        herc_lib, herc_k = _skfolio_herc_ward()
        import skfolio

        records.append(
            {
                "method": "HERC",
                "linkage": "ward",
                "library": f"skfolio {skfolio.__version__}",
                "max_abs_weight_diff": _max_abs_diff(
                    _our_herc_ward(herc_k), herc_lib, tickers
                ),
                "tolerance": HERC_BOUND,
                "explanation": (
                    f"Broj klastera poravnat na skfoliov izbor (K={herc_k}) pa "
                    "razlika nije u K, nego u pravilu alokacije: naš HERC "
                    "(Raffinot 2018) dijeli kapital niz dendrogram po zbroju "
                    "varijanci klastera i koristi naivni risk parity (1/σ) unutar "
                    "klastera, dok skfolio rekurzivno izjednačava doprinos "
                    "varijanci. riskfolio-lib 7.3.0 HERC put je neispravan "
                    "(upstream bug), pa nije referenca."
                ),
            }
        )
        records.append(
            {
                "method": "HRP",
                "linkage": "single",
                "library": f"skfolio {skfolio.__version__}",
                "max_abs_weight_diff": _max_abs_diff(
                    _our_hrp_single(), _skfolio_hrp_single(), tickers
                ),
                "tolerance": float("nan"),
                "explanation": (
                    "Ista udaljenost i veza, ali skfolio bisektira po strukturi "
                    "dendrograma (čvor → dvije grane) umjesto dijeljenjem "
                    "seriiranog poretka napola kako propisuje izvornik → sustavna "
                    "razlika težina (druga varijanta algoritma, ne greška)."
                ),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Testovi
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RISKFOLIO, reason="riskfolio-lib nije instaliran")
def test_hrp_single_matches_riskfolio():
    """HRP single = riskfolio HRP single ≤ 1e-6 (isti algoritam)."""
    _, _, tickers = _synthetic_problem()
    diff = _max_abs_diff(_our_hrp_single(), _riskfolio_hrp_single(), tickers)
    assert diff <= HRP_TOL


@pytest.mark.skipif(not HAS_RISKFOLIO, reason="riskfolio-lib nije instaliran")
def test_nco_matches_riskfolio_within_solver_tolerance():
    """NCO ward = riskfolio NCO na istom rezu stabla, do tolerancije rješavača."""
    _, _, tickers = _synthetic_problem()
    diff = _max_abs_diff(_our_nco_ward(NCO_K), _riskfolio_nco_ward(), tickers)
    assert diff <= NCO_TOL


@pytest.mark.skipif(not HAS_SKFOLIO, reason="skfolio nije instaliran")
def test_herc_close_to_skfolio_with_aligned_k():
    """HERC ward blizu skfolio HERC kad je K poravnat; razlika je metodološka."""
    _, _, tickers = _synthetic_problem()
    herc_lib, herc_k = _skfolio_herc_ward()
    our = _our_herc_ward(herc_k)
    assert (our >= 0).all()
    assert our.sum() == pytest.approx(1.0)
    assert _max_abs_diff(our, herc_lib, tickers) <= HERC_BOUND


@pytest.mark.skipif(
    not (HAS_RISKFOLIO or HAS_SKFOLIO),
    reason="ni riskfolio-lib ni skfolio nisu instalirani",
)
def test_writes_validation_csv():
    """Zapiši usporednu tablicu sa stupcima method, library, max|Δw|, explanation."""
    records = _collect_records()
    assert records, "očekivan barem jedan red usporedbe"

    frame = pd.DataFrame.from_records(records)[
        ["method", "linkage", "library", "max_abs_weight_diff", "tolerance", "explanation"]
    ]

    csv_path = Path(__file__).resolve().parents[1] / _CSV_RELATIVE
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    assert csv_path.exists()
    written = pd.read_csv(csv_path)
    for column in ("method", "library", "max_abs_weight_diff", "explanation"):
        assert column in written.columns
    # Strogi kriterij prihvaćanja F1.4: HRP razlika ≤ 1e-6.
    if HAS_RISKFOLIO:
        hrp_rf = written[
            (written["method"] == "HRP") & written["library"].str.startswith("riskfolio")
        ]
        assert (hrp_rf["max_abs_weight_diff"] <= HRP_TOL).all()

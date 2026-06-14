# Gdje nastaje stilski rizik hijerarhijske alokacije?

Kontrolirana studija koja utvrđuje **gdje u cjevovodu hijerarhijske alokacije**
(HRP, HERC, NCO) nastaje stilski rizik: u *prostoru* u kojem se hijerarhija gradi
(korelacijskom naspram faktorskom) ili u *mehanizmu* alokacije i ograničenja.
Provedeno na point-in-time univerzumu indeksa **S&P 500 (2013. – 2025.)**, uz
projekcijski QP overlay `|w'β_f| ≤ ε` na hijerarhijske težine.

> Jedina manipulirana varijabla je ulaz u izgradnju hijerarhije — korelacijsko
> stablo naspram Wardova stabla na standardiziranim Fama-French petfaktorskim
> značajkama. Metoda veze drži se konstantnom (Wardova veza u oba kraka) pa
> usporedba mijenja samo prostor, a ne i vezu. Dodavanjem overlay ograničenja
> nastaje potpuni **2×2 faktorijal** {prostor hijerarhije} × {bez / sa
> ograničenjem}. Primarni ishod je **koncentracija stila**
> `|β_SMB| + |β_HML| + |β_RMW| + |β_CMA|` iz petfaktorske atribucije ostvarenih
> testnih prinosa.

---

## Glavni rezultat

Unaprijedni backtest, mjesečni prinosi, **13 kliznih prozora (2013. – 2025.)**,
prozor treniranja od 60 mjeseci unatrag s ponovnim procjenjivanjem svakih
12 mjeseci, Ledoit–Wolf sažimanje kovarijance, ograničenje pojedine pozicije
`w_max = 0.05`, primarni broj klastera `K = 10`. Statistika: blok-bootstrap,
Model Confidence Set (Hansen–Lunde–Nason 2011) i Deflated Sharpe Ratio
(Bailey & López de Prado 2014).

**Baza — koncentracija stila i ostvarena volatilnost po alokatoru:**

| Portfelj | Prostor hijerarhije | Konc. stila | God. vol. | Sharpe (neto) | U MCS-u |
|---|---|---:|---:|---:|:---:|
| Jednake težine (1/N) | — (referenca) | 0.59 | 15.6 % | 0.78 | ne |
| Minimalna varijanca | — (referenca) | 0.78 | 12.7 % | 0.72 | da |
| HRP | korelacijski | 0.57 | 13.5 % | 0.82 | da |
| HERC | korelacijski | 0.57 | 14.3 % | 0.73 | ne |
| NCO | korelacijski | 0.80 | 12.3 % | 0.80 | da |
| HRP | faktorski | 0.58 | 13.6 % | 0.83 | da |
| HERC | faktorski | 0.66 | 15.2 % | 0.69 | ne |
| NCO | faktorski | 0.74 | 12.5 % | 0.76 | da |

`konc. stila = |β_SMB| + |β_HML| + |β_RMW| + |β_CMA|`. „U MCS-u” = pripadnost
Model Confidence Setu na neto prinosima, α = 0.10. Brojevi su iz
[`outputs/tables/13_master_table.csv`](outputs/tables/13_master_table.csv)
(37 portfelja × 29 metrika).

**H1 — nasljeđivanje nagiba (dijagnoza).** NCO u cijelosti nasljeđuje skriveni
RMW/CMA nagib minimalne varijance: koncentracija stila `0.80 ≈ min_var 0.78 ≫
1/N 0.59`. HRP i HERC nasljeđuju isti *mehanizam* (težina raste s padom
rezidualne volatilnosti), ali ga raspršuju na razinu `1/N` (`0.57`).

**H2 — prostor ne neutralizira nagib.** Prelazak s korelacijskog na faktorski
prostor hijerarhije ne uklanja nagib (NCO `0.80 → 0.74`, HERC `0.57 → 0.66`,
HRP `0.57 → 0.58`; razlika u bootstrapu obuhvaća nulu). Očekivani dobitak u
stabilnosti klastera ne ostvaruje se pri `K = 10`; jedina konkretna prednost
faktorskog prostora je parsimonija (~34× manje parametara: `~N²/2` korelacija
naspram `6N` beta).

**H3 — overlay je djelotvoran lijek.** Jedini mehanizam koji nagib stvarno
uklanja je izravno linearno ograničenje na bete, primijenjeno kao overlay na
hijerarhijske težine. Potpuni 2×2 faktorijal (prosjek preko HRP/HERC/NCO,
ε = 0):

| Prostor hijerarhije | Bez overlaya | S overlayem (ε = 0) | Δ konc. stila |
|---|---|---|---:|
| korelacijski | konc. stila 0.65 · vol 13.4 % | konc. stila 0.44 · vol 13.7 % | **−0.21 (−33 %)** |
| faktorski | konc. stila 0.66 · vol 13.8 % | konc. stila 0.51 · vol 14.0 % | **−0.15 (−23 %)** |

Overlay smanjuje koncentraciju stila u **oba** prostora (značajno u 5 od 6
pojedinačnih varijanti) uz zanemariv trošak u ostvarenoj volatilnosti i
dominira na granici stil–volatilnost. Brojevi su iz
[`outputs/tables/13_factorial_2x2.csv`](outputs/tables/13_factorial_2x2.csv).

Stari projekt (dijagnoza min-vara i izravna faktorska neutralnost) ugnježđuje se
kao referentna obitelj (`min_var`, `factor_neutral_eε`), ne baca se.

**Figure:**
[pokrivenost članstva](outputs/figures/00_membership_coverage.png) ·
[kumulativni rast](outputs/figures/09_cumulative_growth.png) ·
[koncentracija stila po godini](outputs/figures/10_style_concentration_per_year.png) ·
[težina naspram bete](outputs/figures/10_weight_vs_beta.png) ·
[granica stil–volatilnost](outputs/figures/12_frontier_style_vs_vol.png).

Potpuni izvještaj: [`reports/izvjestaj.pdf`](reports/izvjestaj.pdf)
(izvor [`reports/izvjestaj.tex`](reports/izvjestaj.tex)).

---

## Što se nalazi u repozitoriju

```
.
├── PROJECT_SPEC.md          ← potpuna metodologija i zaključane odluke
├── TASKS.md                 ← fazni plan implementacije (izvor zadataka)
├── PROGRESS.md              ← dnevnik implementacije
├── CLAUDE.md                ← upute za rad na projektu
├── config.yaml              ← jedino mjesto za parametre (prozori, w_max, K, ε, …)
├── requirements.txt         ← fiksirane Python ovisnosti
├── data/
│   ├── raw/                 ← članstvo S&P 500, korekcije oznaka, predmemorija cijena (parquet, izvan gita)
│   └── processed/           ← mjesečni/višak prinosi, faktori, izloženosti, klasteri, članstvo, metapodaci
├── notebooks/
│   ├── 01_data_and_factors.ipynb       ← podaci + petfaktorske regresije po prozoru
│   ├── 02_clustering.ipynb             ← klasteri, silueta, dendrogram, bootstrap stabilnost
│   ├── 03_portfolios.ipynb             ← benchmark portfelji (1/N, min-var) unaprijednim hodom
│   ├── 04_evaluation.ipynb             ← (naslijeđeno) evaluacija starog dizajna
│   ├── 05_factor_neutral.ipynb         ← izravna faktorska neutralnost, ε-prelet (referentna obitelj)
│   ├── 06_summary.ipynb                ← (naslijeđeno) sažetak starog dizajna
│   ├── 07_clustering_methods.ipynb     ← K-means/k-medoid/GMM/DBSCAN + odabir K (robusnost)
│   ├── 09_hierarchical_replication.ipynb  ← HRP/HERC/NCO replikacija + MCS (Faza 1)
│   ├── 10_hierarchical_diagnosis.ipynb    ← dijagnoza skrivenog stilskog rizika (Faza 2, H1)
│   ├── 11_factor_space_intervention.ipynb ← intervencija u faktorskom prostoru (Faza 3, H2)
│   ├── 12_overlay_mechanism.ipynb         ← overlay QP, granica, DSR, finalni MCS (Faza 4, H3)
│   └── 13_final_summary.ipynb             ← master tablica + 2×2 faktorijal (Faza 5)
├── src/
│   ├── config.py            ← učitavanje config.yaml
│   ├── utils.py             ← konstante i putanje (vezane na config)
│   ├── data.py             ← point-in-time članstvo, predmemorija cijena, FF5, sanitacija
│   ├── factors.py          ← OLS petfaktorske regresije po prozoru
│   ├── clustering.py       ← hijerarhijsko klasteriranje + ARI stabilnost
│   ├── clustering_ext.py   ← K-means, k-medoid, GMM, DBSCAN + indeksi odabira K
│   ├── hierarchical.py     ← HRP/HERC/NCO + graditelji korelacijskih stabala (single/ward)
│   ├── portfolio.py        ← Ledoit–Wolf kov., min-var, faktorski neutralan, projekcijski overlay
│   ├── backtest.py         ← klizni prozori + unaprijedni hod (benchmark, hijerarhijski, overlay)
│   ├── evaluation.py       ← rizik, atribucija, troškovi, bootstrap, MCS, DSR
│   └── viz.py              ← zajednički pomoćnici za crtanje
├── tests/                  ← pytest (config, članstvo, cijene, prozori, troškovi, HRP/HERC/NCO, MCS, DSR, overlay, …)
├── scripts/
│   └── run_all.sh          ← jedan skriptirani put: pytest + svi notebookovi glavnog tijeka
├── outputs/
│   ├── tables/             ← CSV paneli i tablice (prefiksi 00, 02, 05, 07, 09–13)
│   └── figures/            ← spremljeni grafovi (PNG)
├── reports/
│   ├── izvjestaj.tex       ← izvještaj (LaTeX)
│   ├── izvjestaj.pdf       ← kompilirani izvještaj
│   └── figures/            ← figure izvještaja (fig01–fig05)
└── archive/                ← naslijeđeno proširenje (notebook 08, supervised.py) — negativan rezultat
```

---

## Reprodukcija

### 1. Okruženje

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Jedan skriptirani put

```bash
bash scripts/run_all.sh
```

Skripta redom pokreće `pytest`, pa izvršava notebookove glavnog tijeka
(01, 02, 03, 05, 07, 09, 10, 11, 12, 13) preko
`jupyter nbconvert --execute --inplace`, te ispisuje ukupno trajanje. Svi
parametri analize čitaju se iz [`config.yaml`](config.yaml).

**Trajanje (približno, ovisi o hardveru i mreži):**

| Pokretanje | Predmemorija cijena | Trajanje |
|---|---|---|
| Prvo | prazna (preuzima ~757 dionica + FF5) | ~30 – 40 min |
| Ponovno | topla (`data/raw/price_cache/`) | ~12 – 18 min |

U oba slučaja notebook 01 (preuzimanje + regresije) i notebook 02 (bootstrap
stabilnost klastera, 500 uzoraka) najskuplji su koraci. Obrađeni CSV-ovi u
`data/processed/` nose dovoljno stanja da se analitički notebookovi pokreću i
bez ponovnog preuzimanja cijena.

> Notebookovi `04` (stara evaluacija) i `06` (stari sažetak) nisu u glavnom
> tijeku — zamijenili su ih `10`/`12` (dijagnoza/mehanizam) i `13` (sažetak) —
> pa ih `run_all.sh` ne izvršava. Notebook `08` (nadzirano proširenje) arhiviran
> je u [`archive/`](archive/).

---

## Izvori podataka

| Stavka | Izvor | Pohranjeno u |
|---|---|---|
| Point-in-time članstvo S&P 500 (oznaka, naziv, interval) | javna rekonstrukcija [`fja05680/sp500`](https://github.com/fja05680/sp500) + spot-provjere poznatih događaja | `data/raw/sp500_membership.csv` |
| Ručne korekcije oznaka i preimenovanja (npr. FB→META, LEHMQ→LEH) | sastavljeno ručno | `data/raw/ticker_overrides.csv` |
| Mjesečne prilagođene cijene | Yahoo Finance (`yfinance`), Stooq kao dopuna za delistane | `data/raw/price_cache/{TICKER}.parquet` |
| Fama–French petfaktorski podaci (mjesečni) | [Knjižnica podataka Kennetha Frencha](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | `data/processed/factors.csv` |

WRDS/CRSP pristup nije bio dostupan, pa članstvo dolazi iz javne rekonstrukcije
(grananje plana dokumentirano u [`PROJECT_SPEC.md`](PROJECT_SPEC.md)). Pokrivenost
po prozoru — koliko je point-in-time članova imalo valjane cijene i ≥ 60 mjeseci
treninga — izmjerena je i izvještena (`data/raw/russell1000_tickers.csv` ostaje
kao povijesna snimka prijašnjeg univerzuma).

---

## Metodologija u kratko

- **Značajke dionice.** Po prozoru treniranja petfaktorska regresija na mjesečnim
  viškovima prinosa daje `(α, β_MKT, β_SMB, β_HML, β_RMW, β_CMA, σ_ε, R²)`.
  Standardizirani petfaktorski vektor je ulaz za faktorsko stablo.
- **Alokatori.** HRP (López de Prado 2016) — kvazidijagonalizacija + rekurzivna
  bisekcija; HERC (Raffinot 2018) — podjela kapitala niz dendrogram po jednakom
  doprinosu riziku; NCO (López de Prado 2019) — ugniježđena min-var unutar i
  među klasterima. Korelacijska verzija gradi stablo iz `d = √(½(1−ρ))`,
  faktorska iz Wardova stabla na petfaktorskim značajkama. Vlastite
  implementacije validirane su naspram `riskfolio-lib`/`skfolio`.
- **Overlay.** Projekcijski QP `min ‖w − w₀‖²_Σ` uz `1'w = 1`, `0 ≤ w ≤ w_max`,
  `|w'β_f| ≤ ε` za `f ∈ {SMB, HML, RMW, CMA}`, `ε ∈ {0.00, 0.05, 0.10, 0.15}`.
- **Kovarijanca.** Ledoit–Wolf sažimanje, jednom po prozoru, dijeljeno među
  alokatorima.
- **Unaprijedni hod.** Podaci 2000. – 2025.; backtest počinje 2008-01 →
  prva testna godina 2013., zadnja 2025. (13 disjunktnih testnih godina);
  univerzum po prozoru = članovi na datum s ≥ 60 mjeseci treninga.

Potpuni detalji, motivacija i ograničenja: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

## Status

| Faza | Opseg | Stanje |
|---|---|---|
| 0 | Podatkovni temelj: point-in-time S&P 500, pokrivenost, troškovi, config, testovi | **gotovo** |
| 1 | Replikacija HRP/HERC/NCO (korelacijski prostor) + validacija + MCS | **gotovo** |
| 2 | Dijagnoza skrivenog stilskog rizika (H1) | **gotovo** |
| 3 | Intervencija u faktorskom prostoru hijerarhije (H2) | **gotovo** |
| 4 | Overlay QP mehanizam i lijek (H3) + DSR + granica + finalni MCS | **gotovo** |
| 5 | Pisanje i pakiranje: izvještaj, master tablica, README, `run_all.sh` | **u tijeku** |

---

## Licenca

Akademska uporaba. Vrijede licence izvornih podataka (Yahoo Finance, Stooq,
knjižnica podataka Kennetha Frencha, javna rekonstrukcija članstva S&P 500).

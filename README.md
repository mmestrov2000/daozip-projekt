# Skriveni stilski rizik u portfeljima minimalne varijance

Kvantitativni projekt otkrivanja znanja koji dijagnosticira skrivenu koncentraciju Fama-French stilskih faktora u portfeljima minimalne varijance sa samo dugim pozicijama — koristeći klasteriranje u prostoru faktora kao dijagnostičku leću, a izravna ograničenja faktorske neutralnosti kao djelotvoran popravak.

> Klasteriraj Russell 1000 prema Fama-French faktorskim izloženostima kako bi se otkrila skrivena stilska struktura univerzuma, dijagnosticiraj zašto se portfelji minimalne varijance tiho opterećuju faktorima profitabilnosti i investiranja, te ispravi nagib izravnim linearnim ograničenjima faktorske neutralnosti — provjereno na 16 kliznih prozora izvan uzorka.

---

## Glavni rezultat

Unaprijedni backtest, mjesečni prinosi, **16 kliznih prozora (2010. – 2025.)**, prozor treniranja od 60 mjeseci unatrag s ponovnim procjenjivanjem svakih 12 mjeseci, Ledoit-Wolf sažimanje kovarijance, ograničenje pojedine pozicije `w_max = 0.02`, grupno ograničenje `u = 0.15`.

**Petfaktorska atribucija prinosa u testnom razdoblju** (glavna dijagnostika):

| Portfelj | β_MKT | β_SMB | β_HML | β_RMW | β_CMA | konc. stila | god. vol. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jednake težine | 1.11 | 0.49 | 0.09 | 0.08 | 0.18 | 0.83 | 21.2 % |
| Minimalna varijanca | 0.66 | 0.20 | −0.12 | **+0.33** | **+0.34** | 0.98 | 12.0 % |
| + ograničenje sektora | 0.68 | 0.21 | −0.14 | +0.26 | +0.36 | 0.97 | 12.1 % |
| + ograničenje korelacijskih klastera | 0.71 | 0.20 | −0.09 | +0.31 | +0.26 | 0.86 | 12.2 % |
| + ograničenje faktorskih klastera | 0.83 | 0.55 | −0.19 | **−0.15** | **+0.58** | 1.47 | 17.9 % |

`style_concentration = |β_SMB| + |β_HML| + |β_RMW| + |β_CMA|`.

**Dijagnoza potvrđena.** Neograničena minimalna varijanca opterećuje se profitabilnošću i niskim investiranjem (β_RMW = +0.33, β_CMA = +0.34). Ograničenja po sektorima i korelacijskim klasterima ostavljaju nagib u biti netaknutim.

**Ograničenje klastera ne ispravlja nagib.** Ograničenje faktorskih klastera neutralizira RMW (`+0.33 → −0.15`), ali preraspodjeljuje težinu u klastere s većim SMB i CMA nagibima. Ukupna koncentracija stila raste s `0.98` na `1.47`, a portfelj je `5.8` postotnih bodova volatilniji. Mehanizam ograničavanja klastera ne cilja nijednu konkretnu faktorsku betu — to je pogrešna poluga za ograničenje.

**Izravna faktorska neutralnost je djelotvoran popravak.** Dodavanje `|w'β_f| ≤ ε` za `f ∈ {SMB, HML, RMW, CMA}` u kvadratni program smanjuje ostvarenu koncentraciju stila za **31 %** pri `ε = 0` bez troška u ostvarenoj volatilnosti:

| Portfelj | β_MKT | β_SMB | β_HML | β_RMW | β_CMA | konc. stila | god. vol. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minimalna varijanca | 0.66 | +0.20 | −0.12 | **+0.33** | **+0.34** | 0.98 | 12.0 % |
| faktorski neutralan, ε = 0.00 | 0.69 | +0.19 | −0.03 | **+0.26** | **+0.20** | **0.67** | **12.1 %** |
| faktorski neutralan, ε = 0.05 | 0.68 | +0.19 | −0.06 | +0.29 | +0.22 | 0.76 | 12.0 % |
| faktorski neutralan, ε = 0.10 | 0.67 | +0.19 | −0.07 | +0.31 | +0.24 | 0.81 | 12.0 % |
| faktorski neutralan, ε = 0.15 | 0.67 | +0.21 | −0.11 | +0.30 | +0.31 | 0.93 | 12.0 % |

**Intervali pouzdanosti blok-bootstrapom u odnosu na `min_var`** (ε = 0): Δ konc. stila = **−0.308**, 95 % CI **[ −0.47, −0.09 ]**, `p = 0.002`. Interval razlike volatilnosti obuhvaća nulu (`p = 0.59`). Popravak donosi upravo onaj kompromis koji je projekt tražio.

Glavna figura: [`outputs/figures/04_style_concentration_per_year.png`](outputs/figures/04_style_concentration_per_year.png).
Granica (konc. stila naspram vol.): [`outputs/figures/05_frontier_style_vs_vol.png`](outputs/figures/05_frontier_style_vs_vol.png).
Bootstrap distribucija: [`outputs/figures/05_bootstrap_factor_neutral_e0.png`](outputs/figures/05_bootstrap_factor_neutral_e0.png).
Raspodjela težina po klasterima: [`outputs/figures/06_cluster_weight_allocation.png`](outputs/figures/06_cluster_weight_allocation.png).

---

## Što se nalazi u ovom repozitoriju

```
.
├── PROJECT_SPEC.md          ← potpuna metodologija i odluke
├── TASKS.md                 ← fazni plan implementacije
├── README.md                ← ova datoteka
├── requirements.txt         ← fiksirane Python ovisnosti
├── data/
│   ├── raw/                 ← snimka univerzuma + predmemorija cijena po dionici (parquet, izvan gita)
│   └── processed/           ← mjesečni prinosi, viškovi prinosa, faktori, izloženosti, klasteri
├── notebooks/
│   ├── 01_data_and_factors.ipynb   ← prikupljanje podataka + faktorske regresije po prozoru
│   ├── 02_clustering.ipynb         ← odabir K, dendrogram, UMAP, bootstrap stabilnost
│   ├── 03_portfolios.ipynb         ← unaprijedni backtest pet portfelja
│   ├── 04_evaluation.ipynb         ← evaluacija izvan uzorka, bootstrap CI, osjetljivost
│   ├── 05_factor_neutral.ipynb     ← izravan popravak faktorske neutralnosti, ε-prelet, hibrid
│   ├── 06_summary.ipynb            ← vizualni rekapitulacijski pregled u pet dijelova (bez ponovnog procjenjivanja)
│   ├── 07_clustering_methods.ipynb ← K-means/k-medoid/GMM/DBSCAN + odabir K Calinski-Harabaszom/Gapom
│   └── 08_drawdown_tree.ipynb      ← proširenje: stablo odlučivanja + ansambli za pad (CV po godini)
├── src/
│   ├── data.py              ← univerzum Russell 1000, obnovljiva predmemorija cijena, FF5 faktori
│   ├── factors.py           ← OLS faktorske regresije po prozoru
│   ├── clustering.py        ← hijerarhijsko klasteriranje + bootstrap stabilnost
│   ├── clustering_ext.py    ← K-means, k-medoid, GMM, DBSCAN + silueta/CH/Davies-Bouldin/Gap
│   ├── portfolio.py         ← Ledoit-Wolf kov., min. varijanca, grupno ograničeni, faktorski neutralan, hibrid
│   ├── backtest.py          ← generator kliznih prozora + unaprijedni hod + prelet faktorske neutralnosti
│   ├── evaluation.py        ← rizik, diversifikacija, faktorska atribucija, bootstrap zajedničkog uzorkovanja
│   ├── supervised.py        ← proširenje: predviđanje pada stablom odlučivanja / ansamblom
│   ├── viz.py               ← zajednički pomoćnici za crtanje (paleta, radar, težine klastera)
│   └── utils.py             ← konstante i putanje projekta
├── scripts/
│   └── build_notebooks.py   ← obnavlja notebookove 01-05 iz jedinstvenog izvora istine
├── outputs/
│   ├── figures/             ← spremljeni grafovi (PNG)
│   └── tables/              ← panel težina, prinosi portfelja, atribucija, bootstrap CI-jevi, osjetljivost
└── reports/
    └── final_report.md      ← dijagnostički izvještaj
```

---

## Reprodukcija

### 1. Okruženje

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pokreni notebookove redom

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_factors.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_portfolios.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_factor_neutral.ipynb
```

Notebook 01 dohvaća univerzum i dnevnu povijest cijena za ~1000 dionica. Prvo pokretanje traje ~14 minuta i sprema parquet predmemoriju po dionici u `data/raw/price_cache/`; sljedeća pokretanja ponovno koriste predmemoriju.

Notebook 02 odabire K, prilagođava klastere na svakom prozoru i provodi analizu stabilnosti s 500 bootstrap uzoraka (~5 minuta na reprezentativnom prozoru).

Notebook 03 pokreće unaprijedni backtest kroz svih 16 prozora (~30 sekundi).

Notebook 04 provodi evaluaciju izvan uzorka, blok-bootstrap zajedničkog uzorkovanja na razlikama metrika i prelet osjetljivosti po procjenitelju kovarijance × `w_max` × `group_cap` (~2 – 3 minute).

Notebook 05 provodi ε-prelet faktorske neutralnosti, hibrid (klaster + faktorska neutralnost) i prošireni bootstrap (~2 – 3 minute).

Ako želiš provjeriti samo analitiku (preskačući preuzimanje cijena), obrađeni CSV-ovi u `data/processed/` nose dovoljno stanja za pokretanje notebookova 02 – 05 zasebno.

---

## Izvori podataka

| Stavka | Izvor | Predmemorirano u |
|---|---|---|
| Univerzum Russell 1000 (oznaka, naziv, GICS sektor) | [Wikipedia „Russell 1000 Index”](https://en.wikipedia.org/wiki/Russell_1000_Index) — tablica sastavnica | `data/raw/russell1000_tickers.csv` |
| Dnevne prilagođene zaključne cijene | Yahoo Finance putem `yfinance` | `data/raw/price_cache/{TICKER}.parquet` |
| Fama-French petfaktorski podaci (mjesečni) | [Knjižnica podataka Kennetha Frencha](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | `data/processed/factors.csv` |
| Metapodaci snimke univerzuma | izračunato u trenutku dohvata | `data/raw/iwb_holdings_meta.json` |

Sloj podataka (`src/data.py`) prvo pokušava dohvatiti iShares IWB CSV udjela; Wikipedijina zamjenska tablica koristi se kad iShares ne posluži CSV koji se može parsirati. Odabrani izvor zabilježen je u `iwb_holdings_meta.json` za izvještaj.

---

## Metodologija u jednoj stranici

**Značajke dionice.** Za svaku dionicu i svaki prozor treniranja prilagođavamo Fama-French petfaktorsku regresiju na mjesečnim viškovima prinosa i bilježimo `(α, β_MKT, β_SMB, β_HML, β_RMW, β_CMA, σ_ε, R²)`. Šestodimenzionalni vektor `(β_MKT, β_SMB, β_HML, β_RMW, β_CMA, σ_ε)` skup je značajki za klasteriranje.

**Klasteriranje (dijagnostički alat).**
* *Faktorski klasteri* — Wardova veza na euklidskoj udaljenosti nakon standardizacije šest značajki. To je glavni dijagnostički objekt: otkriva faktorsku strukturu univerzuma i identificira u kojim se stilskim skupinama optimizator koncentrira. Klasteri *nisu* popravak.
* *Korelacijski klasteri* — potpuna veza na unaprijed izračunatoj korelacijskoj udaljenosti `sqrt(2 (1 – ρ))`. Koristi se kao osnovica.
* *GICS sektori* — treće označavanje, korišteno kao još jedna osnovica.

**Odabir K.** Prosječna silueta u prostoru faktora kroz sve klizne prozore, uz dva ograničenja:
1. Svaki klaster nosi ≥ 4 dionice u svakom prozoru.
2. `K × group_cap ≥ 1`, tako da je portfelj sa samo dugim pozicijama ograničen klasterima potpuno investibilan. Uz `group_cap = 0.15` to daje `K ≥ 7`.

Odabrani `K = 7` ponovno se koristi za korelacijsku osnovicu kako bi usporedbe bile istovrsne.

**Portfelji.** Portfelji sa samo dugim pozicijama, potpuno investirani po prozoru, svaki s `w_max = 0.02`:

*Pet dijagnostičkih portfelja* (notebook 03):
1. Jednake težine (`1/N`).
2. Minimalna varijanca, bez grupnog ograničenja.
3. Minimalna varijanca s `Σ_{i ∈ sector_j} w_i ≤ 0.15`.
4. Minimalna varijanca s `Σ_{i ∈ corr_cluster_k} w_i ≤ 0.15`.
5. Minimalna varijanca s `Σ_{i ∈ factor_cluster_k} w_i ≤ 0.15`.

*Faktorski neutralna obitelj* (notebook 05 — djelotvoran popravak):
6. Minimalna varijanca s `|w' β_f| ≤ ε` za `f ∈ {SMB, HML, RMW, CMA}` pri `ε ∈ {0.00, 0.05, 0.10, 0.15}`.
7. Hibrid: ograničenje klastera + faktorska neutralnost (uključen radi cjelovitosti; dominira ga čista faktorski neutralna obitelj).

Kovarijanca je Ledoit-Wolf sažimanje na mjesečnim prinosima treniranja, anualizirano s × 12.

**Unaprijedni hod.**
* **Prozor projekta:** `2005-01` do `2025-12` (21 godina mjesečnih podataka).
* **Treniranje / testiranje:** prozor od 60 mjeseci unatrag / 12 mjeseci unaprijed.
* **Ritam ponovnog procjenjivanja:** svakih 12 mjeseci → 16 disjunktnih testnih godina (2010. – 2025.).
* **Univerzum po prozoru:** dionice s ≥ 60 valjanih mjeseci treniranja i čistom petfaktorskom prilagodbom (granice kvalitete podataka na betama, rezidualnoj volatilnosti i R²).

**Bootstrap stabilnost.** 500 bootstrap ponovnih uzoraka univerzuma na reprezentativnom prozoru, uspoređujući dobiveno klasteriranje s klasteriranjem na punom uzorku putem Jaccarda zajedničke pripadnosti po parovima. Izvještava se po tipu klasteriranja.

Potpuni detalji, motivacija i ograničenja nalaze se u [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

## Status

| Faza | Opseg | Stanje |
|---|---|---|
| 1 | Prikupljanje podataka, mjesečni prinosi, faktori | **gotovo** |
| 2 | Petfaktorske regresije po prozoru | **gotovo** |
| 3 | Hijerarhijsko klasteriranje + bootstrap stabilnost | **gotovo** |
| 4 | Konstrukcija pet portfelja unaprijednim hodom | **gotovo** |
| 5 | Evaluacija izvan uzorka, bootstrap CI, prelet osjetljivosti | **gotovo** |
| 6 | Izravan popravak faktorske neutralnosti + završni izvještaj | **gotovo** |
| 7 | Klasteriranje usklađeno s kolegijem: K-means/k-medoid/GMM/DBSCAN + odabir K CH/Gapom (notebook 07) | **gotovo** |
| 8 | Nadzirano proširenje: stablo odlučivanja + ansambli na padu (notebook 08) | **gotovo** |

Vidi [`reports/final_report.md`](reports/final_report.md) za potpun dijagnostički izvještaj; § 11 pokriva pokrivenost metoda kolegija (algoritmi klasteriranja i nadzirano stablo). PDF-ovi predavanja nalaze se u [`materials/`](materials/) (vidi [`materials/README.md`](materials/README.md)).

---

## Licenca

Akademska uporaba. Vrijede licence izvornih podataka (Yahoo Finance, knjižnica podataka Kennetha Frencha, Wikipedia).

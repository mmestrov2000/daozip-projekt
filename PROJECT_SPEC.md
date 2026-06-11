# PROJECT_SPEC.md — dizajn i zaključane metodološke odluke

> **Status:** puni dokument (F0.11). Ovo je izvor istine za dizajn i zaključane
> metodološke odluke; `TASKS.md` je izvor istine za provedbu i kriterije
> prihvaćanja; `config.yaml` je jedino mjesto za parametre. Kod sukoba: dizajn
> ovdje, provedba u TASKS.md, brojevi u config.yaml.

---

## 1. Identitet rada

Kontrolirana studija stilskog rizika hijerarhijskih alokatora (HRP/HERC/NCO)
koja utvrđuje **gdje u cjevovodu hijerarhijske alokacije nastaje stilski rizik** —
u prostoru u kojem se hijerarhija gradi ili u mehanizmu alokacije/ograničenja.
Predmet su tri moderna hijerarhijska alokatora: **HRP** (López de Prado 2016),
**HERC** (Raffinot 2018) i **NCO** (López de Prado 2019).

Jedina manipulirana varijabla je **ulaz u izgradnju hijerarhije**: korelacijsko
stablo naspram Wardova stabla na standardiziranim FF5 značajkama. Da usporedba
prostora ne konfundira metodu veze, kontrolirana usporedba drži vezu konstantnom
— **Wardova veza u oba kraka** (`hrp_corr_ward` naspram `hrp_factor`); jednostruka
veza vjerna izvorniku (`hrp_corr_single`) zadržava se isključivo kao replikacijski
red Faze 1. Sve ostalo je identično: point-in-time S&P 500 univerzum 2000.–2025.,
prozori 60/12/12 mj., Ledoit–Wolf Σ za sve korake rizika, isti `w_max`, iste
metrike. U Fazi 4 dodaje se **projekcijski QP overlay** `|w'β_f| ≤ ε` na
hijerarhijske težine → potpuni 2×2 faktorijal {prostor hijerarhije} × {bez/sa
ograničenjem}.

Primarni ishod je **koncentracija stila**
`style_concentration = |β_SMB|+|β_HML|+|β_RMW|+|β_CMA|` iz FF5 atribucije testnih
prinosa. Stari projekt (min-var + izravna faktorska neutralnost) ugnježđuje se kao
benchmark obitelj, ne baca se.

Osoba bez `TASKS.md` treba iz ovog dokumenta razumjeti dizajn; provedbeni
koraci i kriteriji prihvaćanja žive u `TASKS.md`.

---

## 2. Hipoteze (doslovno)

- **H1:** korelacijski HRP/HERC/NCO nasljeđuju skriveni RMW/CMA nagib min-vara
  (mehanizam: inverzno-varijančna alokacija favorizira niskovolatilne grane;
  Scherer 2011, Novy-Marx 2014). — *testira se u Fazi 2.*
- **H2:** ni faktorski prostor hijerarhije ne neutralizira nagib (lekcija
  postojećeg Dijela II: upravljanje težinama skupina ne cilja pojedinačnu betu),
  ali donosi mjerljive dobitke u stabilnosti klastera, obrtaju i
  interpretabilnosti (argument redukcije dimenzije: ~N²/2 korelacija naspram 6N
  beta). — *testira se u Fazi 3.*
- **H3:** jedini mehanizam koji nagib stvarno uklanja je izravno linearno
  ograničenje na bete; primijenjeno kao overlay na hijerarhijske težine, dominira
  na granici (koncentracija stila × ostvarena volatilnost). — *testira se u
  Fazi 4.*

---

## 3. Metodologija

### 3.1 Univerzum i razdoblje

Point-in-time S&P 500, razdoblje projekta 2000-01..2025-12. Panel se gradi preko
**unije svih tickera koji su ikad članovi** u tom razdoblju (`union_universe`),
a po prozoru se reže na članstvo na datum `train_end`.

### 3.2 Klizni prozori

`generate_rolling_windows` (`src/backtest.py`) s parametrima iz `config.yaml`:
trening 60 mj., test 12 mj., refit korak 12 mj. → **21 prozor**; prvi
`label = "2004-12"` (testna godina 2005.), zadnja testna godina završava 2025-12.
`label` je uključivi završni mjesec treniranja (`YYYY-MM`) i kanonski je
identifikator u faktorskim izloženostima, tablicama klasteriranja i panelima
težina. Filtar uloživosti: ≥ 60 mjeseci treniranja (`min_training_months`).

### 3.3 Procjena rizika

**Ledoit–Wolf Σ za sve korake rizika** (zaključana odluka 6); računa se jednom
po prozoru i dijeli među alokatorima. **Bete služe isključivo za izgradnju
hijerarhije, nikad za procjenu rizika.**

### 3.4 Pretpostavka fiksnih težina i transakcijski troškovi (riješeno pitanje 2)

Težine su **fiksne unutar testne godine** (mjesečno se ne rebalansira unutar
prozora; pretpostavka eksplicitno zapisana). Transakcijski trošak obračunava se
**samo na refit obrtaj**: jednostrani obrtaj `0.5·Σ_i |w_{i,t} − w_{i,t−1}|` po
(portfelj, prozor), prvi prozor = 1.0 (puna izgradnja iz gotovine). Trošak
`turnover · tc_bps / 10000` (`tc_bps = 10`) skida se s prinosa **prvog mjeseca
testne godine** (`turnover_per_window`/`apply_costs`, `src/evaluation.py`).
Drift-korigirana verzija ostaje kao stupac robusnosti ako ostane vremena.

### 3.5 Ograničenja portfelja

`w_max = 0.02` po imovini; `group_cap = 0.15`. Nametanje `w_max` (riješeno
pitanje 1): iterativno odsijecanje na `w_max` s renormalizacijom ostatka,
identično za sva tri alokatora i oba prostora. Uz težine se vraća
`capped_weight_share` (udio težine na capu) po (portfelj, prozor), koji runner
bilježi u status tablicu; ako cap često grize, navodi se u ograničenjima rada.

---

## 4. Alokatori

Svi alokatori dijele isti `apply_w_max` post-korak (identičan mehanizam capa kroz
sva tri, uključujući `capped_weight_share`). Ulazno stablo/particija uvijek je
vanjski parametar (`src/hierarchical.py`), pa ista funkcija služi svim krakovima
usporedbe.

### 4.1 HRP (López de Prado 2016)

Kvazidijagonalizacija poretka listova iz linkage matrice + rekurzivna bisekcija s
inverzno-varijančnom podjelom; **rizik uvijek iz proslijeđene Ledoit–Wolf Σ**.

### 4.2 HERC (Raffinot 2018)

Rez stabla na `k` klastera (`fcluster`), top-down podjela kapitala niz stvarni
dendrogram po jednakom doprinosu riziku među klasterima (rizik klastera iz Σ
pod-bloka s naivnim inverzno-varijančnim težinama), naivni risk parity (1/σ_i)
unutar klastera. `k` dolazi iz primarne procedure odabira K.

### 4.3 NCO (López de Prado 2019)

**Particija = rez hijerarhijskog stabla na K** u oba kraka, *umjesto k-meansa iz
izvornog rada* (riješeno pitanje 7); odstupanje se navodi kao fusnota u
replikacijskoj tablici (F1.6); k-means particija ostaje kao dodatna robusnost
(F3.5). Min-var **unutar** svakog klastera (ponovna uporaba `min_variance`) bez
capa; reducirana kovarijanca klastera `Σ_red = W' Σ W`; min-var **među**
klasterima; na kraju isti `apply_w_max` na kombinirane konačne težine.
**Korekcija K2:** unutar-klasterski cap se *ne* izvodi iz težine klastera (težine
klastera poznate tek nakon među-klasterskog koraka), nego se cap primjenjuje
jednom, na kombinirane finalne težine.

### 4.4 Veze (linkage) po kraku usporedbe — korekcija K1

- `hrp_corr_single` — jednostruka veza na d = √(½(1−ρ)) + kvazidijagonalizacija +
  rekurzivna bisekcija; **vjerna replikacija izvornika, isključivo replikacijski
  red** (ne ulazi u uzročnu usporedbu prostora ni u overlay).
- `hrp_corr_ward` — **Wardova veza na korelacijskoj udaljenosti**; nositelj
  kontrolirane usporedbe prostora. Faza 3 manipulira **samo** prostorom jer je
  veza konstantna (Ward) u oba kraka (`hrp_corr_ward` ↔ `hrp_factor`).
- HERC i NCO: Wardova veza u oba kraka; particija/rez stabla na isti K.

Napomena: postojeći `correlation_cluster` (`src/clustering.py`) namjerno odbija
Ward na unaprijed izračunatoj udaljenosti i **ne mijenja se**; stabla za
alokatore gradi zasebna funkcija u `src/hierarchical.py` s dokumentiranim
Lance–Williamsovim opravdanjem (standardna praksa u HERC literaturi).

### 4.5 Faktorski prostor hijerarhije

`tree_space="factor"`: po prozoru se dohvati 6 standardiziranih FF5 značajki
(`FACTOR_FEATURE_COLUMNS`) iz `factor_exposures.csv`, pozove `factor_cluster`
(`src/clustering.py`) koji vraća i linkage matricu; ta linkage ide u HRP/HERC, a
njezin rez na K u NCO. Univerzum, Σ i `w_max` identični su korelacijskoj grani —
jedina promjena je ulazno stablo/particija.

---

## 5. Overlay QP (Faza 4)

`project_factor_neutral(w0, Sigma, factor_betas, w_max, epsilon, norm)`
(`src/portfolio.py`): riješiti

```
min_w (w − w0)' Σ (w − w0)   uz   1'w = 1,   0 ≤ w ≤ w_max,
|w'β_f| ≤ ε   za svaki f ∈ {SMB, HML, RMW, CMA}
```

`norm="sigma"` (primarno) = minimalni tracking error prema hijerarhijskom
portfelju; `norm="euclidean"` (`min ‖w−w0‖²`) kao robusnost. Lanac rješavača
CLARABEL→OSQP→SCS, `psd_wrap`. ε-mreža `[0.0, 0.05, 0.10, 0.15]` → potpuni 2×2
faktorijal {prostor} × {bez/sa overlayem}; ukupno 3 alokatora × 2 prostora × 4 ε
= 24 overlay varijante (uz dokumentirane neizvedivosti za ε=0 u ranim prozorima).
Bete su iz `factor_exposures.csv` istog prozora (bez gledanja unaprijed).
Korelacijska baza za HRP overlay je **`hrp_corr_ward`** (K1); `hrp_corr_single`
ne dobiva overlay.

---

## 6. Metrike

- **Primarni ishod:** `style_concentration = |β_SMB|+|β_HML|+|β_RMW|+|β_CMA|` iz
  FF5 atribucije testnih prinosa (`factor_attribution*`, `src/evaluation.py`).
- **Rizik:** anualizirana volatilnost (×√12), maksimalni drawdown, VaR/CVaR.
- **Prinos prilagođen riziku:** `sharpe_ratio(returns, rf)` (bruto i neto na
  10 bps; `rf` skalar ili Series, npr. FF5 stupac `RF`).
- **Obrtaj:** `turnover_per_window` (jednostrani, vidi §3.4).
- **Diversifikacija/koncentracija:** HHI, efektivni N, omjer diversifikacije.
- **Stabilnost klastera:** ARI između uzastopnih prozora (Faza 3).

---

## 7. Statistika

### 7.1 Blok-bootstrap

Bootstrap pomičnim blokovima (Künsch 1989; `_block_indices`): blok 12 mj., 1000
uzoraka za prinose / 500 za klastere, sjeme 42 (sve iz `config.yaml`). 95 %
percentilni CI i dvostrana p-vrijednost; parne razlike koriste **zajedničko
uzorkovanje** (isti ponovno uzorkovani blok na oba portfelja).

### 7.2 Model Confidence Set (Hansen–Lunde–Nason 2011)

Gubitak po defaultu `l_t = (r_t − r̄)²` (`mcs_loss: "sq_demeaned"`), α = 0.10
(`mcs_alpha`), 1000 blok-bootstrap uzoraka, konfigurabilno. Iterativna
eliminacija po max-t statistici.

**Pravilo za neuravnotežen panel (korekcija K3, zaključano unaprijed):** MCS se
računa na **presjeku mjeseci dostupnih svim uspoređenim varijantama**; broj
ispuštenih mjeseci obavezno se izvještava (`n_months_used`, `n_months_dropped`);
varijanta kojoj presjek odnese **> 20 % njezinih mjeseci isključuje se iz MCS-a**
(ostaje u parnim bootstrap usporedbama) i navodi se u fusnoti.

### 7.3 Deflated Sharpe Ratio (Bailey–López de Prado 2014)

`deflated_sharpe_ratio(returns, n_trials, rf)`: procjena SR uz korekciju za
skewness/kurtosis i za očekivani maksimum preko `n_trials` isprobanih varijanti.
`n_trials` broji **sve** isprobane varijante završne usporedbe (cijela ε-mreža +
obje norme; red veličine ~34+, programatski iz stupaca master panela).

### 7.4 Podjela posla MCS naspram DSR (riješeno pitanje 3)

**MCS** odgovara na pitanje **„najmanja ostvarena varijanca”** (koji su modeli
statistički nerazlučivi od najboljeg po gubitku varijance). **DSR** odgovara na
**riziku prilagođeni prinos** (je li Sharpe statistički značajan nakon korekcije
za višestruko isprobavanje). Dvije metrike, dvije različite tvrdnje.

---

## 8. Izvori podataka

### 8.1 Izvor članstva

**Odluka (2026-06-11): grana (b) — javna rekonstrukcija.** Institucionalni
WRDS/CRSP pristup nije dostupan (potvrđeno 2026-06-11), pa se point-in-time
članstvo S&P 500 rekonstruira iz javnih izvora:

- **Primarni izvor:** GitHub [`fja05680/sp500`](https://github.com/fja05680/sp500),
  datoteka „S&P 500 Historical Components & Changes (Updated).csv” — snapshot
  članova po svakom datumu promjene, 1996-01 do danas, održavana. Parsira se u
  dugu tablicu intervala `ticker, name, start_date, end_date, source` →
  `data/raw/sp500_membership.csv` (funkcija `fetch_sp500_membership`,
  `src/data.py`).
- **Kontrolni izvor:** spot-provjere poznatih indeksnih događaja (izvedene iz
  Wikipedijine tablice promjena „List of S&P 500 companies”) ugrađene u
  `tests/test_membership.py`: TSLA ulazi 2020-12, GM izlazi 2009-06, LEH izlazi
  2008-09, AIG ostaje član nakon 2008; uz to, broj članova na svaki od 21
  `train_end` datuma mora biti u [490, 510]. Sve provjere prolaze na primarnom
  izvoru.
- **EODHD:** neaktiviran — po riješenom pitanju 5 aktivira se samo ako
  spot-provjere padnu, što nije slučaj.

**Napomena o simbolima:** „Updated” datoteka delistane firme vodi pod zadnjim
OTC simbolom (Lehman = LEHMQ, stari GM = MTLQQ), a preimenovanja preživjelih
(npr. FB→META) primjenjuje retroaktivno. Mapa `data/raw/ticker_overrides.csv`
(stupci `source_ticker, ticker, download_ticker, note`) vraća povijesne
simbole (LEHMQ→LEH, MTLQQ→GM); GM zato ima dva disjunktna intervala članstva
(stari GM do 2009-06, novi GM od 2013-06). Ista datoteka služi i za
premošćivanje simbola pri preuzimanju cijena (F0.6).

### 8.2 Cijene

`download_prices_cached` (`src/data.py`): primarno yfinance, sekundarno Stooq
(`{ticker}.US`) kad yahoo vrati prazno; `price_source ∈ {yahoo, stooq, none}` po
tickeru. Parquet predmemorija po tickeru + retry. Pokrivenost po prozoru mjeri se
i izvještava (`00_membership_coverage.csv`, `00_price_source_summary.png`) —
nepoznata pristranost preživjelih postaje izmjerena veličina.

### 8.3 Faktori

FF5 (Ken French) preko `download_ff5_factors`; raspon datuma projekta, logika
netaknuta.

---

## 9. Zaključane metodološke odluke (1–10)

Numeracija prati reference u `TASKS.md`; svaka stavka traceabilna je na riješena
pitanja (RP, §3 TASKS.md, odluke 2026-06-10) ili na narativ dizajna.

1. **Nametanje `w_max`** (RP1): iterativno odsijecanje na `w_max` s
   renormalizacijom ostatka, identično za oba prostora; stupac
   `capped_weight_share` po (portfelj, prozor).
2. **Transakcijski troškovi** (RP2): trošak samo na refit obrtaj
   `0.5·Σ|w_novi − w_stari|`; fiksne težine unutar testne godine; drift-korigirana
   verzija kao robusnost.
3. **MCS gubitak** (RP3): `l_t = (r_t − r̄)²`, α = 0.10, 1000 uzoraka,
   konfigurabilno; podjela posla MCS/DSR (vidi §7.4).
4. **Commit politika izlaza** (RP4): committaju se samo finalne tablice/figure
   koje izvještaj citira; međupaneli ne.
5. **Odabir primarnog K:** postojeća procedura iz notebooka 02 (silueta uz uvjete
   uloživosti), ista za oba prostora; vlastiti-optimalni K po prostoru kao
   robusnost (F3.5).
6. **Ledoit–Wolf Σ za sve korake rizika;** bete služe isključivo za izgradnju
   hijerarhije, ne za rizik.
7. **NCO particija = rez stabla na K** u oba kraka (umjesto k-meansa iz izvornog
   rada; RP7); k-means kao dodatna robusnost.
8. **EODHD provjera članstva** (RP5): aktivira se samo ako spot-provjere javne
   rekonstrukcije padnu; izvor članstva = grana (b), vidi §8.1.
9. **Faktorski neutralna obitelj** (stari projekt, min-var + izravna
   neutralnost): ostaje kao benchmark stupac, ne baca se.
10. **Stari projekt se ugnježđuje kao benchmark obitelj,** ne baca se; nadzirano
    proširenje (notebook 08, `supervised.py`) arhivirano kao negativan rezultat;
    izvještaj i README pišu se tek u zadnjoj fazi.

**Kontrolne korekcije (uzročna usporedba):**

- **K1:** kontrolirana usporedba prostora drži vezu konstantnom (Ward u oba
  kraka); `hrp_corr_ward` je nositelj uzročne usporedbe, `hrp_corr_single` samo
  vjerni replikacijski red.
- **K2:** NCO unutar-klasterski cap se ne izvodi iz težine klastera; jedinstveni
  `apply_w_max` na kombinirane finalne težine.
- **K3:** MCS/DSR na presjeku mjeseci svih varijanti; > 20 % ispuštenih →
  isključenje iz MCS-a (ostaje u parnim usporedbama).

---

## 10. Konvencije imenovanja izlaza

Notebookovi uvoze `src/` module i pišu u `outputs/tables` i `outputs/figures` s
numeričkim prefiksom:

- **00** — podatkovni temelj (pokrivenost članstva, izvori cijena): Faza 0.
- **09** — replikacija u korelacijskom prostoru (HRP/HERC/NCO, validacija,
  MCS replikacije): Faza 1, notebook `09_hierarchical_replication.ipynb`.
- **10** — dijagnoza (FF5 atribucija, bootstrap, skrivene stilske izloženosti):
  Faza 2, notebook `10_hierarchical_diagnosis.ipynb`.
- **11** — intervencija u faktorskom prostoru (parne razlike prostora, ARI,
  obrtaj, K-robusnost): Faza 3, notebook `11_factor_space_intervention.ipynb`.
- **12** — overlay mehanizam (2×2 faktorijal, DSR, granica stil–vol, finalni
  MCS, H3): Faza 4, notebook `12_overlay_mechanism.ipynb`.
- **13** — finalni sažetak (master tablica, 2×2 faktorijalna tablica): Faza 5,
  notebook `13_final_summary.ipynb`.

Datoteke se imenuju `{prefiks}_{opis}.{csv|png}`; overlay varijante:
`{alokator}_{space}_ov_e{ε}`.

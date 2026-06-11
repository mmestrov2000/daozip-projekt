# TASKS.md — plan preformulacije projekta

> **Status:** plan implementacije, faza 0–5. Ovaj dokument je izvor istine za provedbu.
> Svaki task ima ID (`F<faza>.<broj>`), checkbox, popis datoteka, provjerljiv kriterij
> prihvaćanja, ovisnosti i procjenu (S ≤ 2 h, M ≈ pola dana, L ≈ dan).

---

## 1. Sažetak novog dizajna

Projekt se preformulira iz „dijagnoza skrivenog stilskog rizika min-var portfelja”
u **kontroliranu studiju koja utvrđuje gdje u cjevovodu hijerarhijske alokacije
nastaje stilski rizik** — u prostoru u kojem se hijerarhija gradi ili u mehanizmu
alokacije/ograničenja. Predmet su tri moderna hijerarhijska alokatora: **HRP**
(López de Prado 2016), **HERC** (Raffinot 2018) i **NCO** (López de Prado 2019).
Jedina manipulirana varijabla je **ulaz u izgradnju hijerarhije**: korelacijsko
stablo naspram Wardova stabla na standardiziranim FF5 značajkama (kod postoji u
`src/clustering.py::factor_cluster`). Da usporedba prostora ne konfundira metodu
veze, kontrolirana usporedba drži vezu konstantnom — **Wardova veza u oba kraka**
(`hrp_corr_ward` naspram `hrp_factor`); jednostruka veza vjerna izvorniku
(`hrp_corr_single`) zadržava se isključivo kao replikacijski red Faze 1. Sve
ostalo je identično: point-in-time S&P 500 univerzum 2000.–2025., prozori
60/12 mj., Ledoit–Wolf Σ za sve korake rizika, isti `w_max`, iste metrike. U Fazi 4 dodaje se
**projekcijski QP overlay** `|w'β_f| ≤ ε` na hijerarhijske težine → potpuni 2×2
faktorijal {prostor hijerarhije} × {bez/sa ograničenjem}. Primarni ishod je
**koncentracija stila** `|β_SMB|+|β_HML|+|β_RMW|+|β_CMA|` iz FF5 atribucije testnih
prinosa; statistika: postojeći blok-bootstrap + Model Confidence Set + Deflated
Sharpe Ratio. Stari projekt (min-var + izravna faktorska neutralnost) ugnježđuje se
kao benchmark obitelj, ne baca se.

**Hipoteze (doslovno, testiraju se u fazama 2–4):**

- **H1:** korelacijski HRP/HERC/NCO nasljeđuju skriveni RMW/CMA nagib min-vara
  (mehanizam: inverzno-varijančna alokacija favorizira niskovolatilne grane;
  Scherer 2011, Novy-Marx 2014).
- **H2:** ni faktorski prostor hijerarhije ne neutralizira nagib (lekcija
  postojećeg Dijela II: upravljanje težinama skupina ne cilja pojedinačnu betu),
  ali donosi mjerljive dobitke u stabilnosti klastera, obrtaju i
  interpretabilnosti (argument redukcije dimenzije: ~N²/2 korelacija naspram 6N
  beta).
- **H3:** jedini mehanizam koji nagib stvarno uklanja je izravno linearno
  ograničenje na bete; primijenjeno kao overlay na hijerarhijske težine, dominira
  na granici (koncentracija stila × ostvarena volatilnost).

---

## 2. Eksplicitne pretpostavke

1. **Snapshot repoa je nepotpun u odnosu na README.** `PROJECT_SPEC.md`,
   `TASKS.md`, `scripts/build_notebooks.py`, `materials/` i
   `reports/final_report.md` ne postoje u trenutnom stablu iako ih `README.md`
   referencira. Pretpostavljam da su namjerno uklonjeni pri čišćenju; ovaj
   TASKS.md pisan je iznova, a `PROJECT_SPEC.md` se piše iznova u Fazi 0.
2. **Izvor istine za izvještaj je `reports/izvjestaj.tex`** (s figurama u
   `reports/figures/fig01–fig08`), ne `final_report.md`.
3. **Tablice notebookova 01–06 nisu u repou** (`outputs/` sadrži samo `07_*` i
   nije pod gitom) — sve se ionako regenerira na novom univerzumu, pa se stare
   tablice ne migriraju.
4. **Notebookovi su izvor pokretanja analiza** (nema `scripts/`); novi
   notebookovi 09–13 slijede postojeću konvenciju: uvoze `src/` module, pišu u
   `outputs/tables` i `outputs/figures` s numeričkim prefiksom.
5. **`tests/` ne postoji i `pytest` nije u `requirements.txt`** — testna
   infrastruktura se uvodi u Fazi 0 (kriteriji prihvaćanja je pretpostavljaju).
6. **Razdoblje 2000–2025 s prozorima 60/12/12 daje 21 testnu godinu
   (2005.–2025.)** — `generate_rolling_windows` to već podržava čistom promjenom
   parametara (`src/backtest.py:60`), bez izmjene logike.
7. **Veze (linkage) po alokatoru i kraku usporedbe:** vjerna replikacija HRP-a =
   jednostruka veza na d = √(½(1−ρ)) + kvazidijagonalizacija + rekurzivna
   bisekcija (`hrp_corr_single`; samo replikacijski red Faze 1). **Kontrolirana
   usporedba prostora drži vezu konstantnom:** Wardova veza u oba kraka —
   `hrp_corr_ward` naspram `hrp_factor` — inače bi usporedba mijenjala dvije
   stvari odjednom (prostor + vezu). HERC = Wardova veza u oba kraka; NCO =
   particija rezom stabla na K u oba kraka (umjesto k-meansa iz izvornog rada —
   riješeno pitanje 7, fusnota u F1.6). Napomena: `correlation_cluster`
   (`src/clustering.py:125`) namjerno odbija Ward na unaprijed izračunatoj
   udaljenosti i **ne dira se**; stabla za alokatore gradi zasebna funkcija u
   `src/hierarchical.py` (task F1.1b) uz dokumentirano Lance–Williamsovo
   opravdanje (standardna praksa u HERC literaturi).
8. **Necommitana izmjena u `src/clustering_ext.py`** (DBSCAN `eps: 1.5 → 1000.0`)
   izgleda kao zaostatak eksperimenta (eps=1000 u standardiziranom prostoru
   stavlja sve u jedan klaster); vraća se na 1.5 (task F0.12) — potvrđeno
   (riješeno pitanje 6, 2026-06-10).

---

## 3. Riješena pitanja (odluke 2026-06-10)

1. **Nametanje `w_max = 0.02` u HRP/HERC** — *odlučeno 2026-06-10:* prihvaćen
   default — iterativno odsijecanje na `w_max` s renormalizacijom ostatka,
   identično za oba prostora, dokumentirano u `PROJECT_SPEC.md`. **Dodatak:** u
   status tablicu ide stupac `capped_weight_share` („udio težine na capu”) po
   (portfelj, prozor) — vidi F1.1 i F1.5; ako cap često grize, navodi se u
   ograničenjima rada (F5.2).
2. **Obračun transakcijskih troškova** — *odlučeno 2026-06-10:* prihvaćen
   default — trošak samo na refit obrtaj `0.5·Σ|w_novi − w_stari|`;
   drift-korigirana verzija kao stupac robusnosti ako ostane vremena.
   **Dodatak:** pretpostavka fiksnih težina unutar testne godine eksplicitno se
   zapisuje u `PROJECT_SPEC.md` §Metodologija (vidi F0.11).
3. **MCS funkcija gubitka** — *odlučeno 2026-06-10:* prihvaćen default
   `l_t = (r_t − r̄)²`, α = 0.10, 1000 blok-bootstrap uzoraka, konfigurabilno.
   **Dodatak:** `PROJECT_SPEC.md` dobiva rečenicu o podjeli posla — MCS odgovara
   na „najmanja ostvarena varijanca”, DSR na riziku prilagođeni prinos (F0.11).
4. **Commit politika izlaza** — *odlučeno 2026-06-10:* prihvaćen default —
   committaju se samo finalne tablice/figure koje izvještaj citira (F5.5),
   međupaneli ne.
5. **EODHD provjera članstva** — *odlučeno 2026-06-10:* prihvaćen default —
   aktivira se samo ako spot-provjere javne rekonstrukcije (F0.4) padnu.
6. **`clustering_ext.py` eps** — *odlučeno 2026-06-10:* potvrđeno — vratiti na
   1.5; F0.12 ostaje kako jest.
7. **Particija za NCO** — *odlučeno 2026-06-10:* prihvaćen default — particija =
   rez hijerarhijskog stabla na K u oba kraka. **Dodatak:** odstupanje od
   izvornog k-meansa navodi se kao fusnota u replikacijskoj tablici (F1.6) i u
   `PROJECT_SPEC.md` (F0.11); k-means robusnost ostaje u F3.5.

---

## 4. Registar rizika

| Rizik | Vjerojatnost | Učinak | Mitigacija |
|---|---|---|---|
| Yahoo nema cijene za delistane tickere (LEH, WB, …) → pristranost preživjelih ostaje djelomična | visoka | srednji | Stooq fallback (F0.6); pokrivenost po prozoru kao tablica+slika (F0.7) → nepoznata pristranost postaje izmjerena veličina; ograničenje u izvještaju |
| Pogreške u javnoj rekonstrukciji članstva S&P 500 | srednja | visok | spot-provjere poznatih događaja (F0.4); usporedba dvaju javnih izvora; opcija EODHD (riješeno pitanje 5) |
| Ponovna uporaba tickera (isti simbol, druga firma kroz vrijeme) | srednja | srednji | filtar `first_valid_month`/`last_valid_month` u metadata + ručna lista poznatih kolizija u F0.6; postojeći filtri kvalitete beta (`estimate_betas_window`) hvataju većinu artefakata |
| riskfolio-lib/skfolio se ne instaliraju uz pinove `numpy 2.4 / pandas 3.0` | srednja | nizak | validacija u zasebnom venv-u; dovoljna je jednokratna usporedba na sintetičkom primjeru (F1.4), biblioteke nisu runtime ovisnost |
| Numeričke razlike vlastitih HRP/HERC/NCO naspram biblioteka (različite varijante algoritama) | visoka | srednji | sintetički test s dokumentiranom tolerancijom (F1.4); razlike se objašnjavaju u `PROJECT_SPEC.md`, ne skrivaju |
| Trajanje backtesta (21 prozor × ~12 portfelja × ε-mreža × 2 norme) | srednja | srednji | Σ i stabla se računaju jednom po prozoru i dijele među alokatorima (F1.5); paneli se spremaju u CSV pa se faze 2–4 ne preračunavaju |
| Neizvedivost overlay QP-a za ε = 0 u ranim prozorima (poznato iz starog projekta: hibridi 168/180 mj.) | visoka | nizak | status log po prozoru kao u `portfolio_status`; tablice izvještavaju `n_months`; lanac rješavača CLARABEL→OSQP→SCS već postoji u `_solve_min_variance`; za MCS vrijedi unaprijed zaključano pravilo presjeka mjeseci s pragom 20 % (K3; F1.7/F4.6) |
| Ward na unaprijed izračunatoj korelacijskoj udaljenosti metodološki je upitan (postojeći `correlation_cluster` ga zato eksplicitno odbija) | sigurna (svjesna odluka) | srednji | gradnja stabla u zasebnoj funkciji `src/hierarchical.py` (F1.1b) s dokumentiranim Lance–Williamsovim opravdanjem (standard u HERC literaturi); guard u `src/clustering.py` se ne dira; `hrp_corr_single` zadržan kao vjerna replikacija pa je svako odstupanje vidljivo u F1.6 |
| MCS/DSR implementacijske greške | srednja | visok | sintetički testovi s poznatim ishodom (F1.7, F4.4); DSR provjeren na ručno izračunatom primjeru |
| Veći union univerzum (~1100–1300 tickera) → duže prvo preuzimanje | visoka | nizak | postojeća parquet predmemorija po tickeru (`download_prices_cached`) + retry; jednokratan trošak |

---

# Faza 0 — Podatkovni temelj i čišćenje

**Cilj:** zamijeniti trenutni Russell 1000 point-in-time S&P 500 univerzumom
2000.–2025. s izmjerenom pokrivenošću, uvesti sloj troškova, konfiguraciju i
testnu infrastrukturu, te očistiti opseg (arhiviranje notebooka 08).

**Čega se NE diramo u ovoj fazi:** `src/clustering.py`, `src/portfolio.py`,
`src/backtest.py`, `src/evaluation.py` (osim čistog dodavanja sloja troškova),
notebookovi 02–07, `reports/izvjestaj.tex`.

### Taskovi

- [x] **F0.1 — Konfiguracijska datoteka projekta** (M)
  - Opis: stvoriti `config.yaml` u korijenu (PyYAML je već u requirements) s:
    `project_start: "2000-01"`, `project_end: "2025-12"`, prozori
    (60/12/12), `w_max: 0.02`, `group_cap: 0.15`, ε-mreža
    `[0.0, 0.05, 0.10, 0.15]`, `tc_bps: 10`, bootstrap parametri
    (1000/500/12), `random_seed: 42`, `mcs_alpha: 0.10`, MCS gubitak.
    Novi modul `src/config.py` učitava YAML; **konstante u `src/utils.py`
    (retci 22–41) vežu se na učitane vrijednosti** tako da svi postojeći
    importi (`from src.utils import W_MAX, …`) rade bez izmjene.
  - Datoteke: `config.yaml` (novo), `src/config.py` (novo), `src/utils.py`.
  - Prihvaćanje: `python -c "from src.utils import PROJECT_START; print(PROJECT_START)"`
    ispisuje `2000-01`; svi postojeći importi rade; promjena vrijednosti u YAML-u
    mijenja konstantu bez diranja koda.
  - Ovisnosti: —
  - *Odstupanje:* uz retke 22–41 dodane su i konstante `EPSILON_GRID, TC_BPS, MCS_ALPHA, MCS_LOSS` (vežu se na iste config ključeve koje F0.1 propisuje); environment nema `python` alias pa je korišten `.venv/bin/python`.
- [x] **F0.2 — Testna infrastruktura** (S)
  - Opis: dodati `pytest` u `requirements.txt`, stvoriti `tests/` s
    `tests/test_config.py` (smoke: konstante se učitavaju, ε-mreža ima 4 člana,
    sjeme je int). Svi kasniji taskovi s kriterijem „test prolazi” dodaju
    datoteke ovdje.
  - Datoteke: `requirements.txt`, `tests/test_config.py` (novo).
  - Prihvaćanje: `python -m pytest tests/ -q` prolazi zeleno.
  - Ovisnosti: F0.1
- [x] **F0.3 — Arhiviranje nadziranog proširenja** (S)
  - Opis: `git mv notebooks/08_drawdown_tree.ipynb archive/` i
    `git mv src/supervised.py archive/`; premjestiti postojeće `08_*` izlaze ako
    se pojave; dodati `archive/README.md` s jednom rečenicom (negativan rezultat:
    oznaka klastera ne dodaje ništa povrh beta; u radu ostaje jedna rečenica u
    raspravi).
  - Datoteke: `notebooks/08_drawdown_tree.ipynb`, `src/supervised.py`,
    `archive/` (novo).
  - Prihvaćanje: `grep -rn "supervised" src/ notebooks/ --include="*.py" --include="*.ipynb"`
    ne vraća aktivne importe; `archive/` sadrži obje datoteke i README.
  - Ovisnosti: —
- [x] **F0.4 — Provjera WRDS/CRSP pristupa (grananje plana)** (S)
  - Opis: poslati upit mentoru/knjižnici PMF-a o institucionalnom WRDS pristupu.
    **Grananje:** (a) ako pristup postoji → F0.5 koristi CRSP point-in-time
    članstvo i delistane prinose (tada F0.6 postaje opcionalan); (b) ako ne (ili
    nema odgovora u 5 radnih dana) → javna rekonstrukcija: GitHub
    `fja05680/sp500` („S&P 500 Historical Components & Changes”) kao primarni
    izvor + revizije Wikipedijine stranice „List of S&P 500 companies” (tablica
    promjena) kao kontrolni; (c) EODHD samo po riješenom pitanju 5.
  - Datoteke: odluka se dokumentira u `PROJECT_SPEC.md` §Podaci (F0.11).
  - Prihvaćanje: u `PROJECT_SPEC.md` postoji odlomak „Izvor članstva” s odabranom
    granom i datumom odluke.
  - Ovisnosti: —
  - *Odstupanje:* nema WRDS pristupa (potvrđeno 2026-06-11) → grana (b): `fja05680/sp500` (Updated) primarni, kontrola = spot-provjere poznatih događaja u `tests/test_membership.py` (prolaze, EODHD neaktiviran); odlomak „Izvor članstva” zapisan u minimalni stub `PROJECT_SPEC.md` (puni dokument ostaje F0.11).
- [x] **F0.5 — Point-in-time sloj članstva u `src/data.py`** (L)
  - Opis: nove funkcije `fetch_sp500_membership()` (preuzima/parsira izvor iz
    F0.4 u dugu tablicu `ticker, name, start_date, end_date, source` →
    `data/raw/sp500_membership.csv`) i `membership_on(date)` /
    `universe_for_window(window)` (vraća članove na datum `window.train_end`).
    GICS sektor po tickeru zadržava se iz postojećeg metadata sloja gdje postoji,
    inače iz izvora članstva. Stari `fetch_russell1000_tickers` ostaje u modulu
    (povijesna usporedivost), ali ga `preprocess` više ne zove.
  - Datoteke: `src/data.py`, `data/raw/sp500_membership.csv` (novo),
    `tests/test_membership.py` (novo).
  - Prihvaćanje: test sa spot-provjerama prolazi: TSLA ulazi 2020-12, GM izlazi
    2009-06, LEH izlazi 2008-09, AIG ostaje član nakon 2008; za svaki od 21
    prozora `len(universe_for_window(w))` je u [490, 510] — raspon se odnosi na
    **point-in-time članstvo prije filtara kvalitete** (investabilni univerzum
    nakon filtra ≥ 60 mj. treninga bit će manji, osobito u ranim prozorima, i
    test zbog toga ne smije pasti).
  - Ovisnosti: F0.1, F0.2, F0.4
  - *Odstupanje:* izvor delistane vodi pod zadnjim OTC simbolom (LEHMQ, MTLQQ) pa je `data/raw/ticker_overrides.csv` (formalno F0.6) uveden već ovdje s mapom LEHMQ→LEH, MTLQQ→GM; GM zato ima dva disjunktna intervala (do 2009-06 i od 2013-06).
- [x] **F0.6 — Rukovanje delistanim tickerima (Stooq dopuna)** (M)
  - Opis: u `download_prices_cached` (`src/data.py:282`) dodati sekundarni izvor:
    ako yfinance ne vrati ništa, pokušati Stooq (`{ticker}.US` dnevni CSV);
    zabilježiti `price_source ∈ {yahoo, stooq, none}` po tickeru. Dodati malu
    ručnu mapu poznatih kolizija/preimenovanja (npr. FB→META) u
    `data/raw/ticker_overrides.csv`.
  - Datoteke: `src/data.py`, `data/raw/ticker_overrides.csv` (novo),
    `tests/test_prices.py` (novo; mockani odgovori).
  - Prihvaćanje: test prolazi (fallback se aktivira kad yahoo vrati prazno);
    tablica `outputs/tables/00_price_source_summary.csv` postoji sa stupcima
    `source, n_tickers, pct`.
  - Ovisnosti: F0.5
  - *Odstupanje:* Stooq endpoint s ovog stroja vraća anti-bot JS stranicu → fallback validiran mockom (kako task i propisuje); na stvarnoj uniji (1080 tickera) summary: yahoo 759 (70,3 %), none 321 (29,7 %), stooq 0 — pokrivenost delistanih mjeri se u F0.7.
- [x] **F0.7 — Izvještaj pokrivenosti po prozoru (tablica + slika)** (M)
  - Opis: za svaki od 21 prozora izračunati: broj point-in-time članova, broj s
    valjanim cijenama, broj koji prolazi filtar ≥ 60 mj. treninga, postotke.
    **Obavezno** po zaključanoj odluci 1 — nepoznata pristranost mora postati
    izmjerena veličina. Slika: linije pokrivenosti kroz prozore (stil postojeće
    `01_universe_coverage.png`).
  - Datoteke: `notebooks/01_data_and_factors.ipynb` (nova sekcija),
    `outputs/tables/00_membership_coverage.csv`,
    `outputs/figures/00_membership_coverage.png`.
  - Prihvaćanje: CSV postoji sa stupcima
    `train_window, n_members, n_with_prices, n_with_60m, pct_prices, pct_60m`
    i 21 retkom; PNG postoji.
  - Ovisnosti: F0.5, F0.6, F0.8
  - *Odstupanje:* logika izdvojena u `membership_coverage_report` (`src/data.py`) radi ponovljivosti; `n_with_prices` = ≥1 valjani mjesečni prinos u prozoru treniranja; figura je linijski graf (članovi / s cijenama / ≥60 mj.). Mjereno: udio s ≥60 mj. raste s 54,75 % (prozor 2004-12) na 95,63 % (2024-12). Dodan `tests/test_coverage.py` (izvan popisa datoteka, podupire kriterij).
- [x] **F0.8 — Adaptacija `preprocess()` na novi univerzum i razdoblje** (M)
  - Opis: `preprocess` (`src/data.py:423`) gradi panel preko **unije svih
    tickera koji su ikad članovi** u 2000.–2025., piše iste izlazne datoteke
    (`monthly_returns.csv`, `excess_returns.csv`, `factors.csv`, `metadata.csv`)
    + novu `membership` tablicu; `metadata.csv` dobiva stupce
    `price_source, member_from, member_to`. FF5 preuzimanje
    (`download_ff5_factors`) ostaje netaknuto, samo novi raspon datuma.
    Pokrenuti notebook 01 na novom univerzumu (prvo preuzimanje ~20–30 min).
  - Datoteke: `src/data.py`, `notebooks/01_data_and_factors.ipynb`,
    `data/processed/*.csv` (regenerirano).
  - Prihvaćanje: `data/processed/monthly_returns.csv` pokriva 2000-01..2025-12;
    `factors.csv` isto; `metadata.csv` ima nove stupce; notebook 01 prolazi
    `nbconvert --execute` bez greške.
  - Ovisnosti: F0.5, F0.6
  - *Odstupanje:* univerzum = `union_universe` (tickeri čiji se interval članstva preklapa s 2000.–2025.) → 1080 imena; nakon preuzimanja 757 zadržano (322 delistana bez Yahoo povijesti → `price_source=none`, mjereno u F0.7). `drop_reason` „yfinance_failed” → „price_download_failed” (sada pokriva yahoo+stooq). Sektor preuzet iz starog Russell popisa gdje postoji, inače „Unknown”. Cache je držao samo 2005+ pa je trebalo jednokratno `force_prices=True` (cijene od 1999-12 dohvaćene s Yahooa); korišteni `.venv/bin/python` i `.venv/bin/jupyter`.
- [ ] **F0.9 — Sloj transakcijskih troškova** (M)
  - Opis: nove funkcije u `src/evaluation.py`:
    `turnover_per_window(weights_panel)` → jednostrani obrtaj
    `0.5·Σ_i |w_i,t − w_i,t−1|` po (portfelj, prozor) iz dugog panela
    (`train_window, portfolio, ticker, weight` — format koji već proizvodi
    `run_walk_forward`); prvi prozor = 1.0 (puna izgradnja, dokumentirano);
    `apply_costs(port_returns_panel, turnover, tc_bps)` → neto panel (trošak se
    skida od prvog mjeseca testne godine). Dodati i `sharpe_ratio(returns, rf)`
    (bruto/neto), jer `risk_metrics` ima `ann_return/ann_vol`, ali ne Sharpe.
  - Datoteke: `src/evaluation.py`, `tests/test_costs.py` (novo).
  - Prihvaćanje: ručni primjer u testu (2 prozora, 3 imovine, poznat obrtaj)
    daje točan obrtaj i neto prinos na 10 bps; test prolazi.
  - Ovisnosti: F0.1, F0.2
- [ ] **F0.10 — Validacija prozora 2000–2025** (S)
  - Opis: smoke test da `generate_rolling_windows("2000-01","2025-12",60,12,12)`
    (`src/backtest.py:60`) vrati točno 21 prozor, prvi `label="2004-12"`
    (test 2005.), zadnji test završava 2025-12.
  - Datoteke: `tests/test_windows.py` (novo).
  - Prihvaćanje: test prolazi.
  - Ovisnosti: F0.1, F0.2
- [ ] **F0.11 — Novi `PROJECT_SPEC.md`** (M)
  - Opis: napisati iznova (datoteka ne postoji u snapshotu): identitet rada,
    hipoteze H1–H3 doslovno, zaključane odluke 1–10, definicije alokatora
    (HRP/HERC/NCO; korelacijska i faktorska verzija, uklj. razliku
    `hrp_corr_single` naspram `hrp_corr_ward` iz K1), overlay QP, metrike,
    statistika (bootstrap/MCS/DSR), izvori podataka s grananjem iz F0.4,
    konvencije imenovanja izlaza (prefiksi 00, 09–13). **Obavezne eksplicitne
    rečenice (iz riješenih pitanja):** (i) §Metodologija — pretpostavka fiksnih
    težina unutar testne godine, trošak samo na refit obrtaj (riješeno
    pitanje 2); (ii) §Statistika — podjela posla: MCS odgovara na „najmanja
    ostvarena varijanca”, DSR na riziku prilagođeni prinos (riješeno pitanje 3);
    (iii) §Alokatori — NCO particija rezom stabla umjesto izvornog k-meansa
    (riješeno pitanje 7); (iv) §Statistika — MCS pravilo presjeka mjeseci s
    pragom 20 % (korekcija K3).
  - Datoteke: `PROJECT_SPEC.md` (novo).
  - Prihvaćanje: dokument postoji i sadrži sve gore navedene sekcije; osoba bez
    ovog TASKS.md može iz njega razumjeti dizajn.
  - Ovisnosti: F0.4
- [x] **F0.12 — Čišćenje necommitane izmjene `clustering_ext.py`** (S)
  - Opis: vratiti DBSCAN `eps` default s 1000.0 na 1.5
    (`src/clustering_ext.py:127`) — potvrđeno (riješeno pitanje 6).
  - Datoteke: `src/clustering_ext.py`.
  - Prihvaćanje: `git diff src/clustering_ext.py` prazan.
  - Ovisnosti: —
  - *Odstupanje:* nema izmjene — `eps` je već 1.5 u radnom stablu (nigdje 1000.0), `git diff` prazan; kriterij zadovoljen bez promjene koda.

**Definicija završetka Faze 0:** `data/processed/` regeneriran na point-in-time
S&P 500 2000–2025; `outputs/tables/00_membership_coverage.csv` +
`00_price_source_summary.csv` + `outputs/figures/00_membership_coverage.png`
postoje; `pytest` zeleno (config, membership, prices, costs, windows); notebook
08 i `supervised.py` u `archive/`; `PROJECT_SPEC.md` i `config.yaml` postoje.

---

# Faza 1 — Replikacija (korelacijski prostor)

**Cilj:** vlastite, validirane implementacije HRP/HERC/NCO na korelacijskom
stablu, integrirane u postojeći walk-forward, s replikacijskom tablicom naspram
1/N i min-var benchmarka.

**Čega se NE diramo:** prostor hijerarhije (sve je korelacijsko), metrike stila
(to je Faza 2), `src/clustering.py` logika (samo se poziva).

### Taskovi

- [ ] **F1.0 — Regeneracija beta, klastera i benchmark portfelja na novom univerzumu** (M)
  - Opis: pokrenuti notebookove 02, 03 i 05 na novim podacima iz Faze 0:
    `estimate_betas_all_windows` (`src/factors.py:131`) → `factor_exposures.csv`;
    postojeća procedura odabira K iz notebooka 02 (silueta + uvjeti uloživosti)
    → novi primarni K (dokumentirati ako ≠ 7); `cluster_all_windows`
    (`src/clustering.py:376`) → `factor_clusters.csv`/`correlation_clusters.csv`;
    `run_walk_forward` (`src/backtest.py:173`) → benchmark paneli (1/N, min_var,
    min_var_sector, min_var_corr_cluster, min_var_factor_cluster);
    `run_factor_neutral_sweep` (`src/backtest.py:423`) → faktorski neutralna
    obitelj (zaključana odluka 9: ostaje kao stupac).
  - Datoteke: `notebooks/02_clustering.ipynb`, `notebooks/03_portfolios.ipynb`,
    `notebooks/05_factor_neutral.ipynb`, `data/processed/*.csv`,
    `outputs/tables/{weights_panel,port_returns_panel,portfolio_status}.csv`,
    `outputs/tables/05_*.csv`.
  - Prihvaćanje: paneli postoje i pokrivaju 21 testnu godinu (uz dokumentirane
    neizvedive prozore u `portfolio_status`); `02_silhouette_by_k.csv` postoji s
    odabranim K.
  - Ovisnosti: F0.8
- [ ] **F1.1 — `src/hierarchical.py`: HRP** (L)
  - Opis: novi modul. Funkcije: `quasi_diagonal_order(linkage_matrix)` → poredak
    listova; `hrp_weights(Sigma, order)` → rekurzivna bisekcija s
    inverzno-varijančnom podjelom, **rizik uvijek iz proslijeđene Σ**
    (Ledoit–Wolf, zaključana odluka 6); `apply_w_max(weights, w_max)` →
    iterativni clip + renormalizacija (riješeno pitanje 1), koja **uz težine
    vraća i `capped_weight_share`** (udio težine na capu) — runner ga bilježi u
    status tablicu po (portfelj, prozor). **Ulazno stablo je uvijek vanjski
    parametar** (za korelacijski prostor gradi ga F1.1b, za faktorski
    `factor_cluster`) → ista funkcija služi svim krakovima usporedbe.
  - Datoteke: `src/hierarchical.py` (novo), `tests/test_hierarchical.py` (novo).
  - Prihvaćanje: jedinični testovi prolaze: težine ≥ 0, Σw = 1, max ≤ w_max; na
    dijagonalnoj Σ HRP = inverzno-varijančne težine (analitički provjerljivo);
    `capped_weight_share` = 0 kad cap ne grize, > 0 na konstruiranom primjeru
    gdje grize.
  - Ovisnosti: F0.1, F0.2
- [ ] **F1.1b — `src/hierarchical.py`: korelacijska stabla (single + ward)** (M)
  - Opis: `correlation_distance(corr)` → d = √(½(1−ρ));
    `build_correlation_tree(returns_or_corr, linkage ∈ {"single","ward"})` →
    SciPy linkage matrica. `single` = vjerna replikacija López de Prada 2016
    (samo za `hrp_corr_single`); `ward` = krak kontrolirane usporedbe (K1: ista
    veza kao faktorsko stablo, pa Faza 3 manipulira **samo** prostorom).
    Napomena: postojeći `correlation_cluster` (`src/clustering.py:125`) namjerno
    odbija Ward na unaprijed izračunatoj udaljenosti i **ne mijenja se**; ova
    funkcija živi u `src/hierarchical.py` s docstringom koji dokumentira
    Lance–Williamsovo opravdanje (standardna praksa u HERC literaturi) i
    upozorenje o metodološkoj razlici.
  - Datoteke: `src/hierarchical.py`, `tests/test_hierarchical.py`.
  - Prihvaćanje: testovi prolaze: obje veze vraćaju valjanu linkage matricu
    (oblik (n−1, 4), monotone udaljenosti za ward); na konstruiranom primjeru
    single i ward daju različit poredak listova; `correlation_cluster` guard u
    `src/clustering.py` netaknut (`git diff` prazan za tu datoteku).
  - Ovisnosti: F0.1, F0.2
- [ ] **F1.2 — `src/hierarchical.py`: HERC** (L)
  - Opis: `herc_weights(Sigma, linkage_matrix, k)` — rez stabla na `k` klastera
    (`fcluster`, ista konvencija kao `src/clustering.py:103`), top-down podjela
    kapitala **niz stvarni dendrogram** po jednakom doprinosu riziku među
    klasterima (rizik klastera iz Σ pod-bloka s naivnim inverzno-varijančnim
    težinama unutar klastera), naivni risk parity (1/σ_i) unutar klastera;
    `apply_w_max` na kraju. `k` dolazi iz primarne procedure (F1.0).
  - Datoteke: `src/hierarchical.py`, `tests/test_hierarchical.py`.
  - Prihvaćanje: test: za k=2 i blok-dijagonalnu Σ s dva jednaka bloka podjela
    kapitala je 50/50; težine valjane (≥0, Σ=1, ≤ w_max).
  - Ovisnosti: F1.1
- [ ] **F1.3 — `src/hierarchical.py`: NCO** (M)
  - Opis: `nco_weights(Sigma, labels, w_max)` — particija = oznake klastera
    (rez stabla na K; pretpostavka 7 / riješeno pitanje 7); min-var **unutar**
    svakog klastera ponovnom uporabom `min_variance` (`src/portfolio.py:178`) na
    Σ pod-bloku, **bez capa** (tj. `w_max=1.0` ili labavi tehnički cap samo radi
    numerike) — unutar-klasterski cap se *ne* izvodi iz težine klastera jer su
    težine klastera poznate tek nakon među-klasterskog koraka (korekcija K2);
    reducirana kovarijanca (klaster = jedna varijabla): `Σ_red = W' Σ W` gdje su
    W unutar-klasterske težine; min-var **među** klasterima na `Σ_red`; na kraju
    **isti `apply_w_max` post-korak na kombinirane konačne težine** kao kod
    HRP/HERC — identičan mehanizam capa kroz sva tri alokatora (uklj.
    `capped_weight_share` u status tablici).
  - Datoteke: `src/hierarchical.py`, `tests/test_hierarchical.py`.
  - Prihvaćanje: test: za K=1 i **nevezujući cap** (w_max=1) NCO = min_variance
    na punoj Σ (razlika ≤ 1e-8); za blok-dijagonalnu Σ kombinirane težine =
    analitičko rješenje po blokovima; nakon `apply_w_max` sve težine ≤ w_max uz
    Σw = 1.
  - Ovisnosti: F1.1
- [ ] **F1.4 — Validacija protiv riskfolio-lib / skfolio** (M)
  - Opis: sintetički primjer (15 imovina, fiksno sjeme, poznata Σ iz faktorskog
    modela). Usporediti: naš `hrp_corr_single` (jednostruka veza — ista
    konfiguracija kao biblioteke) naspram `riskfolio.HCPortfolio(model="HRP")` i/ili
    `skfolio` HRP — **tolerancija ≤ 1e-6** (isti algoritam); HERC/NCO naspram
    bibliotečnih varijanti — dokumentirati izvor svake razlike (linkage, način
    podjele, capovi) u docstringu i `PROJECT_SPEC.md`, s kvantificiranom
    maksimalnom apsolutnom razlikom težina. Biblioteke u zasebnom venv-u ako se
    ne slažu s pinovima (registar rizika).
  - Datoteke: `tests/test_validation_libs.py` (novo; `@pytest.mark.skipif` ako
    biblioteka nije instalirana), `outputs/tables/09_validation_vs_libraries.csv`.
  - Prihvaćanje: HRP razlika ≤ 1e-6 na sintetičkom primjeru; CSV s usporedbom
    (metoda, biblioteka, max |Δw|, objašnjenje) postoji.
  - Ovisnosti: F1.1, F1.1b, F1.2, F1.3
- [ ] **F1.5 — Integracija u walk-forward** (L)
  - Opis: nova funkcija `run_hierarchical_walk_forward(...)` u `src/backtest.py`
    po uzoru na `run_walk_forward` (`src/backtest.py:173`): isti presjek
    univerzuma, ista Ledoit–Wolf Σ (jednom po prozoru, dijeli se među
    alokatorima), isti format izlaza (`weights_panel`, `port_returns_panel`,
    `portfolio_status`). Parametar `tree_space ∈ {"correlation", "factor"}`
    (faza 3 samo mijenja taj argument). Ova faza: `hrp_corr_single` (vjerna
    replikacija, jednostruka veza), `hrp_corr_ward` (krak kontrolirane
    usporedbe, K1), `herc_corr`, `nco_corr` — korelacijska stabla iz F1.1b.
    Status tablica dobiva stupac `capped_weight_share` po (portfelj, prozor)
    (riješeno pitanje 1). Pokretanje u novom notebooku
    `notebooks/09_hierarchical_replication.ipynb`.
  - Datoteke: `src/backtest.py`, `notebooks/09_hierarchical_replication.ipynb`
    (novo), `outputs/tables/09_weights_panel_hierarchical.csv`,
    `outputs/tables/09_port_returns_panel_hierarchical.csv`,
    `outputs/tables/09_portfolio_status_hierarchical.csv`.
  - Prihvaćanje: tri CSV-a postoje; panel prinosa ima stupce
    `hrp_corr_single, hrp_corr_ward, herc_corr, nco_corr` s ≥ 240 mjeseci;
    status log bez neobjašnjenih `failed` i sa stupcem `capped_weight_share`.
  - Ovisnosti: F1.0, F1.1, F1.1b, F1.2, F1.3
- [ ] **F1.6 — Replikacijska tablica** (M)
  - Opis: tablica s OOS godišnjom volatilnošću, Sharpeom (bruto i neto na
    10 bps), maksimalnim drawdownom, obrtajem i `n_months` za: 1/N, min_var,
    hrp_corr_single, hrp_corr_ward, herc_corr, nco_corr (+ postojeći stupci
    min_var_sector / corr_cluster / factor_cluster radi kontinuiteta —
    zaključana odluka 10). Koristi `risk_metrics`, `sharpe_ratio`,
    `turnover_per_window`, `apply_costs` iz `src/evaluation.py`. **Obavezne
    fusnote uz tablicu:** (a) `hrp_corr_single` = vjerna replikacija izvornika
    (jednostruka veza) i ostaje isključivo replikacijski red — kontrolirana
    usporedba prostora ide preko `hrp_corr_ward` (K1); (b) NCO particija = rez
    stabla na K umjesto k-meansa iz izvornog rada (riješeno pitanje 7).
  - Datoteke: `notebooks/09_hierarchical_replication.ipynb`,
    `outputs/tables/09_replication_summary.csv`,
    `outputs/figures/09_cumulative_growth.png`.
  - Prihvaćanje: CSV postoji sa stupcima
    `portfolio, ann_vol, sharpe_gross, sharpe_net, max_drawdown, turnover, n_months`
    i ≥ 9 redaka; obje fusnote prisutne (markdown ćelija notebooka + bilješka u
    tablici za izvještaj); figura postoji.
  - Ovisnosti: F0.9, F1.5
- [ ] **F1.7 — Model Confidence Set (Hansen–Lunde–Nason 2011)** (L)
  - Opis: nova funkcija `model_confidence_set(returns_panel, loss="sq_demeaned",
    alpha, n_bootstraps, block_size, seed)` u `src/evaluation.py`: gubitak po
    defaultu `l_t = (r_t − r̄)²` (riješeno pitanje 3); iterativna eliminacija po
    max-t statistici s blok-bootstrap distribucijom (ponovno iskoristiti
    `_block_indices`, `src/evaluation.py:410`); vraća tablicu
    `portfolio, in_mcs, p_value, rank, n_months_used, n_months_dropped`.
    **Pravilo za neuravnotežen panel (K3, zaključano unaprijed):** MCS se računa
    na **presjeku mjeseci dostupnih svim uspoređenim varijantama**; broj
    ispuštenih mjeseci obavezno se izvještava u izlaznoj tablici; varijanta
    kojoj presjek odnese > 20 % njezinih mjeseci **isključuje se iz MCS-a**
    (ostaje u parnim bootstrap usporedbama) i navodi se u fusnoti. Primijeniti
    na replikacijski skup.
  - Datoteke: `src/evaluation.py`, `tests/test_mcs.py` (novo),
    `outputs/tables/09_mcs_replication.csv`.
  - Prihvaćanje: sintetički test prolazi (3 niza: dva jednaka + jedan očito
    dominiran s 2× varijancom → dominirani ispada iz MCS-a na α=0.10); test
    neuravnoteženog panela prolazi (varijanta s > 20 % nedostajućih mjeseci u
    presjeku označena `excluded`, presjek i brojevi ispuštenih mjeseci točni);
    CSV postoji sa stupcima `n_months_used, n_months_dropped`.
  - Ovisnosti: F1.5

**Definicija završetka Faze 1:** `src/hierarchical.py` postoji s validiranim
HRP/HERC/NCO i graditeljima korelacijskih stabala single + ward (testovi +
usporedba s bibliotekama); notebook 09 izvršava se kraja;
`09_replication_summary.csv`, `09_validation_vs_libraries.csv`,
`09_mcs_replication.csv` i prateći paneli postoje.

---

# Faza 2 — Dijagnoza

**Cilj:** izmjeriti skrivene stilske izloženosti korelacijskih HRP/HERC/NCO i
testirati H1: nasljeđuju li RMW/CMA nagib min-vara.

**Čega se NE diramo:** alokatori (nikakva izmjena `src/hierarchical.py`), prostor
hijerarhije, benchmark paneli.

### Taskovi

- [ ] **F2.1 — FF5 atribucija hijerarhijskih alokatora** (S)
  - Opis: primijeniti `factor_attribution_full_sample` i
    `factor_attribution_per_window` (`src/evaluation.py:314, 265`) na spojeni
    panel (benchmarki + `hrp_corr_single`, `hrp_corr_ward`, `herc_corr`,
    `nco_corr` + faktorski neutralna obitelj). Obje HRP varijante se atribuiraju
    (replikacijska radi potpunosti, ward varijanta je nositelj kontrolirane
    usporedbe u Fazi 3 — K1). Novi notebook
    `notebooks/10_hierarchical_diagnosis.ipynb`.
  - Datoteke: `notebooks/10_hierarchical_diagnosis.ipynb` (novo),
    `outputs/tables/10_attribution_full_sample.csv`,
    `outputs/tables/10_attribution_per_year.csv`.
  - Prihvaćanje: tablice postoje; svaki redak ima α, 5 beta, R²,
    `style_concentration`, `n_months`; pokriveni svi alokatori iz Faze 1.
  - Ovisnosti: F1.5
- [ ] **F2.2 — Bootstrap intervali i parne razlike (test H1)** (M)
  - Opis: `block_bootstrap_metric` (`src/evaluation.py:550`) → 95 % CI
    koncentracije stila za svaki alokator; `block_bootstrap_diff`
    (`src/evaluation.py:465`, zajedničko uzorkovanje) → parne razlike svakog
    korelacijskog hijerarhijskog alokatora (obje HRP varijante, HERC, NCO)
    naspram min_var i naspram equal_weight za
    `style_concentration` i `annualized_vol`. H1 se podržava ako je CI razlike
    naspram min_var ≈ 0 (nagib naslijeđen) ili točkasta procjena pozitivna, a
    razlika naspram 1/N značajno pozitivna — formulirati zaključak na
    razini RMW i CMA beta pojedinačno + ukupne koncentracije.
  - Datoteke: `notebooks/10_hierarchical_diagnosis.ipynb`,
    `outputs/tables/10_bootstrap_ci.csv`,
    `outputs/tables/10_paired_diffs_vs_minvar.csv`.
  - Prihvaćanje: tablice postoje sa stupcima
    `(portfolio[, compared_to], metric, point, ci_low, ci_high, p_two_sided)`;
    1000 uzoraka, blok 12, sjeme iz configa.
  - Ovisnosti: F2.1
- [ ] **F2.3 — Mehanička dijagnostika inverzno-varijančne alokacije** (M)
  - Opis: poveznica s mehanizmom H1 (Scherer 2011, Novy-Marx 2014): po prozoru
    regresirati/korelirati težine alokatora na karakteristike dionica
    (β_RMW, β_CMA, residual_vol iz `factor_exposures.csv`); prikazati prosjek
    kroz prozore. Pokazuje *zašto* nagib nastaje (preferencija niskovolatilnih
    dionica s visokim RMW/CMA).
  - Datoteke: `notebooks/10_hierarchical_diagnosis.ipynb`,
    `outputs/tables/10_weight_characteristic_corr.csv`,
    `outputs/figures/10_weight_vs_beta.png`.
  - Prihvaćanje: tablica (portfolio × karakteristika → prosječna korelacija
    težina) i figura postoje.
  - Ovisnosti: F2.1
- [ ] **F2.4 — Glavna isporuka: tablica skrivenih stilskih izloženosti + figura** (S)
  - Opis: konsolidirana tablica (alokator × {β_RMW, β_CMA, konc. stila [CI],
    ann_vol}) — ekvivalent README tablice starog projekta, sada za hijerarhijske
    alokatore; figura koncentracije stila po godini (stil postojeće
    `04_style_concentration_per_year.png`) s hijerarhijskim alokatorima.
  - Datoteke: `notebooks/10_hierarchical_diagnosis.ipynb`,
    `outputs/tables/10_hidden_style_exposures.csv`,
    `outputs/figures/10_style_concentration_per_year.png`.
  - Prihvaćanje: oboje postoji; tablica sadrži min_var i 1/N kao benchmarke;
    zaključak o H1 (podržana/odbačena, s brojevima) zapisan u markdown ćeliji
    notebooka.
  - Ovisnosti: F2.2

**Definicija završetka Faze 2:** notebook 10 izvršava se do kraja; postoje
`10_attribution_full_sample.csv`, `10_bootstrap_ci.csv`,
`10_paired_diffs_vs_minvar.csv`, `10_hidden_style_exposures.csv` i obje figure;
H1 ima eksplicitan, kvantificiran zaključak.

---

# Faza 3 — Intervencija (faktorski prostor)

**Cilj:** faktorske verzije sva tri alokatora (jedina promjena = ulazno
stablo/particija iz Wardova klasteriranja na standardiziranim FF5 značajkama) i
test H2 parnim razlikama korelacijska↔faktorska po alokatoru.

**Čega se NE diramo:** korelacijski paneli iz Faze 1 (ostaju zamrznuti), Σ i svi
koraci rizika (bete služe isključivo za izgradnju hijerarhije — zaključana
odluka 6), benchmark obitelj.

### Taskovi

- [ ] **F3.1 — Faktorsko stablo kao ulaz u alokatore** (M)
  - Opis: u runneru iz F1.5 implementirati granu `tree_space="factor"`: po
    prozoru dohvatiti 6 značajki (`FACTOR_FEATURE_COLUMNS`,
    `src/clustering.py:21`) iz `factor_exposures.csv`, pozvati `factor_cluster`
    (`src/clustering.py:70`) koji već vraća **i linkage matricu** — ta linkage
    ide u `quasi_diagonal_order` (HRP), `herc_weights` (HERC), a njezin rez na K
    u `nco_weights` (NCO). Univerzum, Σ, w_max identični korelacijskoj grani.
  - Datoteke: `src/backtest.py`, `tests/test_hierarchical.py` (test: oba
    prostora vraćaju valjane težine na istom sintetičkom prozoru, a razlikuju se
    samo kad se stabla razlikuju).
  - Prihvaćanje: test prolazi; `run_hierarchical_walk_forward(tree_space="factor")`
    vraća panele istog formata.
  - Ovisnosti: F1.5
- [ ] **F3.2 — Walk-forward faktorskih verzija** (M)
  - Opis: pokrenuti `hrp_factor`, `herc_factor`, `nco_factor` kroz svih 21
    prozor; novi notebook `notebooks/11_factor_space_intervention.ipynb`; spojiti
    s panelima Faze 1 u jedinstveni prošireni panel prinosa.
  - Datoteke: `notebooks/11_factor_space_intervention.ipynb` (novo),
    `outputs/tables/11_port_returns_panel_factor.csv`,
    `outputs/tables/11_weights_panel_factor.csv`,
    `outputs/tables/11_portfolio_status_factor.csv`.
  - Prihvaćanje: paneli postoje sa stupcima `hrp_factor, herc_factor, nco_factor`
    i ≥ 240 mjeseci; status log čist.
  - Ovisnosti: F3.1
- [ ] **F3.3 — Parne razlike korelacijska↔faktorska po alokatoru (test H2, dio 1)** (M)
  - Opis: kontrolirane parne usporedbe s **vezom konstantnom u oba kraka**
    (korekcija K1): `hrp_factor − hrp_corr_ward` (Ward u oba kraka;
    `hrp_corr_single` je isključen iz uzročne usporedbe jer bi konfundirao vezu
    — fusnota u tablici), `herc_factor − herc_corr`, `nco_factor − nco_corr`.
    `block_bootstrap_diff` (zajedničko uzorkovanje) za primarni ishod
    (koncentracija stila) i sekundarne (ostvarena volatilnost — **test
    ne-inferiornosti**: gornja granica 95 % CI razlike vol ≤ +1 p.b.;
    Sharpe bruto/neto; MDD). H2-dio-nagib se podržava ako CI razlike
    koncentracije stila obuhvaća 0 (faktorski prostor NE neutralizira nagib).
  - Datoteke: `notebooks/11_factor_space_intervention.ipynb`,
    `outputs/tables/11_paired_diffs_space.csv`.
  - Prihvaćanje: CSV sa stupcima `(allocator, metric, point, ci_low, ci_high,
    p_two_sided, n_months)` za 3 alokatora × ≥ 4 metrike; HRP redak uspoređuje
    `hrp_factor` s `hrp_corr_ward`, a fusnota o isključenju `hrp_corr_single`
    postoji.
  - Ovisnosti: F3.2
- [ ] **F3.4 — Stabilnost klastera (ARI) i obrtaj (test H2, dio 2)** (M)
  - Opis: nova funkcija `ari_between_consecutive_windows(cluster_table,
    label_column)` u `src/clustering.py` (koristi već uvezeni
    `adjusted_rand_score`): za uzastopne prozore, ARI na presjeku tickera; za
    korelacijska i faktorska stabla rezana na isti K (korelacijsko stablo =
    **ward varijanta** iz F1.1b, dosljedno K1 — veza konstantna, mijenja se samo
    prostor). Usporediti i obrtaj (`turnover_per_window`) korelacijskih naspram
    faktorskih verzija po alokatoru (za HRP: `hrp_corr_ward`). Dokumentirati argument redukcije dimenzije (~N²/2 korelacija
    naspram 6N beta) kao opisnu tablicu (broj procijenjenih veličina po prozoru).
  - Datoteke: `src/clustering.py`, `tests/test_ari_stability.py` (novo),
    `notebooks/11_factor_space_intervention.ipynb`,
    `outputs/tables/11_ari_stability.csv`,
    `outputs/tables/11_turnover_by_space.csv`.
  - Prihvaćanje: test prolazi (identične oznake → ARI=1); CSV-ovi postoje;
    ARI tablica ima 20 redaka po prostoru (prijelazi između 21 prozora).
  - Ovisnosti: F3.2
- [ ] **F3.5 — Odluka o K: primarno + robusnost** (M)
  - Opis: primarna analiza s istim K za obje verzije (postojeća procedura iz
    notebooka 02 — silueta uz uvjete uloživosti; zaključana odluka 5). Robusnost:
    vlastiti-optimalni K svake verzije (korelacijska: silueta na korelacijskoj
    udaljenosti; faktorska: postojeći prelet) → ponoviti HERC/NCO s tim K i
    izvijestiti osjetljivost ključnih metrika. K-means particija za NCO kao
    dodatna robusnost (riješeno pitanje 7).
  - Datoteke: `notebooks/11_factor_space_intervention.ipynb`,
    `outputs/tables/11_k_robustness.csv`.
  - Prihvaćanje: CSV s metrikama (konc. stila, ann_vol, obrtaj) za
    {primarni K, vlastiti K korelacijski, vlastiti K faktorski[, kmeans-NCO]} ×
    {HERC, NCO}.
  - Ovisnosti: F3.2
- [ ] **F3.6 — Konsolidirani zaključak H2** (S)
  - Opis: markdown sekcija u notebooku 11 + konsolidirana tablica: nagib NIJE
    neutraliziran (iz F3.3), ALI dobitci u stabilnosti (F3.4 ARI), obrtaju
    (F3.4) i interpretabilnosti (opisna usporedba dimenzija) — svaki dio s
    brojkom i intervalom.
  - Datoteke: `notebooks/11_factor_space_intervention.ipynb`,
    `outputs/tables/11_h2_summary.csv`.
  - Prihvaćanje: tablica postoji; zaključak eksplicitan po sva tri dijela H2.
  - Ovisnosti: F3.3, F3.4, F3.5

**Definicija završetka Faze 3:** notebook 11 izvršava se do kraja; postoje
`11_port_returns_panel_factor.csv`, `11_paired_diffs_space.csv`,
`11_ari_stability.csv`, `11_turnover_by_space.csv`, `11_k_robustness.csv`,
`11_h2_summary.csv`; H2 ima kvantificiran troslojni zaključak.

---

# Faza 4 — Mehanizam i lijek

**Cilj:** projekcijski QP overlay s ε-preletom na obje verzije svih alokatora →
potpuni 2×2 faktorijal, proširena granica stil–volatilnost, DSR, test H3.

**Čega se NE diramo:** alokatori i njihovi paneli bez overlaya (zamrznuti);
definicije metrika.

### Taskovi

- [ ] **F4.1 — Projekcijski QP u `src/portfolio.py`** (L)
  - Opis: nova funkcija `project_factor_neutral(w0, Sigma, factor_betas, w_max,
    epsilon, norm="sigma")`: riješiti
    `min_w (w−w0)' Σ (w−w0)` uz `1'w = 1`, `0 ≤ w ≤ w_max`,
    `|w'β_f| ≤ ε ∀f ∈ {SMB, HML, RMW, CMA}`. Σ-norma = minimalni tracking error
    prema hijerarhijskom portfelju (primarno); `norm="euclidean"` →
    `min ‖w−w0‖²` (robusnost). Ponovno iskoristiti obrasce iz
    `_solve_min_variance` (`src/portfolio.py:93`): lanac rješavača
    CLARABEL→OSQP→SCS, `psd_wrap`, te `_align_factor_betas`
    (`src/portfolio.py:251`) za poravnanje beta.
  - Datoteke: `src/portfolio.py`, `tests/test_overlay.py` (novo).
  - Prihvaćanje: testovi prolaze: (1) ako w0 već zadovoljava ograničenja →
    vraća w0 (max |Δw| ≤ 1e-6); (2) izlaz uvijek zadovoljava sve uvjete;
    (3) Σ-norma i euklidska norma daju različita rješenja na konstruiranom
    primjeru s korelacijama.
  - Ovisnosti: F0.1, F0.2
- [ ] **F4.2 — Overlay walk-forward s ε-preletom (2×2 faktorijal)** (L)
  - Opis: nova funkcija `run_overlay_sweep(...)` u `src/backtest.py` po uzoru na
    `run_factor_neutral_sweep` (`src/backtest.py:423`): učitati spremljene
    panele težina iz Faza 1 i 3 (`09_weights_panel_hierarchical.csv`,
    `11_weights_panel_factor.csv`), za svaki (alokator × prostor × prozor)
    primijeniti `project_factor_neutral` s ε ∈ {0, 0.05, 0.10, 0.15} (bete iz
    `factor_exposures.csv` istog prozora — bez gledanja unaprijed), izračunati
    testne prinose istom logikom kao postojeći runneri. Korelacijska baza za
    HRP je **`hrp_corr_ward`** (krak kontrolirane usporedbe, K1;
    `hrp_corr_single` ne dobiva overlay). Imenovanje:
    `{alokator}_{space}_ov_e{ε}`. Status log za neizvedive (ε=0) prozore.
    Novi notebook `notebooks/12_overlay_mechanism.ipynb`.
  - Datoteke: `src/backtest.py`, `notebooks/12_overlay_mechanism.ipynb` (novo),
    `outputs/tables/12_port_returns_panel_overlay.csv`,
    `outputs/tables/12_weights_panel_overlay.csv`,
    `outputs/tables/12_overlay_status.csv`.
  - Prihvaćanje: paneli postoje za 3 alokatora × 2 prostora × 4 ε = 24 overlay
    varijante (uz dokumentirane neizvedivosti); Σ-norma primarna.
  - Ovisnosti: F4.1, F1.5, F3.2
- [ ] **F4.3 — Euklidska norma kao robusnost** (S)
  - Opis: ponoviti F4.2 s `norm="euclidean"` za reprezentativni podskup
    (npr. najbolji alokator × oba prostora × ε ∈ {0, 0.10}); usporediti metrike
    sa Σ-normom.
  - Datoteke: `notebooks/12_overlay_mechanism.ipynb`,
    `outputs/tables/12_norm_robustness.csv`.
  - Prihvaćanje: CSV s usporedbom Σ naspram euklidske norme po metrikama.
  - Ovisnosti: F4.2
- [ ] **F4.4 — Deflated Sharpe Ratio (Bailey–López de Prado 2014)** (M)
  - Opis: nova funkcija `deflated_sharpe_ratio(returns, n_trials, rf)` u
    `src/evaluation.py`: procjena SR, korekcija za skewness/kurtosis i za
    očekivani maksimum preko `n_trials` isprobanih varijanti; vraća DSR
    p-vrijednost. **`n_trials` broji SVE isprobane varijante završne
    usporedbe, uključujući cijelu ε-mrežu i obje norme** — formula:
    `n_trials = |benchmarki: 1/N, min_var| (2) + |min-var faktorski neutralna
    obitelj| (4 ε) + |hijerarhijske baze: hrp_corr_single, hrp_corr_ward,
    herc_corr, nco_corr, hrp_factor, herc_factor, nco_factor| (7) + |overlay
    Σ-norma: 3 alokatora × 2 prostora × 4 ε| (24) + |euklidska robusnost|
    (~4)` ≈ **37–41 s trenutnom mrežom (red veličine ~34+, nikako ~10)**;
    točan broj računa se programatski iz popisa stupaca master panela i
    zapisuje u `12_dsr.csv` (stupac `n_trials` + popis varijanti u pratećoj
    metadata ćeliji notebooka). Primijeniti na sve portfelje završne usporedbe
    (bruto i neto).
  - Datoteke: `src/evaluation.py`, `tests/test_dsr.py` (novo),
    `outputs/tables/12_dsr.csv`.
  - Prihvaćanje: test naspram ručno izračunatog primjera (poznate vrijednosti
    SR, γ3, γ4, T, N → DSR na 4 decimale); CSV postoji i sadrži stupac
    `n_trials` ≥ 34 čija vrijednost odgovara stvarnom broju varijanti u master
    panelu.
  - Ovisnosti: F4.2
- [ ] **F4.5 — Proširena granica stil–volatilnost (nova Slika 7)** (M)
  - Opis: nova verzija postojeće granice
    (`reports/figures/fig07-style-volatility-frontier.png`, generirana logikom
    `05_frontier_style_vs_vol.png` iz notebooka 05): x = ostvarena godišnja
    volatilnost, y = koncentracija stila; točke = svi alokatori (korelacijski,
    faktorski, s overlayem po ε, min-var obitelj, 1/N); ε-putanje spojene
    linijama; paretova fronta istaknuta.
  - Datoteke: `notebooks/12_overlay_mechanism.ipynb`,
    `outputs/figures/12_frontier_style_vs_vol.png`,
    `outputs/tables/12_frontier_points.csv`.
  - Prihvaćanje: figura i tablica točaka postoje; svaka točka ima
    `portfolio, ann_vol, style_concentration, n_months`.
  - Ovisnosti: F4.2
- [ ] **F4.6 — MCS preko svih alokatora + test H3** (M)
  - Opis: `model_confidence_set` (F1.7) preko cijelog završnog skupa (bruto i
    neto), uz **pravilo presjeka iz K3**: računa se na presjeku mjeseci
    dostupnih svim uspoređenim varijantama; broj ispuštenih mjeseci izvještava
    se u `12_mcs_final.csv`; overlay varijante (tipično ε=0) kojima presjek
    odnese > 20 % mjeseci isključuju se iz MCS-a s fusnotom, ali ostaju u
    parnim bootstrap usporedbama u `12_h3_paired_diffs.csv`. Parne razlike
    overlay naspram baze po alokatoru
    (`block_bootstrap_diff`, koncentracija stila i vol). H3 se podržava ako:
    (1) overlay smanjuje koncentraciju stila statistički značajno za obje
    verzije prostora; (2) razlika vol obuhvaća nulu (analogno starom rezultatu
    −31 %, p=0.002 za min-var); (3) overlay točke dominiraju na granici iz F4.5.
  - Datoteke: `notebooks/12_overlay_mechanism.ipynb`,
    `outputs/tables/12_mcs_final.csv`,
    `outputs/tables/12_h3_paired_diffs.csv`.
  - Prihvaćanje: oba CSV-a postoje; `12_mcs_final.csv` sadrži stupce
    `n_months_used, n_months_dropped` i oznaku `excluded` za varijante iznad
    praga 20 %; zaključak H3 (po sva tri kriterija) eksplicitan u markdown
    ćeliji.
  - Ovisnosti: F4.2, F4.4, F4.5, F1.7

**Definicija završetka Faze 4:** notebook 12 izvršava se do kraja; postoje
`12_port_returns_panel_overlay.csv`, `12_overlay_status.csv`, `12_dsr.csv`,
`12_frontier_style_vs_vol.png`, `12_mcs_final.csv`, `12_h3_paired_diffs.csv`,
`12_norm_robustness.csv`; H3 ima kvantificiran zaključak; 2×2 faktorijal potpun.

---

# Faza 5 — Pisanje i pakiranje

**Cilj:** finalne imenovane tablice/slike, restrukturiran izvještaj s hipotezama
unaprijed, novi README i jedan skriptirani put od sirovih podataka do svih
rezultata.

**Čega se NE diramo:** brojčani rezultati (nikakvo preračunavanje u ovoj fazi —
samo konsolidacija, tekst i pakiranje).

### Taskovi

- [ ] **F5.1 — Finalni sažetak i master tablica** (M)
  - Opis: novi notebook `notebooks/13_final_summary.ipynb` (uloga starog 06):
    učitava spremljene panele (bez preračunavanja), gradi master tablicu svih
    portfelja × svih metrika (β-ovi, konc. stila [CI], vol, Sharpe bruto/neto,
    MDD, obrtaj, eff. N, DSR, MCS članstvo, n_months) i 2×2 faktorijalnu
    tablicu {prostor} × {overlay}.
  - Datoteke: `notebooks/13_final_summary.ipynb` (novo),
    `outputs/tables/13_master_table.csv`,
    `outputs/tables/13_factorial_2x2.csv`.
  - Prihvaćanje: oba CSV-a postoje; master tablica pokriva ≥ 20 portfelja;
    brojevi se podudaraju s izvornim tablicama faza 1–4 (provjera spot-uzorkom).
  - Ovisnosti: F4.6
- [ ] **F5.2 — Restrukturiranje izvještaja** (L)
  - Opis: prerada `reports/izvjestaj.tex`: uvod s pregledom literature (López de
    Prado 2016, 2019; Raffinot 2018; Scherer 2011; Novy-Marx 2014; Hansen et al.
    2011; Bailey & López de Prado 2014) i **hipotezama H1–H3 unaprijed**;
    metodologija (univerzum s pokrivenošću, alokatori, overlay, statistika);
    rezultati po fazama 1→4; stari rezultati (min-var dijagnoza, neuspjeh
    grupnih ograničenja, izravna neutralnost) ugrađeni kao motivacija i
    benchmark; ograničenja: pokrivenost delistanih (brojke iz F0.7), šum beta iz
    60 mj., jednostavan model troškova (10 bps linearno), te — ako
    `capped_weight_share` iz status tablica pokaže da cap često grize —
    nametanje `w_max` clip-postupkom (riješeno pitanje 1); notebook 07 kao
    dodatak robusnosti (zaključana odluka 10); notebook 08 jedna rečenica u
    raspravi. Figure iz `outputs/figures/{00,09,10,11,12}_*.png` kopiraju se u
    `reports/figures/` s postojećom `figNN-` konvencijom.
  - Datoteke: `reports/izvjestaj.tex`, `reports/figures/`.
  - Prihvaćanje: PDF se kompilira bez grešaka; svaka tvrdnja u rezultatima ima
    referencu na tablicu/figuru koja postoji u `outputs/`.
  - Ovisnosti: F5.1
- [ ] **F5.3 — Novi README** (M)
  - Opis: napisati iznova (tek u zadnjoj fazi — zaključana odluka 10): nova
    identitetska rečenica, glavna tablica (iz `13_master_table.csv`, skraćena),
    struktura repoa (bez referenci na nepostojeće datoteke — pretpostavka 1),
    upute za reprodukciju kroz F5.4, izvori podataka (S&P 500 članstvo, yfinance
    + Stooq, Ken French), status faza.
  - Datoteke: `README.md`.
  - Prihvaćanje: README ne referencira nijednu nepostojeću datoteku
    (provjera: svaki spomenuti put postoji); glavni brojevi se podudaraju s
    `13_master_table.csv`.
  - Ovisnosti: F5.1, F5.4
- [ ] **F5.4 — Reproducibilnost: jedan skriptirani put** (M)
  - Opis: `scripts/run_all.sh` (ili `Makefile`): redom `pytest`, pa
    `jupyter nbconvert --to notebook --execute --inplace` za notebookove
    01, 02, 03, 05, 07, 09, 10, 11, 12, 13 (04 zadržati u nizu ako ostaje u
    glavnom tijeku, inače dokumentirati); ispis ukupnog trajanja. Dokumentirati
    očekivano trajanje prvog (s preuzimanjem) i ponovnog pokretanja.
  - Datoteke: `scripts/run_all.sh` (novo).
  - Prihvaćanje: jedna naredba na čistom okruženju (uz postojeću parquet
    predmemoriju) regenerira sve tablice i figure koje izvještaj citira.
  - Ovisnosti: F5.1
- [ ] **F5.5 — Završno čišćenje i commit politika izlaza** (S)
  - Opis: committati finalne tablice/figure koje izvještaj citira (riješeno
    pitanje 4), ažurirati `.gitignore` za međupanele; ukloniti zastarjele
    artefakte (stari `russell1000_tickers.csv` ostaje u `data/raw/` kao
    povijesna snimka s napomenom u README).
  - Datoteke: `.gitignore`, `outputs/`.
  - Prihvaćanje: `git status` čist nakon commita; repo se klonira i README
    upute prolaze.
  - Ovisnosti: F5.2, F5.3, F5.4

**Definicija završetka Faze 5:** izvještaj (PDF) kompiliran s novim dizajnom i
hipotezama unaprijed; README nov i točan; `scripts/run_all.sh` regenerira sve;
`13_master_table.csv` i `13_factorial_2x2.csv` committani; repo čist.

---

## Redoslijed i kritični put

```
F0.1 → F0.2 → {F0.5 → F0.6 → F0.8 → F0.7} → F1.0 → F1.5 → F2.* → F3.* → F4.2 → F4.6 → F5.*
              {F0.3/F0.4, F0.9, F0.10, F0.11, F0.12 paralelno}
              {F1.1 → F1.2/F1.3; F1.1b paralelno; → F1.4 — sve paralelno s F1.0}
              {F1.7, F4.1 neovisni — mogu ranije}
```

F1.5 ovisi o F1.0, F1.1, F1.1b, F1.2 i F1.3 (korelacijska stabla single + ward
dolaze iz F1.1b).

Najveći pojedinačni rizik na kritičnom putu je F0.5/F0.6 (kvaliteta
point-in-time podataka) — zato pokrivenost (F0.7) dolazi odmah iza, prije nego
što se išta gradi na novom univerzumu.

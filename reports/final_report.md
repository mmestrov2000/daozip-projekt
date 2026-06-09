# Skriveni stilski rizik u portfeljima minimalne varijance — završni izvještaj

Dijagnoza putem faktorskog klasteriranja na Russell 1000, 2005. – 2025.

---

## 1. Uvod

Portfelji minimalne varijance sa samo dugim pozicijama često se prikazuju
kao prirodna „diversificirana” referenca u pasivnom upravljanju: po
konstrukciji raspoređuju težinu kako bi minimizirali ostvarenu varijancu,
bez izričitog stava o prinosima. Hipoteza ovog projekta jest da se, unatoč
toj mehaničkoj diversifikaciji, ti portfelji tiho opterećuju
Fama-French *stilskim* faktorima — osobito profitabilnošću (RMW) i
investiranjem (CMA) — te da je to opterećenje nevidljivo diversifikaciji
po sektorima ili po korelacijama.

Projekt teče u tri faze. Najprije klasteriramo Russell 1000
u prostoru faktorske izloženosti i koristimo dobivenu strukturu za dijagnozu
skrivenog nagiba. Zatim testiramo intuitivan popravak koji klasteriranje
sugerira — ograničavanje težine unutar svakog faktorskog klastera — i izvještavamo
negativan rezultat. Treće, vođeni razumijevanjem zašto je taj popravak zakazao, primjenjujemo
ograničenje izravno na faktorske bete portfelja i pokazujemo da
*taj* popravak djeluje. Tri faze odgovaraju trima činovima
ovog izvještaja: otkriće (§§ 4 – 5), neuspio popravak (§§ 6 – 7) i
djelotvoran popravak (§ 9).

## 2. Podaci

- **Univerzum.** Trenutačni Russell 1000 (1.002 dionice s GICS sektorima).
  Pristranost preživljavanja se priznaje: povijesna analiza koristi
  trenutačno članstvo u Russell 1000.
- **Cijene.** Dnevna prilagođena zaključna cijena iz Yahoo Financea putem `yfinance`,
  predmemorirana po dionici, preuzorkovana na kraj mjeseca.
- **Faktori.** Fama-French petfaktorski mjesečni prinosi iz knjižnice
  podataka Kennetha Frencha.
- **Razdoblje.** 2005-01 do 2025-12 (21 godina mjesečnih podataka). Prvih
  pet godina hrani prvi prozor treniranja; testne godine teku od 2010.
  do 2025.

Pokrivenost univerzuma po prozoru treniranja (broj dionica s ≥ 60
valjanih mjeseci treniranja i čistom petfaktorskom prilagodbom) prikazana je u nastavku.
Univerzum raste s ~ 600 u prozoru 2009-12 na ~ 880 u prozoru
2024-12 kako dionice s kasnijim IPO-om postaju prihvatljive.

![Pokrivenost univerzuma kroz klizne prozore](../outputs/figures/01_universe_coverage.png)

## 3. Metodologija

**Značajke dionice.** Za svaki `(prozor, dionica)` pokrećemo OLS
petfaktorsku regresiju na viškovima prinosa i bilježimo `(α, β_MKT, β_SMB,
β_HML, β_RMW, β_CMA, σ_ε, R²)`. Oznake čija prilagodba krši
granice kvalitete podataka (`|β| ≤ 5`, `σ_ε ≤ 0.50`, `R² ≥ 0.05`) izbacuju se
iz tog prozora — ti neuspjesi koncentrirani su u nekolicini
dionica nakon spajanja ili prije IPO-a čije povijesti prinosa proizvode nestabilne
OLS prilagodbe.

**Klasteriranje.**
- *Faktorski klasteri* — Wardova veza na euklidskoj udaljenosti nakon
  standardizacije šest značajki `(β_MKT, β_SMB, β_HML, β_RMW, β_CMA, σ_ε)`.
  To je glavni objekt istraživanja.
- *Korelacijski klasteri* — potpuna veza na unaprijed izračunatoj
  korelacijskoj udaljenosti `sqrt(2(1 − ρ))`. Koristi se kao osnovica.
- *GICS sektori* — treće označavanje, korišteno kao još jedna osnovica.

`K = 7` najviša je postavka po prosječnoj silueti koja zadovoljava oba
ograničenja: svaki klaster ima ≥ 4 dionice u svakom prozoru, i
`K × group_cap ≥ 1` (tako da je portfelj sa samo dugim pozicijama ograničen klasterima
potpuno investibilan uz `group_cap = 0.15`).

**Portfelji.** Portfelji sa samo dugim pozicijama, potpuno investirani po prozoru,
svaki s `w_max = 0.02`:

1. Jednake težine `1/N`.
2. Minimalna varijanca bez grupnog ograničenja.
3. Minimalna varijanca s `Σ w_i ≤ 0.15` unutar svakog GICS sektora.
4. Minimalna varijanca s istim grupnim ograničenjem na korelacijskim klasterima.
5. Minimalna varijanca s istim grupnim ograničenjem na faktorskim klasterima.
6. Faktorski neutralna minimalna varijanca: `|w' β_f| ≤ ε` za
   `f ∈ {SMB, HML, RMW, CMA}`, uz prelet `ε ∈ {0.00, 0.05, 0.10, 0.15}`.

Kovarijanca je Ledoit-Wolf sažimanje na mjesečnim prinosima treniranja,
anualizirano s `× 12`. Uzoračka kovarijanca izvještava se kao provjera
osjetljivosti (§ 8).

**Unaprijedni hod.** Prozor treniranja od 60 mjeseci unatrag, testni
horizont od 12 mjeseci, ponovno procjenjivanje svakih 12 mjeseci → 16 disjunktnih testnih godina (2010. – 2025.).

**Bootstrap.** Blok-bootstrap (Künsch 1989.) s pomičnim
blokovima od 12 mjeseci, 1000 ponovnih uzoraka, korišten za konstrukciju CI-jeva za razliku
metrika između parova portfelja.

---

## Čin I — Otkriće

## 4. Klasteriranje otkriva faktorsku strukturu univerzuma

Faktorsko klasteriranje particionira univerzum u sedam skupina, svaku
koherentnu u prostoru Fama-French izloženosti. Dendrogram reprezentativnog prozora
(2024-12) i radar-grafovi centroida po klasteru pokazuju
unutarnju strukturu:

![Dendrogram faktorskih klastera za reprezentativni prozor](../outputs/figures/02_dendrogram_representative.png)

![Radar-grafovi centroida klastera — standardizirani faktorski profili](../outputs/figures/02_cluster_centroids_radar.png)

UMAP ugnježđenje, obojeno na tri načina, pokazuje da faktorski klasteri
režu univerzum po linijama koje ne vide ni korelacijski klasteri ni
GICS sektori:

![UMAP prostora faktora obojen po faktorskom klasteru, korelacijskom klasteru i GICS sektoru](../outputs/figures/02_factor_umap_three_panels.png)

Stabilnost klastera pod bootstrap ponovnim uzorkovanjem (500 ponovnih uzoraka)
umjerena je — srednji Jaccard zajedničke pripadnosti je `0.30` za faktorske klastere
naspram `0.26` za korelacijske klastere — dovoljno da potvrdi da
klasteri nisu artefakti unutar uzorka.

![Stabilnost klastera: faktorski naspram korelacijskih](../outputs/figures/02_cluster_stability_bar.png)

Ta struktura klasteriranja postaje leća kroz koju se čita ostatak
analize: čini faktorsku geometriju univerzuma
vidljivom i identificira u koje se stilske skupine optimizator privlači.

## 5. Skriveni nagib — minimalna varijanca opterećuje se profitabilnošću i investiranjem

Petfaktorska atribucija prinosa testnog razdoblja na punom uzorku:

| Portfelj | β_MKT | β_SMB | β_HML | β_RMW | β_CMA | konc. stila | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jednake težine | 1.11 | 0.49 | 0.09 | 0.08 | 0.18 | 0.83 | 0.77 |
| Minimalna varijanca | 0.66 | 0.20 | −0.12 | **+0.33** | **+0.34** | 0.98 | 0.72 |
| + ograničenje sektora | 0.68 | 0.21 | −0.14 | +0.26 | +0.36 | 0.97 | 0.76 |
| + ograničenje korelacijskih klastera | 0.71 | 0.20 | −0.09 | +0.31 | +0.26 | 0.86 | 0.78 |

Minimalna varijanca nosi jasno pozitivno opterećenje na RMW (`+0.33`) i CMA
(`+0.34`) — kvintesencijalan „profitabilan, nisko investirajući” stil.
`style_concentration = |β_SMB| + |β_HML| + |β_RMW| + |β_CMA|` iznosi
`0.98`, materijalno iznad `0.83` jednakih težina.

Raščlamba po testnim godinama pokazuje da je nagib postojan kroz oporavak
nakon globalne financijske krize, kvalitetnu rally fazu nakon 2014., šok COVID-a i
kamatni ciklus 2022. – 2023.:

![Koncentracija stila po testnoj godini — pet izvornih portfelja](../outputs/figures/04_style_concentration_per_year.png)

Sektorski ograničeni portfelj smješten je gotovo točno povrh
neograničene linije minimalne varijance (`β_RMW = 0.26`, `β_CMA = 0.36`,
`style_conc = 0.97`). GICS sektori nisu poravnati s faktorskim
izloženostima na način koji optimizator može iskoristiti.
Korelacijsko-klasterski portfelj umjereno smanjuje ukupnu koncentraciju stila
(`0.86`), ali blok-bootstrap CI na razlici pokriva
nulu (CI95% `[−0.14, +0.12]`, `p ≈ 0.99`), pa smanjenje nije
statistički razlučivo od šuma.

Dijagnoza je potvrđena: obična minimizacija rizika na Russell
1000 tiho se opterećuje profitabilnim / nisko investirajućim stilom, a
ni sektorska ni korelacijska diversifikacija to ne ispravljaju. Pitanje
postaje može li se struktura faktorskih klastera — koja *vidi*
nagib — iskoristiti da ga se ograniči.

---

## Čin II — Hipoteza ograničenja klastera i njezin neuspjeh

## 6. Predloženi popravak — ograničenje faktorskih klastera

Faktorski klasteri particioniraju univerzum u skupine koje su
po konstrukciji koherentne u prostoru stila. Intuitivna hipoteza bila je
da bi ograničavanje težine portfelja svakog klastera na 15 % spriječilo
optimizator da se prekomjerno koncentrira u bilo kojoj pojedinoj stilskoj skupini, čime bi se
smanjio neto faktorski nagib portfelja. Mehanizam je
jednostavan: dodaj `Σ_{i ∈ cluster_k} w_i ≤ 0.15` u kvadratni program za
svaki klaster `k`.

Rasuđivanje je bilo uvjerljivo: ako skriveni nagib nastaje jer se
optimizator opterećuje specifičnim područjem prostora faktora, a klasteri
ocrtavaju ta područja, tada bi gornja granica na težini svakog područja trebala
prisiliti optimizator da se raširi po cijelom faktorskom krajoliku i
razrijedi nagib. To je popravak predložen u `PROJECT_SPEC.md` § 6.

## 7. Rezultati — popravak ne uspijeva

**Glavni test, `min_var_factor_cluster − min_var`**:

| Metrika | Točkasta procjena | Blok-bootstrap 95% CI | `p` |
|---|---:|---:|---:|
| Koncentracija stila | **+0.49** | `[ −0.16,  +0.71 ]` | 0.30 |
| Anualizirana vol. | **+0.058** | `[ +0.023, +0.098 ]` | 0.000 |
| Maksimalni pad | −0.030 | `[ −0.119, +0.014 ]` | 0.13 |

Faktorsko-klasterski portfelj `5.8` postotnih bodova je volatilniji godišnje od
neograničene minimalne varijance, značajno uz `p ≈ 0.000` pod blok
bootstrapom. Razlika koncentracije stila je pozitivna (popravak *povećava*
koncentraciju stila), iako CI pokriva nulu.

| Portfelj | β_MKT | β_SMB | β_HML | β_RMW | β_CMA | konc. stila |
|---|---:|---:|---:|---:|---:|---:|
| Minimalna varijanca | 0.66 | 0.20 | −0.12 | **+0.33** | **+0.34** | 0.98 |
| + ograničenje faktorskih klastera | 0.83 | 0.55 | −0.19 | −0.15 | +0.58 | 1.47 |

Ograničenje faktorskih klastera *jest* neutraliziralo RMW nagib
(`+0.33 → −0.15`) — upravo opterećenje koje je dijagnoza označila — ali je
gurnulo težinu u klastere s jačim SMB i CMA izloženostima
(`β_SMB: 0.20 → 0.55`, `β_CMA: 0.34 → 0.58`). Neto je portfelj
s *više* ukupne koncentracije stila od neograničene
osnovice, uz osjetno višu ostvarenu volatilnost.

![Bootstrap distribucija: razlika koncentracije stila (faktorski klaster − min. var.)](../outputs/figures/04_bootstrap_style_gap.png)

**Dekompozicija rizika** (srednji udio faktorske varijance kroz 16 testnih
godina):

| Portfelj | Udio faktorske varijance |
|---|---:|
| Jednake težine | 0.967 |
| Minimalna varijanca | 0.868 |
| + ograničenje sektora | 0.865 |
| + ograničenje korelacijskih klastera | 0.862 |
| + ograničenje faktorskih klastera | 0.890 |

![Srednja faktorska / idiosinkratska varijanca po portfelju](../outputs/figures/04_risk_decomposition_stacked.png)

**Zašto je ograničenje klastera zakazalo.** Strukturni razlog jest da su Wardovi
klasteri u prostoru faktora po konstrukciji tijesni — svaki je klaster
koherentna stilska skupina. Ograničavanje težine svakog klastera diversificira
*preko* granica klastera, ali svaki klaster pojedinačno nosi
stilski nagib, a neto izloženost portfelja težinski je prosjek
tih nagiba koji se ne moraju poništiti. Ograničenje grupne težine ne cilja
nijednu konkretnu faktorsku betu; ograničava *količinu* težine u
svakoj stilskoj skupini, ali ne govori ništa o *smjeru* faktorske
izloženosti te skupine. Optimizator, prisiljen raširiti težinu, jednostavno
je preraspodjeljuje u klastere čiji vlastiti nagibi zbrojeno daju veću ukupnu
koncentraciju stila nego prije.

Taj prijelaz ključni je uvid: klasteri ispravno identificiraju
*gdje* faktorska struktura živi — oni su dobra dijagnostika — ali
su pogrešna *poluga* za ograničenje. Da bi se stilska izloženost stvarno smanjila,
ograničenje mora djelovati izravno na faktorske bete.

---

## Čin III — Djelotvoran popravak

## 8. Robusnost dijagnoze

Prije nego prijeđemo na popravak, potvrđujemo da dijagnoza preživljava svaku
provjeru robusnosti.

**Prelet osjetljivosti**
(`outputs/tables/04_sensitivity_sweep.csv`). Koncentracija stila
pod svakom kombinacijom `(cov_estimator, w_max, group_cap)`:
faktorsko-klasterski portfelj ima najvišu koncentraciju stila u svakoj
izvedivoj ćeliji osim najopuštenije (`w_max = 0.015`, `u = 0.20`).
Korelacijsko-klasterski portfelj dosljedno je najniži. Sektorski
portfelj u biti se ne razlikuje od minimalne varijance. Prelazak
s Ledoit-Wolf na uzoračku kovarijancu ostavlja poredak nepromijenjenim.

![Toplinska karta osjetljivosti: koncentracija stila kroz mrežu parametara](../outputs/figures/04_sensitivity_style_heatmap.png)

**Stabilnost klastera.** Na reprezentativnom prozoru, srednji bootstrap
Jaccard zajedničke pripadnosti (500 ponovnih uzoraka): faktorsko klasteriranje `0.30`,
korelacijsko klasteriranje `0.26` (`outputs/tables/02_cluster_stability.csv`).
Faktorski klasteri umjereno su stabilniji od korelacijskih klastera
pod ponovnim uzorkovanjem, pa *struktura* ograničenja nije artefakt unutar
uzorka — no sama stabilnost ne spašava popravak.

**Kumulativni rast i pad** kroz puno razdoblje unaprijednog hoda
potvrđuju da viša volatilnost faktorsko-klasterskog portfelja nije
kompenzirana boljim prinosima na rizikom prilagođenoj osnovi:

![Kumulativni rast i pad, 2010. – 2025.](../outputs/figures/04_cumulative_and_drawdown.png)

**Klizne 12-mjesečne bete** pokazuju da je stilski nagib postojan, a ne
vođen jednom epizodom:

![Klizne 12-mjesečne bete portfelja](../outputs/figures/04_rolling_betas.png)

**Robusnost na metodologiju klasteriranja** (notebook 07). Faktorska struktura
ne ovisi o izboru Wardove veze: ponovno klasteriranje
reprezentativnog prozora s **K-meansom** vraća u biti iste
skupine (prilagođeni Randov indeks ≈ 0.55, gotovo identična silueta i
Calinski-Harabasz), a na stršeće vrijednosti otporan **k-medoid** prati ih
izbliza. Četiri indeksa interne validacije koja se uče na kolegiju —
silueta, Calinski-Harabasz, Davies-Bouldin i Tibshiranijeva **Gap
statistika** — svi upućuju na *statistički* prirodan `K` od 3 – 5, a ne
7. `K = 7` korišten kroz cijeli rad stoga je iskreno
ograničenje investibilnosti portfelja (`K × group_cap ≥ 1`), položeno povrh
grublje faktorske geometrije, a **ne** optimum siluete — što je
dio razloga zašto je popravak ograničenjem klastera imao tako malo strukture za uhvatiti (§ 7).
**DBSCAN** ne nalazi prirodne praznine gustoće u prostoru faktora (njegovo označavanje
jedva se slaže s particijskim metodama, ARI ≈ 0), potvrđujući da su
metode s nametnutim `K` pravi alat ovdje i otkrivajući aureolu
istinski idiosinkratskih „bezstilskih” dionica kao šum.

![Indeksi odabira K i Gap statistika](../outputs/figures/07_kselection_indices.png)

## 9. Proširenje — izravna faktorska neutralnost (djelotvoran popravak)

Neuspjeh ograničenja klastera ukazao je na lijek: ograničenje
mora djelovati na geometriji koju je klasteriranje otkrilo — faktorskim betama
portfelja — a ne na strukturi grupnih težina. Proširujemo
kvadratni program minimalne varijance linearnim ograničenjima

$$
|\, w^{\!\top} \beta_f \,| \le \varepsilon, \qquad f \in \{\mathrm{SMB},\ \mathrm{HML},\ \mathrm{RMW},\ \mathrm{CMA}\},
$$

gdje je `β_f` faktorska izloženost dionice za faktor `f`, procijenjena na istom
prozoru treniranja. `ε = 0` nameće strogu faktorsku neutralnost na strani
treniranja; pozitivan `ε` dopušta kontrolirani nagib. Ograničenje je
linearno, funkcija cilja je nepromijenjena, kvadratni program ostaje konveksan.

Obitelj `factor_neutral_eX` prelijeće `ε ∈ {0.00, 0.05, 0.10, 0.15}`
i izgrađena je u `notebooks/05_factor_neutral.ipynb`.

**Petfaktorska atribucija na punom uzorku i ostvarena vol.** (prinosi testnog razdoblja,
192 mjesečna opažanja):

| Portfelj | β_MKT | β_SMB | β_HML | β_RMW | β_CMA | konc. stila | god. vol. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minimalna varijanca | 0.66 | +0.20 | −0.12 | **+0.33** | **+0.34** | 0.98 | 12.0 % |
| faktorski neutralan, ε = 0.00 | 0.69 | +0.19 | −0.03 | **+0.26** | **+0.20** | **0.67** | **12.1 %** |
| faktorski neutralan, ε = 0.05 | 0.68 | +0.19 | −0.06 | +0.29 | +0.22 | 0.76 | 12.0 % |
| faktorski neutralan, ε = 0.10 | 0.67 | +0.19 | −0.07 | +0.31 | +0.24 | 0.81 | 12.0 % |
| faktorski neutralan, ε = 0.15 | 0.67 | +0.21 | −0.11 | +0.30 | +0.31 | 0.93 | 12.0 % |

**Zašto ostvarene bete ne dosežu nulu pri ε = 0.** Ograničenje
faktorske neutralnosti aktivno je na betama *prozora treniranja*: ono
prisiljava `|w' β_f^{train}| ≤ 0` u trenutku kad je portfelj
konstruiran. Ali testna regresija procjenjuje *drukčiju*
`β_f^{test}` — onu koja odražava 12 mjeseci podataka koje optimizator
nikada nije vidio. Običan pomak procjene između 60-mjesečnog prozora
treniranja i 12-mjesečnog testnog prozora dovoljan je da proizvede nenulte
ostvarene bete čak i kad je ograničenje na strani treniranja točno
zadovoljeno. Smjer je ipak monoton: stroži `ε`
proizvodi nižu ostvarenu koncentraciju stila u svakoj testnoj godini.

**Blok-bootstrap CI-jevi naspram `min_var`** (pomični blokovi od 12 mjeseci, 1000
ponovnih uzoraka, faktori ponovno uzorkovani zajedno s prinosima portfelja):

| Usporedba | Δ konc. stila | 95% CI | `p` | Δ god. vol. | 95% CI | `p` |
|---|---:|---:|---:|---:|---:|---:|
| faktorski neutralan, ε = 0.00 − min_var | **−0.308** | **[ −0.47, −0.09 ]** | **0.002** | +0.001 | [ −0.003, +0.005 ] | 0.59 |
| faktorski neutralan, ε = 0.05 − min_var | −0.225 | [ −0.39, −0.06 ] | 0.004 | +0.000 | [ −0.003, +0.003 ] | 0.74 |
| faktorski neutralan, ε = 0.10 − min_var | −0.171 | [ −0.32, −0.03 ] | 0.006 | −0.000 | [ −0.003, +0.002 ] | 0.98 |
| faktorski neutralan, ε = 0.15 − min_var | −0.054 | [ −0.10, −0.01 ] | 0.020 | +0.000 | [ −0.001, +0.002 ] | 0.60 |

(`outputs/tables/05_bootstrap_ci_extended.csv`.)

Stroga ćelija `ε = 0` smanjuje ostvarenu koncentraciju stila za 31 %
(`0.98 → 0.67`) uz u biti nulti trošak u ostvarenoj volatilnosti (`+0.001`
anualizirano, CI tijesno oko nule). CI razlike koncentracije stila leži
u potpunosti ispod nule pri svakom `ε`, pa je smanjenje statistički
razlučivo od šuma čak i nakon blok bootstrapa sa
zajedničkim ponovnim uzorkovanjem faktora. CI-jevi razlike volatilnosti obuhvaćaju nulu, pa je popravak
neutralan na volatilnost. To je kompromis koji je dijagnoza tražila.

![Bootstrap distribucija: Δ koncentracije stila (faktorski neutralan ε=0 − min_var)](../outputs/figures/05_bootstrap_factor_neutral_e0.png)

**Koncentracija stila po testnoj godini**, s faktorski neutralnom obitelji
preklopljenom na pet izvornih, potvrđuje da je smanjenje postojano
kroz režime:

![Koncentracija stila po testnoj godini s preklopljenom faktorski neutralnom obitelji](../outputs/figures/05_style_concentration_per_year_with_neutral.png)

**Hibrid (ograničenje klastera + faktorska neutralnost).**
`factor_cluster_neutral_eX` spaja Wardovo ograničenje klastera iz § 6
s novim ograničenjem faktorske neutralnosti. Pri `ε ≥ 0.10` doseže
najnižu ostvarenu koncentraciju stila u pokretanju (`0.57 – 0.63`),
ali ograničenje klastera gura kvadratni program u kutna rješenja čije
ostvarene bete daleko odlutaju od beta treniranja, uz trošak od ~3 postotna boda
anualizirane vol. naspram minimalne varijance. Pri manjem `ε` hibrid je
nestabilan (ostvarena vol. >30 % anualizirano pri `ε ∈ {0, 0.05}`). Čista
faktorski neutralna obitelj smještena je na efikasnoj granici; ograničenje klastera
ne obavlja dodatan posao jednom kad su ograničenja po faktoru
postavljena.

**Zašto je hibrid pri ε = 0 degeneriran.** Kad se ograničenje klastera i
stroga faktorska neutralnost (`ε = 0`) nameću istodobno, izvediv
skup smanjuje se na tanku iverku: kvadratni program mora pronaći težine koje
zbrojeno daju 1, poštuju ograničenje pozicije od 2 %, ostaju ispod 15 % po klasteru *i*
poništavaju četiri faktorske bete — sve na univerzumu čiji su klasteri
i sami korelirani s tim betama. Optimizator je prisiljen u
ekstremna kutna rješenja koncentrirana u nekolicini dionica koje
slučajno leže na sjecištu svih ograničenja, proizvodeći
portfelj s 86 % anualizirane ostvarene volatilnosti — u biti
nediversificiran. Pri `ε = 0.05` hibrid se dovoljno opušta da izbjegne
neizvedivost u većini prozora, ali i dalje ostvaruje 34 % vol., potvrđujući
da se dvije obitelji ograničenja međusobno bore umjesto da se
nadopunjuju.

**Granica: koncentracija stila naspram ostvarene vol.**

![Koncentracija stila naspram ostvarene vol. — sve obitelji portfelja](../outputs/figures/05_frontier_style_vs_vol.png)

## 10. Raspodjela težina po klasterima

Da bismo priču o faktorskoj strukturi učinili konkretnom, figura ispod prikazuje
kako svaki portfelj raspoređuje težinu po sedam faktorskih klastera
na najnovijem prozoru treniranja (2024-12). Klasteri su označeni
prema svoje dvije dominantne standardizirane faktorske izloženosti.

![Raspodjela težina po klasterima po portfelju — reprezentativni prozor (2024-12)](../outputs/figures/06_cluster_weight_allocation.png)

Obrazac potvrđuje naraciju. Minimalna varijanca koncentrira 47 %
težine u C4 (niska β_MKT, niska β_SMB — defenzivni klaster velikih
dionica) i 37 % u C5 (visok CMA, nizak HML — nisko investirajući
klaster). Zajedno ta dva klastera čine 84 % težine
portfelja, a njihov kombinirani faktorski profil upravo je RMW/CMA
nagib koji je dijagnoza identificirala. Ograničenje sektora jedva mijenja
ovu sliku. Ograničenje faktorskih klastera spljošti svaki stupac na gornju granicu od 15 %,
gurajući težinu u klastere koje bi optimizator inače
izbjegavao (C1, C2, C6, C7), ali ne može kontrolirati *neto* faktorsku
izloženost koja iz toga proizlazi. Faktorski neutralan portfelj (ε = 0) i dalje
koncentrira u C4 (54 %) — optimizator i dalje traži nisku varijancu —
ali pomiče težinu s C5 prema C1, proizvodeći mješavinu klastera
čiji se faktorski nagibi djelomično poništavaju. Jednake težine, kako se i očekuje,
prate sastav univerzuma.

## 11. Pokrivenost metoda kolegija — algoritmi klasteriranja i nadzirano stablo

Dva dodatka smještaju projekt čvrsto na metode koje se uče na
kolegiju i podvrgavaju njegove izbore stres-testu.

**Algoritmi klasteriranja i odabir K** (notebook 07, predavanja 9b/10).
Povrh Wardova hijerarhijskog klasteriranja glavnog protoka, prostor faktora
ponovno se klasterira **particijskim** (K-means, k-medoid),
**probabilističkim** (Gaussova mješavina / EM) i **na gustoći baziranim**
(DBSCAN) metodama, a `K` se ocjenjuje **siluetom, Calinski-Harabaszom,
Davies-Bouldinom i Gap statistikom**. Glavni zaključci nalaze se u
§ 8: struktura je robusna na algoritam, prirodan `K` je 3 – 5, a
`K = 7` ograničenje je investibilnosti. Toplinska karta prilagođenog Randova indeksa
po parovima i indeksi po metodi spremljeni su u
`outputs/tables/07_method_ari.csv` i
`outputs/tables/07_method_comparison_scores.csv`.

![Usporedba algoritama klasteriranja — ARI i silueta](../outputs/figures/07_method_comparison.png)

**Nadzirano proširenje — predviđanje pada** (notebook 08, predavanja
3 – 4; `PROJECT_SPEC.md` § 9). Za svaku dionicu i klizni testni period
označavamo „visok pad” (gori od medijana presjeka te godine) i
predviđamo ga iz značajki prozora treniranja **stablima odlučivanja**
(entropija / porast informacije i Gini, dubina 3) i **ansamblima** (slučajna
šuma, gradijentni boosting), evaluirano s **GroupKFold po testnoj godini**.
Uspoređuju se tri skupa značajki: faktorske bete, `+` GICS sektor, `+`
oznaka faktorskog klastera.

| Skup značajki | najbolji ROC-AUC | najbolja točnost |
|---|---:|---:|
| Samo faktorske bete | 0.69 | 0.64 |
| + GICS sektor | 0.69 | 0.63 |
| + oznaka faktorskog klastera | 0.68 | 0.64 |

(Najbolji model po retku je gradijentni boosting; puna mreža u
`outputs/tables/08_cv_results.csv`.) Pad sljedeće godine *umjereno* je
predvidljiv iz faktorskih izloženosti prozora treniranja (AUC ≈ 0.69 naspram
linije slučaja od 0.5), ali važnosti iz slučajne šume pokazuju da signal
dominira `residual_vol` (idiosinkratski rizik) i tržišna beta — a ne
stilski faktori. Dodavanje GICS sektora ili oznake faktorskog klastera **ne**
poboljšava nad sirovim betama (AUC ≈ 0.68 – 0.69 u oba slučaja): klaster
je izvrstan *deskriptivni* objekt, ali **ne** i dodatna *prediktivna*
značajka. To se izvještava kakvo jest — deskriptivna
struktura ne udvostručuje se kao prediktor.

![CV rezultati predviđanja pada](../outputs/figures/08_cv_results.png)

**Karta metoda kolegija.**

| Tema kolegija | Predavanje | Gdje u ovom projektu |
|---|---|---|
| Statistika / standardizacija | 1–2 | z-standardizirane faktorske značajke |
| Stabla odlučivanja (entropija/porast info., Gini) | 3 | notebook 08 |
| Ansambli (bagging, boosting) | 4 | notebook 08 |
| Particijsko klasteriranje (K-means, varijante) | 9b | notebook 07 |
| Odabir K (silueta, Calinski-Harabasz, Gap) | 9b | notebookovi 02, 07 |
| Hijerarhijsko klasteriranje (Ward, Lance-Williams) | 10 | notebook 02 (glavni protok) |
| Probabilističko klasteriranje (GMM / EM) | 10 | notebook 07 |
| Klasteriranje bazirano na gustoći (DBSCAN) | 10 | notebook 07 |
| Validacija / stabilnost klastera | 9b–10 | notebook 02 (bootstrap Jaccard) |

## 12. Zaključak

**Što smo otkrili:** portfelji minimalne varijance na Russell
1000 nose skriven, postojan nagib prema faktorima profitabilnosti (RMW)
i investiranja (CMA) Fama-French — koncentraciju stila
nevidljivu diversifikaciji po sektorima ili korelacijama, ali jasno
ocrtanu klasteriranjem u prostoru faktora.

**Što popravak postiže:** dodavanje linearnih ograničenja
`|w' β_f| ≤ ε` izravno u kvadratni program smanjuje ostvarenu koncentraciju stila za
31 % pri ε = 0 bez vidljivog troška u ostvarenoj volatilnosti
(`p ≈ 0.002` za razliku stila; CI razlike vol. obuhvaća nulu).

Struktura faktorskih klastera izgrađena u §§ 3 – 4 bila je prava
dijagnostika — učinila je skriveni nagib vidljivim i identificirala u kojem
području prostora faktora se optimizator koncentrira. No
ograničenje klastera (§§ 6 – 7) bilo je pogrešna poluga: ograničavanje grupne
težine ne cilja nijednu konkretnu betu. Djelotvoran popravak stavlja
ograničenje točno ondje gdje je klasteriranje reklo da je problem — na
same faktorske bete.

**Ograničenja.**

- Pristranost preživljavanja: univerzum je trenutačni Russell 1000, a ne
  indeks u stvarnoj točki vremena. Dionice s kasnijim IPO-om ulaze kako njihova povijest
  sazrijeva, no izbačene dionice nikada se ne pojavljuju.
- Bete su procijenjene iz 60 mjesečnih opažanja po prozoru — dovoljno malo
  da budu šumovite, osobito za dionice s niskim `R²`. Filtar kvalitete
  podataka (§ 3) obrezuje najgore prijestupnike, ali ne uklanja
  šum procjene.
- Transakcijski troškovi i obrtaj zanemareni su.
- Procjenitelj kovarijance i univerzum specifični su izbori; kvalitativna
  dijagnoza vjerojatno nije specifična za Ledoit-Wolf
  (prelet osjetljivosti potvrđuje), ali generalizacija na neameričke ili
  univerzume manjih dionica nije testirana.
- K klastera drži se fiksnim kroz prozore radi usporedivosti; odabir K po prozoru
  mogao bi dati drukčije zaključke, iako je
  krajolik siluete dosta plosnat za `K ∈ {5..10}`.
- Predviđanje prinosa nije cilj; brojeve kumulativnog rasta u
  dodatku treba čitati kao nuspojavu, a ne rezultat.
- Korelacijsko-klasterski portfelj i hibridi ε ∈ {0, 0.05}
  agregirani su preko manje testnih mjeseci (168 / 180 naspram 192) jer je njihovo
  ograničenje bilo neizvedivo u najranijem prozoru(ima) — zabilježeno u
  `portfolio_status` — pa njihove agregatne metrike nisu nad
  identičnim uzorkom kao kod ostalih portfelja.
- Blok-bootstrap ponovno uzorkuje blokove od 12 mjeseci; za o putanji ovisnu
  metriku maksimalnog pada dobiveni CI premiješa putanju pada i
  treba ga čitati kao indikativan, a ne točan (CI-jevi koncentracije stila
  i volatilnosti, koji su neovisni o redoslijedu, time nisu zahvaćeni).

---

### Dodatak — potpuni sažetak rizika

| Portfelj | God. prinos | God. vol. | Maksimalni pad | Konc. stila |
|---|---:|---:|---:|---:|
| Jednake težine | 19.0 % | 21.2 % | −27.7 % | 0.83 |
| Minimalna varijanca | 14.0 % | 12.0 % | −19.5 % | 0.98 |
| + ograničenje sektora | 14.8 % | 12.1 % | −18.2 % | 0.97 |
| + ograničenje korelacijskih klastera | 13.0 % | 12.2 % | −28.6 % | 0.86 |
| + ograničenje faktorskih klastera | 20.6 % | 17.9 % | −22.4 % | 1.47 |
| faktorski neutralan, ε = 0.00 | 14.0 % | 12.1 % | −20.2 % | **0.67** |
| faktorski neutralan, ε = 0.05 | 13.9 % | 12.0 % | −19.9 % | 0.76 |
| faktorski neutralan, ε = 0.10 | 13.8 % | 12.0 % | −19.7 % | 0.81 |
| faktorski neutralan, ε = 0.15 | 14.2 % | 12.0 % | −19.6 % | 0.93 |
| hibrid (klaster + faktorski neutralan), ε = 0.10 | 18.4 % | 15.4 % | −22.6 % | 0.63 |
| hibrid (klaster + faktorski neutralan), ε = 0.15 | 18.4 % | 14.9 % | −22.4 % | **0.57** |

(Agregirano kroz unaprijedni hod 2010. – 2025.;
`outputs/tables/04_risk_summary.csv`,
`outputs/tables/05_risk_summary_all.csv`. Hibrid pri `ε ∈ {0, 0.05}`
izostavljen je iz tablice — nestabilan je, s ostvarenom anualiziranom
vol. od 86 % odnosno 34 %.)

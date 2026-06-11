# PROJECT_SPEC.md

> **Status:** minimalni stub. Puni dokument (identitet rada, hipoteze H1–H3,
> zaključane odluke, definicije alokatora, overlay QP, metrike, statistika,
> konvencije imenovanja) piše se iznova u tasku **F0.11**. Ovdje je zasad samo
> odlomak „Izvor članstva” koji dokumentira odluku grananja iz **F0.4**.

## Podaci

### Izvor članstva

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

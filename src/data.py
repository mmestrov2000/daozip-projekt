"""Prikupljanje i pretprocesiranje podataka za projekt klasteriranja portfelja."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.utils import (
    MIN_TRAINING_MONTHS,
    MONTHLY_RETURN_CLIP_HIGH,
    MONTHLY_RETURN_CLIP_LOW,
    PRICE_CACHE_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_END,
    PROJECT_START,
    RAW_DATA_DIR,
    TABLES_DIR,
)


LOGGER = logging.getLogger(__name__)

IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
RUSSELL_1000_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SP500_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
SP500_CURRENT_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
TICKER_OVERRIDES_PATH = RAW_DATA_DIR / "ticker_overrides.csv"
SP500_MEMBERSHIP_PATH = RAW_DATA_DIR / "sp500_membership.csv"
PRICE_SOURCE_LOG_PATH = RAW_DATA_DIR / "price_source.csv"
FF5_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

_USER_AGENT = "daozip-factor-clustering/1.0 (academic project)"


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp(how="start")


def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp(how="end").normalize()


def _yahoo_ticker(ticker: str) -> str:
    """Pretvori oznake poput BRK.B u Yahoov format BRK-B."""
    return ticker.replace(".", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# Univerzum Russell 1000 putem iShares IWB
# ---------------------------------------------------------------------------


def _parse_iwb_csv(raw_text: str) -> tuple[pd.DataFrame, str | None]:
    """Parsiraj iShares IWB CSV udjela, vraćajući (holdings, fund_date).

    iShares izvoz dodaje blok metapodataka prije retka koji počinje
    s ``Ticker,Name,Sector,...``. Pronalazimo taj redak zaglavlja i čitamo
    tablicu od njega. Datum stanja fonda pojavljuje se u bloku metapodataka
    pod oznakom ``Fund Holdings as of``.
    """
    lines = raw_text.splitlines()
    fund_date: str | None = None
    header_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"Fund Holdings as of"'):
            parts = [part.strip(' "') for part in stripped.split(",", maxsplit=1)]
            if len(parts) == 2:
                fund_date = parts[1]
        if stripped.lower().startswith("ticker,"):
            header_index = index
            break

    if header_index is None:
        raise ValueError("Nije moguće pronaći zaglavlje s oznakama u IWB CSV-u.")

    data_lines: list[str] = [lines[header_index]]
    for line in lines[header_index + 1 :]:
        if not line.strip():
            break
        data_lines.append(line)

    holdings = pd.read_csv(StringIO("\n".join(data_lines)))
    holdings.columns = [str(column).strip() for column in holdings.columns]
    return holdings, fund_date


def _fetch_iwb_holdings_csv(url: str) -> tuple[pd.DataFrame, str | None, str]:
    """Preuzmi IWB CSV udjela. Diže iznimku ako iShares vrati stranicu za botove."""
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 "
                "Safari/537.36"
            ),
            "Accept": "text/csv,application/csv,*/*",
            "Referer": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf",
        },
        timeout=60,
    )
    response.raise_for_status()
    raw_text = response.text
    if raw_text.lstrip().lower().startswith("<!doctype html") or "Ticker," not in raw_text[:200_000]:
        raise RuntimeError(
            "Krajnja točka iShares IWB CSV-a vratila je HTML stranicu umjesto "
            "datoteke udjela (stranica blokira skriptirana preuzimanja). Koristi "
            "Wikipedijinu zamjenu ili stavi ručno preuzetu kopiju u "
            "data/raw/russell1000_tickers.csv."
        )
    holdings, fund_date = _parse_iwb_csv(raw_text)
    return holdings, fund_date, raw_text


def fetch_russell1000_tickers(
    output_path: str | Path | None = None,
    url: str = IWB_HOLDINGS_URL,
    raw_dir: str | Path = RAW_DATA_DIR,
    fallback_to_wikipedia: bool = True,
) -> pd.DataFrame:
    """Dohvati trenutačni univerzum Russell 1000.

    Prvo pokušava iShares IWB CSV udjela (kanonski izvor iz specifikacije).
    Ako iShares vrati svoju HTML stranicu za detekciju botova (što čini za većinu
    skriptiranih zahtjeva), vraća se na Wikipedijinu tablicu sastavnica
    „Russell 1000 Index”. Odabrani izvor bilježi se u pratećoj datoteci
    metapodataka kako bi ga izvještaj mogao pošteno dokumentirati.

    Vraća DataFrame sa stupcima ``ticker, name, sector`` sortiran po
    oznaci dionice.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    source: str
    fund_date: str | None = None
    raw_text: str | None = None
    holdings: pd.DataFrame

    try:
        holdings, fund_date, raw_text = _fetch_iwb_holdings_csv(url)
        if "Asset Class" in holdings.columns:
            holdings = holdings[
                holdings["Asset Class"].astype(str).str.strip().eq("Equity")
            ]
        column_map = {"Ticker": "ticker", "Name": "name", "Sector": "sector"}
        missing = [col for col in column_map if col not in holdings.columns]
        if missing:
            raise RuntimeError(f"IWB datoteci udjela nedostaju stupci: {missing}")
        result = holdings.rename(columns=column_map)[["ticker", "name", "sector"]]
        source = "ishares_iwb"
        (raw_dir / "iwb_holdings_raw.csv").write_text(raw_text)
    except Exception as iwb_error:
        if not fallback_to_wikipedia:
            raise
        LOGGER.warning("IWB preuzimanje nije uspjelo (%s); vraćam se na Wikipediju.", iwb_error)
        wiki_response = requests.get(
            RUSSELL_1000_WIKIPEDIA_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=60,
        )
        wiki_response.raise_for_status()
        raw_text = wiki_response.text
        tables = pd.read_html(StringIO(raw_text))
        components: pd.DataFrame | None = None
        for table in tables:
            cols = {str(column).strip().lower(): column for column in table.columns}
            if {"company", "symbol", "gics sector"}.issubset(cols):
                components = table.rename(
                    columns={
                        cols["company"]: "name",
                        cols["symbol"]: "ticker",
                        cols["gics sector"]: "sector",
                    }
                )[["ticker", "name", "sector"]]
                break
        if components is None:
            raise RuntimeError(
                "Nije moguće pronaći tablicu sastavnica Russell 1000 na Wikipediji."
            ) from iwb_error
        result = components
        source = "wikipedia_russell1000"

    result = result.assign(
        ticker=lambda frame: frame["ticker"].astype(str).str.strip(),
        name=lambda frame: frame["name"].astype(str).str.strip(),
        sector=lambda frame: frame["sector"].astype(str).str.strip(),
    )
    result = result.replace({"ticker": {"-": np.nan, "": np.nan}}).dropna(subset=["ticker"])
    result = result[~result["sector"].str.lower().isin({"cash and/or derivatives", "-"})]
    result = (
        result.drop_duplicates(subset="ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    meta = {
        "source": source,
        "source_url": url if source == "ishares_iwb" else RUSSELL_1000_WIKIPEDIA_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "fund_holdings_as_of": fund_date,
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if raw_text is not None
        else None,
        "n_holdings": int(len(result)),
    }
    (raw_dir / "iwb_holdings_meta.json").write_text(json.dumps(meta, indent=2))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


# ---------------------------------------------------------------------------
# Point-in-time članstvo S&P 500 (F0.4 grana b: javna rekonstrukcija)
# ---------------------------------------------------------------------------


def load_ticker_overrides(path: str | Path = TICKER_OVERRIDES_PATH) -> pd.DataFrame:
    """Učitaj ručnu mapu poznatih kolizija/preimenovanja tickera.

    Vraća DataFrame sa stupcima ``source_ticker, ticker, download_ticker, note``;
    prazan DataFrame ako datoteka ne postoji. Komentar-retci (``#``) se preskaču.
    """
    path = Path(path)
    columns = ["source_ticker", "ticker", "download_ticker", "note"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    overrides = pd.read_csv(path, comment="#").reindex(columns=columns)
    for column in columns:
        overrides[column] = overrides[column].fillna("").astype(str).str.strip()
    return overrides


def _fetch_current_sp500_names(url: str = SP500_CURRENT_URL) -> dict[str, str]:
    """Dohvati imena trenutnih članova S&P 500 (sp500.csv istog repozitorija)."""
    try:
        response = requests.get(url, timeout=60, headers={"User-Agent": _USER_AGENT})
        response.raise_for_status()
        current = pd.read_csv(StringIO(response.text))
        return dict(
            zip(
                current["Symbol"].astype(str).str.strip(),
                current["Security"].astype(str).str.strip(),
            )
        )
    except Exception as error:
        LOGGER.warning("Dohvat imena trenutnih članova nije uspio: %s", error)
        return {}


def fetch_sp500_membership(
    output_path: str | Path | None = SP500_MEMBERSHIP_PATH,
    url: str = SP500_MEMBERSHIP_URL,
    raw_dir: str | Path = RAW_DATA_DIR,
    overrides_path: str | Path = TICKER_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Preuzmi i parsiraj point-in-time članstvo S&P 500 u dugu tablicu intervala.

    Izvor (fja05680/sp500, „Updated” datoteka) drži po jedan redak po datumu
    promjene s popisom svih članova na taj dan. Ovdje se snapshoti pretvaraju u
    intervale ``ticker, name, start_date, end_date, source``: ticker koji nestane
    na snapshotu ``d`` dobiva ``end_date = d − 1 dan``; prisutan u zadnjem
    snapshotu ostaje otvoren (``end_date`` prazan). Ticker smije imati više
    intervala (izlazak pa ponovni ulazak, npr. GM).

    Izvor delistane firme vodi pod zadnjim OTC simbolom (LEHMQ, MTLQQ);
    mapa iz ``ticker_overrides.csv`` vraća povijesne simbole (LEH, GM).
    Imena se popunjavaju iz postojećeg ``data/processed/metadata.csv`` gdje
    postoje, inače iz trenutnog ``sp500.csv``; za ostale delistane ostaju prazna.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=120, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    raw_text = response.text

    snapshots = pd.read_csv(StringIO(raw_text))
    if not {"date", "tickers"}.issubset(snapshots.columns):
        raise ValueError("Izvor članstva nema očekivane stupce 'date,tickers'.")
    snapshots["date"] = pd.to_datetime(snapshots["date"])
    snapshots = snapshots.sort_values("date").reset_index(drop=True)

    overrides = load_ticker_overrides(overrides_path)
    rename_map = {
        row["source_ticker"]: row["ticker"]
        for _, row in overrides.iterrows()
        if row["source_ticker"] and row["ticker"]
    }

    open_intervals: dict[str, pd.Timestamp] = {}
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    previous: set[str] = set()
    for _, snapshot in snapshots.iterrows():
        members = {
            rename_map.get(symbol, symbol)
            for symbol in (part.strip() for part in str(snapshot["tickers"]).split(","))
            if symbol
        }
        date = snapshot["date"]
        for ticker in members - previous:
            open_intervals[ticker] = date
        for ticker in previous - members:
            rows.append((ticker, open_intervals.pop(ticker), date - pd.Timedelta(days=1)))
        previous = members
    for ticker, start in open_intervals.items():
        rows.append((ticker, start, pd.NaT))

    membership = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])

    name_map = _fetch_current_sp500_names()
    metadata_path = Path(PROCESSED_DATA_DIR) / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        name_map.update(
            dict(
                zip(
                    metadata["ticker"].astype(str).str.strip(),
                    metadata["name"].astype(str).str.strip(),
                )
            )
        )
    membership["name"] = membership["ticker"].map(name_map).fillna("")
    membership["source"] = "fja05680_sp500_updated"
    membership = (
        membership[["ticker", "name", "start_date", "end_date", "source"]]
        .sort_values(["ticker", "start_date"])
        .reset_index(drop=True)
    )

    meta = {
        "source": "fja05680_sp500_updated",
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "n_snapshots": int(len(snapshots)),
        "n_intervals": int(len(membership)),
        "n_tickers": int(membership["ticker"].nunique()),
    }
    (raw_dir / "sp500_membership_meta.json").write_text(json.dumps(meta, indent=2))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        membership.to_csv(output_path, index=False)

    return membership


_MEMBERSHIP_CACHE: pd.DataFrame | None = None


def _load_membership() -> pd.DataFrame:
    """Lijeno učitaj tablicu članstva s diska (modularni cache)."""
    global _MEMBERSHIP_CACHE
    if _MEMBERSHIP_CACHE is None:
        if not SP500_MEMBERSHIP_PATH.exists():
            raise FileNotFoundError(
                f"{SP500_MEMBERSHIP_PATH} ne postoji; pokreni fetch_sp500_membership()."
            )
        _MEMBERSHIP_CACHE = pd.read_csv(
            SP500_MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"]
        )
    return _MEMBERSHIP_CACHE


def membership_on(
    date: str | pd.Timestamp,
    membership: pd.DataFrame | None = None,
) -> list[str]:
    """Vrati sortirane članove S&P 500 na zadani datum (point-in-time)."""
    if membership is None:
        membership = _load_membership()
    date = pd.Timestamp(date)
    active = membership[
        (membership["start_date"] <= date)
        & (membership["end_date"].isna() | (membership["end_date"] >= date))
    ]
    return sorted(active["ticker"].unique())


def universe_for_window(window, membership: pd.DataFrame | None = None) -> list[str]:
    """Vrati point-in-time univerzum za prozor: članovi na ``window.train_end``.

    Prima ``RollingWindow`` (ili bilo što s atributom ``train_end``), odnosno
    izravno datum — bez uvoza ``src.backtest`` u ovaj modul.
    """
    date = getattr(window, "train_end", window)
    return membership_on(date, membership=membership)


def union_universe(
    start: str | pd.Timestamp = PROJECT_START,
    end: str | pd.Timestamp = PROJECT_END,
    membership: pd.DataFrame | None = None,
) -> list[str]:
    """Unija svih tickera koji su ikad bili članovi S&P 500 u [start, end] (F0.8).

    Ticker ulazi u uniju ako mu se bar jedan interval članstva preklapa s
    razdobljem projekta. Ovo je panel-univerzum nad kojim ``preprocess`` gradi
    prinose; point-in-time presjek po prozoru i dalje radi ``universe_for_window``.
    """
    if membership is None:
        membership = _load_membership()
    start_ts = _month_start(start)
    end_ts = _month_end(end)
    overlap = membership[
        (membership["start_date"] <= end_ts)
        & (membership["end_date"].isna() | (membership["end_date"] >= start_ts))
    ]
    return sorted(overlap["ticker"].unique())


def _membership_bounds(membership: pd.DataFrame) -> pd.DataFrame:
    """Po tickeru sažmi intervale članstva u (member_from, member_to).

    ``member_from`` = najraniji ulazak; ``member_to`` = najkasniji izlazak, ili
    ``NaT`` ako je bilo koji interval još otvoren (ticker je trenutačno član).
    """

    def _last_end(end_dates: pd.Series) -> pd.Timestamp:
        return pd.NaT if end_dates.isna().any() else end_dates.max()

    grouped = membership.groupby("ticker")
    return pd.DataFrame(
        {
            "member_from": grouped["start_date"].min(),
            "member_to": grouped["end_date"].apply(_last_end),
        }
    )


def membership_coverage_report(
    monthly_returns: pd.DataFrame,
    windows: Iterable | None = None,
    membership: pd.DataFrame | None = None,
    min_training_months: int = MIN_TRAINING_MONTHS,
    output_path: str | Path | None = TABLES_DIR / "00_membership_coverage.csv",
) -> pd.DataFrame:
    """Pokrivenost point-in-time univerzuma po prozoru treniranja (F0.7).

    Za svaki prozor računa: broj point-in-time članova na ``train_end``, koliko
    ih ima valjane cijene (bar jedan ne-NaN mjesečni prinos u prozoru
    treniranja), koliko prolazi filtar ≥ ``min_training_months`` valjanih
    mjeseci, te pripadne postotke. Po zaključanoj odluci 1 — nepoznata
    pristranost preživjelih postaje izmjerena veličina.

    Vraća DataFrame sa stupcima ``train_window, n_members, n_with_prices,
    n_with_60m, pct_prices, pct_60m``.
    """
    if membership is None:
        membership = _load_membership()
    if windows is None:
        from src.backtest import generate_rolling_windows

        # Dijagnostika pokrivenosti namjerno pokriva PUNO razdoblje projekta
        # (2000—2025, 21 prozor), ne podrezani backtest prozor — upravo gradijent
        # pokrivenosti u ranim godinama opravdava podrezivanje (PROJECT_SPEC §3.1).
        windows = generate_rolling_windows(PROJECT_START, PROJECT_END)

    records: list[dict] = []
    for window in windows:
        members = universe_for_window(window, membership)
        present = [ticker for ticker in members if ticker in monthly_returns.columns]
        train_panel = monthly_returns.loc[
            window.train_start : window.train_end, present
        ]
        valid_counts = train_panel.notna().sum(axis=0)
        n_members = len(members)
        n_with_prices = int((valid_counts > 0).sum())
        n_with_60m = int((valid_counts >= min_training_months).sum())
        records.append(
            {
                "train_window": window.label,
                "n_members": n_members,
                "n_with_prices": n_with_prices,
                "n_with_60m": n_with_60m,
                "pct_prices": round(100.0 * n_with_prices / max(n_members, 1), 2),
                "pct_60m": round(100.0 * n_with_60m / max(n_members, 1), 2),
            }
        )

    coverage = pd.DataFrame(
        records,
        columns=[
            "train_window",
            "n_members",
            "n_with_prices",
            "n_with_60m",
            "pct_prices",
            "pct_60m",
        ],
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(output_path, index=False)
    return coverage


# ---------------------------------------------------------------------------
# Obnovljiva predmemorija cijena po dionici
# ---------------------------------------------------------------------------


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("/", "_")
    return cache_dir / f"{safe}.parquet"


def _download_single_ticker(
    yahoo_ticker: str,
    start: str,
    end: str,
    max_retries: int = 3,
) -> pd.Series | None:
    """Preuzmi dnevnu prilagođenu zaključnu cijenu za jednu dionicu uz jednostavno ponavljanje."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            history = yf.download(
                yahoo_ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
        except Exception as error:  # mrežna/parserska greška — ponovi
            last_error = error
            time.sleep(1.0 + attempt)
            continue

        if history is None or history.empty:
            return None
        if isinstance(history.columns, pd.MultiIndex):
            if "Adj Close" not in history.columns.get_level_values(0):
                return None
            series = history["Adj Close"].iloc[:, 0]
        elif "Adj Close" in history.columns:
            series = history["Adj Close"]
        else:
            return None
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        series.name = "adj_close"
        return series.dropna()

    if last_error is not None:
        LOGGER.warning("Preuzimanje %s nije uspjelo nakon ponavljanja: %s", yahoo_ticker, last_error)
    return None


def _stooq_symbol(ticker: str) -> str:
    """Pretvori oznaku u Stooqov format za američke dionice (npr. BRK.B → brk-b.us)."""
    return ticker.replace(".", "-").replace("/", "-").lower() + ".us"


def _download_single_ticker_stooq(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 2,
) -> pd.Series | None:
    """Preuzmi dnevnu zaključnu cijenu sa Stooqa (sekundarni izvor, F0.6).

    Stooqove povijesne cijene su prilagođene za splitove i dividende, pa se
    stupac ``Close`` koristi kao ``adj_close``. Endpoint povremeno vraća
    anti-bot HTML stranicu umjesto CSV-a — takav odgovor tretira se kao
    neuspjeh (None), što ``download_prices_cached`` bilježi kao izvor ``none``.
    """
    params = {
        "s": _stooq_symbol(ticker),
        "d1": pd.Timestamp(start).strftime("%Y%m%d"),
        "d2": pd.Timestamp(end).strftime("%Y%m%d"),
        "i": "d",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                STOOQ_DAILY_URL,
                params=params,
                timeout=60,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
        except Exception as error:
            last_error = error
            time.sleep(1.0 + attempt)
            continue

        text = response.text
        if not text or text.lstrip().startswith("<") or not text.lstrip().startswith("Date"):
            return None
        try:
            table = pd.read_csv(StringIO(text))
        except Exception:
            return None
        if "Date" not in table.columns or "Close" not in table.columns or table.empty:
            return None
        series = pd.Series(
            pd.to_numeric(table["Close"], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(pd.to_datetime(table["Date"])).normalize(),
            name="adj_close",
        )
        return series.dropna().sort_index()

    if last_error is not None:
        LOGGER.warning("Stooq preuzimanje %s nije uspjelo: %s", ticker, last_error)
    return None


def _update_price_source_log(
    sources: dict[str, str],
    source_log_path: str | Path,
) -> None:
    """Upiši/ažuriraj zapis ``ticker, price_source`` po tickeru (F0.6)."""
    source_log_path = Path(source_log_path)
    existing: dict[str, str] = {}
    if source_log_path.exists():
        log = pd.read_csv(source_log_path)
        existing = dict(zip(log["ticker"].astype(str), log["price_source"].astype(str)))
    existing.update(sources)
    source_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        sorted(existing.items()), columns=["ticker", "price_source"]
    ).to_csv(source_log_path, index=False)


def download_prices_cached(
    tickers: Iterable[str],
    start: str | pd.Timestamp = PROJECT_START,
    end: str | pd.Timestamp = PROJECT_END,
    cache_dir: str | Path = PRICE_CACHE_DIR,
    force: bool = False,
    max_workers: int = 8,
    source_log_path: str | Path = PRICE_SOURCE_LOG_PATH,
    overrides_path: str | Path = TICKER_OVERRIDES_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Preuzmi dnevnu prilagođenu zaključnu cijenu po dionici u obnovljivu parquet predmemoriju.

    Zaseban parquet po dionici, ključan po simbolu, drži puni dnevni niz za
    razdoblje projekta. Postojeće predmemorije ponovno se koriste osim ako je ``force=True``. Promašaji
    se preuzimaju paralelno putem ``max_workers`` dretvi. Ako yfinance ne vrati
    ništa, pokušava se Stooq kao sekundarni izvor (F0.6); izvor po tickeru
    (``yahoo``/``stooq``/``none``) bilježi se u ``source_log_path``. Tickeri u
    predmemoriji bez zapisa o izvoru naknadno se vode kao ``yahoo`` (svi su
    povijesno preuzeti yfinanceom). Simbol za preuzimanje može se premostiti
    stupcem ``download_ticker`` u ``ticker_overrides.csv``. Oznake čije
    preuzimanje ne uspije bilježe se i isključuju iz vraćenog DataFramea.
    Vraća ``(daily_prices, failed_tickers)``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    project_tickers = list(dict.fromkeys(str(ticker).strip() for ticker in tickers))
    # Dohvati od jedan mjesec prije ``start`` kako bi prvi mjesec prozora
    # projekta imao nenedostajući jednostavni prinos nakon ``pct_change``.
    download_start = _month_start(start) - pd.DateOffset(months=1)
    start_str = download_start.strftime("%Y-%m-%d")
    end_str = (_month_end(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    overrides = load_ticker_overrides(overrides_path)
    download_map = {
        row["ticker"]: row["download_ticker"]
        for _, row in overrides.iterrows()
        if row["ticker"] and row["download_ticker"]
    }

    known_sources: dict[str, str] = {}
    source_log_path = Path(source_log_path)
    if source_log_path.exists():
        log = pd.read_csv(source_log_path)
        known_sources = dict(zip(log["ticker"].astype(str), log["price_source"].astype(str)))

    series_by_ticker: dict[str, pd.Series] = {}
    failed: list[str] = []
    misses: list[str] = []
    sources: dict[str, str] = {}

    for ticker in project_tickers:
        cache_path = _cache_path(ticker, cache_dir)
        if cache_path.exists() and not force:
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    series_by_ticker[ticker] = cached["adj_close"]
                    sources[ticker] = known_sources.get(ticker, "yahoo")
                    continue
            except Exception as error:
                LOGGER.warning("Odbacujem nečitljivu predmemoriju za %s: %s", ticker, error)
                cache_path.unlink(missing_ok=True)
        misses.append(ticker)

    def worker(ticker: str) -> tuple[str, pd.Series | None, str]:
        download_ticker = download_map.get(ticker, ticker)
        series = _download_single_ticker(_yahoo_ticker(download_ticker), start_str, end_str)
        if series is not None and not series.empty:
            return ticker, series, "yahoo"
        series = _download_single_ticker_stooq(download_ticker, start_str, end_str)
        if series is not None and not series.empty:
            return ticker, series, "stooq"
        return ticker, None, "none"

    if misses:
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker, ticker) for ticker in misses]
            for future in as_completed(futures):
                ticker, series, source = future.result()
                completed_count += 1
                sources[ticker] = source
                if series is None or series.empty:
                    failed.append(ticker)
                else:
                    series.to_frame("adj_close").to_parquet(_cache_path(ticker, cache_dir))
                    series_by_ticker[ticker] = series
                if completed_count % 50 == 0 or completed_count == len(misses):
                    LOGGER.info(
                        "Preuzeto %d/%d novih oznaka (%d neuspjelih dosad)",
                        completed_count,
                        len(misses),
                        len(failed),
                    )

    _update_price_source_log(sources, source_log_path)

    if not series_by_ticker:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date")), failed

    daily_prices = pd.concat(series_by_ticker, axis=1)
    daily_prices.columns.name = None
    daily_prices = daily_prices.sort_index()
    daily_prices.index.name = "date"
    return daily_prices, failed


def write_price_source_summary(
    tickers: Iterable[str] | None = None,
    source_log_path: str | Path = PRICE_SOURCE_LOG_PATH,
    output_path: str | Path = TABLES_DIR / "00_price_source_summary.csv",
) -> pd.DataFrame:
    """Agregiraj zapis izvora cijena u tablicu ``source, n_tickers, pct`` (F0.6).

    ``tickers`` ograničava sažetak na zadani univerzum (ticker bez zapisa u
    logu broji se kao ``none``); bez argumenta sažima cijeli log.
    """
    source_log_path = Path(source_log_path)
    known_sources: dict[str, str] = {}
    if source_log_path.exists():
        log = pd.read_csv(source_log_path)
        known_sources = dict(zip(log["ticker"].astype(str), log["price_source"].astype(str)))

    if tickers is None:
        universe = sorted(known_sources)
    else:
        universe = list(dict.fromkeys(str(ticker).strip() for ticker in tickers))

    per_ticker = pd.Series(
        {ticker: known_sources.get(ticker, "none") for ticker in universe},
        name="price_source",
    )
    summary = (
        per_ticker.value_counts()
        .rename_axis("source")
        .reset_index(name="n_tickers")
        .sort_values("source")
        .reset_index(drop=True)
    )
    summary["pct"] = (100.0 * summary["n_tickers"] / max(len(universe), 1)).round(2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def daily_to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Preuzorkuj dnevne prilagođene zaključne cijene na kraj mjeseca i izračunaj jednostavne prinose."""
    if daily_prices.empty:
        return daily_prices
    monthly = daily_prices.resample("ME").last()
    monthly.index = monthly.index.normalize()
    returns = monthly.pct_change()
    returns.index.name = "date"
    return returns


def clip_implausible_returns(
    returns: pd.DataFrame,
    low: float = MONTHLY_RETURN_CLIP_LOW,
    high: float = MONTHLY_RETURN_CLIP_HIGH,
) -> pd.DataFrame:
    """Postavi na NaN mjesečne prinose izvan ``[low, high]`` (sanitacija podataka).

    Mjesečni prinos člana S&P 500 izvan ``[low, high]`` (default
    ``[-0.90, 3.0]`` iz ``config.yaml``) fizički je nemoguć i potječe iz
    podatkovne greške: neadekvatno premošten split, jednodnevni pogrešan ispis
    cijene (uzorak „skok pa povrat”) ili ponovo iskorišten ticker. Takve
    vrijednosti postaju ``NaN`` (imovina ne doprinosi tom mjesecu) umjesto da
    se ukliještenjem ubaci lažan prinos. Granica je konzervativna: stvarni
    ekstremi unutar pojasa (npr. AIG +245 % u 2009-08, kratki skvíz) ostaju.

    Vraća kopiju; postojeći ``NaN`` ostaju ``NaN``.
    """
    if returns.empty:
        return returns
    return returns.where((returns >= low) & (returns <= high))


# ---------------------------------------------------------------------------
# Fama-French petfaktorski podaci
# ---------------------------------------------------------------------------


def download_ff5_factors(
    start: str | pd.Timestamp = PROJECT_START,
    end: str | pd.Timestamp = PROJECT_END,
    url: str = FF5_MONTHLY_URL,
) -> pd.DataFrame:
    """Preuzmi i parsiraj Fama-French petfaktorske mjesečne podatke (decimalni prinosi)."""
    response = requests.get(url, timeout=60, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Fama-French zip arhiva ne sadrži CSV datoteku.")
        raw_text = archive.read(csv_names[0]).decode("utf-8", errors="replace")

    lines = raw_text.splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith(",Mkt-RF,SMB,HML,RMW,CMA,RF")
    )
    data_lines: list[str] = []
    for line in lines[header_index + 1 :]:
        first_value = line.split(",", maxsplit=1)[0].strip()
        if not (first_value.isdigit() and len(first_value) == 6):
            break
        data_lines.append(line)

    factors = pd.read_csv(
        BytesIO(("\n".join([lines[header_index], *data_lines])).encode("utf-8")),
        index_col=0,
    )
    factors.index = (
        pd.PeriodIndex(factors.index.astype(str), freq="M")
        .to_timestamp(how="end")
        .normalize()
    )
    factors = factors.rename_axis("date")
    factors = factors[FACTOR_COLUMNS].apply(pd.to_numeric, errors="raise") / 100.0
    return factors.loc[_month_end(start) : _month_end(end)]


# ---------------------------------------------------------------------------
# Orkestrator
# ---------------------------------------------------------------------------


def preprocess(
    start: str | pd.Timestamp = PROJECT_START,
    end: str | pd.Timestamp = PROJECT_END,
    raw_dir: str | Path = RAW_DATA_DIR,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    cache_dir: str | Path = PRICE_CACHE_DIR,
    force_universe: bool = False,
    force_prices: bool = False,
) -> dict[str, pd.DataFrame]:
    """Pokreni pretprocesiranje i zapiši CSV izlaze projekta (F0.8).

    Univerzum je unija svih tickera koji su ikad bili članovi point-in-time
    S&P 500 u razdoblju projekta (``union_universe`` nad tablicom članstva iz
    F0.5), a ne više trenutačni Russell 1000. Preuzima dnevne cijene po dionici
    (obnovljiva parquet predmemorija, uz Stooq dopunu iz F0.6), preuzorkuje na
    mjesečno, poravnava s Fama-French faktorima i zapisuje obrađene CSV-ove
    (``monthly_returns``, ``excess_returns``, ``factors``, ``metadata``) te
    tablicu ``membership``. ``metadata`` dobiva stupce ``price_source,
    member_from, member_to``.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    cache_dir = Path(cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    membership_path = raw_dir / SP500_MEMBERSHIP_PATH.name
    if force_universe or not membership_path.exists():
        fetch_sp500_membership(output_path=membership_path, raw_dir=raw_dir)
    membership = pd.read_csv(
        membership_path, parse_dates=["start_date", "end_date"]
    )
    union = union_universe(start, end, membership)

    # Ime iz tablice članstva; GICS sektor iz postojećih slojeva gdje postoji
    # (stari Russell popis / prethodni metadata), inače "Unknown".
    name_map = {
        str(ticker).strip(): str(name).strip()
        for ticker, name in zip(membership["ticker"], membership["name"].fillna(""))
    }
    sector_map: dict[str, str] = {}
    russell_path = raw_dir / "russell1000_tickers.csv"
    if russell_path.exists():
        russell = pd.read_csv(russell_path)
        sector_map.update(
            dict(
                zip(
                    russell["ticker"].astype(str).str.strip(),
                    russell["sector"].astype(str).str.strip(),
                )
            )
        )
    old_metadata_path = processed_dir / "metadata.csv"
    if old_metadata_path.exists():
        old_metadata = pd.read_csv(old_metadata_path)
        if "sector" in old_metadata.columns:
            sector_map.update(
                dict(
                    zip(
                        old_metadata["ticker"].astype(str).str.strip(),
                        old_metadata["sector"].astype(str).str.strip(),
                    )
                )
            )
    universe = pd.DataFrame({"ticker": union})
    universe["name"] = universe["ticker"].map(name_map).fillna("")
    universe["sector"] = (
        universe["ticker"].map(sector_map).fillna("Unknown").replace("", "Unknown")
    )

    daily_prices, failed = download_prices_cached(
        universe["ticker"].tolist(),
        start=start,
        end=end,
        cache_dir=cache_dir,
        force=force_prices,
    )
    if failed:
        LOGGER.warning(
            "%d oznaka nije uspjelo preuzeti: %s",
            len(failed),
            ", ".join(failed[:20]) + ("..." if len(failed) > 20 else ""),
        )

    monthly_returns = daily_to_monthly_returns(daily_prices)
    monthly_returns = monthly_returns.loc[_month_end(start) : _month_end(end)]

    factors = download_ff5_factors(start=start, end=end)

    common_index = monthly_returns.index.intersection(factors.index)
    monthly_returns = monthly_returns.loc[common_index].sort_index()
    factors = factors.loc[common_index].sort_index()

    # Sanitacija: fizički nemogući mjesečni prinosi (split/ponovo iskorišten
    # ticker/pogrešan ispis cijene) postaju NaN prije izvedenih veličina.
    monthly_returns = clip_implausible_returns(monthly_returns)

    # Izbaci oznake koje su potpuno NaN kroz prozor projekta (preuzimanje nije uspjelo
    # usred niza, oznaka je prenova itd.).
    column_has_data = monthly_returns.notna().any(axis=0)
    usable_tickers = column_has_data[column_has_data].index.tolist()
    monthly_returns = monthly_returns[usable_tickers]
    excess_returns = monthly_returns.sub(factors["RF"], axis=0)

    n_obs_per_ticker = monthly_returns.notna().sum(axis=0)
    first_date_per_ticker = (
        monthly_returns.apply(lambda s: s.first_valid_index())
        .rename("first_valid_month")
    )
    last_date_per_ticker = (
        monthly_returns.apply(lambda s: s.last_valid_index())
        .rename("last_valid_month")
    )

    bounds = _membership_bounds(membership)
    source_log: dict[str, str] = {}
    if PRICE_SOURCE_LOG_PATH.exists():
        log = pd.read_csv(PRICE_SOURCE_LOG_PATH)
        source_log = dict(
            zip(log["ticker"].astype(str), log["price_source"].astype(str))
        )

    failed_set = set(failed)
    metadata_out = (
        universe.assign(
            yahoo_ticker=lambda frame: frame["ticker"].map(_yahoo_ticker),
            price_source=lambda frame: frame["ticker"].map(source_log).fillna("none"),
            member_from=lambda frame: frame["ticker"].map(bounds["member_from"]),
            member_to=lambda frame: frame["ticker"].map(bounds["member_to"]),
            download_failed=lambda frame: frame["ticker"].isin(failed_set),
            n_valid_months=lambda frame: frame["ticker"]
            .map(n_obs_per_ticker)
            .fillna(0)
            .astype(int),
            first_valid_month=lambda frame: frame["ticker"].map(first_date_per_ticker),
            last_valid_month=lambda frame: frame["ticker"].map(last_date_per_ticker),
        )
        .assign(
            retained=lambda frame: ~frame["download_failed"]
            & frame["ticker"].isin(monthly_returns.columns),
            drop_reason=lambda frame: np.where(
                frame["download_failed"],
                "price_download_failed",
                np.where(
                    frame["ticker"].isin(monthly_returns.columns),
                    "",
                    "no_monthly_observations",
                ),
            ),
        )[
            [
                "ticker",
                "name",
                "sector",
                "yahoo_ticker",
                "price_source",
                "member_from",
                "member_to",
                "retained",
                "n_valid_months",
                "first_valid_month",
                "last_valid_month",
                "drop_reason",
            ]
        ]
    )

    membership_out = (
        membership[membership["ticker"].isin(union)]
        .sort_values(["ticker", "start_date"])
        .reset_index(drop=True)
    )

    monthly_returns.to_csv(processed_dir / "monthly_returns.csv", index_label="date")
    excess_returns.to_csv(processed_dir / "excess_returns.csv", index_label="date")
    factors.to_csv(processed_dir / "factors.csv", index_label="date")
    metadata_out.to_csv(processed_dir / "metadata.csv", index=False)
    membership_out.to_csv(processed_dir / "membership.csv", index=False)

    return {
        "monthly_returns": monthly_returns,
        "excess_returns": excess_returns,
        "factors": factors,
        "metadata": metadata_out,
        "membership": membership_out,
        "failed_downloads": failed,
    }

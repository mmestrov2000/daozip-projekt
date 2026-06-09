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
    PRICE_CACHE_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_END,
    PROJECT_START,
    RAW_DATA_DIR,
)


LOGGER = logging.getLogger(__name__)

IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
RUSSELL_1000_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
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


def download_prices_cached(
    tickers: Iterable[str],
    start: str | pd.Timestamp = PROJECT_START,
    end: str | pd.Timestamp = PROJECT_END,
    cache_dir: str | Path = PRICE_CACHE_DIR,
    force: bool = False,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """Preuzmi dnevnu prilagođenu zaključnu cijenu po dionici u obnovljivu parquet predmemoriju.

    Zaseban parquet po dionici, ključan po simbolu, drži puni dnevni niz za
    razdoblje projekta. Postojeće predmemorije ponovno se koriste osim ako je ``force=True``. Promašaji
    se preuzimaju paralelno putem ``max_workers`` dretvi. Oznake čije
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

    series_by_ticker: dict[str, pd.Series] = {}
    failed: list[str] = []
    misses: list[str] = []

    for ticker in project_tickers:
        cache_path = _cache_path(ticker, cache_dir)
        if cache_path.exists() and not force:
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    series_by_ticker[ticker] = cached["adj_close"]
                    continue
            except Exception as error:
                LOGGER.warning("Odbacujem nečitljivu predmemoriju za %s: %s", ticker, error)
                cache_path.unlink(missing_ok=True)
        misses.append(ticker)

    def worker(ticker: str) -> tuple[str, pd.Series | None]:
        return ticker, _download_single_ticker(_yahoo_ticker(ticker), start_str, end_str)

    if misses:
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker, ticker) for ticker in misses]
            for future in as_completed(futures):
                ticker, series = future.result()
                completed_count += 1
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

    if not series_by_ticker:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date")), failed

    daily_prices = pd.concat(series_by_ticker, axis=1)
    daily_prices.columns.name = None
    daily_prices = daily_prices.sort_index()
    daily_prices.index.name = "date"
    return daily_prices, failed


def daily_to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Preuzorkuj dnevne prilagođene zaključne cijene na kraj mjeseca i izračunaj jednostavne prinose."""
    if daily_prices.empty:
        return daily_prices
    monthly = daily_prices.resample("ME").last()
    monthly.index = monthly.index.normalize()
    returns = monthly.pct_change()
    returns.index.name = "date"
    return returns


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
    """Pokreni pretprocesiranje faze 1 i zapiši CSV izlaze projekta.

    Dohvaća Russell 1000 s iSharesa (predmemorirano na disku), preuzima dnevne
    cijene po dionici (obnovljiva parquet predmemorija), preuzorkuje na mjesečno,
    poravnava s Fama-French faktorima i zapisuje obrađene CSV-ove.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    cache_dir = Path(cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers_path = raw_dir / "russell1000_tickers.csv"
    if force_universe or not tickers_path.exists():
        universe = fetch_russell1000_tickers(output_path=tickers_path, raw_dir=raw_dir)
    else:
        universe = pd.read_csv(tickers_path)

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

    failed_set = set(failed)
    metadata_out = (
        universe.assign(
            yahoo_ticker=lambda frame: frame["ticker"].map(_yahoo_ticker),
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
                "yfinance_failed",
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
                "retained",
                "n_valid_months",
                "first_valid_month",
                "last_valid_month",
                "drop_reason",
            ]
        ]
    )

    monthly_returns.to_csv(processed_dir / "monthly_returns.csv", index_label="date")
    excess_returns.to_csv(processed_dir / "excess_returns.csv", index_label="date")
    factors.to_csv(processed_dir / "factors.csv", index_label="date")
    metadata_out.to_csv(processed_dir / "metadata.csv", index=False)

    return {
        "monthly_returns": monthly_returns,
        "excess_returns": excess_returns,
        "factors": factors,
        "metadata": metadata_out,
        "failed_downloads": failed,
    }

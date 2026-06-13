"""Testovi Stooq fallbacka i zapisa izvora cijena (F0.6) — mockani odgovori."""

import pandas as pd
import pytest

import numpy as np

import src.data as data_module
from src.data import (
    _download_single_ticker_stooq,
    clip_implausible_returns,
    download_prices_cached,
    write_price_source_summary,
)


def _fake_series() -> pd.Series:
    index = pd.date_range("2007-01-02", periods=5, freq="B")
    return pd.Series([10.0, 10.5, 10.2, 10.8, 11.0], index=index, name="adj_close")


def _read_log(path) -> dict[str, str]:
    log = pd.read_csv(path)
    return dict(zip(log["ticker"], log["price_source"]))


def test_clip_implausible_returns_nans_only_out_of_band():
    """Prinosi izvan [-90%, +300%] -> NaN; plauzibilni ekstremi i NaN ostaju."""
    index = pd.period_range("2010-01", periods=4, freq="M").to_timestamp(how="end")
    returns = pd.DataFrame(
        {
            "GLITCH": [1778.88, -0.999, 0.05, np.nan],  # skok pa povrat -> oba van pojasa
            "REAL": [2.45, -0.50, 0.10, 0.02],  # AIG-tip ekstrem unutar pojasa
        },
        index=index,
    )

    clipped = clip_implausible_returns(returns, low=-0.90, high=3.0)

    assert np.isnan(clipped.loc[index[0], "GLITCH"])  # +177888%
    assert np.isnan(clipped.loc[index[1], "GLITCH"])  # -99.9% povrat
    assert clipped.loc[index[2], "GLITCH"] == pytest.approx(0.05)
    assert clipped["REAL"].tolist() == pytest.approx([2.45, -0.50, 0.10, 0.02])
    # postojeći NaN ostaje NaN, ne diže iznimku
    assert np.isnan(clipped.loc[index[3], "GLITCH"])


def test_stooq_fallback_activates_when_yahoo_empty(monkeypatch, tmp_path):
    """Yahoo ne vraća ništa → Stooq fallback puni predmemoriju i bilježi izvor."""
    monkeypatch.setattr(data_module, "_download_single_ticker", lambda *a, **k: None)
    monkeypatch.setattr(
        data_module, "_download_single_ticker_stooq", lambda *a, **k: _fake_series()
    )
    log_path = tmp_path / "price_source.csv"

    prices, failed = download_prices_cached(
        ["LEH"],
        start="2007-01",
        end="2007-12",
        cache_dir=tmp_path / "cache",
        source_log_path=log_path,
    )

    assert failed == []
    assert "LEH" in prices.columns
    assert (tmp_path / "cache" / "LEH.parquet").exists()
    assert _read_log(log_path)["LEH"] == "stooq"


def test_source_none_when_both_fail(monkeypatch, tmp_path):
    """Oba izvora prazna → ticker u failed, izvor zabilježen kao none."""
    monkeypatch.setattr(data_module, "_download_single_ticker", lambda *a, **k: None)
    monkeypatch.setattr(
        data_module, "_download_single_ticker_stooq", lambda *a, **k: None
    )
    log_path = tmp_path / "price_source.csv"

    prices, failed = download_prices_cached(
        ["XXXX"],
        start="2007-01",
        end="2007-12",
        cache_dir=tmp_path / "cache",
        source_log_path=log_path,
    )

    assert failed == ["XXXX"]
    assert prices.empty
    assert _read_log(log_path)["XXXX"] == "none"


def test_yahoo_success_skips_stooq(monkeypatch, tmp_path):
    """Yahoo uspijeva → izvor yahoo, Stooq se uopće ne zove."""
    monkeypatch.setattr(
        data_module, "_download_single_ticker", lambda *a, **k: _fake_series()
    )

    def _fail(*args, **kwargs):
        raise AssertionError("Stooq se ne smije zvati kad yahoo uspije.")

    monkeypatch.setattr(data_module, "_download_single_ticker_stooq", _fail)
    log_path = tmp_path / "price_source.csv"

    prices, failed = download_prices_cached(
        ["AAPL"],
        start="2007-01",
        end="2007-12",
        cache_dir=tmp_path / "cache",
        source_log_path=log_path,
    )

    assert failed == []
    assert "AAPL" in prices.columns
    assert _read_log(log_path)["AAPL"] == "yahoo"


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_stooq_parser_on_csv(monkeypatch):
    """Kanned Stooq CSV → ispravna serija prilagođenih zaključnih cijena."""
    csv_text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2007-01-03,10.0,10.6,9.9,10.5,1000\n"
        "2007-01-04,10.5,10.9,10.4,10.8,1100\n"
    )
    monkeypatch.setattr(
        data_module.requests, "get", lambda *a, **k: _FakeResponse(csv_text)
    )
    series = _download_single_ticker_stooq("LEH", "2007-01-01", "2007-12-31")
    assert series is not None
    assert series.name == "adj_close"
    assert series.tolist() == [10.5, 10.8]
    assert series.index[0] == pd.Timestamp("2007-01-03")


def test_stooq_parser_rejects_html(monkeypatch):
    """Anti-bot HTML stranica umjesto CSV-a → None (izvor none, bez iznimke)."""
    monkeypatch.setattr(
        data_module.requests,
        "get",
        lambda *a, **k: _FakeResponse("<!DOCTYPE html><html>verify</html>"),
    )
    assert _download_single_ticker_stooq("LEH", "2007-01-01", "2007-12-31") is None


def test_price_source_summary_columns(tmp_path):
    """Sažetak izvora ima stupce source, n_tickers, pct i točne udjele."""
    log_path = tmp_path / "price_source.csv"
    pd.DataFrame(
        {"ticker": ["A", "B", "C", "D"], "price_source": ["yahoo", "yahoo", "stooq", "none"]}
    ).to_csv(log_path, index=False)
    output_path = tmp_path / "00_price_source_summary.csv"

    summary = write_price_source_summary(
        ["A", "B", "C", "D", "E"], source_log_path=log_path, output_path=output_path
    )

    assert output_path.exists()
    assert list(summary.columns) == ["source", "n_tickers", "pct"]
    by_source = summary.set_index("source")
    assert by_source.loc["yahoo", "n_tickers"] == 2
    assert by_source.loc["stooq", "n_tickers"] == 1
    # E nema zapisa u logu → broji se kao none.
    assert by_source.loc["none", "n_tickers"] == 2
    assert pytest.approx(summary["pct"].sum(), abs=0.05) == 100.0

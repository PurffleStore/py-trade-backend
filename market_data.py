"""market_data.py

yfinance-only utilities for fetching OHLCV data used by TA and AI modules.
"""
from typing import Optional
import pandas as pd
import time

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance optional
    yf = None

_cache = {}


def download(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV for `symbol` using yfinance.

    Returns a DataFrame with columns: ['Open','High','Low','Close','AdjClose','Volume'] and a DatetimeIndex.
    """
    if yf is None:
        # Avoid non-ASCII characters in error message which can trigger UnicodeDecodeError
        raise RuntimeError("yfinance is required for market_data.download - install with `pip install yfinance`")

    key = (symbol, period, interval)
    # simple in-memory cache with ttl of 60 seconds to avoid rapid repeated calls during development
    entry = _cache.get(key)
    if entry and (time.time() - entry[0] < 60):
        return entry[1].copy()

    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No market data returned for {symbol} (period={period}, interval={interval})")

    # standardize column names
    df = df.rename(columns={
        'Adj Close': 'AdjClose',
        'Close': 'Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Volume': 'Volume'
    })

    expected = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[expected].copy()
    _cache[key] = (time.time(), out.copy())
    return out


def get_ohlcv(symbol: str, days: int = 180) -> pd.DataFrame:
    """Helper to retrieve approximately `days` of daily OHLCV data.

    This function chooses an appropriate `period` for yfinance based on `days`.
    """
    # choose nearest period label supported by yfinance
    if days <= 7:
        period = '7d'
    elif days <= 30:
        period = '1mo'
    elif days <= 90:
        period = '3mo'
    elif days <= 180:
        period = '6mo'
    elif days <= 365:
        period = '1y'
    else:
        period = '2y'

    return download(symbol, period=period, interval='1d')

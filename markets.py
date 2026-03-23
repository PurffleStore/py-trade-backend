# markets.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Any

from flask import Blueprint, jsonify, request
import yfinance as yf

# All endpoints will start with /api/markets/...
markets_bp = Blueprint("markets", __name__, url_prefix="/api/markets")

# -------------------------------------------
# Index configuration (you can add more)
# -------------------------------------------
INDEX_TICKERS: Dict[str, Dict[str, str]] = {
    "NIFTY_50": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
    },
    "BANK_NIFTY": {
        "name": "BANK NIFTY",
        "symbol": "^NSEBANK",
    },
    "SENSEX": {
        "name": "SENSEX",
        "symbol": "^BSESN",
    },
    "NIFTY_MIDCAP": {
        "name": "NIFTY Midcap 100",
        "symbol": "^CRSMID",
    },
    "NIFTY_SMALLCAP": {
        "name": "NIFTY Smallcap 100",
        "symbol": "^CNXSC",
    },
}

# Example company lists for demo.
# Replace / expand with your real lists when you are ready.
INDEX_COMPANIES: Dict[str, List[str]] = {
    # NIFTY 50 sample
    "NIFTY_50": [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
    ],
    # Smallcap examples from your screenshot
    "NIFTY_SMALLCAP": [
        "TANLA.NS",
        "MAPMYINDIA.NS",
        "KEI.NS",
        "PNCINFRA.NS",
        "TARC.NS",
    ],
    # Fill others later if you like
    "BANK_NIFTY": [],
    "SENSEX": [],
    "NIFTY_MIDCAP": [],
}


# -------------------------------------------
# Helpers
# -------------------------------------------
def _index_snapshot(symbol: str) -> Dict[str, Any] | None:
    """
    Return last price, change, % change and sparkline (list of prices)
    for a given index symbol.
    """
    # Try intraday first
    hist = yf.download(
        symbol,
        period="1d",
        interval="5m",
        progress=False,
    )

    if hist.empty:
        # Fallback to last few daily candles
        hist = yf.download(
            symbol,
            period="5d",
            interval="1d",
            progress=False,
        )

    if hist.empty:
        return None

    first_row = hist.iloc[0]
    last_row = hist.iloc[-1]

    open_price = float(first_row["Open"])
    last_price = float(last_row["Close"])

    change = last_price - open_price if open_price else 0.0
    pct_change = (change / open_price * 100.0) if open_price else 0.0

    # Use last 30 closes for sparkline
    closes = hist["Close"].tail(30).tolist()
    closes = [float(x) for x in closes]

    return {
        "last_price": round(last_price, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
        "sparkline": closes,
    }


def _stock_snapshot(symbol: str) -> Dict[str, Any] | None:
    """
    Basic snapshot for a single stock.
    """
    ticker = yf.Ticker(symbol)

    # 1 day daily data to compute today move
    hist = ticker.history(period="1d", interval="1d")
    if hist.empty:
        return None

    row = hist.iloc[-1]
    open_price = float(row["Open"])
    last_price = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])

    change = last_price - open_price if open_price else 0.0
    pct_change = (change / open_price * 100.0) if open_price else 0.0

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        # Avoid breaking if Yahoo blocks info() sometimes
        info = {}

    name = info.get("shortName") or symbol

    return {
        "name": name,
        "symbol": symbol,
        "ltp": round(last_price, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "pct_change": round(pct_change, 2),
    }


# -------------------------------------------
# Routes
# -------------------------------------------

@markets_bp.route("/indices", methods=["GET"])
def get_indices():
    """
    Return live snapshot for all configured indices.

    Response:
    {
      "indices": [
        {
          "code": "NIFTY_50",
          "name": "NIFTY 50",
          "symbol": "^NSEI",
          "last_price": ...,
          "change": ...,
          "pct_change": ...,
          "sparkline": [ ... floats ... ]
        },
        ...
      ]
    }
    """
    result = []

    for code, meta in INDEX_TICKERS.items():
        snap = _index_snapshot(meta["symbol"])
        if not snap:
            continue

        result.append(
            {
                "code": code,
                "name": meta["name"],
                "symbol": meta["symbol"],
                **snap,
            }
        )

    return jsonify({"indices": result})


@markets_bp.route("/index-companies", methods=["GET"])
def get_index_companies():
    """
    Return company table for a selected index.

    Query: ?index=NIFTY_50  (default NIFTY_50)

    Response:
    {
      "index": "NIFTY_50",
      "companies": [
        {
          "name": "...",
          "symbol": "RELIANCE.NS",
          "ltp": ...,
          "high": ...,
          "low": ...,
          "pct_change": ...
        },
        ...
      ]
    }
    """
    index_code = request.args.get("index", "NIFTY_50")
    tickers = INDEX_COMPANIES.get(index_code, [])

    companies = []
    for symbol in tickers:
        snap = _stock_snapshot(symbol)
        if snap:
            companies.append(snap)

    return jsonify(
        {
            "index": index_code,
            "companies": companies,
        }
    )


@markets_bp.route("/stock-intraday", methods=["GET"])
def get_stock_intraday():
    """
    Return intraday series for the right-side chart.

    Query:
      ?symbol=RELIANCE.NS
      &interval=15m  (optional, default 15m)

    Response:
    {
      "symbol": "RELIANCE.NS",
      "points": [
        {"ts": "2025-11-20T10:15:00+05:30", "close": 2450.2},
        ...
      ]
    }
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    interval = request.args.get("interval", "15m")

    hist = yf.download(
        symbol,
        period="5d",
        interval=interval,
        progress=False,
    )

    if hist.empty:
        return jsonify({"symbol": symbol, "points": []})

    points = []
    # hist index is DatetimeIndex
    for idx, row in hist.iterrows():
        ts = idx.isoformat()
        points.append(
            {
                "ts": ts,
                "close": float(row["Close"]),
            }
        )

    return jsonify({"symbol": symbol, "points": points})

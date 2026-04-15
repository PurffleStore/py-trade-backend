# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, io, json, time, os
from typing import Dict, List, Any, Optional
from pathlib import Path
from io import StringIO

import requests

# optional (for Wikipedia tables)
try:
    import pandas as pd  # requires: pip install pandas lxml
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ---------- configuration (unchanged names) ----------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127 Safari/537.36"
REFERER = "https://www.niftyindices.com/indices/equity/broad-based-indices"
TTL_SECONDS = 60 * 60 * 12  # 12h cache
DEFAULT_CACHE_DIR = os.getenv("CACHE_DIR", "/data/cache")
CACHE_DIR = Path(DEFAULT_CACHE_DIR if DEFAULT_CACHE_DIR else ".").expanduser()
if CACHE_DIR == Path("."):
    CACHE_DIR = Path(__file__).with_name("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Official CSV endpoints for NSE indices (unchanged name)
NIFTY_URLS: Dict[str, str] = {
    "NIFTY50":      "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY100":     "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY200":     "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "NIFTYMID100":  "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "NIFTY500":     "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

# Filters payload for the UI (unchanged variable name)
MARKETS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "India": {
        "NSE (National Stock Exchange)": [
            {"code": "NIFTY50",     "name": "NIFTY 50"},
            {"code": "NIFTY100",    "name": "NIFTY 100"},
            {"code": "NIFTY200",    "name": "NIFTY 200"},
            {"code": "NIFTYMID100", "name": "NIFTY Midcap 100"},
            {"code": "NIFTY500",    "name": "NIFTY 500"},
        ]
    }
}

# ---------- extras (new, additive) ----------
WIKI_PAGES: Dict[str, str] = {
    "NASDAQ100": "https://en.wikipedia.org/wiki/NASDAQ-100",
    "DAX40":     "https://en.wikipedia.org/wiki/DAX",
    "OMXS30":    "https://en.wikipedia.org/wiki/OMX_Stockholm_30",
}

EXTRA_MARKETS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "United States": {
        "NASDAQ": [
            {"code": "NASDAQ100", "name": "NASDAQ-100"}
        ]
    },
    "Germany": {
        "XETRA (Deutsche Börse)": [
            {"code": "DAX40", "name": "DAX 40"}
        ]
    },
    "Sweden": {
        "OMX Stockholm": [
            {"code": "OMXS30", "name": "OMX Stockholm 30"}
        ]
    }
}

# ---------- utilities (kept original names) ----------
def http_get_text(url: str, accept: str = "text/csv,*/*") -> str:
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Referer": REFERER, "Accept": accept})
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text

def parse_nifty_csv(text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    rdr = csv.DictReader(io.StringIO(text))
    for row in rdr:
        sym = (row.get("Symbol") or "").strip()
        name = (row.get("Company Name") or "").strip()
        if sym and name:
            out.append({"symbol": f"{sym}.NS", "company": name})
    return out

def cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code.lower()}.json"

def load_cache(code: str) -> Any | None:
    fp = cache_path(code)
    if not fp.exists():
        return None
    age = time.time() - fp.stat().st_mtime
    if age > TTL_SECONDS:
        return None
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(code: str, payload: Any) -> None:
    fp = cache_path(code)
    with fp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---- Wikipedia helpers ----
def _fetch_wiki_tables(url: str):
    if not HAS_PANDAS:
        raise RuntimeError("pandas/lxml not installed. Run: pip install pandas lxml")
    html = http_get_text(url, accept="text/html,*/*")
    return pd.read_html(StringIO(html))

def _pick_table_and_columns(tables, ticker_candidates, company_candidates):
    for t in tables:
        cols_map = {str(c).strip().lower(): c for c in t.columns}
        ticker_col = next((cols_map[c] for c in ticker_candidates if c in cols_map), None)
        company_col = next((cols_map[c] for c in company_candidates if c in cols_map), None)
        if ticker_col is not None and company_col is not None:
            return t, ticker_col, company_col
    raise RuntimeError(
        f"No suitable table found. Ticker in {ticker_candidates}, company in {company_candidates}."
    )

def _parse_wiki_constituents(url: str, ticker_candidates, company_candidates, suffix: str, upper_tickers: bool) -> List[Dict[str, str]]:
    tables = _fetch_wiki_tables(url)
    df, t_col, c_col = _pick_table_and_columns(tables, ticker_candidates, company_candidates)
    rows: List[Dict[str, str]] = []
    for sym, name in zip(df[t_col], df[c_col]):
        s = str(sym).strip()
        n = str(name).strip()
        if not s or not n:
            continue
        if upper_tickers:
            s = s.upper()
        # Strip the suffix if Wikipedia already includes it, then re-append cleanly
        if suffix and s.endswith(suffix.upper() if upper_tickers else suffix):
            s = s[: -len(suffix)]
        rows.append({"symbol": f"{s}{suffix}", "company": n})
    if not rows:
        raise RuntimeError("Parsed zero rows from Wikipedia table.")
    return rows

def _parse_nasdaq100():
    url = WIKI_PAGES["NASDAQ100"]
    rows = _parse_wiki_constituents(
        url,
        ticker_candidates=["ticker", "symbol"],
        company_candidates=["company", "name"],
        suffix="",
        upper_tickers=True,
    )
    return rows, "NASDAQ", "US", "USD", url

def _parse_dax40():
    url = WIKI_PAGES["DAX40"]
    rows = _parse_wiki_constituents(
        url,
        ticker_candidates=["ticker symbol", "ticker", "symbol"],
        company_candidates=["company", "name"],
        suffix=".DE",
        upper_tickers=True,
    )
    return rows, "XETRA", "DE", "EUR", url

def _parse_omxs30():
    url = WIKI_PAGES["OMXS30"]
    rows = _parse_wiki_constituents(
        url,
        ticker_candidates=["ticker", "symbol"],
        company_candidates=["company", "name"],
        suffix=".ST",
        upper_tickers=True,
    )
    return rows, "OMX Stockholm", "SE", "SEK", url

# ---------- public helpers ----------
def get_markets() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """
    Return filters structure for UI.
    Does not mutate MARKETS; returns MARKETS + EXTRA_MARKETS merged.
    """
    # FIX: removed an extra ']' here
    merged: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    # deep copy MARKETS
    for country, exchanges in MARKETS.items():
        merged[country] = {ex: refs[:] for ex, refs in exchanges.items()}
    # merge extras
    for country, exchanges in EXTRA_MARKETS.items():
        merged.setdefault(country, {})
        for ex, refs in exchanges.items():
            merged[country].setdefault(ex, [])
            merged[country][ex].extend(refs)
    return merged

def _all_supported_index_codes(markets: Dict[str, Dict[str, List[Dict[str, str]]]]) -> List[str]:
    codes: List[str] = []
    for _country, exchanges in markets.items():
        for _exch, refs in exchanges.items():
            for ref in refs:
                codes.append(ref["code"])
    return codes

def _index_display_name(code: str, markets: Dict[str, Dict[str, List[Dict[str, str]]]]) -> str:
    cu = code.upper()
    for _country, exchanges in markets.items():
        for _exch, refs in exchanges.items():
            for ref in refs:
                if ref["code"].upper() == cu:
                    return ref.get("name", cu)
    return cu

def search_companies(q: str,
                     indices: Optional[List[str]] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
    """
    Global search across supported indices (cached via build_companies_payload).
    Returns items: {symbol, company, indexCode, indexName, exchange, country}
    """
    q_norm = (q or "").strip().lower()
    if not q_norm:
        return []

    markets = get_markets()
    index_codes = indices or _all_supported_index_codes(markets)

    results: List[Dict[str, Any]] = []
    for code in index_codes:
        try:
            payload = build_companies_payload(code)
        except Exception:
            continue
        idx_name = _index_display_name(code, markets)
        for row in payload.get("constituents", []):
            sym = str(row.get("symbol", "")).strip()
            com = str(row.get("company", "")).strip()
            if not sym or not com:
                continue
            if q_norm in sym.lower() or q_norm in com.lower():
                results.append({
                    "symbol": sym,
                    "company": com,
                    "indexCode": payload.get("code"),
                    "indexName": idx_name,
                    "exchange": payload.get("exchange"),
                    "country": payload.get("country"),
                })
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break

    def rank(item):
        sym, com = item["symbol"].lower(), item["company"].lower()
        if sym == q_norm or com == q_norm:
            return 0
        if sym.startswith(q_norm) or com.startswith(q_norm):
            return 1
        return 2

    results.sort(key=rank)
    return results[:limit]

# ---------- core (unchanged name, extended) ----------
def build_companies_payload(code: str) -> Dict[str, Any]:
    code = (code or "").upper().strip()
    if not code:
        raise ValueError("Index code is required.")

    cached = load_cache(code)
    if cached:
        return cached

    if code in NIFTY_URLS:
        url = NIFTY_URLS[code]
        text = http_get_text(url)
        rows = parse_nifty_csv(text)
        exchange, country, currency, source = "NSE", "IN", "INR", url
    elif code == "NASDAQ100":
        rows, exchange, country, currency, source = _parse_nasdaq100()
    elif code == "DAX40":
        rows, exchange, country, currency, source = _parse_dax40()
    elif code == "OMXS30":
        rows, exchange, country, currency, source = _parse_omxs30()
    else:
        raise ValueError(f"Unknown index code: {code}")

    payload = {
        "code": code,
        "exchange": exchange,
        "country": country,
        "currency": currency,
        "asOf": _now_iso_utc(),
        "count": len(rows),
        "constituents": rows,
        "source": source,
    }
    save_cache(code, payload)
    return payload

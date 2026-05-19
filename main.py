"""
APEX Markets — Backend API v6 (EODHD + Yahoo Finance + FRED edition)
"""
import os
import re
import json
import time
import string
import random
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from auth import verify_jwt
from supabase_client import supabase as sb_client
from yahoo_finance import fetch_yahoo_quote   # TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
from fred_api import fetch_fred_series, fetch_fred_cpi_yoy, fetch_fred_indpro_yoy, fetch_fred_yield_history
from ecb_api import fetch_ecb_series, fetch_ecb_yield_history
import httpx
import asyncio

# ─── Config ───────────────────────────────────────────────────────────────────

EODHD_API_KEY    = "69ee10907be601.18560848"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EODHD_BASE    = "https://eodhd.com/api"

BENCHMARK = "SPY"

SECTOR_ETFS = {
    "Technologie":     "XLK",
    "Kommunikation":   "XLC",
    "Finanzen":        "XLF",
    "Gesundheit":      "XLV",
    "Industrie":       "XLI",
    "Konsum_Zyklisch": "XLY",
    "Konsum_Basis":    "XLP",
    "Energie":         "XLE",
    "Materialien":     "XLB",
    "Immobilien":      "XLRE",
    "Versorger":       "XLU",
}

TICKER_SECTOR_MAP = {
    # Technologie
    "AAPL": "Technologie", "MSFT": "Technologie", "NVDA": "Technologie",
    "GOOGL": "Technologie", "GOOG": "Technologie", "META": "Technologie",
    "AMZN": "Technologie", "ORCL": "Technologie", "CRM": "Technologie",
    "ADBE": "Technologie", "SAP": "Technologie", "ASML": "Technologie",
    "TSM": "Technologie", "AVGO": "Technologie", "AMD": "Technologie",
    "INTC": "Technologie", "CSCO": "Technologie", "IBM": "Technologie",
    "QCOM": "Technologie", "TXN": "Technologie", "AMAT": "Technologie",
    "MU": "Technologie", "LRCX": "Technologie", "KLAC": "Technologie",
    "SNPS": "Technologie", "CDNS": "Technologie", "NOW": "Technologie",
    "INTU": "Technologie", "PANW": "Technologie", "CRWD": "Technologie",
    "SNOW": "Technologie", "NET": "Technologie", "DDOG": "Technologie",
    "SHOP": "Technologie", "SE": "Technologie", "BABA": "Technologie",
    "IFX": "Technologie",
    # Gesundheit
    "JNJ": "Gesundheit", "PFE": "Gesundheit", "UNH": "Gesundheit",
    "LLY": "Gesundheit", "MRK": "Gesundheit", "ABBV": "Gesundheit",
    "NVO": "Gesundheit", "NOVO": "Gesundheit", "TMO": "Gesundheit",
    "ABT": "Gesundheit", "DHR": "Gesundheit", "BMY": "Gesundheit",
    "AMGN": "Gesundheit", "GILD": "Gesundheit", "ISRG": "Gesundheit",
    "SYK": "Gesundheit", "MDT": "Gesundheit", "BSX": "Gesundheit",
    "ELV": "Gesundheit", "CVS": "Gesundheit", "HUM": "Gesundheit",
    "BAYN": "Gesundheit", "SHL": "Gesundheit", "FRE": "Gesundheit",
    # Finanzen
    "JPM": "Finanzen", "BAC": "Finanzen", "GS": "Finanzen",
    "MS": "Finanzen", "V": "Finanzen", "MA": "Finanzen",
    "BRK-B": "Finanzen", "BRK-A": "Finanzen", "BLK": "Finanzen",
    "WFC": "Finanzen", "C": "Finanzen", "AXP": "Finanzen",
    "SCHW": "Finanzen", "SPGI": "Finanzen", "MCO": "Finanzen",
    "ALV": "Finanzen", "MUV2": "Finanzen", "DBK": "Finanzen",
    "CBK": "Finanzen", "AXA": "Finanzen", "BNP": "Finanzen",
    "ING": "Finanzen", "INGA": "Finanzen", "SAN": "Finanzen",
    "CS": "Finanzen", "UBS": "Finanzen",
    # Konsum
    "TSLA": "Konsum", "HD": "Konsum", "NKE": "Konsum",
    "MCD": "Konsum", "SBUX": "Konsum", "LULU": "Konsum",
    "PG": "Konsum", "KO": "Konsum", "PEP": "Konsum",
    "WMT": "Konsum", "COST": "Konsum", "TGT": "Konsum",
    "AMZN": "Konsum", "EBAY": "Konsum", "ETSY": "Konsum",
    "F": "Konsum", "GM": "Konsum", "TM": "Konsum",
    "BMW": "Konsum", "MBG": "Konsum", "VOW3": "Konsum",
    "PAH3": "Konsum", "RACE": "Konsum", "OR": "Konsum",
    "MC": "Konsum", "CDI": "Konsum", "HEN3": "Konsum",
    "MDLZ": "Konsum", "GIS": "Konsum", "HSY": "Konsum",
    "CL": "Konsum", "EL": "Konsum",
    # Industrie
    "BA": "Industrie", "CAT": "Industrie", "GE": "Industrie",
    "HON": "Industrie", "UPS": "Industrie", "LMT": "Industrie",
    "RTX": "Industrie", "MMM": "Industrie", "DE": "Industrie",
    "SIE": "Industrie", "AIR": "Industrie", "BAS": "Industrie",
    "FDX": "Industrie", "NOC": "Industrie", "GD": "Industrie",
    "EMR": "Industrie", "ETN": "Industrie", "PH": "Industrie",
    "ROK": "Industrie", "CMI": "Industrie", "PCAR": "Industrie",
    "WM": "Industrie", "RSG": "Industrie", "UBER": "Industrie",
    "MTX": "Industrie", "RHM": "Industrie",
    # Kommunikation
    "NFLX": "Kommunikation", "DIS": "Kommunikation", "T": "Kommunikation",
    "VZ": "Kommunikation", "CMCSA": "Kommunikation", "TMUS": "Kommunikation",
    "GOOGL": "Kommunikation", "GOOG": "Kommunikation", "META": "Kommunikation",
    "CHTR": "Kommunikation", "PARA": "Kommunikation", "WBD": "Kommunikation",
    "SPOT": "Kommunikation", "SNAP": "Kommunikation", "PINS": "Kommunikation",
    "DTE": "Kommunikation", "TEF": "Kommunikation", "ORAN": "Kommunikation",
    # Energie
    "XOM": "Energie", "CVX": "Energie", "COP": "Energie",
    "SHEL": "Energie", "BP": "Energie", "TTE": "Energie",
    "EOG": "Energie", "SLB": "Energie", "PSX": "Energie",
    "MPC": "Energie", "VLO": "Energie", "OXY": "Energie",
    "ENB": "Energie", "ET": "Energie",
    # Materialien
    "LIN": "Materialien", "FCX": "Materialien", "NEM": "Materialien",
    "BHP": "Materialien", "RIO": "Materialien", "APD": "Materialien",
    "SHW": "Materialien", "ECL": "Materialien", "DD": "Materialien",
    "PPG": "Materialien", "ALB": "Materialien",
    # Immobilien
    "PLD": "Immobilien", "AMT": "Immobilien", "EQIX": "Immobilien",
    "VNA": "Immobilien", "CCI": "Immobilien", "PSA": "Immobilien",
    "WELL": "Immobilien", "DLR": "Immobilien", "O": "Immobilien",
    # Versorger
    "NEE": "Versorger", "DUK": "Versorger", "SO": "Versorger",
    "EOAN": "Versorger", "RWE": "Versorger", "AEP": "Versorger",
    "EXC": "Versorger", "XEL": "Versorger", "D": "Versorger",
    "ED": "Versorger", "EIX": "Versorger",
}

# EODHD ticker format:
#   US stocks/ETFs : symbol only (SPY, AAPL)
#   Crypto         : BTC-USD.CC
#   Forex          : EURUSD.FOREX
#   Indices        : GDAXI.INDX, ATX.INDX, STOXX50E.INDX

# EODHD — indices, crypto, forex (working correctly on current plan)
MACRO_TICKERS_EODHD = {
    "S&P 500":       "GSPC.INDX",
    "Volatility":    "VIXY",
    "Bitcoin":       "BTC-USD.CC",
    "EUR/USD":       "EURUSD.FOREX",
    "DAX":           "GDAXI.INDX",
    "Euro Stoxx 50": "STOXX50E.INDX",
    "ATX":           "ATX.INDX",
}

# Yahoo Finance — commodities and DXY (ETF surrogates removed)
# TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
MACRO_TICKERS_YAHOO = {
    "Gold":        "GC=F",      # Gold Spot Futures (~$4500)
    "Brent_Crude": "BZ=F",      # Brent Crude Futures (~$109)
    "WTI_Crude":   "CL=F",      # WTI Crude Futures (~$107)
    "US_Dollar":   "DX-Y.NYB",  # US Dollar Index DXY (~99)
}

# FRED — US Treasury yields (replaces broken EODHD TYX/IRX proxies)
MACRO_TICKERS_FRED = {
    "10Y_Treasury": "DGS10",
    "30Y_Treasury": "DGS30",
}

# Plausibility bounds {label: (min, max)} — return error if value is outside range
MACRO_PLAUSIBILITY = {
    "Gold":        (500,  10000),
    "Brent_Crude": (20,   300),
    "WTI_Crude":   (20,   300),
    "US_Dollar":   (70,   150),
    "10Y_Treasury":(0,    15),
    "30Y_Treasury":(0,    15),
    "Volatility":  (5,    100),
}

# Keep legacy alias so any code still referencing MACRO_TICKERS doesn't break
MACRO_TICKERS = {**MACRO_TICKERS_EODHD}

# ── Global indices ─────────────────────────────────────────────────────────────
# EODHD tickers for major international indices + ETF (MSCI World)
GLOBAL_INDEX_ITEMS = [
    {"label": "DAX",           "ticker": "GDAXI.INDX",   "region": "EU"},
    {"label": "Euro Stoxx 50", "ticker": "STOXX50E.INDX", "region": "EU"},
    {"label": "FTSE 100",      "ticker": "ISF.LSE",      "region": "EU"},   # ISF.LSE = iShares Core FTSE 100 ETF (GBX, London) — UKX.INDX/FTSE.INDX not on EODHD plan
    {"label": "Nikkei 225",    "ticker": "N225.INDX",    "region": "Asia"},
    {"label": "Hang Seng",     "ticker": "HSI.INDX",     "region": "Asia"},
    {"label": "ATX",           "ticker": "ATX.INDX",     "region": "EU"},
    {"label": "SMI",           "ticker": "SSMI.INDX",    "region": "EU"},
    {"label": "MSCI World",    "ticker": "URTH",         "region": "Global"},
]

GLOBAL_INDEX_PLAUSIBILITY = {
    "DAX":           (5_000,  30_000),
    "Euro Stoxx 50": (2_000,   8_000),
    "FTSE 100":      (5_000, 15_000),   # ISF.LSE × 10 approximation of FTSE 100 index level
    "Nikkei 225":   (15_000,  70_000),
    "Hang Seng":    (10_000,  40_000),
    "ATX":           (1_000,   8_000),
    "SMI":           (5_000,  20_000),
    "MSCI World":   (    50,    300),   # ETF price
}

# ECB Yield Curve maturities (Euro Area AAA government bonds, Svensson model)
ECB_YIELD_MATURITIES = {
    "2Y":  "SR_2Y",
    "5Y":  "SR_5Y",
    "10Y": "SR_10Y",
    "30Y": "SR_30Y",
}

SCREEN_UNIVERSE = [
    # Technology
    "AAPL",  "MSFT",  "NVDA",  "GOOGL", "META",  "AVGO",  "AMD",   "ORCL",  "CRM",
    # Communication
    "NFLX",  "DIS",
    # Consumer Discretionary
    "AMZN",  "TSLA",  "HD",    "MCD",
    # Healthcare
    "LLY",   "UNH",   "JNJ",   "ABBV",
    # Financials
    "JPM",   "V",     "MA",    "GS",    "BAC",
    # Energy
    "XOM",   "CVX",
    # Industrials
    "CAT",   "HON",   "BA",
    # Materials / Real Estate / Utilities / Staples
    "LIN",   "AMT",   "NEE",   "PG",    "WMT",
]

TICKER_META = {
    "AAPL":  ("Apple Inc.",         "Technology"),
    "MSFT":  ("Microsoft",          "Technology"),
    "NVDA":  ("NVIDIA",             "Technology"),
    "GOOGL": ("Alphabet",           "Technology"),
    "META":  ("Meta Platforms",     "Technology"),
    "AVGO":  ("Broadcom",           "Technology"),
    "AMD":   ("Advanced Micro Dev.","Technology"),
    "ORCL":  ("Oracle",             "Technology"),
    "CRM":   ("Salesforce",         "Technology"),
    "NFLX":  ("Netflix",            "Communication"),
    "DIS":   ("Walt Disney",        "Communication"),
    "AMZN":  ("Amazon",             "Consumer"),
    "TSLA":  ("Tesla",              "Consumer"),
    "HD":    ("Home Depot",         "Consumer"),
    "MCD":   ("McDonald's",         "Consumer"),
    "LLY":   ("Eli Lilly",          "Healthcare"),
    "UNH":   ("UnitedHealth",       "Healthcare"),
    "JNJ":   ("Johnson & Johnson",  "Healthcare"),
    "ABBV":  ("AbbVie",             "Healthcare"),
    "JPM":   ("JPMorgan Chase",     "Finance"),
    "V":     ("Visa",               "Finance"),
    "MA":    ("Mastercard",         "Finance"),
    "GS":    ("Goldman Sachs",      "Finance"),
    "BAC":   ("Bank of America",    "Finance"),
    "XOM":   ("ExxonMobil",         "Energy"),
    "CVX":   ("Chevron",            "Energy"),
    "CAT":   ("Caterpillar",        "Industrials"),
    "HON":   ("Honeywell",          "Industrials"),
    "BA":    ("Boeing",             "Industrials"),
    "LIN":   ("Linde",              "Materials"),
    "AMT":   ("American Tower",     "Real Estate"),
    "NEE":   ("NextEra Energy",     "Utilities"),
    "PG":    ("Procter & Gamble",   "Staples"),
    "WMT":   ("Walmart",            "Staples"),
}

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="APEX Markets API",
    description="Marktanalyse-Backend mit EODHD Daten",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://finscope.markets",
        "https://www.finscope.markets",
        "https://finscope-phi.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Cache ────────────────────────────────────────────────────────────────────

class TimedCache:
    def __init__(self):
        self._store: dict = {}

    def get(self, key: str, ttl_seconds: int):
        entry = self._store.get(key)
        if entry and time.time() - entry[1] < ttl_seconds:
            return entry[0]
        return None

    def set(self, key: str, value):
        self._store[key] = (value, time.time())

cache = TimedCache()

# ─── Ticker format resolution ─────────────────────────────────────────────────

def _eodhd_ticker_formats(ticker: str) -> list[str]:
    """Return EODHD formats to try in order.
    Tickers that already contain a dot (BTC-USD.CC, GDAXI.INDX) are tried as-is only."""
    if "." in ticker:
        return [ticker]
    return [ticker, f"{ticker}.US", f"{ticker}.NASDAQ"]


def _fetch_rt_one(fmt: str) -> dict | None:
    """Single real-time lookup for one EODHD ticker format. Returns normalised quote or None."""
    try:
        data  = eodhd_get(f"/real-time/{fmt}")
        items = data if isinstance(data, list) else [data]
        item  = items[0] if items else {}
        price = float(item.get("close") or item.get("previousClose") or 0)
        if not price:
            return None
        prev = float(item.get("previousClose") or price)
        return {
            "ticker":         item.get("code", fmt),
            "symbol":         item.get("code", fmt),
            "shortName":      item.get("name", fmt),
            "price":          round(price, 4),
            "previous_close": round(prev, 4),
            "change":         round(float(item.get("change")   or 0), 4),
            "change_pct":     round(float(item.get("change_p") or 0), 4),
            "volume":         int(item.get("volume") or 0),
        }
    except Exception:
        return None


# ─── EODHD request layer ──────────────────────────────────────────────────────

_last_eodhd_call: float = 0.0
_EODHD_MIN_INTERVAL = 2.0  # conservative — avoids burst issues on any plan tier

def eodhd_get(path: str, params: dict | None = None):
    """Rate-limited GET to EODHD API. Always appends api_token and fmt=json."""
    global _last_eodhd_call

    elapsed = time.time() - _last_eodhd_call
    if elapsed < _EODHD_MIN_INTERVAL:
        time.sleep(_EODHD_MIN_INTERVAL - elapsed)

    p = dict(params or {})
    p["api_token"] = EODHD_API_KEY
    p["fmt"]       = "json"

    try:
        resp = requests.get(f"{EODHD_BASE}{path}", params=p, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"EODHD network error: {e}")
    finally:
        _last_eodhd_call = time.time()

    data = resp.json()

    # EODHD returns plain {"message": "..."} for auth errors
    if isinstance(data, dict) and "message" in data and not any(
        k in data for k in ("open", "close", "code", "date")
    ):
        raise HTTPException(status_code=502, detail=data["message"])

    return data

# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_realtime(tickers: list[str]) -> dict[str, dict]:
    """
    Batch real-time quotes via EODHD /real-time endpoint.
    Tickers missing from the batch response are retried individually with
    .US and .NASDAQ suffixes before giving up.
    Cached 15 min per ticker; only uncached tickers are fetched.
    """
    missing = [t for t in tickers if cache.get(f"rt:{t}", 900) is None]
    result  = {t: cache.get(f"rt:{t}", 900) for t in tickers if t not in missing}

    if not missing:
        return result

    found: set[str] = set()

    # Batch call — path uses first ticker; all go into &s= as well
    try:
        path_ticker = missing[0]
        data  = eodhd_get(f"/real-time/{path_ticker}", {"s": ",".join(missing)})
        items = data if isinstance(data, list) else [data]

        for item in items:
            sym = item.get("code", "")
            if not sym:
                continue

            price      = float(item.get("close")         or item.get("previousClose") or 0)
            prev_close = float(item.get("previousClose") or price)
            entry = {
                "ticker":         sym,
                "symbol":         sym,
                "price":          round(price, 4),
                "previous_close": round(prev_close, 4),
                "change":         round(float(item.get("change")   or 0), 4),
                "change_pct":     round(float(item.get("change_p") or 0), 4),
                "volume":         int(item.get("volume") or 0),
            }
            cache.set(f"rt:{sym}", entry)
            result[sym] = entry
            found.add(sym)
            # Also index by base ticker if EODHD returned an exchange-suffixed code
            base = sym.rsplit(".", 1)[0] if "." in sym else sym
            if base != sym:
                cache.set(f"rt:{base}", entry)
                result[base] = entry
                found.add(base)
    except Exception:
        pass  # batch failed; per-ticker fallback below covers everything

    # Per-ticker fallback for anything still absent after the batch call.
    # NOTE: we do NOT skip fmt == t — dotted tickers (GDAXI.INDX, BTC-USD.CC)
    # have only one format candidate and the batch often misses them silently.
    for t in missing:
        if t in found:
            continue
        for fmt in _eodhd_ticker_formats(t):
            entry = _fetch_rt_one(fmt)
            if entry:
                cache.set(f"rt:{t}",   entry)
                cache.set(f"rt:{fmt}", entry)
                result[t]   = entry
                result[fmt] = entry
                found.add(t)
                break

    return result


def fetch_eod(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Historical daily OHLCV via EODHD /eod endpoint.
    Tries bare ticker first, then .US and .NASDAQ suffixes if no data is returned.
    EODHD returns data in ascending order (oldest first). Cached 24h.
    """
    for fmt in _eodhd_ticker_formats(ticker):
        cache_key = f"eod:{fmt}:{from_date}:{to_date}"
        cached    = cache.get(cache_key, ttl_seconds=86400)
        if cached is not None:
            return cached

        try:
            data = eodhd_get(f"/eod/{fmt}", {"from": from_date, "to": to_date})
        except HTTPException:
            continue

        if not isinstance(data, list) or len(data) == 0:
            continue

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={
            "open":           "Open",
            "high":           "High",
            "low":            "Low",
            "adjusted_close": "Close",   # use split/dividend-adjusted close
            "volume":         "Volume",
        }, inplace=True)

        if "Close" not in df.columns and "close" in df.columns:
            df.rename(columns={"close": "Close"}, inplace=True)

        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce").ffill()
        df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0).astype(int)

        cache.set(cache_key, df)
        return df

    raise HTTPException(status_code=502, detail=f"No EOD data for {ticker}")


def _date_range(days: int) -> tuple[str, str]:
    """(from_date, to_date) as YYYY-MM-DD strings."""
    to  = datetime.utcnow()
    frm = to - timedelta(days=days)
    return frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d")


def fetch_history(ticker: str, days: int = 370) -> pd.DataFrame:
    frm, to = _date_range(days)
    return fetch_eod(ticker, frm, to)

# ─── Scoring ──────────────────────────────────────────────────────────────────

def calc_relative_strength(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(ticker_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    t_cum = (1 + ticker_returns).prod() - 1
    b_cum = (1 + benchmark_returns).prod() - 1
    return float((t_cum - b_cum) * 100)


def calc_momentum_score(prices: pd.Series) -> float:
    if len(prices) < 21:
        return 50.0
    try:
        r_1m  = prices.iloc[-1] / prices.iloc[-21]  - 1
        r_3m  = (prices.iloc[-1] / prices.iloc[-63]  - 1) if len(prices) >= 63  else r_1m
        r_6m  = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else r_3m
        r_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) if len(prices) >= 252 else r_6m
        weighted = 0.4 * r_1m + 0.3 * r_3m + 0.2 * r_6m + 0.1 * r_12m
        return float(np.clip(50 + np.tanh(weighted * 5) * 50, 0, 100))
    except Exception:
        return 50.0


def calc_composite_score(prices: pd.Series, benchmark_prices: pd.Series) -> dict:
    if len(prices) < 21:
        return {"composite": 50.0, "momentum": 50.0, "rs": 0.0}

    momentum   = calc_momentum_score(prices)
    common_idx = prices.index.intersection(benchmark_prices.index)
    if len(common_idx) < 21:
        return {"composite": momentum, "momentum": momentum, "rs": 0.0}

    p  = prices.loc[common_idx].iloc[-63:]           if len(common_idx) >= 63 else prices.loc[common_idx]
    b  = benchmark_prices.loc[common_idx].iloc[-63:] if len(common_idx) >= 63 else benchmark_prices.loc[common_idx]
    rs = calc_relative_strength(p.pct_change().dropna(), b.pct_change().dropna())

    composite = 0.5 * momentum + 0.5 * (50 + np.tanh(rs / 10) * 50)
    return {
        "composite": round(float(composite), 1),
        "momentum":  round(float(momentum), 1),
        "rs":        round(float(rs), 2),
    }


def calc_returns(prices: pd.Series) -> dict:
    if len(prices) == 0:
        return {"1M": 0.0, "3M": 0.0, "6M": 0.0, "YTD": 0.0, "1Y": 0.0}
    last = prices.iloc[-1]
    out  = {
        "1M":  round(float((last / prices.iloc[-21]  - 1) * 100), 2) if len(prices) >= 21  else 0.0,
        "3M":  round(float((last / prices.iloc[-63]  - 1) * 100), 2) if len(prices) >= 63  else 0.0,
        "6M":  round(float((last / prices.iloc[-126] - 1) * 100), 2) if len(prices) >= 126 else 0.0,
        "1Y":  round(float((last / prices.iloc[-252] - 1) * 100), 2) if len(prices) >= 252 else 0.0,
    }
    ytd        = prices[prices.index.year == prices.index[-1].year]
    out["YTD"] = round(float((ytd.iloc[-1] / ytd.iloc[0] - 1) * 100), 2) if len(ytd) > 1 else 0.0
    return out


def trend_indicator(prices: pd.Series) -> str:
    if len(prices) < 50:
        return "flat"
    sma_20 = prices.iloc[-20:].mean()
    sma_50 = prices.iloc[-50:].mean()
    if sma_20 > sma_50 * 1.01:
        return "up"
    if sma_20 < sma_50 * 0.99:
        return "down"
    return "flat"

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":   "APEX Markets API",
        "version":   "5.0.0 (EODHD)",
        "status":    "online",
        "endpoints": [
            "/health", "/quote/{ticker}", "/search/{ticker}",
            "/history/{ticker}", "/sectors", "/screener/top",
            "/macro", "/portfolio/analyze",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/quote/{ticker}")
def quote(ticker: str):
    t     = ticker.strip().upper()
    snaps = fetch_realtime([t])
    data  = snaps.get(t)
    if not data:
        raise HTTPException(status_code=404, detail=f"No quote for {ticker}")
    return {**data, "symbol": data.get("symbol", t)}


@app.get("/search/{ticker}")
def search_ticker(ticker: str):
    """
    Resolve a ticker to its working EODHD format by trying bare ticker, then .US, then .NASDAQ.
    Returns {"found": true, "ticker_eodhd": "SOXX.US", "price": 234.5} or {"found": false}.
    Used by the frontend to validate custom tickers before adding them to the RRG chart.
    """
    t = ticker.strip().upper()
    for fmt in _eodhd_ticker_formats(t):
        entry = _fetch_rt_one(fmt)
        if entry and entry.get("price"):
            return {
                "found":        True,
                "ticker":       t,
                "ticker_eodhd": fmt,
                "price":        entry["price"],
                "change_pct":   entry["change_pct"],
                "symbol":       entry.get("symbol", fmt),
                "shortName":    entry.get("shortName", fmt),
            }
    return {"found": False, "ticker": t}


def _fetch_fundamentals_highlights(eodhd_t: str) -> dict:
    """
    Fetch EODHD Highlights for a ticker. Tries ?filter=Highlights first,
    falls back to the full fundamentals object. Logs outcome for Railway diagnostics.
    Returns {} on failure.
    """
    # Attempt 1: filtered endpoint (smaller, faster — if supported by plan)
    try:
        data = eodhd_get(f"/fundamentals/{eodhd_t}", {"filter": "Highlights"})
        if isinstance(data, dict):
            result = data.get("Highlights") or data
            if isinstance(result, dict):
                has_data = any(
                    result.get(k) not in (None, "", "None")
                    for k in ("MarketCapitalization", "PERatio", "EarningsShare")
                )
                logger.info(
                    "Fundamentals filtered %s: has_cap=%s has_pe=%s has_eps=%s",
                    eodhd_t,
                    result.get("MarketCapitalization") is not None,
                    result.get("PERatio") is not None,
                    result.get("EarningsShare") is not None,
                )
                if has_data:
                    return result
                logger.warning("Fundamentals filtered response missing key fields for %s", eodhd_t)
    except Exception as exc:
        logger.error("Fundamentals filtered fetch failed for %s: %s", eodhd_t, exc)

    # Attempt 2: full fundamentals object, extract Highlights section
    try:
        data = eodhd_get(f"/fundamentals/{eodhd_t}")
        if isinstance(data, dict):
            result = data.get("Highlights", {})
            logger.info(
                "Fundamentals full fallback %s: has_cap=%s",
                eodhd_t,
                result.get("MarketCapitalization") is not None,
            )
            return result
    except Exception as exc:
        logger.error("Fundamentals full fetch failed for %s: %s", eodhd_t, exc)

    logger.error("Fundamentals completely unavailable for %s", eodhd_t)
    return {}


@app.get("/research/memo/{ticker}")
def research_memo(ticker: str):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Anthropic API key not configured on server")

    t         = ticker.strip().upper()
    today_str = datetime.utcnow().strftime("%d.%m.%Y")

    # ── 1. Fetch live data (all non-fatal) ────────────────────────────────────
    quote_data: dict = {}
    try:
        snaps      = fetch_realtime([t])
        quote_data = snaps.get(t) or {}
    except Exception:
        pass

    hist_closes: list = []
    try:
        df          = fetch_history(t, days=370)
        hist_closes = df["Close"].dropna().tolist()
    except Exception:
        pass

    eodhd_t            = t if "." in t else f"{t}.US"
    highlights: dict   = _fetch_fundamentals_highlights(eodhd_t)

    # ── 2. Derive stats ───────────────────────────────────────────────────────
    def safe_f(key):
        v = highlights.get(key)
        try:
            return float(v) if v not in (None, "", "None") else None
        except (ValueError, TypeError):
            return None

    def fmt_cap(v):
        if v is None:
            return None
        if v >= 1e12:
            return f"${v / 1e12:.2f} Bio."
        if v >= 1e9:
            return f"${v / 1e9:.1f} Mrd."
        return f"${v / 1e6:.0f} Mio."

    price      = quote_data.get("price")
    change_pct = quote_data.get("change_pct")

    high52w = low52w = perf1y = perf1m = None
    if hist_closes:
        last    = hist_closes[-1]
        high52w = round(max(hist_closes), 2)
        low52w  = round(min(hist_closes), 2)
        ref1y   = hist_closes[-252] if len(hist_closes) >= 252 else hist_closes[0]
        perf1y  = round(((last / ref1y) - 1) * 100, 1)
        if len(hist_closes) >= 21:
            perf1m = round(((last / hist_closes[-21]) - 1) * 100, 1)

    market_cap   = safe_f("MarketCapitalization")
    pe_ratio     = safe_f("PERatio")
    eps          = safe_f("EarningsShare")
    revenue_ttm  = safe_f("RevenueTTM")
    target_price = safe_f("WallStreetTargetPrice")

    # ── 3. Build live-data block ──────────────────────────────────────────────
    lines = [f"TICKER: {t}", f"Datum: {today_str}"]
    if price         is not None: lines.append(f"Aktueller Kurs: {price:.2f} USD")
    if change_pct    is not None: lines.append(f"Tagesveränderung: {change_pct:.2f}%")
    if high52w       is not None: lines.append(f"52W-Hoch: {high52w} USD")
    if low52w        is not None: lines.append(f"52W-Tief: {low52w} USD")
    if perf1y        is not None: lines.append(f"1J Performance: {perf1y}%")
    if perf1m        is not None: lines.append(f"1M Performance: {perf1m}%")
    cap_str = fmt_cap(market_cap)
    if cap_str:                   lines.append(f"Marktkapitalisierung: {cap_str}")
    if pe_ratio      is not None: lines.append(f"KGV (P/E): {pe_ratio:.1f}")
    if eps           is not None: lines.append(f"EPS: ${eps:.2f}")
    rev_str = fmt_cap(revenue_ttm)
    if rev_str:                   lines.append(f"Umsatz TTM: {rev_str}")
    if target_price  is not None: lines.append(f"Analysten-Kursziel (Ø): ${target_price:.2f}")

    has_live   = len(lines) > 2
    live_block = "\n".join(lines)

    # ── 4. Prompts ────────────────────────────────────────────────────────────
    data_section = (
        f"LIVE-DATEN — Stand: {today_str}:\n{live_block}\n\n"
        f"Nutze AUSSCHLIESSLICH diese Live-Daten für alle Zahlen, Kurse und Bewertungen."
    ) if has_live else (
        f"Ticker: {t}\nDatum: {today_str}\n\n"
        f"Hinweis: Live-Marktdaten waren nicht abrufbar. "
        f"Kennzeichne alle Bewertungszahlen als geschätzte Modellwerte."
    )

    user_prompt = (
        f"Erstelle ein vollständiges Investment Memo für **{t}** (Stand: {today_str}).\n\n"
        f"{data_section}\n\n"
        f"Strukturiere das Memo mit diesen Abschnitten (## als Header):\n\n"
        f"## Unternehmensprofil & Geschäftsmodell\n"
        f"## Kursentwicklung & Technische Analyse\n"
        f"## Bewertung\n"
        f"## Katalysatoren (Bull Case)\n"
        f"## Risiken (Bear Case)\n"
        f"## Base Case & Kursziel (12 Monate)\n"
        f"## Risiko-Rendite-Profil\n"
        f"## Disclaimer\n\n"
        f"Verwende Aufzählungspunkte (- ) für Listen. Sei präzise und konkret."
    )

    system_prompt = (
        f"Du bist ein erfahrener Finanzanalyst mit CFA-Qualifikation. "
        f"Du erstellst faktengenaue, professionelle Investment Memos auf Basis bereitgestellter Live-Marktdaten.\n\n"
        f"PFLICHTREGELN:\n"
        f"- Verwende AUSSCHLIESSLICH die übergebenen Live-Daten für alle Kurse, KGVs und Bewertungszahlen\n"
        f"- Das heutige Datum ist {today_str} — verwende NICHT dein Trainingsdaten-Wissen für Datumsangaben\n"
        f"- KEINE EMOJIS in Überschriften oder Text — professioneller Bloomberg-Stil, ## Markdown-Header\n"
        f"- Kein 'Kaufen' / 'Verkaufen' / 'Halten' / 'Buy' / 'Sell' / 'Hold' — keine Anlageempfehlungen\n"
        f"- Abschnitt 'Risiko-Rendite-Profil': neutrale Charakterisierung, keine Empfehlung\n"
        f"- Du schreibst ausschließlich auf Deutsch, sachlich und direkt\n"
        f"- Das Memo endet mit Disclaimer: keine individuelle Anlageberatung"
    )

    # ── 5. Call Anthropic ─────────────────────────────────────────────────────
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":    "application/json",
                "x-api-key":       ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-5-20250929",
                "max_tokens": 4096,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_prompt}],
            },
            timeout=90,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API network error: {exc}")

    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(
            status_code=502,
            detail=err.get("error", {}).get("message", f"Anthropic API Fehler {resp.status_code}"),
        )

    memo_text = resp.json()["content"][0]["text"]

    # ── 6. Strip emojis from ## headers ──────────────────────────────────────
    _emoji_re = re.compile(
        r"[\U0001F300-\U0001FFFF\U00002600-\U000027FF\U00002702-\U000027B0]+",
        flags=re.UNICODE,
    )
    cleaned = []
    for line in memo_text.split("\n"):
        if line.startswith("## "):
            line = "## " + _emoji_re.sub("", line[3:]).strip()
        cleaned.append(line)

    return {
        "ticker":        t,
        "memo":          "\n".join(cleaned),
        "has_live_data": has_live,
        "date":          today_str,
    }


@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|3y|5y|10y|max)$"),
):
    period_days = {
        "1mo": 35, "3mo": 95, "6mo": 185,
        "1y": 370, "2y": 740, "3y": 1100,
        "5y": 1830, "10y": 3660, "max": 7300,
    }
    df = fetch_history(ticker.upper(), days=period_days[period])

    out = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 4)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "period": period, "data": out}


@app.get("/sectors")
def sectors():
    """
    Sector ETF quotes (1 batch call) + EOD history per ETF for multi-timeframe
    returns and composite scoring. Full result cached 1h; history cached 24h.
    """
    cache_key = "sectors:full"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    sector_tickers = list(SECTOR_ETFS.values())
    snaps          = fetch_realtime(sector_tickers + [BENCHMARK])
    benchmark_df   = fetch_history(BENCHMARK, days=370)
    benchmark_close = benchmark_df["Close"]

    results = []
    for name, ticker in SECTOR_ETFS.items():
        try:
            snap    = snaps.get(ticker, {})
            df      = fetch_history(ticker, days=370)
            close   = df["Close"]
            returns = calc_returns(close)
            scores  = calc_composite_score(close, benchmark_close)
            trend   = trend_indicator(close)

            results.append({
                "name":           name,
                "ticker":         ticker,
                "price":          snap.get("price", round(float(close.iloc[-1]), 2)),
                "performance":    snap.get("change_pct", returns["1M"]),  # 1d % for bar chart
                "perf_1d":        snap.get("change_pct", 0.0),
                "perf_1M":        returns["1M"],
                "perf_3M":        returns["3M"],
                "perf_YTD":       returns["YTD"],
                "perf_1Y":        returns["1Y"],
                "score":          scores["composite"],
                "momentum_score": scores["momentum"],
                "rs_vs_spy":      scores["rs"],
                "trend":          trend,
            })
        except Exception as e:
            results.append({
                "name":        name,
                "ticker":      ticker,
                "performance": snaps.get(ticker, {}).get("change_pct", 0.0),
                "perf_1d":     snaps.get(ticker, {}).get("change_pct", 0.0),
                "error":       str(e),
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    output = {"timestamp": datetime.utcnow().isoformat(), "sectors": results}
    cache.set(cache_key, output)
    return output


@app.get("/screener/top")
def screener_top(limit: int = Query(20, ge=1, le=50)):
    """
    Top stocks by composite score (momentum + RS vs SPY).
    Full universe computed once and cached 4h; limit slices the sorted result.
    """
    full_cache_key = "screener:full"
    cached         = cache.get(full_cache_key, ttl_seconds=14400)
    if cached is not None:
        return {"timestamp": cached["timestamp"], "results": cached["results"][:limit]}

    snaps           = fetch_realtime(SCREEN_UNIVERSE + [BENCHMARK])
    benchmark_df    = fetch_history(BENCHMARK, days=370)
    benchmark_close = benchmark_df["Close"]

    results = []
    for ticker in SCREEN_UNIVERSE:
        try:
            snap    = snaps.get(ticker, {})
            df      = fetch_history(ticker, days=370)
            close   = df["Close"]
            scores  = calc_composite_score(close, benchmark_close)
            returns = calc_returns(close)

            curr     = snap.get("price", round(float(close.iloc[-1]), 2))
            high_52w = round(float(close.max()), 2)
            low_52w  = round(float(close.min()), 2)
            pos_52w  = round((curr - low_52w) / (high_52w - low_52w) * 100, 1) if high_52w != low_52w else 50.0

            meta_name, meta_sector = TICKER_META.get(ticker, (ticker, "Other"))

            results.append({
                "ticker":     ticker,
                "name":       meta_name,
                "sector":     meta_sector,
                "price":      curr,
                "change_pct": snap.get("change_pct", 0.0),
                "score":      scores["composite"],
                "momentum":   scores["momentum"],
                "rs_vs_spy":  scores["rs"],
                "perf_1M":    returns["1M"],
                "perf_3M":    returns["3M"],
                "perf_6M":    returns["6M"],
                "perf_1Y":    returns["1Y"],
                "perf_YTD":   returns["YTD"],
                "pos_52w":    pos_52w,
                "high_52w":   high_52w,
                "low_52w":    low_52w,
                "signal": (
                    "Strong Momentum" if scores["composite"] >= 80 else
                    "Momentum"        if scores["composite"] >= 65 else
                    "Neutral"         if scores["composite"] >= 45 else
                    "Schwach"
                ),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    data_date = benchmark_df.index[-1].strftime("%Y-%m-%d") if len(benchmark_df) > 0 else None
    full = {"timestamp": datetime.utcnow().isoformat(), "data_date": data_date, "results": results}
    cache.set(full_cache_key, full)
    return {"timestamp": full["timestamp"], "data_date": data_date, "results": results[:limit]}


def _plausibility_check(label: str, ticker: str, value: float) -> dict | None:
    """Returns error dict if value is outside expected range, else None."""
    bounds = MACRO_PLAUSIBILITY.get(label)
    if bounds is None:
        return None
    lo, hi = bounds
    if not (lo <= value <= hi):
        logger.error(
            "Plausibility FAIL %-15s (%s): %.4f not in [%.0f, %.0f]",
            label, ticker, value, lo, hi,
        )
        return {"label": label, "ticker": ticker, "error": "implausible_value", "raw_value": value}
    return None


@app.get("/macro")
def macro():
    """
    Macro indicators — mixed sources:
      EODHD : indices, crypto, forex (S&P 500, DAX, Euro Stoxx 50, ATX, EUR/USD, Bitcoin, Volatility)
      Yahoo  : Gold (GC=F), Brent_Crude (BZ=F), WTI_Crude (CL=F), US_Dollar (DX-Y.NYB)
      FRED   : 10Y_Treasury (DGS10), 30Y_Treasury (DGS30)
    Cached 5 min. Plausibility checks guard against implausible values.
    TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
    """
    cache_key = "macro:v2"
    cached    = cache.get(cache_key, ttl_seconds=300)
    if cached is not None:
        return cached

    out = []

    # ── EODHD batch: indices, crypto, forex ─────────────────────────────────
    snaps = fetch_realtime(list(MACRO_TICKERS_EODHD.values()))
    for label, ticker in MACRO_TICKERS_EODHD.items():
        snap = snaps.get(ticker)
        if snap:
            value = snap["price"]
            err   = _plausibility_check(label, ticker, value)
            if err:
                out.append(err)
            else:
                out.append({
                    "label":      label,
                    "ticker":     ticker,
                    "value":      value,
                    "change":     snap["change"],
                    "change_pct": snap["change_pct"],
                })
            logger.info("Macro EODHD OK   %-15s (%s): %.4f", label, ticker, value)
        else:
            logger.warning("Macro EODHD MISS %-15s (%s): no data", label, ticker)

    # ── Yahoo Finance: Gold, Brent_Crude, WTI_Crude, US_Dollar ──────────────
    # TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
    for label, yf_symbol in MACRO_TICKERS_YAHOO.items():
        quote = fetch_yahoo_quote(yf_symbol)
        if quote:
            value = quote["price"]
            err   = _plausibility_check(label, yf_symbol, value)
            if err:
                out.append(err)
            else:
                out.append({
                    "label":      label,
                    "ticker":     yf_symbol,
                    "value":      value,
                    "change":     quote["change"],
                    "change_pct": quote["change_pct"],
                })
            logger.info("Macro Yahoo OK   %-15s (%s): %.4f", label, yf_symbol, value)
        else:
            out.append({"label": label, "ticker": yf_symbol, "error": "yahoo_unavailable"})
            logger.warning("Macro Yahoo MISS %-15s (%s): no data", label, yf_symbol)

    # ── FRED: 10Y_Treasury, 30Y_Treasury ────────────────────────────────────
    for label, fred_id in MACRO_TICKERS_FRED.items():
        data = fetch_fred_series(fred_id)
        if data:
            value = data["value"]
            err   = _plausibility_check(label, fred_id, value)
            if err:
                out.append(err)
            else:
                prev   = data["prev_value"]
                change = round(value - prev, 4)
                out.append({
                    "label":      label,
                    "ticker":     fred_id,
                    "value":      value,
                    "change":     change,
                    "change_pct": round((change / prev * 100) if prev else 0.0, 4),
                })
            logger.info("Macro FRED OK    %-15s (%s): %.4f", label, fred_id, value)
        else:
            out.append({"label": label, "ticker": fred_id, "error": "fred_unavailable"})
            logger.warning("Macro FRED MISS  %-15s (%s): no data", label, fred_id)

    result = {"timestamp": datetime.utcnow().isoformat(), "indicators": out}
    cache.set(cache_key, result)
    return result


# ─── Portfolio ─────────────────────────────────────────────────────────────────

class Position(BaseModel):
    ticker:   str
    shares:   float
    avg_cost: float

class PortfolioRequest(BaseModel):
    positions: list[Position]

@app.post("/portfolio/analyze")
def portfolio_analyze(req: PortfolioRequest):
    if not req.positions:
        raise HTTPException(status_code=400, detail="Keine Positionen übergeben")

    tickers           = [p.ticker.upper() for p in req.positions]
    snaps             = fetch_realtime(tickers + [BENCHMARK])
    benchmark_df      = fetch_history(BENCHMARK, days=370)
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

    enriched    = []
    total_value = 0.0
    total_cost  = 0.0

    for pos in req.positions:
        ticker = pos.ticker.upper()
        try:
            snap    = snaps.get(ticker, {})
            current = snap.get("price") or 0.0
            if not current:
                raise ValueError("No price available")

            value   = current * pos.shares
            cost    = pos.avg_cost * pos.shares
            pnl     = value - cost
            pnl_pct = (pnl / cost * 100) if cost else 0

            df  = fetch_history(ticker, days=370)
            ret = df["Close"].pct_change().dropna()

            enriched.append({
                "ticker":        ticker,
                "shares":        pos.shares,
                "avg_cost":      pos.avg_cost,
                "current_price": current,
                "value":         round(value, 2),
                "cost_basis":    round(cost, 2),
                "pnl":           round(pnl, 2),
                "pnl_pct":       round(pnl_pct, 2),
                "weight":        0,
                "_returns":      ret,
            })
            total_value += value
            total_cost  += cost
        except Exception as e:
            enriched.append({"ticker": ticker, "error": str(e)})

    port_returns = None
    if total_value > 0:
        for p in enriched:
            if "error" not in p:
                p["weight"] = round(p["value"] / total_value * 100, 2)

        weighted = pd.Series(0.0, index=benchmark_returns.index)
        for p in enriched:
            if "error" in p:
                continue
            common = p["_returns"].index.intersection(weighted.index)
            weighted.loc[common] += p["_returns"].loc[common] * (p["weight"] / 100)
        port_returns = weighted.dropna()

    risk = {}
    if port_returns is not None and len(port_returns) > 30:
        common = port_returns.index.intersection(benchmark_returns.index)
        p_ret  = port_returns.loc[common]
        b_ret  = benchmark_returns.loc[common]

        cov   = np.cov(p_ret, b_ret)[0, 1]
        var_b = np.var(b_ret)
        beta  = cov / var_b if var_b else 1.0

        excess = p_ret - (0.044 / 252)
        sharpe = (excess.mean() / p_ret.std() * np.sqrt(252)) if p_ret.std() else 0
        vol    = p_ret.std() * np.sqrt(252) * 100
        cum    = (1 + p_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100

        risk = {
            "beta":                  round(float(beta), 2),
            "sharpe_ratio":          round(float(sharpe), 2),
            "volatility_annualized": round(float(vol), 2),
            "max_drawdown":          round(float(max_dd), 2),
        }

    for p in enriched:
        p.pop("_returns", None)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_value":    round(total_value, 2),
            "total_cost":     round(total_cost, 2),
            "total_pnl":      round(total_value - total_cost, 2),
            "total_pnl_pct":  round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0,
            "position_count": len([p for p in enriched if "error" not in p]),
        },
        "positions":    enriched,
        "risk_metrics": risk,
    }


# ─── Yield Curve & Macro Indicators ──────────────────────────────────────────

# FRED series for full 7-maturity yield curve (replaces broken CBOE INDX proxies)
YIELD_FRED_SERIES = {
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}


@app.get("/yield-curve")
def yield_curve():
    """
    US Treasury yield curve (FRED) + EU AAA government bond curve (ECB Yield Curve dataset).
    Response includes both {us, eu} nested objects AND legacy flat keys for backward compat.
    US: 7 maturities (3M–30Y). EU: 4 maturities (2Y–30Y), Svensson model spot rates.
    Cached 1 hour.
    """
    cache_key = "yield_curve:v3"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    # ── US Treasury from FRED ─────────────────────────────────────────────────
    cur, m3, y1 = {}, {}, {}
    fetched = 0

    for mat, series_id in YIELD_FRED_SERIES.items():
        history = fetch_fred_yield_history(series_id, limit=400)
        if not history:
            logger.warning("Yield curve US MISS %s (%s): no FRED data", mat, series_id)
            continue
        cur[mat] = history[0]["value"]
        fetched += 1
        if len(history) > 63:
            m3[mat] = history[63]["value"]
        if len(history) > 252:
            y1[mat] = history[252]["value"]
        logger.info("Yield curve US OK %s (%s): %.3f%%", mat, series_id, cur[mat])

    if fetched == 0:
        logger.error("Yield curve: no FRED data available for any US maturity")
        return {"error": "yield_curve_unavailable", "timestamp": datetime.utcnow().isoformat()}

    y10 = cur.get("10Y")
    y2  = cur.get("2Y")
    spread = round(y10 - y2, 3) if y10 is not None and y2 is not None else None
    status = ("Invers" if spread is not None and spread < -0.1
              else "Flach"  if spread is not None and spread < 0.2
              else "Normal" if spread is not None
              else "Unbekannt")

    us_data = {
        "maturities":   ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"],
        "current":      cur,
        "3m_ago":       m3,
        "1y_ago":       y1,
        "spread_2y10y": spread,
        "status":       status,
    }

    # ── EU Bund proxy from ECB Yield Curve (AAA-rated Euro area gov bonds) ────
    eu_cur:  dict = {}
    eu_fetched = 0

    for mat, mk in ECB_YIELD_MATURITIES.items():
        hist = fetch_ecb_yield_history(mk, limit=300)
        if hist:
            eu_cur[mat] = round(hist[0]["value"], 3)
            eu_fetched += 1
        else:
            logger.warning("Yield curve EU MISS %s: no ECB data", mat)

    eu_data = None
    if eu_fetched > 0:
        eu10 = eu_cur.get("10Y")
        eu2  = eu_cur.get("2Y")
        eu_spread = round(eu10 - eu2, 3) if eu10 is not None and eu2 is not None else None
        eu_status = ("Invers" if eu_spread is not None and eu_spread < -0.1
                     else "Flach"  if eu_spread is not None and eu_spread < 0.2
                     else "Normal" if eu_spread is not None
                     else "Unbekannt")
        eu_data = {
            "maturities":   ["2Y", "5Y", "10Y", "30Y"],
            "current":      eu_cur,
            "spread_2y10y": eu_spread,
            "status":       eu_status,
        }
        logger.info("Yield curve EU: %d maturities fetched, 10Y=%.3f%%", eu_fetched, eu10 or 0)
    else:
        logger.warning("Yield curve EU: no ECB data for any maturity")

    result = {
        # New nested structure
        "us":           us_data,
        "eu":           eu_data,
        # Backward-compat flat keys (= US data, old frontend reads these)
        "maturities":   us_data["maturities"],
        "current":      cur,
        "3m_ago":       m3,
        "1y_ago":       y1,
        "spread_2y10y": spread,
        "status":       status,
        "timestamp":    datetime.utcnow().isoformat(),
    }
    cache.set(cache_key, result)
    return result


@app.get("/macro-indicators")
def macro_indicators():
    """
    Live macro indicators: US (FRED + Yahoo Finance) + EU (ECB SDW).
    Each indicator carries a 'region' flag: 'US' or 'EU'.
    Trend: 'up' if current > previous + 0.5, 'down' if < previous − 0.5, else 'neutral'.
    Cached 1 hour.
    TODO PRE-LAUNCH: DXY via Yahoo Finance — Replace with licensed source before commercial launch
    """
    cache_key = "macro_indicators:v3"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    def _trend(current: float, prev: float) -> str:
        diff = current - prev
        if diff > 0.5:  return "up"
        if diff < -0.5: return "down"
        return "neutral"

    def _ok(label: str, unit: str, desc: str, data: dict, region: str = "US") -> dict:
        v, p = data["value"], data["prev_value"]
        return {
            "label": label, "unit": unit, "desc": desc, "region": region,
            "value": v, "change": data["change"], "change_pct": data["change_pct"],
            "trend": _trend(v, p),
        }

    def _err(label: str, unit: str, desc: str, reason: str = "unavailable", region: str = "US") -> dict:
        return {"label": label, "unit": unit, "desc": desc, "error": reason, "region": region}

    results = []

    # ── US: Fed Funds Rate ────────────────────────────────────────────────────
    d = fetch_fred_series("FEDFUNDS")
    results.append(_ok("Fed Funds Rate", "%", "US-Leitzins (Federal Reserve)", d, "US")
                   if d else _err("Fed Funds Rate", "%", "US-Leitzins (Federal Reserve)", "fred_unavailable", "US"))

    # ── US: CPI YoY ───────────────────────────────────────────────────────────
    d = fetch_fred_cpi_yoy()
    results.append(_ok("CPI YoY", "%", "Inflationsrate USA (YoY)", d, "US")
                   if d else _err("CPI YoY", "%", "Inflationsrate USA (YoY)", "fred_unavailable", "US"))

    # ── US: Unemployment ──────────────────────────────────────────────────────
    d = fetch_fred_series("UNRATE")
    results.append(_ok("Arbeitslosigkeit", "%", "US-Arbeitslosenquote", d, "US")
                   if d else _err("Arbeitslosigkeit", "%", "US-Arbeitslosenquote", "fred_unavailable", "US"))

    # ── US: Industrial Production YoY (INDPRO) ───────────────────────────────
    d = fetch_fred_indpro_yoy()
    results.append(_ok("Industrieproduktion (YoY)", "%", "US-Industrieproduktion Year-over-Year", d, "US")
                   if d else _err("Industrieproduktion (YoY)", "%", "US-Industrieproduktion Year-over-Year", "fred_unavailable", "US"))

    # ── US: M2 Money Supply ───────────────────────────────────────────────────
    d = fetch_fred_series("M2SL")
    results.append(_ok("M2 Geldmenge", "Mrd. $", "US-Geldmenge M2", d, "US")
                   if d else _err("M2 Geldmenge", "Mrd. $", "US-Geldmenge M2", "fred_unavailable", "US"))

    # ── US: Dollar Index (Yahoo Finance) ─────────────────────────────────────
    # TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source before commercial launch
    dxy = fetch_yahoo_quote("DX-Y.NYB")
    if dxy:
        v   = dxy["price"]
        chg = dxy["change"]
        results.append({
            "label": "US Dollar Index", "unit": "Punkte", "desc": "US Dollar Index (DXY)",
            "value": v, "change": chg, "change_pct": dxy["change_pct"], "region": "US",
            "trend": "up" if chg > 0.5 else "down" if chg < -0.5 else "neutral",
        })
    else:
        results.append(_err("US Dollar Index", "Punkte", "US Dollar Index (DXY)", "yahoo_unavailable", "US"))

    # ── EU: ECB Main Refinancing Rate ─────────────────────────────────────────
    d = fetch_ecb_series("FM", "D.U2.EUR.4F.KR.MRR_FR.LEV")
    results.append(_ok("ECB Leitzins", "%", "EZB Hauptrefinanzierungssatz", d, "EU")
                   if d else _err("ECB Leitzins", "%", "EZB Hauptrefinanzierungssatz", "ecb_unavailable", "EU"))

    # ── EU: Euro Area HICP Inflation ──────────────────────────────────────────
    d = fetch_ecb_series("ICP", "M.U2.N.000000.4.ANR")
    results.append(_ok("EU Inflation (HICP)", "%", "Eurozone Inflation YoY (HICP)", d, "EU")
                   if d else _err("EU Inflation (HICP)", "%", "Eurozone Inflation YoY (HICP)", "ecb_unavailable", "EU"))

    # ── EU: Euro Area Unemployment ────────────────────────────────────────────
    d = fetch_ecb_series("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T")
    results.append(_ok("EU Arbeitslosigkeit", "%", "Eurozone-Arbeitslosenquote", d, "EU")
                   if d else _err("EU Arbeitslosigkeit", "%", "Eurozone-Arbeitslosenquote", "ecb_unavailable", "EU"))

    output = {"timestamp": datetime.utcnow().isoformat(), "indicators": results}
    cache.set(cache_key, output)
    return output


@app.get("/global-indices")
def global_indices():
    """
    Current values for major global stock indices via EODHD.
    Includes plausibility checks — impossible values returned as error objects.
    Regions: EU, Asia, Global. Cached 5 min.
    """
    cache_key = "global_indices:v3"
    cached    = cache.get(cache_key, ttl_seconds=300)
    if cached is not None:
        return cached

    tickers = [item["ticker"] for item in GLOBAL_INDEX_ITEMS]
    snaps   = fetch_realtime(tickers)

    out = []
    for item in GLOBAL_INDEX_ITEMS:
        ticker = item["ticker"]
        label  = item["label"]
        region = item["region"]
        snap   = snaps.get(ticker)

        if not snap:
            logger.warning("Global index MISS %-15s (%s): no data", label, ticker)
            out.append({"label": label, "ticker": ticker, "region": region, "error": "no_data"})
            continue

        value  = snap["price"]
        change = snap["change"]
        note   = None

        # ISF.LSE tracks FTSE 100 at ~1:10 (GBX pence). Scale up to approximate index level.
        if ticker == "ISF.LSE":
            value  = round(value  * 10, 2)
            change = round(change * 10, 4)
            note   = "Berechnet aus ISF.LSE × 10 (Tracking ETF, ±0.5% Tracking-Error)"

        bounds = GLOBAL_INDEX_PLAUSIBILITY.get(label)
        if bounds and not (bounds[0] <= value <= bounds[1]):
            logger.error(
                "Global index PLAUSIBILITY FAIL %-15s: %.2f not in [%.0f, %.0f]",
                label, value, bounds[0], bounds[1],
            )
            out.append({
                "label": label, "ticker": ticker, "region": region,
                "error": "implausible_value", "raw_value": value,
            })
            continue

        entry = {
            "label":      label,
            "ticker":     ticker,
            "value":      value,
            "change":     change,
            "change_pct": snap["change_pct"],
            "region":     region,
        }
        if note:
            entry["note"] = note
        out.append(entry)
        logger.info("Global index OK %-15s (%s): %.2f", label, ticker, value)

    result = {"timestamp": datetime.utcnow().isoformat(), "indices": out}
    cache.set(cache_key, result)
    return result


# ─── Sector Holdings ─────────────────────────────────────────────────────────

@app.get("/sector-holdings/{ticker}")
def sector_holdings(ticker: str):
    t         = ticker.upper()
    cache_key = f"sector_holdings:{t}"
    cached    = cache.get(cache_key, ttl_seconds=86400)  # holdings change rarely
    if cached is not None:
        return cached

    is_eu = "XETRA" in t or t.startswith("EXH")
    eodhd_fmt = t if is_eu else f"{t}.US"

    try:
        data = eodhd_get(f"/fundamentals/{eodhd_fmt}", {"filter": "Components"})
    except Exception:
        result = {"available": False, "reason": "Holdings für diesen ETF nicht im EODHD-Plan verfügbar"}
        cache.set(cache_key, result)
        return result

    if not isinstance(data, dict) or not data:
        result = {"available": False, "reason": "Keine Holdings-Daten von EODHD"}
        cache.set(cache_key, result)
        return result

    holdings = []
    for code, comp in data.items():
        # EODHD returns weight as "Assets_%" string or "Weight" float
        raw = comp.get("Weight") or comp.get("Assets_%") or 0
        try:
            w = float(raw)
        except (ValueError, TypeError):
            w = 0
        # Values < 1 are fractions (0.21 = 21%); values > 1 are already percentages
        weight = round(w * 100 if w <= 1 else w, 2)
        if weight <= 0:
            continue
        holdings.append({
            "ticker": comp.get("Code", code),
            "name":   comp.get("Name", code),
            "weight": weight,
        })

    holdings.sort(key=lambda x: x["weight"], reverse=True)
    top10 = holdings[:10]

    result = {
        "ticker":       t,
        "holdings":     top10,
        "top10_weight": round(sum(h["weight"] for h in top10), 1),
    }
    cache.set(cache_key, result)
    return result


# ─── EU Sector Phase Detection ───────────────────────────────────────────────

@app.get("/sectors/eu-phase")
def sectors_eu_phase():
    """
    EU sector phase detection: EXH ETFs vs STOXX 50 benchmark (90-day RS).
    Returns phase, dominant sector, RS values. Cached 1 hour.
    """
    cache_key = "eu_phase:v1"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    EU_ETFS   = ["EXH4.XETRA", "EXH8.XETRA", "EXH3.XETRA", "EXH1.XETRA"]
    BENCH_KEY = "STOXX50E.INDX"
    PHASE_NAMES   = ["Frühzyklus", "Mittezyklus", "Spätzyklus", "Rezession"]
    DOMINANT_MAP  = {0: "EXH4 (Tech)", 1: "EXH7 (Industrie)", 2: "EXH8 (Energie)", 3: "EXH3 (Healthcare)"}

    try:
        frm, to   = _date_range(110)
        bench_df  = fetch_eod(BENCH_KEY, frm, to)
        if len(bench_df) < 20:
            raise ValueError("Insufficient benchmark data")
        bench_close = bench_df["Close"]

        rs_values: dict[str, float] = {}
        trends:    dict[str, str]   = {}

        for etf in EU_ETFS:
            try:
                etf_df  = fetch_eod(etf, frm, to)
                if len(etf_df) < 20:
                    continue
                common  = etf_df.index.intersection(bench_close.index)
                if len(common) < 20:
                    continue
                etf_c   = etf_df["Close"].loc[common].iloc[-60:]
                ben_c   = bench_close.loc[common].iloc[-60:]
                rs_ser  = (etf_c / etf_c.iloc[0] - 1) * 100 - (ben_c / ben_c.iloc[0] - 1) * 100
                rs_values[etf] = round(float(rs_ser.iloc[-1]), 2)
                if len(rs_ser) >= 40:
                    trends[etf] = "up" if rs_ser.iloc[-20:].mean() > rs_ser.iloc[-40:-20].mean() else "down"
                else:
                    trends[etf] = "flat"
            except Exception as e:
                logger.warning("EU phase RS failed for %s: %s", etf, e)

        tech   = rs_values.get("EXH4.XETRA", 0)
        energy = rs_values.get("EXH8.XETRA", 0)
        health = rs_values.get("EXH3.XETRA", 0)
        fin    = rs_values.get("EXH1.XETRA", 0)

        if health > max(tech, energy, fin) and health > 0:
            phase_idx = 3
        elif energy > 0 and fin > 0 and trends.get("EXH8.XETRA") == "up":
            phase_idx = 2
        elif tech > 0 and trends.get("EXH4.XETRA") == "up":
            phase_idx = 0
        else:
            phase_idx = 1

        result = {
            "phase":          PHASE_NAMES[phase_idx],
            "phase_idx":      phase_idx,
            "dominant_sector": DOMINANT_MAP[phase_idx],
            "rs_values":      {k.split(".")[0]: v for k, v in rs_values.items()},
            "trends":         {k.split(".")[0]: v for k, v in trends.items()},
            "benchmark":      BENCH_KEY,
            "timestamp":      datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("EU phase detection failed: %s", e)
        result = {
            "phase": "Mittezyklus", "phase_idx": 1,
            "dominant_sector": "EXH7 (Industrie)",
            "rs_values": {}, "trends": {}, "benchmark": BENCH_KEY,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }

    cache.set(cache_key, result)
    return result


# ─── Ticker Fundamentals ──────────────────────────────────────────────────────

@app.get("/ticker-fundamentals/{ticker}")
def ticker_fundamentals(ticker: str):
    t         = ticker.upper()
    cache_key = f"ticker_fundamentals:{t}"

    # Only serve from cache when fundamentals fields are non-null; if the cached
    # result has all-null fundamentals it means a previous call failed — skip it
    # so we retry EODHD on the next request.
    cached = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        has_cached_fund = any(cached.get(k) is not None for k in ("market_cap", "pe_ratio", "eps"))
        if has_cached_fund:
            return cached
        logger.info("Skipping stale null-fundamentals cache for %s, retrying EODHD", t)

    eodhd_t    = t if "." in t else f"{t}.US"
    highlights = _fetch_fundamentals_highlights(eodhd_t)

    def _safe(key):
        v = highlights.get(key)
        try: return float(v) if v not in (None, "", "None") else None
        except (ValueError, TypeError): return None

    # Real-time quote — always available
    quote = _fetch_rt_one(eodhd_t)

    # Fallback 52W range from 1Y history when fundamentals unavailable
    high_52w = _safe("52WeekHigh")
    low_52w  = _safe("52WeekLow")
    perf_1y  = None
    if high_52w is None or low_52w is None:
        try:
            frm, to = _date_range(380)
            df      = fetch_eod(eodhd_t, frm, to)
            series  = df["Close"].dropna()
            if len(series) >= 10:
                high_52w = round(float(series.max()), 4)
                low_52w  = round(float(series.min()), 4)
                if len(series) >= 252:
                    perf_1y = round((float(series.iloc[-1]) / float(series.iloc[-252]) - 1) * 100, 2)
        except Exception:
            pass

    if quote is None and high_52w is None:
        raise HTTPException(status_code=404, detail=f"No data available for {t}")

    result = {
        "ticker":        t,
        "price":         quote["price"]      if quote else None,
        "change":        quote["change"]     if quote else None,
        "change_pct":    quote["change_pct"] if quote else None,
        "market_cap":    _safe("MarketCapitalization"),
        "pe_ratio":      _safe("PERatio"),
        "eps":           _safe("EarningsShare"),
        "high_52w":      high_52w,
        "low_52w":       low_52w,
        "revenue_ttm":   _safe("RevenueTTM"),
        "target_price":  _safe("WallStreetTargetPrice"),
        "profit_margin": _safe("ProfitMargin"),
        "perf_1y":       perf_1y,
    }

    # Only cache with full TTL when we actually got fundamentals; if all null,
    # cache with a very short effective TTL by back-dating the timestamp so the
    # entry expires in ~60 s instead of 3600 s.
    cache.set(cache_key, result)
    has_fundamentals = any(result.get(k) is not None for k in ("market_cap", "pe_ratio", "eps"))
    if not has_fundamentals:
        cache._store[cache_key] = (result, time.time() - 3540)  # expires in ~60 s

    return result


# ─── Autocomplete ────────────────────────────────────────────────────────────

@app.get("/autocomplete/{query}")
def autocomplete(query: str):
    """
    Ticker/name autocomplete via EODHD search. Returns up to 8 results with
    ticker (Code.Exchange), name, and asset type. Cached 1 h per query.
    """
    q = query.strip()
    if len(q) < 2:
        return {"results": []}

    cache_key = f"autocomplete:{q.upper()}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    try:
        data = eodhd_get(f"/search/{q}", {"limit": 10})
    except Exception as exc:
        logger.warning("Autocomplete search failed for %r: %s", q, exc)
        return {"results": []}

    if not isinstance(data, list):
        return {"results": []}

    results = []
    for item in data[:8]:
        code     = item.get("Code", "")
        exchange = item.get("Exchange", "") or ""
        if not code:
            continue
        ticker = f"{code}.{exchange}" if exchange and exchange.upper() not in ("", "NONE") else code
        results.append({
            "ticker":   ticker,
            "name":     item.get("Name", ""),
            "type":     item.get("Type", ""),
            "exchange": exchange,
        })

    result = {"results": results}
    cache.set(cache_key, result)
    return result


# ─── Risk Metrics ─────────────────────────────────────────────────────────────

@app.get("/risk-metrics/{ticker}")
def risk_metrics(ticker: str):
    t         = ticker.strip().upper()
    cache_key = f"risk_metrics:{t}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    eodhd_t = t if "." in t else f"{t}.US"

    try:
        df    = fetch_history(t, days=380)
        close = df["Close"].dropna()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No history for {t}: {e}")

    try:
        spy_df    = fetch_history(BENCHMARK, days=380)
        spy_close = spy_df["Close"].dropna()
    except Exception:
        spy_close = pd.Series(dtype=float)

    quote = None
    try:
        snaps = fetch_realtime([t])
        quote = snaps.get(t)
    except Exception:
        pass

    n              = len(close)
    daily_returns  = close.pct_change().dropna()

    perf_1m = round(float((close.iloc[-1] / close.iloc[-21]  - 1) * 100), 2) if n >= 21  else None
    perf_3m = round(float((close.iloc[-1] / close.iloc[-63]  - 1) * 100), 2) if n >= 63  else None
    perf_1y = round(float((close.iloc[-1] / close.iloc[-252] - 1) * 100), 2) if n >= 252 else None

    vol_1y = None
    if len(daily_returns) >= 30:
        window = daily_returns.iloc[-252:] if len(daily_returns) >= 252 else daily_returns
        vol_1y = round(float(window.std() * np.sqrt(252) * 100), 2)

    max_drawdown = None
    if n >= 30:
        prices_win = close.iloc[-252:] if n >= 252 else close
        cum_max    = prices_win.cummax()
        drawdown   = (prices_win - cum_max) / cum_max
        max_drawdown = round(float(drawdown.min() * 100), 2)

    beta_vs_spy = None
    if len(spy_close) >= 30 and len(daily_returns) >= 30:
        spy_ret = spy_close.pct_change().dropna()
        common  = daily_returns.index.intersection(spy_ret.index)
        if len(common) >= 30:
            s = daily_returns.loc[common].iloc[-252:]
            b = spy_ret.loc[common].iloc[-252:]
            c2 = s.index.intersection(b.index)
            s, b = s.loc[c2], b.loc[c2]
            var = float(np.var(b))
            if var:
                beta_vs_spy = round(float(np.cov(s, b)[0, 1]) / var, 2)

    high_52w = round(float(close.max()), 4) if n >= 1 else None
    low_52w  = round(float(close.min()), 4) if n >= 1 else None

    result = {
        "ticker":          t,
        "price":           quote["price"]      if quote else None,
        "change_pct":      quote["change_pct"] if quote else None,
        "perf_1m":         perf_1m,
        "perf_3m":         perf_3m,
        "perf_1y":         perf_1y,
        "volatility_1y":   vol_1y,
        "max_drawdown_1y": max_drawdown,
        "beta_vs_spy":     beta_vs_spy,
        "high_52w":        high_52w,
        "low_52w":         low_52w,
    }
    cache.set(cache_key, result)
    return result


# ─── User: Pydantic models ────────────────────────────────────────────────────

class WatchlistAdd(BaseModel):
    ticker: str

class PositionAdd(BaseModel):
    ticker: str
    shares: float
    cost_basis: float

class IpsUpsert(BaseModel):
    goal: str | None = None
    time_horizon: str | None = None
    risk_tolerance: str | None = None
    asset_allocation: dict | None = None
    exclusions: str | None = None

class MemoSave(BaseModel):
    ticker: str
    memo_text: str

# ─── User: Watchlist ──────────────────────────────────────────────────────────

@app.get("/user/watchlist")
def get_watchlist(user_id: str = Depends(verify_jwt)):
    result = sb_client.table("watchlists").select("ticker,added_at").eq("user_id", user_id).order("added_at").execute()
    return result.data


@app.post("/user/watchlist", status_code=201)
def add_watchlist(body: WatchlistAdd, user_id: str = Depends(verify_jwt)):
    ticker = body.ticker.strip().upper()
    try:
        result = sb_client.table("watchlists").insert({"user_id": user_id, "ticker": ticker}).execute()
        return result.data[0]
    except Exception as e:
        msg = str(e).lower()
        if "duplicate" in msg or "unique" in msg or "23505" in msg:
            raise HTTPException(status_code=409, detail="Ticker bereits in Watchlist")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/user/watchlist/{ticker}")
def remove_watchlist(ticker: str, user_id: str = Depends(verify_jwt)):
    sb_client.table("watchlists").delete().eq("user_id", user_id).eq("ticker", ticker.upper()).execute()
    return {"ok": True}

# ─── User: Portfolio ──────────────────────────────────────────────────────────

@app.get("/user/portfolio")
def get_portfolio(user_id: str = Depends(verify_jwt)):
    portfolios = sb_client.table("portfolios").select("id,name,created_at").eq("user_id", user_id).execute()
    if not portfolios.data:
        new_p = sb_client.table("portfolios").insert({"user_id": user_id, "name": "Hauptdepot"}).execute()
        p = new_p.data[0]
        return [{"id": p["id"], "name": p["name"], "positions": []}]
    result = []
    for p in portfolios.data:
        positions = sb_client.table("positions").select("id,ticker,shares,cost_basis,added_at").eq("portfolio_id", p["id"]).execute()
        result.append({"id": p["id"], "name": p["name"], "positions": positions.data})
    return result


@app.post("/user/portfolio/{portfolio_id}/position", status_code=201)
def add_position(portfolio_id: str, body: PositionAdd, user_id: str = Depends(verify_jwt)):
    owner = sb_client.table("portfolios").select("id").eq("id", portfolio_id).eq("user_id", user_id).execute()
    if not owner.data:
        raise HTTPException(status_code=404, detail="Portfolio nicht gefunden")
    result = sb_client.table("positions").insert({
        "portfolio_id": portfolio_id,
        "ticker": body.ticker.strip().upper(),
        "shares": body.shares,
        "cost_basis": body.cost_basis,
    }).execute()
    return result.data[0]


@app.delete("/user/portfolio/position/{position_id}")
def delete_position(position_id: str, user_id: str = Depends(verify_jwt)):
    pos = sb_client.table("positions").select("portfolio_id").eq("id", position_id).execute()
    if not pos.data:
        raise HTTPException(status_code=404, detail="Position nicht gefunden")
    owner = sb_client.table("portfolios").select("id").eq("id", pos.data[0]["portfolio_id"]).eq("user_id", user_id).execute()
    if not owner.data:
        raise HTTPException(status_code=403, detail="Keine Berechtigung")
    sb_client.table("positions").delete().eq("id", position_id).execute()
    return {"ok": True}

# ─── User: IPS ────────────────────────────────────────────────────────────────

@app.get("/user/ips")
def get_ips(user_id: str = Depends(verify_jwt)):
    result = sb_client.table("ips").select("*").eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


@app.put("/user/ips")
def upsert_ips(body: IpsUpsert, user_id: str = Depends(verify_jwt)):
    data = {
        "user_id": user_id,
        "goal": body.goal,
        "time_horizon": body.time_horizon,
        "risk_tolerance": body.risk_tolerance,
        "asset_allocation": body.asset_allocation,
        "exclusions": body.exclusions,
        "updated_at": datetime.utcnow().isoformat(),
    }
    result = sb_client.table("ips").upsert(data, on_conflict="user_id").execute()
    return result.data[0]

# ─── User: Memos ──────────────────────────────────────────────────────────────

@app.post("/user/memos", status_code=201)
def save_memo(body: MemoSave, user_id: str = Depends(verify_jwt)):
    result = sb_client.table("saved_memos").insert({
        "user_id": user_id,
        "ticker": body.ticker.strip().upper(),
        "memo_text": body.memo_text,
    }).execute()
    return result.data[0]


@app.get("/user/memos")
def get_memos(user_id: str = Depends(verify_jwt)):
    result = (
        sb_client.table("saved_memos")
        .select("id,ticker,memo_text,generated_at")
        .eq("user_id", user_id)
        .order("generated_at", desc=True)
        .execute()
    )
    return result.data


@app.delete("/user/memos/{memo_id}")
def delete_memo(memo_id: str, user_id: str = Depends(verify_jwt)):
    memo = sb_client.table("saved_memos").select("id").eq("id", memo_id).eq("user_id", user_id).execute()
    if not memo.data:
        raise HTTPException(status_code=404, detail="Memo nicht gefunden")
    sb_client.table("saved_memos").delete().eq("id", memo_id).execute()
    return {"ok": True}

# ─── Academy ──────────────────────────────────────────────────────────────────

class ProgressBody(BaseModel):
    lesson_id: str
    quiz_score: int = 0

class SpecializationBody(BaseModel):
    track_id: str

ACADEMY_TRACKS = [
    {
        "id": "foundation",
        "title": "Foundation",
        "subtitle": "Grundlagen des Investierens",
        "description": "Verstehen Sie die Bausteine der Kapitalmärkte: Aktien, Börsen, Indizes und Bewertungsmethoden.",
        "icon": "📚",
        "color": "#4C2EE5",
        "total_lessons": 15,
        "status": "available",
        "category": "foundation",
    },
    {
        "id": "analyst",
        "title": "Analyst",
        "subtitle": "Fundamentalanalyse & Unternehmensbewertung",
        "description": "DCF-Modelle, Bilanzen lesen, Moats identifizieren — wie ein professioneller Aktienanalyst denken.",
        "icon": "🔍",
        "color": "#4C2EE5",
        "total_lessons": 10,
        "status": "available",
        "category": "specialization",
        "prerequisite": "Empfohlen: Foundation Track abgeschlossen",
    },
    {
        "id": "swing-trader",
        "title": "Swing Trader",
        "subtitle": "Technische Analyse & kurzfristiges Trading",
        "description": "Chartmuster, Momentum-Indikatoren, Stop-Loss und Position Sizing für aktive Trader.",
        "icon": "📊",
        "color": "#00d4a0",
        "total_lessons": 9,
        "status": "available",
        "category": "specialization",
        "prerequisite": "Empfohlen: Foundation Track abgeschlossen",
    },
    {
        "id": "macro_investor",
        "title": "Makro Investor",
        "subtitle": "Große wirtschaftliche Kräfte verstehen",
        "description": "Fed-Zyklen, Yield Curve, Währungen und Rohstoffe — verstehe die Makro-Kräfte die alle Märkte bewegen.",
        "icon": "🌍",
        "color": "#0ea5e9",
        "total_lessons": 6,
        "status": "available",
        "category": "specialization",
        "prerequisite": "Empfohlen: Foundation Track abgeschlossen",
    },
    {
        "id": "dividenden_investor",
        "title": "Dividenden Investor",
        "subtitle": "Passives Einkommen durch Qualitätsdividenden",
        "description": "Aufbau eines stabilen Einkommensstroms: Dividend Aristocrats, Payout Ratio, DRIP und Portfoliokonstruktion.",
        "icon": "💰",
        "color": "#22c55e",
        "total_lessons": 6,
        "status": "available",
        "category": "specialization",
        "prerequisite": "Empfohlen: Foundation Track abgeschlossen",
    },
    {
        "id": "praktische_uebungen",
        "title": "Praktische Übungen",
        "subtitle": "Echte Fallstudien — analysiere wie ein Profi",
        "description": "15 interaktive Fallstudien mit echten Unternehmenskennzahlen. Von PEG-Ratio-Rechnung bis Wirecard-Red-Flags — lerne durch Anwendung statt durch Theorie.",
        "icon": "🔬",
        "color": "#f59e0b",
        "total_lessons": 15,
        "status": "available",
        "category": "practical",
    },
    {
        "id": "technical",
        "title": "Technische Analyse",
        "subtitle": "Charts lesen und interpretieren",
        "description": "Candlestick-Muster, Indikatoren und Trendanalyse.",
        "icon": "📈",
        "color": "#00d4a0",
        "total_lessons": 8,
        "status": "coming_soon",
        "category": "coming_soon",
    },
]

ACADEMY_LESSONS_DATA = {
    "foundation": [
        {
            "id": "foundation-1.1",
            "track_id": "foundation",
            "title": "Was ist eine Aktie?",
            "subtitle": "Eigenkapital, Stimmrechte und Dividenden",
            "duration_min": 8,
            "sort_order": 1,
            "status": "available",
            "learning_goals": [
                "Verstehen, was eine Aktie rechtlich und wirtschaftlich darstellt",
                "Den Unterschied zwischen Stammaktien und Vorzugsaktien kennen",
                "Wissen, wie Dividenden entstehen und ausgezahlt werden",
            ],
            "sections": [
                {
                    "type": "text",
                    "heading": "Was ist eine Aktie?",
                    "body": "Eine Aktie ist ein Wertpapier, das einen Eigentumsanteil an einem Unternehmen verbrieft. Wenn Sie eine Aktie von Apple kaufen, werden Sie zu einem Miteigentümer von Apple Inc. — auch wenn Ihr Anteil nur einen winzigen Bruchteil des gesamten Unternehmens ausmacht.\n\nAktiengesellschaften (AG) teilen ihr Eigenkapital in gleich große Anteile auf, die als Aktien bezeichnet werden. Die Gesamtzahl dieser Anteile ergibt die sogenannte Aktienanzahl oder \"Shares Outstanding\"."
                },
                {
                    "type": "info_box",
                    "heading": "Kernbegriff: Aktie",
                    "body": "Eine Aktie = ein Bruchteil des Eigenkapitals eines Unternehmens. Als Aktionär sind Sie Miteigentümer, haben Stimmrechte auf Hauptversammlungen und nehmen an Gewinnen (Dividenden) sowie Wertsteigerungen teil."
                },
                {
                    "type": "live_widget",
                    "widget_type": "ticker_card",
                    "ticker": "AAPL",
                    "label": "Apple Inc. — eines der wertvollsten Unternehmen der Welt"
                },
                {
                    "type": "text",
                    "heading": "Warum gehen Unternehmen an die Börse?",
                    "body": "Wenn ein Unternehmen wächst und Kapital benötigt, kann es an die Börse gehen — ein Vorgang, der als Börsengang (IPO — Initial Public Offering) bezeichnet wird. Das Unternehmen gibt neue Aktien aus und erhält dafür frisches Kapital von Investoren.\n\nFür Anleger entsteht so die Möglichkeit, an der Wertentwicklung erfolgreicher Unternehmen zu partizipieren. Der Preis einer Aktie spiegelt dabei wider, was Marktteilnehmer bereit sind, für diesen Eigentumsanteil zu zahlen."
                },
                {
                    "type": "info_box",
                    "heading": "Stammaktie vs. Vorzugsaktie",
                    "body": "Stammaktien (Common Shares) geben dem Inhaber Stimmrechte auf der Hauptversammlung. Vorzugsaktien (Preferred Shares) haben häufig kein Stimmrecht, dafür aber eine höhere Dividendenpriorität. Die meisten an der Börse gehandelten Aktien sind Stammaktien."
                },
                {
                    "type": "app_link",
                    "label": "→ Anwenden in Finscope: AAPL im Screener analysieren",
                    "target_tab": "screener",
                    "target_ticker": "AAPL"
                },
            ],
            "quiz": [
                {
                    "id": "q1",
                    "type": "multiple_choice",
                    "question": "Was stellt eine Aktie rechtlich dar?",
                    "options": [
                        "Ein Darlehen an das Unternehmen",
                        "Einen Eigentumsanteil am Unternehmen",
                        "Eine Versicherungspolice gegen Kursverluste",
                        "Ein Recht auf feste Zinszahlungen",
                    ],
                    "correct": 1,
                    "explanation": "Richtig! Eine Aktie ist ein Eigentumsanteil. Als Aktionär sind Sie Miteigentümer des Unternehmens — mit allen Chancen und Risiken, die das mit sich bringt."
                },
                {
                    "id": "q2",
                    "type": "multiple_choice",
                    "question": "Was ist eine Dividende?",
                    "options": [
                        "Eine Strafgebühr bei Kursverlusten",
                        "Der Kursgewinn beim Verkauf einer Aktie",
                        "Eine Gewinnausschüttung des Unternehmens an seine Aktionäre",
                        "Die Differenz zwischen Kauf- und Verkaufspreis",
                    ],
                    "correct": 2,
                    "explanation": "Korrekt. Eine Dividende ist ein Teil des Unternehmensgewinns, den das Management beschließt, an die Aktionäre auszuschütten. Nicht alle Unternehmen zahlen Dividenden — viele wachstumsstarke Firmen reinvestieren ihre Gewinne lieber."
                },
                {
                    "id": "q3",
                    "type": "multiple_choice",
                    "question": "Was passiert bei einem IPO (Initial Public Offering)?",
                    "options": [
                        "Ein Unternehmen gibt neue Aktien aus und erhält Kapital von Investoren",
                        "Ein Unternehmen kauft eigene Aktien vom Markt zurück",
                        "Zwei Unternehmen fusionieren zu einem neuen Unternehmen",
                        "Ein Unternehmen wird von der Börse genommen",
                    ],
                    "correct": 0,
                    "explanation": "Genau. Beim IPO — dem Börsengang — bietet ein Unternehmen erstmals Aktien an der Börse an. Es erhält frisches Kapital, und Anleger erhalten die Möglichkeit, Miteigentümer zu werden."
                },
            ],
        },
        {
            "id": "foundation-1.2",
            "track_id": "foundation",
            "title": "Wie funktioniert die Börse?",
            "subtitle": "Marktmechanismen, Ordertypen und Liquidität",
            "duration_min": 10,
            "sort_order": 2,
            "status": "available",
            "learning_goals": [
                "Verstehen, wie Käufer und Verkäufer an der Börse zusammentreffen",
                "Die wichtigsten Ordertypen (Market, Limit, Stop) kennen",
                "Den Bid-Ask-Spread verstehen und seine Bedeutung einschätzen",
            ],
            "sections": [
                {
                    "type": "text",
                    "heading": "Die Börse als organisierter Marktplatz",
                    "body": "Eine Börse ist ein regulierter Marktplatz, auf dem Wertpapiere wie Aktien, Anleihen und ETFs gehandelt werden. Die bekanntesten Börsen der Welt sind die New York Stock Exchange (NYSE) und die NASDAQ in den USA sowie die Deutsche Börse (XETRA) in Frankfurt.\n\nModerne Börsen funktionieren elektronisch: Millionen von Kauf- und Verkaufsaufträgen werden von Computern in Millisekunden verarbeitet und zusammengeführt."
                },
                {
                    "type": "info_box",
                    "heading": "Bid & Ask — das Herzstück jedes Marktes",
                    "body": "Der Bid-Kurs ist der höchste Preis, den ein Käufer gerade zu zahlen bereit ist. Der Ask-Kurs (auch Brief) ist der niedrigste Preis, zu dem ein Verkäufer gerade verkaufen möchte. Die Differenz zwischen beiden nennt sich Spread. Ein enger Spread bedeutet hohe Liquidität."
                },
                {
                    "type": "live_widget",
                    "widget_type": "mini_chart",
                    "ticker": "SPY",
                    "label": "S&P 500 ETF (SPY) — der meistgehandelte ETF der Welt"
                },
                {
                    "type": "text",
                    "heading": "Ordertypen: Wie Sie handeln",
                    "body": "**Market Order:** Eine Market Order wird sofort zum aktuellen Marktpreis ausgeführt. Sie ist schnell, aber Sie haben keine Kontrolle über den genauen Preis.\n\n**Limit Order:** Sie legen einen Maximalpreis (beim Kauf) oder Minimalpreis (beim Verkauf) fest. Die Order wird nur ausgeführt, wenn der Marktpreis Ihren Limit erreicht.\n\n**Stop-Loss Order:** Wenn der Kurs unter einen bestimmten Wert fällt, wird automatisch verkauft. Dient dem Verlustbegrenzen."
                },
                {
                    "type": "info_box",
                    "heading": "Wann ist die Börse geöffnet?",
                    "body": "NYSE/NASDAQ: Montag–Freitag, 9:30–16:00 Uhr Eastern Time (15:30–22:00 Uhr MEZ). XETRA (Frankfurt): 9:00–17:30 Uhr MEZ. Außerhalb dieser Zeiten gibt es Pre-Market und After-Hours Handel mit geringerer Liquidität."
                },
                {
                    "type": "app_link",
                    "label": "→ Anwenden in Finscope: Aktuelle Marktdaten im Dashboard",
                    "target_tab": "dashboard",
                    "target_ticker": None
                },
            ],
            "quiz": [
                {
                    "id": "q1",
                    "type": "multiple_choice",
                    "question": "Was ist der Bid-Ask-Spread?",
                    "options": [
                        "Die jährliche Rendite einer Aktie",
                        "Die Differenz zwischen dem höchsten Kaufpreis und dem niedrigsten Verkaufspreis",
                        "Die Verwaltungsgebühr eines ETFs",
                        "Der Unterschied zwischen Eröffnungs- und Schlusskurs",
                    ],
                    "correct": 1,
                    "explanation": "Genau. Der Bid-Ask-Spread ist die Differenz zwischen dem Bid (höchster Kaufpreis) und dem Ask (niedrigster Verkaufspreis). Ein enger Spread signalisiert hohe Liquidität und niedrige Transaktionskosten."
                },
                {
                    "id": "q2",
                    "type": "multiple_choice",
                    "question": "Was ist der Vorteil einer Limit Order gegenüber einer Market Order?",
                    "options": [
                        "Sie wird immer sofort ausgeführt",
                        "Sie gibt dem Anleger Kontrolle über den Ausführungspreis",
                        "Sie ist immer günstiger als eine Market Order",
                        "Sie schützt automatisch vor Kursverlusten",
                    ],
                    "correct": 1,
                    "explanation": "Richtig. Mit einer Limit Order legen Sie den Maximalpreis (Kauf) oder Mindestpreis (Verkauf) fest. Sie vermeiden Preisüberraschungen, riskieren aber, dass die Order nicht ausgeführt wird, wenn der Marktpreis Ihr Limit nicht erreicht."
                },
                {
                    "id": "q3",
                    "type": "multiple_choice",
                    "question": "Was ist Liquidität an der Börse?",
                    "options": [
                        "Das Bargeld, das ein Unternehmen in der Kasse hat",
                        "Die Möglichkeit, ein Wertpapier schnell zu einem fairen Preis zu kaufen oder zu verkaufen",
                        "Die Dividendenrendite einer Aktie",
                        "Der Buchwert des Eigenkapitals",
                    ],
                    "correct": 1,
                    "explanation": "Korrekt. Liquidität beschreibt, wie leicht ein Wertpapier gehandelt werden kann, ohne den Kurs stark zu beeinflussen. Hochliquide Wertpapiere wie Apple-Aktien oder SPY-ETFs haben enge Spreads und hohe Handelsvolumen."
                },
            ],
        },
        {
            "id": "foundation-1.3",
            "track_id": "foundation",
            "title": "Verstehe den S&P 500",
            "subtitle": "Aufbau, Gewichtung und historische Bedeutung",
            "duration_min": 7,
            "sort_order": 3,
            "status": "available",
            "learning_goals": [
                "Den Aufbau und die Funktion des S&P 500 verstehen",
                "Das Konzept der marktkapitalisierungsgewichteten Indizes kennen",
                "Die wichtigsten Sektoren des S&P 500 benennen können",
            ],
            "sections": [
                {
                    "type": "text",
                    "heading": "Der S&P 500 — Maßstab der US-Wirtschaft",
                    "body": "Der S&P 500 (Standard & Poor's 500) ist der wichtigste Aktienindex der USA und gilt weltweit als Referenzmaßstab für die Entwicklung des Aktienmarkts. Er umfasst die 500 größten börsennotierten US-Unternehmen nach Marktkapitalisierung.\n\nDer Index wird von S&P Dow Jones Indices verwaltet und quartalsweise überprüft. Unternehmen, die nicht mehr die Kriterien erfüllen, werden durch neue ersetzt."
                },
                {
                    "type": "live_widget",
                    "widget_type": "ticker_card",
                    "ticker": "SPY",
                    "label": "SPDR S&P 500 ETF Trust (SPY) — bildet den S&P 500 nach"
                },
                {
                    "type": "info_box",
                    "heading": "Marktkapitalisierungsgewichtung",
                    "body": "Im S&P 500 werden Unternehmen nach ihrer Marktkapitalisierung gewichtet. Marktkapitalisierung = Aktienkurs × Anzahl ausstehender Aktien. Größere Unternehmen wie Apple, Microsoft und NVIDIA haben deshalb einen höheren Einfluss auf die Indexentwicklung als kleinere Unternehmen."
                },
                {
                    "type": "text",
                    "heading": "Die 11 Sektoren des S&P 500",
                    "body": "Der S&P 500 ist in 11 Sektoren aufgeteilt. Derzeit (Stand 2025) dominiert Technologie mit über 30% Gewichtung, gefolgt von Finanzwesen und Gesundheit.\n\nDiese Sektorverteilung ist für Anleger wichtig, weil sie zeigt, welche Branchen den US-Markt antreiben — und welche Risiken damit verbunden sind. Wenn Technologieaktien fallen, fällt der S&P 500 meist stärker als andere Indizes."
                },
                {
                    "type": "app_link",
                    "label": "→ Anwenden in Finscope: Sektor-Rotation im Sektoren-Tab analysieren",
                    "target_tab": "sektoren",
                    "target_ticker": None
                },
            ],
            "quiz": [
                {
                    "id": "q1",
                    "type": "multiple_choice",
                    "question": "Wie viele Unternehmen sind im S&P 500 enthalten?",
                    "options": ["100", "500", "1000", "30"],
                    "correct": 1,
                    "explanation": "Richtig! Der S&P 500 umfasst die 500 größten börsennotierten US-Unternehmen nach Marktkapitalisierung."
                },
                {
                    "id": "q2",
                    "type": "multiple_choice",
                    "question": "Was bedeutet marktkapitalisierungsgewichteter Index?",
                    "options": [
                        "Alle Unternehmen haben das gleiche Gewicht",
                        "Größere Unternehmen haben einen stärkeren Einfluss auf den Index",
                        "Der Index wird täglich neu gewichtet",
                        "Nur profitable Unternehmen werden berücksichtigt",
                    ],
                    "correct": 1,
                    "explanation": "Genau. Bei einem marktkapitalisierungsgewichteten Index wie dem S&P 500 bestimmt die Marktkapitalisierung (Kurs × Aktienanzahl) das Gewicht. Apple ist daher deutlich einflussreicher als ein kleineres Unternehmen."
                },
                {
                    "id": "q3",
                    "type": "multiple_choice",
                    "question": "Welcher Sektor hat derzeit das größte Gewicht im S&P 500?",
                    "options": ["Gesundheit", "Energie", "Technologie", "Finanzen"],
                    "correct": 2,
                    "explanation": "Korrekt. Technologie dominiert den S&P 500 mit über 30% Gewichtung (Stand 2025). Das bedeutet: Wenn Tech-Aktien wie Apple, Microsoft und NVIDIA fallen, zieht das den gesamten Index nach unten."
                },
            ],
        },
    ]
}


def _load_academy_content():
    """Load lesson JSON files from academy_content/ and merge into ACADEMY_LESSONS_DATA.
    Supports both single-lesson dicts and arrays of lessons per file."""
    import glob as _glob
    content_dir = os.path.join(os.path.dirname(__file__), "academy_content")
    existing_ids = {
        l["id"]
        for lessons in ACADEMY_LESSONS_DATA.values()
        for l in lessons
    }
    for filepath in sorted(_glob.glob(os.path.join(content_dir, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            lessons = data if isinstance(data, list) else [data]
            for lesson in lessons:
                if lesson.get("id") in existing_ids:
                    continue
                track_id = lesson.get("track_id", "foundation")
                ACADEMY_LESSONS_DATA.setdefault(track_id, []).append(lesson)
                existing_ids.add(lesson["id"])
        except Exception as exc:
            print(f"Warning: could not load {filepath}: {exc}")
    for track_lessons in ACADEMY_LESSONS_DATA.values():
        track_lessons.sort(key=lambda l: l.get("sort_order", 999))


_load_academy_content()


@app.get("/academy/tracks")
def get_tracks():
    return ACADEMY_TRACKS


@app.get("/academy/lessons/{track_id}")
def get_lessons(track_id: str):
    lessons = ACADEMY_LESSONS_DATA.get(track_id)
    if lessons is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' nicht gefunden")
    return [
        {k: v for k, v in l.items() if k not in ("sections", "quiz")}
        for l in lessons
    ]


@app.get("/academy/lesson/{lesson_id}")
def get_lesson(lesson_id: str):
    for track_lessons in ACADEMY_LESSONS_DATA.values():
        for l in track_lessons:
            if l["id"] == lesson_id:
                if l["status"] == "coming_soon":
                    raise HTTPException(status_code=403, detail="Diese Lesson ist noch nicht verfügbar")
                return l
    raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' nicht gefunden")


@app.get("/academy/progress")
def get_progress(user_id: str = Depends(verify_jwt)):
    result = (
        sb_client.table("academy_progress")
        .select("lesson_id,completed_at,quiz_score")
        .eq("user_id", user_id)
        .execute()
    )
    return result.data


@app.post("/academy/progress", status_code=201)
def save_progress(body: ProgressBody, user_id: str = Depends(verify_jwt)):
    try:
        sb_client.table("academy_progress").upsert(
            {"user_id": user_id, "lesson_id": body.lesson_id, "quiz_score": body.quiz_score},
            on_conflict="user_id,lesson_id"
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


@app.get("/academy/dashboard")
def get_academy_dashboard(user_id: str = Depends(verify_jwt)):
    from datetime import date as _date
    all_available = [
        l for lessons in ACADEMY_LESSONS_DATA.values() for l in lessons
        if l["status"] == "available"
    ]
    lesson_by_id = {l["id"]: l for l in all_available}

    try:
        rows = (
            sb_client.table("academy_progress")
            .select("lesson_id,completed_at,quiz_score")
            .eq("user_id", user_id)
            .execute()
        ).data or []
    except Exception:
        rows = []

    completed_ids = {r["lesson_id"] for r in rows}
    today_str = _date.today().isoformat()
    today_rows = [r for r in rows if (r.get("completed_at") or "")[:10] == today_str]
    today_minutes = sum(
        lesson_by_id[r["lesson_id"]]["duration_min"]
        for r in today_rows if r["lesson_id"] in lesson_by_id
    )

    track_completions: dict[str, int] = {}
    for r in rows:
        if r["lesson_id"] in lesson_by_id:
            tid = lesson_by_id[r["lesson_id"]]["track_id"]
            track_completions[tid] = track_completions.get(tid, 0) + 1

    active_tracks = list(track_completions.keys())
    search_tracks = active_tracks if active_tracks else ["foundation"]

    next_lesson = None
    for tid in search_tracks:
        for l in ACADEMY_LESSONS_DATA.get(tid, []):
            if l["status"] == "available" and l["id"] not in completed_ids:
                next_lesson = {k: l[k] for k in ("id", "title", "subtitle", "track_id", "duration_min", "sort_order")}
                break
        if next_lesson:
            break

    track_progress = {
        tid: {
            "completed": sum(1 for l in lessons if l["status"] == "available" and l["id"] in completed_ids),
            "available": sum(1 for l in lessons if l["status"] == "available"),
        }
        for tid, lessons in ACADEMY_LESSONS_DATA.items()
    }

    return {
        "total_completed": len(completed_ids),
        "total_available": len(all_available),
        "today_minutes": today_minutes,
        "today_lessons": [r["lesson_id"] for r in today_rows],
        "active_tracks": active_tracks,
        "next_lesson": next_lesson,
        "track_progress": track_progress,
    }


@app.post("/academy/evaluate-answer")
async def evaluate_answer(request: Request):
    body = await request.json()
    lesson_title = body.get("lesson_title", "")
    question     = body.get("question", "")
    user_answer  = body.get("answer", "")
    user_id      = body.get("user_id", "")

    prompt = f"""Du bist ein strenger aber fairer Finanz-Tutor. Bewerte die folgende Antwort eines Lernenden.

Lektion: {lesson_title}
Frage: {question}
Antwort des Lernenden: {user_answer}

Bewerte auf einer Skala von 0-100 und antworte NUR als JSON (kein Markdown, keine Codeblöcke):
{{
  "score": <0-100>,
  "passed": <true wenn score >= 60>,
  "feedback": "<1-2 Sätze Gesamtfeedback auf Deutsch>",
  "strengths": "<was gut war, 1 Satz>",
  "improvement": "<was verbessert werden könnte, 1 Satz, oder null wenn passed>"
}}"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-5-20250929",
                "max_tokens": 400,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API network error: {exc}")

    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(status_code=502, detail=err.get("error", {}).get("message", f"Anthropic Fehler {resp.status_code}"))

    raw = resp.json()["content"][0]["text"].strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        result = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=502, detail="Ungültige JSON-Antwort von Anthropic")

    if user_id:
        try:
            sb_client.table("open_answer_attempts").insert({
                "user_id":      user_id,
                "lesson_title": lesson_title,
                "question":     question,
                "answer":       user_answer,
                "score":        result["score"],
                "passed":       result["passed"],
            }).execute()
        except Exception:
            pass  # Don't fail the request if DB insert fails

    return result


@app.post("/academy/tutor-chat")
async def tutor_chat(request: Request):
    body = await request.json()
    lesson_title   = body.get("lesson_title", "")
    lesson_context = body.get("lesson_context", "")
    user_message   = body.get("message", "")
    chat_history   = body.get("history", [])

    system_prompt = (
        f"Du bist ein präziser Finanz-Tutor für die Finscope Academy.\n"
        f"Aktuelle Lektion: {lesson_title}\n"
        f"Lektionsinhalt: {lesson_context}\n\n"
        f"Regeln:\n"
        f"- Antworte auf Deutsch, max 4 Sätze\n"
        f"- Bleib strikt beim Lektionsthema\n"
        f"- Bei Off-Topic: freundlich zurücklenken\n"
        f"- Keine Anlageempfehlungen\n"
        f"- Erkläre Konzepte klar und praxisnah"
    )

    messages = chat_history + [{"role": "user", "content": user_message}]

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-5-20250929",
                "max_tokens": 300,
                "system":     system_prompt,
                "messages":   messages,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API network error: {exc}")

    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(status_code=502, detail=err.get("error", {}).get("message", f"Anthropic Fehler {resp.status_code}"))

    return {"reply": resp.json()["content"][0]["text"]}


@app.get("/academy/specializations")
def get_specializations(user_id: str = Depends(verify_jwt)):
    try:
        rows = (
            sb_client.table("user_specializations")
            .select("track_id,selected_at")
            .eq("user_id", user_id)
            .execute()
        ).data or []
    except Exception:
        rows = []
    return rows


@app.post("/academy/specialization", status_code=201)
def set_specialization(body: SpecializationBody, user_id: str = Depends(verify_jwt)):
    valid_ids = {t["id"] for t in ACADEMY_TRACKS if t.get("category") == "specialization"}
    if body.track_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"'{body.track_id}' ist keine gueltige Spezialisierung")
    try:
        sb_client.table("user_specializations").upsert(
            {"user_id": user_id, "track_id": body.track_id},
            on_conflict="user_id,track_id"
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


# ─── Gamification ─────────────────────────────────────────────────────────────

class CompleteLessonBody(BaseModel):
    user_id:    str
    lesson_id:  str
    score:      int = 0
    xp_earned:  int = 0


def _league_for_xp(xp: int) -> str:
    if xp >= 15000: return "black_swan"
    if xp >= 5000:  return "portfolio_manager"
    if xp >= 2000:  return "hedge_fund_analyst"
    if xp >= 500:   return "floor_broker"
    return "rookie_trader"


def _award_achievement(user_id: str, achievement_id: str, name: str, results: list) -> None:
    try:
        existing = sb_client.table("user_achievements") \
            .select("id").eq("user_id", user_id).eq("achievement_id", achievement_id).execute()
        if not existing.data:
            sb_client.table("user_achievements").insert({
                "user_id": user_id,
                "achievement_id": achievement_id,
                "achievement_name": name,
            }).execute()
            results.append({"id": achievement_id, "name": name})
    except Exception:
        pass


def _check_achievements(user_id: str, streak_days: int, league: str) -> list:
    new_achievements: list = []
    try:
        # Sharp Ratio: 10+ open_answer_attempts with score >= 80
        high = sb_client.table("open_answer_attempts") \
            .select("id", count="exact").eq("user_id", user_id).gte("score", 80).execute()
        if (high.count or 0) >= 10:
            _award_achievement(user_id, "sharp_ratio", "Sharp Ratio", new_achievements)

        # 100 Day Streak
        if streak_days >= 100:
            _award_achievement(user_id, "100_day_streak", "100 Day Streak", new_achievements)

        # Black Swan league
        if league == "black_swan":
            _award_achievement(user_id, "black_swan", "Black Swan", new_achievements)

        # Full Analyst: all available analyst lessons completed
        analyst_ids = [
            l["id"]
            for l in ACADEMY_LESSONS_DATA.get("analyst", [])
            if l.get("status") == "available"
        ]
        if analyst_ids:
            done = sb_client.table("academy_progress") \
                .select("lesson_id").eq("user_id", user_id).in_("lesson_id", analyst_ids).execute()
            if len(done.data or []) >= len(analyst_ids):
                _award_achievement(user_id, "full_analyst", "Full Analyst", new_achievements)
    except Exception:
        pass
    return new_achievements


@app.post("/academy/complete-lesson")
async def complete_lesson(body: CompleteLessonBody):
    from datetime import date as _date, timedelta as _td

    user_id = body.user_id
    score   = body.score
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id erforderlich")

    # XP: 50 base + bonus for passing score
    xp = 50
    if score >= 90:
        xp += 20
    elif score >= 60:
        xp += 10

    today     = _date.today().isoformat()
    yesterday = (_date.today() - _td(days=1)).isoformat()
    two_ago   = (_date.today() - _td(days=2)).isoformat()

    # ── Fetch existing progress row ──────────────────────────────────────────
    try:
        row     = sb_client.table("user_progress").select("*").eq("user_id", user_id).execute()
        current = row.data[0] if row.data else None
    except Exception as e:
        logger.error(f"[complete-lesson] SELECT failed user={user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"DB-Lesefehler: {e}")

    # ── Compute new values ───────────────────────────────────────────────────
    if current:
        last_date   = (current.get("last_activity_date") or "")[:10]
        streak_days = current.get("streak_days", 1)
        use_freeze  = False

        if last_date != today:
            if last_date == yesterday:
                streak_days += 1
            elif last_date == two_ago and current.get("streak_freeze_available"):
                use_freeze = True
            else:
                streak_days = 1

        xp_total     = (current.get("xp_total") or 0) + xp
        xp_this_week = (current.get("xp_this_week") or 0) + xp
        league       = _league_for_xp(xp_total)

        update_data: dict = {
            "xp_total":           xp_total,
            "xp_this_week":       xp_this_week,
            "streak_days":        streak_days,
            "last_activity_date": today,
            "league":             league,
        }
        if use_freeze:
            update_data["streak_freeze_available"] = False

        try:
            sb_client.table("user_progress").update(update_data).eq("user_id", user_id).execute()
            logger.info(f"[complete-lesson] UPDATE user={user_id} +{xp}XP → total={xp_total} league={league}")
        except Exception as e:
            logger.error(f"[complete-lesson] UPDATE failed user={user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"XP-Update fehlgeschlagen: {e}")
    else:
        xp_total     = xp
        xp_this_week = xp
        streak_days  = 1
        league       = _league_for_xp(xp_total)

        try:
            sb_client.table("user_progress").insert({
                "user_id":                 user_id,
                "xp_total":                xp_total,
                "xp_this_week":            xp_this_week,
                "streak_days":             streak_days,
                "last_activity_date":      today,
                "league":                  league,
                "streak_freeze_available": True,
            }).execute()
            logger.info(f"[complete-lesson] INSERT new user={user_id} +{xp}XP league={league}")
        except Exception as e:
            logger.error(f"[complete-lesson] INSERT failed user={user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"XP-Insert fehlgeschlagen: {e}")

    new_achievements = _check_achievements(user_id, streak_days, league)

    return {
        "xp_earned":        xp,
        "xp_total":         xp_total,
        "streak_days":      streak_days,
        "league":           league,
        "new_achievements": new_achievements,
    }


@app.get("/academy/my-progress")
def get_my_progress(user_id: str = Depends(verify_jwt)):
    try:
        prog = sb_client.table("user_progress").select("*").eq("user_id", user_id).execute()
        progress = prog.data[0] if prog.data else {}
    except Exception:
        progress = {}

    try:
        ach = sb_client.table("user_achievements").select("*").eq("user_id", user_id).execute()
        achievements = ach.data or []
    except Exception:
        achievements = []

    return {"progress": progress, "achievements": achievements}


@app.get("/academy/leaderboard")
def get_leaderboard(league: str = "all"):
    try:
        q = sb_client.table("user_progress") \
            .select("user_id, xp_this_week, streak_days, league") \
            .order("xp_this_week", desc=True) \
            .limit(50)
        if league != "all":
            q = q.eq("league", league)
        rows = q.execute().data or []
    except Exception:
        return []

    if not rows:
        return []

    user_ids = [r["user_id"] for r in rows]
    try:
        profiles_res = sb_client.table("profiles") \
            .select("id, username, first_name, last_name") \
            .in_("id", user_ids).execute()
        profiles_map = {p["id"]: p for p in (profiles_res.data or [])}
    except Exception:
        profiles_map = {}

    result = []
    for i, row in enumerate(rows):
        p    = profiles_map.get(row["user_id"], {})
        name = (
            p.get("username")
            or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            or f"Trader #{i + 1}"
        )
        result.append({
            "rank":         i + 1,
            "user_id":      row["user_id"],
            "display_name": name,
            "xp_this_week": row.get("xp_this_week", 0),
            "streak_days":  row.get("streak_days", 0),
            "league":       row.get("league", "rookie_trader"),
        })
    return result


@app.post("/academy/use-streak-freeze")
async def use_streak_freeze(request: Request):
    body    = await request.json()
    user_id = body.get("user_id", "")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id erforderlich")

    try:
        row = sb_client.table("user_progress") \
            .select("streak_freeze_available").eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row.data:
        raise HTTPException(status_code=404, detail="Nutzer nicht gefunden")
    if not row.data[0].get("streak_freeze_available"):
        raise HTTPException(status_code=400, detail="Kein Streak Freeze verfügbar")

    sb_client.table("user_progress") \
        .update({"streak_freeze_available": False}).eq("user_id", user_id).execute()
    return {"ok": True}


# ─── Academy — Helpers ────────────────────────────────────────────────────────

def _get_sector_summary() -> str:
    try:
        data = cache.get("sectors:full", ttl_seconds=3600)
        if data:
            secs = data.get("sectors", [])
            if secs:
                best  = secs[0]
                worst = secs[-1]
                return (
                    f"Bester Sektor heute: {best.get('name')} "
                    f"({best.get('perf_1d', 0):+.1f}%, ETF: {best.get('ticker')}). "
                    f"Schlechtester Sektor: {worst.get('name')} "
                    f"({worst.get('perf_1d', 0):+.1f}%, ETF: {worst.get('ticker')})."
                )
    except Exception:
        pass
    return "S&P 500 Sektoren mit gemischter Performance. Referenz: SPY als Benchmark."


def _claude_json(prompt: str, max_tokens: int = 500) -> dict:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":      "claude-sonnet-4-5-20250929",
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=40,
    )
    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(status_code=502, detail=err.get("error", {}).get("message", f"Anthropic {resp.status_code}"))
    raw = resp.json()["content"][0]["text"].strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw).strip()
    raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        raise HTTPException(status_code=502, detail="Ungueltige JSON-Antwort von Claude")


# ─── Academy — Daily Challenge ────────────────────────────────────────────────

@app.get("/academy/daily-challenge/today")
async def daily_challenge_today(user_id: str = ""):
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        existing = sb_client.table("daily_challenges") \
            .select("*").eq("date", today_str).eq("is_boss", False).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if existing.data:
        challenge = existing.data[0]
    else:
        sector_summary = _get_sector_summary()
        prompt = (
            "Generiere eine praezise Finanz-Frage auf Deutsch basierend auf diesen Marktdaten: "
            f"{sector_summary} "
            "Die Frage soll einen konkreten Marktbezug haben und in 3-5 Saetzen beantwortbar sein. "
            "Antworte NUR als JSON (kein Markdown, keine Codeblocks):\n"
            '{"question": "<die Frage>", "context": "<1-2 Saetze Marktkontext>", '
            '"market_ticker": "<relevantes ETF/Ticker-Symbol>"}'
        )
        gen = _claude_json(prompt, max_tokens=400)
        try:
            row = sb_client.table("daily_challenges").insert({
                "date":          today_str,
                "question":      gen.get("question", ""),
                "context":       gen.get("context", ""),
                "market_ticker": gen.get("market_ticker", "SPY"),
                "xp_reward":     75,
                "is_boss":       False,
            }).execute()
            challenge = row.data[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    already_attempted = False
    if user_id:
        try:
            att = sb_client.table("daily_challenge_attempts") \
                .select("id").eq("user_id", user_id).eq("challenge_id", challenge["id"]).execute()
            already_attempted = len(att.data) > 0
        except Exception:
            pass

    return {
        "id":                challenge["id"],
        "question":          challenge["question"],
        "context":           challenge.get("context", ""),
        "market_ticker":     challenge.get("market_ticker", "SPY"),
        "xp_reward":         challenge.get("xp_reward", 75),
        "already_attempted": already_attempted,
    }


@app.post("/academy/daily-challenge/submit")
async def daily_challenge_submit(request: Request):
    body         = await request.json()
    user_id      = body.get("user_id", "")
    challenge_id = body.get("challenge_id", "")
    answer       = body.get("answer", "")

    if not user_id or not challenge_id or not answer:
        raise HTTPException(status_code=400, detail="user_id, challenge_id und answer erforderlich")

    try:
        ch = sb_client.table("daily_challenges").select("*").eq("id", challenge_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not ch.data:
        raise HTTPException(status_code=404, detail="Challenge nicht gefunden")
    challenge = ch.data[0]

    try:
        att = sb_client.table("daily_challenge_attempts") \
            .select("id").eq("user_id", user_id).eq("challenge_id", challenge_id).execute()
        if att.data:
            raise HTTPException(status_code=400, detail="Challenge bereits beantwortet")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    eval_prompt = (
        "Du bist ein strenger aber fairer Finanz-Tutor. Bewerte die folgende Antwort eines Lernenden.\n\n"
        f"Tages-Challenge: {challenge['question']}\n"
        f"Kontext: {challenge.get('context', '')}\n"
        f"Antwort des Lernenden: {answer}\n\n"
        "Bewerte auf einer Skala von 0-100 und antworte NUR als JSON (kein Markdown):\n"
        '{"score": <0-100>, "passed": <true wenn score >= 60>, '
        '"feedback": "<1-2 Saetze Gesamtfeedback auf Deutsch>", '
        '"strengths": "<was gut war, 1 Satz>", '
        '"improvement": "<was verbessert werden koennte, 1 Satz, oder null wenn passed>"}'
    )
    result = _claude_json(eval_prompt, max_tokens=400)

    try:
        sb_client.table("daily_challenge_attempts").insert({
            "user_id":      user_id,
            "challenge_id": challenge_id,
            "answer":       answer,
            "score":        result.get("score", 0),
            "passed":       result.get("passed", False),
        }).execute()
    except Exception:
        pass

    xp_earned = 0
    if result.get("passed"):
        xp_reward = challenge.get("xp_reward", 75)
        try:
            prog = sb_client.table("user_progress") \
                .select("xp_total, xp_this_week").eq("user_id", user_id).execute()
            if prog.data:
                current_xp   = prog.data[0].get("xp_total", 0)
                current_week = prog.data[0].get("xp_this_week", 0)
                new_xp       = current_xp + xp_reward
                sb_client.table("user_progress").update({
                    "xp_total":     new_xp,
                    "xp_this_week": current_week + xp_reward,
                    "league":       _league_for_xp(new_xp),
                }).eq("user_id", user_id).execute()
                xp_earned = xp_reward
        except Exception:
            pass

    return {**result, "xp_earned": xp_earned}


# ─── Academy — Profile ────────────────────────────────────────────────────────

@app.get("/academy/profile/{user_id}")
async def get_profile(user_id: str):
    try:
        prog     = sb_client.table("user_progress").select("*").eq("user_id", user_id).execute()
        achv     = sb_client.table("user_achievements").select("*").eq("user_id", user_id).execute()
        lessons  = sb_client.table("academy_progress").select("id").eq("user_id", user_id).execute()
        profile  = sb_client.table("profiles").select("user_id, display_name, created_at") \
                       .eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    profile_row = profile.data[0] if profile.data else {}
    return {
        "progress":     prog.data[0] if prog.data else {},
        "achievements": achv.data or [],
        "lesson_count": len(lessons.data) if lessons.data else 0,
        "member_since": profile_row.get("created_at", ""),
        "display_name": profile_row.get("display_name", ""),
    }


# ─── Academy — Weekly Boss ────────────────────────────────────────────────────

def _monday_this_week() -> str:
    today  = datetime.utcnow()
    monday = today - timedelta(days=today.weekday())
    return monday.strftime("%Y-%m-%d")

def _sunday_this_week() -> str:
    today  = datetime.utcnow()
    sunday = today + timedelta(days=(6 - today.weekday()))
    return sunday.replace(hour=23, minute=59, second=59).isoformat()


@app.get("/academy/weekly-boss")
async def weekly_boss(user_id: str = ""):
    import uuid as _uuid
    monday = _monday_this_week()

    try:
        existing = sb_client.table("daily_challenges") \
            .select("*").eq("is_boss", True).gte("date", monday).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if existing.data:
        questions = sorted(existing.data, key=lambda x: x.get("boss_question_index", 0))
    else:
        sector_summary = _get_sector_summary()
        prompt = (
            "Erstelle eine Weekly Boss Challenge fuer Finanz-Profis auf Deutsch. "
            "3 aufeinander aufbauende Fragen die zusammen ein komplexes Marktthema abdecken. "
            f"Basierend auf: {sector_summary}\n"
            "Antworte NUR als JSON (kein Markdown):\n"
            '{"title": "<Titel der Boss Challenge>", "theme": "<Marktthema 3-5 Worte>", '
            '"questions": ['
            '{"question": "<Frage 1>", "context": "<Kontext 1>"},'
            '{"question": "<Frage 2>", "context": "<Kontext 2>"},'
            '{"question": "<Frage 3>", "context": "<Kontext 3>"}'
            ']}'
        )
        gen      = _claude_json(prompt, max_tokens=800)
        boss_id  = str(_uuid.uuid4())
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        title     = gen.get("title", "Weekly Boss Challenge")
        theme     = gen.get("theme", "Marktanalyse")

        questions = []
        for i, q in enumerate(gen.get("questions", [])[:3]):
            try:
                row = sb_client.table("daily_challenges").insert({
                    "date":                 today_str,
                    "question":             q.get("question", ""),
                    "context":              q.get("context", ""),
                    "market_ticker":        "SPY",
                    "xp_reward":            300,
                    "is_boss":              True,
                    "boss_id":              boss_id,
                    "boss_question_index":  i,
                    "boss_title":           title,
                    "boss_theme":           theme,
                }).execute()
                questions.append(row.data[0])
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    if not questions:
        raise HTTPException(status_code=500, detail="Keine Boss-Fragen verfuegbar")

    boss_id  = questions[0].get("boss_id")
    title    = questions[0].get("boss_title", "Weekly Boss Challenge")
    theme    = questions[0].get("boss_theme", "Marktanalyse")

    already_attempted = False
    if user_id:
        try:
            att = sb_client.table("weekly_boss_attempts") \
                .select("id").eq("user_id", user_id).eq("boss_id", boss_id).execute()
            already_attempted = len(att.data) > 0
        except Exception:
            pass

    return {
        "boss_id":           boss_id,
        "title":             title,
        "theme":             theme,
        "questions":         [{"question": q["question"], "context": q.get("context", "")} for q in questions],
        "xp_reward":         300,
        "available_until":   _sunday_this_week(),
        "already_attempted": already_attempted,
    }


@app.post("/academy/weekly-boss/submit")
async def weekly_boss_submit(request: Request):
    body    = await request.json()
    user_id = body.get("user_id", "")
    boss_id = body.get("boss_id", "")
    answers = body.get("answers", [])

    if not user_id or not boss_id or len(answers) < 3:
        raise HTTPException(status_code=400, detail="user_id, boss_id und 3 answers erforderlich")

    try:
        att = sb_client.table("weekly_boss_attempts") \
            .select("id").eq("user_id", user_id).eq("boss_id", boss_id).execute()
        if att.data:
            raise HTTPException(status_code=400, detail="Boss bereits herausgefordert diese Woche")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        qs = sb_client.table("daily_challenges") \
            .select("*").eq("boss_id", boss_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not qs.data or len(qs.data) < 3:
        raise HTTPException(status_code=404, detail="Boss nicht gefunden")

    questions = sorted(qs.data, key=lambda x: x.get("boss_question_index", 0))
    theme     = questions[0].get("boss_theme", "Marktanalyse")

    eval_prompt = (
        f"Bewerte diese 3 Antworten zur Weekly Boss Challenge '{theme}'.\n\n"
        f"Frage 1: {questions[0]['question']}\nAntwort 1: {answers[0]}\n\n"
        f"Frage 2: {questions[1]['question']}\nAntwort 2: {answers[1]}\n\n"
        f"Frage 3: {questions[2]['question']}\nAntwort 3: {answers[2]}\n\n"
        "Bewerte jede Antwort auf 0-100 und antworte NUR als JSON (kein Markdown):\n"
        '{"scores": [<score1>, <score2>, <score3>], "total_score": <summe 0-300>, '
        '"passed": <true wenn total_score >= 180>, '
        '"overall_feedback": "<2-3 Saetze Gesamtfeedback auf Deutsch>", '
        '"question_feedback": ["<Feedback F1>", "<Feedback F2>", "<Feedback F3>"]}'
    )
    result = _claude_json(eval_prompt, max_tokens=600)

    try:
        sb_client.table("weekly_boss_attempts").insert({
            "user_id":     user_id,
            "boss_id":     boss_id,
            "answers":     answers,
            "scores":      result.get("scores", []),
            "total_score": result.get("total_score", 0),
            "passed":      result.get("passed", False),
        }).execute()
    except Exception:
        pass

    xp_earned = 0
    if result.get("passed"):
        try:
            prog = sb_client.table("user_progress") \
                .select("xp_total, xp_this_week").eq("user_id", user_id).execute()
            if prog.data:
                current_xp   = prog.data[0].get("xp_total", 0)
                current_week = prog.data[0].get("xp_this_week", 0)
                new_xp       = current_xp + 300
                sb_client.table("user_progress").update({
                    "xp_total":     new_xp,
                    "xp_this_week": current_week + 300,
                    "league":       _league_for_xp(new_xp),
                }).eq("user_id", user_id).execute()
                xp_earned = 300
        except Exception:
            pass

    return {**result, "xp_earned": xp_earned}


# ─── Fundamentals (Alpha Vantage API) ────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"
AV_KEY  = os.environ.get("ALPHA_VANTAGE_KEY", "")

fundamentals_cache: dict = {}  # { "AAPL": {"data": {...}, "expires": datetime} }


@app.get("/sector/{ticker}")
async def get_sector(ticker: str):
    cache_key = f"sector_{ticker.upper()}"
    if cache_key in fundamentals_cache:
        cached = fundamentals_cache[cache_key]
        if datetime.now() < cached["expires"]:
            return cached["data"]
    try:
        t = ticker.replace(".US", "").replace(".XETRA", "").replace(".CC", "").upper()
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{AV_BASE}?function=OVERVIEW&symbol={t}&apikey={AV_KEY}")
        data   = r.json()
        result = {
            "ticker":   ticker,
            "sector":   data.get("Sector", "Sonstige") or "Sonstige",
            "industry": data.get("Industry", ""),
            "name":     data.get("Name", ticker),
        }
        fundamentals_cache[cache_key] = {"data": result, "expires": datetime.now() + timedelta(days=7)}
        return result
    except Exception:
        return {"ticker": ticker, "sector": "Sonstige", "industry": "", "name": ticker}


@app.get("/fundamentals/{ticker}")
async def get_fundamentals(ticker: str):
    cache_key = ticker.upper()
    cached = fundamentals_cache.get(cache_key)
    if cached and datetime.now() < cached["expires"]:
        return cached["data"]

    t = ticker.replace(".US", "").replace(".XETRA", "").replace(".CC", "").replace(".FOREX", "").upper()

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            overview_r = await client.get(f"{AV_BASE}?function=OVERVIEW&symbol={t}&apikey={AV_KEY}")
            await asyncio.sleep(13)
            income_r   = await client.get(f"{AV_BASE}?function=INCOME_STATEMENT&symbol={t}&apikey={AV_KEY}")
            await asyncio.sleep(13)
            bs_r       = await client.get(f"{AV_BASE}?function=BALANCE_SHEET&symbol={t}&apikey={AV_KEY}")
            await asyncio.sleep(13)
            cf_r       = await client.get(f"{AV_BASE}?function=CASH_FLOW&symbol={t}&apikey={AV_KEY}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Alpha Vantage Netzwerkfehler: {e}")

    ov          = overview_r.json() if overview_r.status_code == 200 else {}
    income_data = income_r.json()   if income_r.status_code   == 200 else {}
    bs_data     = bs_r.json()       if bs_r.status_code       == 200 else {}
    cf_data     = cf_r.json()       if cf_r.status_code       == 200 else {}

    if not ov.get("Symbol"):
        raise HTTPException(status_code=429, detail="Alpha Vantage Limit erreicht. Bitte 60 Sekunden warten.")

    inc_reports = income_data.get("annualReports", [])
    bs_reports  = bs_data.get("annualReports", [])
    cf_reports  = cf_data.get("annualReports", [])

    inc0 = inc_reports[0] if inc_reports else {}
    bs0  = bs_reports[0]  if bs_reports  else {}
    cf0  = cf_reports[0]  if cf_reports  else {}

    def to_float(val):
        try:
            return float(val) if val and val != "None" else None
        except Exception:
            return None

    def av_series(reports, key):
        result = {}
        for r in reversed(reports):
            year = r.get("fiscalDateEnding", "")[:4]
            val  = to_float(r.get(key))
            if year and val is not None:
                result[year] = val
        return result

    total_debt     = to_float(bs0.get("shortLongTermDebtTotal")) or (
                         (to_float(bs0.get("shortTermDebt")) or 0) +
                         (to_float(bs0.get("longTermDebt")) or 0)
                     )
    cash           = to_float(bs0.get("cashAndCashEquivalentsAtCarryingValue"))
    net_debt       = (total_debt or 0) - (cash or 0)
    revenue        = to_float(inc0.get("totalRevenue"))
    net_income     = to_float(inc0.get("netIncome"))
    gross_profit   = to_float(inc0.get("grossProfit"))
    ebitda         = to_float(inc0.get("ebitda"))
    operating_inc  = to_float(inc0.get("operatingIncome"))
    gross_margin   = (gross_profit   / revenue) if gross_profit   and revenue else None
    net_margin     = (net_income     / revenue) if net_income     and revenue else None
    operating_margin = (operating_inc / revenue) if operating_inc and revenue else None
    operating_cf   = to_float(cf0.get("operatingCashflow"))
    capex          = to_float(cf0.get("capitalExpenditures"))
    free_cf        = (operating_cf - abs(capex)) if operating_cf and capex else None
    shares         = to_float(ov.get("SharesOutstanding"))
    fcf_per_share  = (free_cf / shares) if free_cf and shares else None
    total_equity   = to_float(bs0.get("totalShareholderEquity"))
    total_assets   = to_float(bs0.get("totalAssets"))
    roe            = (net_income / total_equity) if net_income and total_equity else None
    roa            = (net_income / total_assets) if net_income and total_assets else None

    result = {
        "ticker":    ticker,
        "yf_ticker": t,
        "company": {
            "name":        ov.get("Name", ticker),
            "sector":      ov.get("Sector", ""),
            "industry":    ov.get("Industry", ""),
            "country":     ov.get("Country", ""),
            "employees":   to_float(ov.get("FullTimeEmployees")),
            "description": (ov.get("Description", ""))[:500],
        },
        "market": {
            "price":      to_float(ov.get("50DayMovingAverage")),
            "market_cap": to_float(ov.get("MarketCapitalization")),
            "52w_high":   to_float(ov.get("52WeekHigh")),
            "52w_low":    to_float(ov.get("52WeekLow")),
            "beta":       to_float(ov.get("Beta")),
            "avg_volume": to_float(ov.get("10DayAverageTradingVolume")),
        },
        "valuation": {
            "pe_ratio":   to_float(ov.get("TrailingPE")),
            "forward_pe": to_float(ov.get("ForwardPE")),
            "peg_ratio":  to_float(ov.get("PEGRatio")),
            "pb_ratio":   to_float(ov.get("PriceToBookRatio")),
            "ps_ratio":   to_float(ov.get("PriceToSalesRatioTTM")),
            "ev_ebitda":  to_float(ov.get("EVToEBITDA")),
            "ev_revenue": to_float(ov.get("EVToRevenue")),
        },
        "profitability": {
            "gross_margin":     gross_margin,
            "operating_margin": operating_margin,
            "net_margin":       net_margin,
            "roe":              roe,
            "roa":              roa,
            "roic":             to_float(ov.get("ReturnOnCapitalEmployedTTM")),
        },
        "balance_sheet": {
            "cash":                 cash,
            "total_cash":           cash,
            "total_debt":           total_debt,
            "net_debt":             net_debt,
            "debt_to_equity":       to_float(ov.get("DebtToEquityRatio")),
            "current_ratio":        to_float(ov.get("CurrentRatio")),
            "quick_ratio":          None,
            "total_assets":         total_assets,
            "total_equity":         total_equity,
            "book_value_per_share": to_float(ov.get("BookValue")),
            "history": {
                "total_debt": av_series(bs_reports, "shortLongTermDebtTotal"),
                "cash":       av_series(bs_reports, "cashAndCashEquivalentsAtCarryingValue"),
                "equity":     av_series(bs_reports, "totalShareholderEquity"),
            },
        },
        "income": {
            "revenue_ttm":    revenue,
            "revenue_growth": to_float(ov.get("QuarterlyRevenueGrowthYOY")),
            "gross_profit":   gross_profit,
            "ebitda":         ebitda,
            "net_income":     net_income,
            "eps_ttm":        to_float(ov.get("EPS")),
            "eps_forward":    to_float(ov.get("ForwardEPS")),
            "history": {
                "revenue":    av_series(inc_reports, "totalRevenue"),
                "net_income": av_series(inc_reports, "netIncome"),
                "ebitda":     av_series(inc_reports, "ebitda"),
            },
        },
        "cash_flow": {
            "operating_cf":   operating_cf,
            "capex":          capex,
            "free_cash_flow": free_cf,
            "fcf_per_share":  fcf_per_share,
            "dividend_yield": to_float(ov.get("DividendYield")),
            "payout_ratio":   to_float(ov.get("PayoutRatio")),
            "history": {
                "operating_cf":   av_series(cf_reports, "operatingCashflow"),
                "capex":          av_series(cf_reports, "capitalExpenditures"),
                "free_cash_flow": {},
            },
        },
        "growth": {
            "revenue_growth_yoy":        to_float(ov.get("QuarterlyRevenueGrowthYOY")),
            "earnings_growth_yoy":       to_float(ov.get("QuarterlyEarningsGrowthYOY")),
            "earnings_quarterly_growth": None,
        },
        "analyst": {
            "recommendation": ov.get("AnalystRatingStrongBuy"),
            "target_price":   to_float(ov.get("AnalystTargetPrice")),
            "target_high":    None,
            "target_low":     None,
            "num_analysts":   to_float(ov.get("AnalystRatingBuy")),
        },
    }

    fundamentals_cache[cache_key] = {"data": result, "expires": datetime.now() + timedelta(hours=48)}
    return result


@app.post("/fundamentals/ai-analysis")
async def fundamentals_ai_analysis(request: Request):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY nicht konfiguriert.")

    body   = await request.json()
    ticker = body.get("ticker", "").upper()
    d      = body.get("data", {})

    if not ticker or not d:
        raise HTTPException(status_code=400, detail="ticker und data erforderlich.")

    co  = d.get("company", {})
    mkt = d.get("market", {})
    val = d.get("valuation", {})
    pro = d.get("profitability", {})
    bs  = d.get("balance_sheet", {})
    inc = d.get("income", {})
    cf  = d.get("cash_flow", {})
    gr  = d.get("growth", {})

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "n/a"

    def fmt(v, decimals=2):
        return f"{v:,.{decimals}f}" if v is not None else "n/a"

    def bn(v):
        return f"${v/1e9:.2f}B" if v is not None else "n/a"

    user_prompt = (
        f"Erstelle eine strukturierte Fundamentalanalyse für {ticker} ({co.get('name', ticker)}).\n\n"
        f"UNTERNEHMEN\n"
        f"Sektor: {co.get('sector', 'n/a')} | Branche: {co.get('industry', 'n/a')} | Land: {co.get('country', 'n/a')}\n"
        f"Mitarbeiter: {fmt(co.get('employees'), 0)}\n\n"
        f"MARKTDATEN\n"
        f"Market Cap: {bn(mkt.get('market_cap'))} | Beta: {fmt(mkt.get('beta'))}\n"
        f"52W Hoch: {fmt(mkt.get('52w_high'))} | 52W Tief: {fmt(mkt.get('52w_low'))}\n\n"
        f"BEWERTUNG\n"
        f"KGV (TTM): {fmt(val.get('pe_ratio'))} | Forward KGV: {fmt(val.get('forward_pe'))}\n"
        f"KBV: {fmt(val.get('pb_ratio'))} | KUV: {fmt(val.get('ps_ratio'))}\n"
        f"EV/EBITDA: {fmt(val.get('ev_ebitda'))} | PEG: {fmt(val.get('peg_ratio'))}\n\n"
        f"PROFITABILITÄT\n"
        f"Bruttomarge: {pct(pro.get('gross_margin'))} | Betriebsmarge: {pct(pro.get('operating_margin'))}\n"
        f"Nettomarge: {pct(pro.get('net_margin'))} | ROE: {pct(pro.get('roe'))} | ROA: {pct(pro.get('roa'))}\n\n"
        f"BILANZ\n"
        f"Cash: {bn(bs.get('total_cash'))} | Schulden: {bn(bs.get('total_debt'))} | Nettoverschuldung: {bn(bs.get('net_debt'))}\n"
        f"Debt/Equity: {fmt(bs.get('debt_to_equity'))} | Current Ratio: {fmt(bs.get('current_ratio'))}\n\n"
        f"GEWINN & UMSATZ\n"
        f"Umsatz (TTM): {bn(inc.get('revenue_ttm'))} | Umsatzwachstum YoY: {pct(inc.get('revenue_growth'))}\n"
        f"EBITDA: {bn(inc.get('ebitda'))} | Nettogewinn: {bn(inc.get('net_income'))}\n"
        f"EPS (TTM): {fmt(inc.get('eps_ttm'))} | Forward EPS: {fmt(inc.get('eps_forward'))}\n\n"
        f"CASHFLOW\n"
        f"Operating CF: {bn(cf.get('operating_cf'))} | Capex: {bn(cf.get('capex'))}\n"
        f"Free Cashflow: {bn(cf.get('free_cash_flow'))} | FCF/Aktie: {fmt(cf.get('fcf_per_share'))}\n"
        f"Dividendenrendite: {pct(cf.get('dividend_yield'))}\n\n"
        f"WACHSTUM\n"
        f"Umsatzwachstum YoY: {pct(gr.get('revenue_growth_yoy'))} | Gewinnwachstum YoY: {pct(gr.get('earnings_growth_yoy'))}\n"
    )

    system_prompt = (
        "Du bist ein erfahrener Finanzanalyst und erstellst präzise, faktenbasierte Fundamentalanalysen. "
        "Strukturiere deine Analyse mit diesen Abschnitten (## als Markdown-Header):\n"
        "## Unternehmensüberblick\n"
        "## Bewertung\n"
        "## Profitabilität & Effizienz\n"
        "## Bilanzstärke\n"
        "## Wachstum & Cashflow\n"
        "## Risiko-Rendite-Profil\n\n"
        "Regeln:\n"
        "- Verwende '• ' für Bullet Points\n"
        "- Keine Anlageempfehlungen (kein Kaufen/Verkaufen/Halten)\n"
        "- Sachlich, direkt, auf Deutsch\n"
        "- Endet mit: 'Disclaimer: Diese Analyse dient ausschließlich zu Informationszwecken und stellt keine individuelle Anlageberatung dar.'"
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 2048,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": user_prompt}],
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic Netzwerkfehler: {exc}")

    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(
            status_code=502,
            detail=err.get("error", {}).get("message", f"Anthropic API Fehler {resp.status_code}"),
        )

    return {"analysis": resp.json()["content"][0]["text"]}


@app.post("/portfolio/ai-analysis")
async def portfolio_ai_analysis(request: Request):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY nicht konfiguriert.")

    body      = await request.json()
    positions = body.get("positions", [])
    metrics   = body.get("metrics", {})

    if not positions:
        raise HTTPException(status_code=400, detail="positions erforderlich.")

    positions_text = "\n".join([
        f"- {p.get('symbol','?')}: {p.get('shares','?')} Stück, "
        f"Kaufkurs {p.get('buyPrice','?')}, "
        f"Aktuell {p.get('currentPrice','?')}, "
        f"Gewichtung {p.get('weight','?')}%, "
        f"Sektor: {p.get('sector','?')}"
        for p in positions
    ])

    prompt = (
        f"Du bist ein erfahrener Portfolio-Manager. "
        f"Analysiere folgendes Portfolio objektiv auf Deutsch.\n\n"
        f"PORTFOLIO KENNZAHLEN:\n"
        f"- Gesamtwert: {metrics.get('totalValue', 'n/a')}\n"
        f"- Gesamt P&L: {metrics.get('totalPnL', 'n/a')}\n"
        f"- Anzahl Positionen: {len(positions)}\n"
        f"- Portfolio Beta: {metrics.get('beta', 'n/a')}\n\n"
        f"POSITIONEN:\n{positions_text}\n\n"
        f"Erstelle eine strukturierte Analyse mit exakt diesen 5 Sektionen:\n\n"
        f"## DIVERSIFIKATION\n"
        f"2-3 Sätze: Wie gut ist das Portfolio diversifiziert? Klumpenrisiken?\n\n"
        f"## STÄRKEN\n"
        f"2-3 konkrete Stärken als Stichpunkte (• ...)\n\n"
        f"## RISIKEN\n"
        f"2-3 konkrete Risiken als Stichpunkte (• ...)\n\n"
        f"## OPTIMIERUNG\n"
        f"2-3 konkrete Verbesserungsvorschläge als Stichpunkte (• ...)\n\n"
        f"## FAZIT\n"
        f"1-2 Sätze Gesamtbewertung.\n\n"
        f"Disclaimer: Keine Anlageempfehlung.\n\n"
        f"Ton: Objektiv, professionell, faktenbasiert. Keine Kaufempfehlungen."
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 800,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic Netzwerkfehler: {exc}")

    if not resp.ok:
        err = resp.json() if resp.content else {}
        raise HTTPException(
            status_code=502,
            detail=err.get("error", {}).get("message", f"Anthropic API Fehler {resp.status_code}"),
        )

    return {"analysis": resp.json()["content"][0]["text"]}


# ─── News Feed ────────────────────────────────────────────────────────────────

@app.get("/news/market")
async def market_news():
    cache_key = "market_news"
    if cache_key in fundamentals_cache:
        cached = fundamentals_cache[cache_key]
        if datetime.now() < cached["expires"] and cached["data"].get("news"):
            return cached["data"]
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{AV_BASE}?function=NEWS_SENTIMENT&topics=financial_markets,economy_macro,finance&limit=20&apikey={AV_KEY}"
            )
        data = r.json()
        feed = data.get("feed", [])
        if not feed:
            print(f"[news] empty feed. keys={list(data.keys())} body={str(data)[:200]}")
            return {"news": [], "debug": str(data)[:200]}
        news = [{
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source": item.get("source", ""),
            "time": item.get("time_published", ""),
            "sentiment": item.get("overall_sentiment_label", "Neutral"),
            "summary": (item.get("summary", "") or "")[:200],
        } for item in feed[:8]]
        result = {"news": news}
        if news:
            fundamentals_cache[cache_key] = {
                "data": result,
                "expires": datetime.now() + timedelta(hours=1),
            }
        return result
    except Exception as e:
        return {"news": [], "error": str(e)}


# ─── Earnings Calendar ────────────────────────────────────────────────────────

@app.get("/earnings/today")
async def earnings_today():
    cache_key = "earnings_today"
    if cache_key in fundamentals_cache:
        cached = fundamentals_cache[cache_key]
        if datetime.now() < cached["expires"]:
            return cached["data"]

    KNOWN_TICKERS = {
        # US Tech
        'AAPL','MSFT','GOOGL','GOOG','AMZN','META','NVDA','TSLA','NFLX',
        'AMD','INTC','CRM','ORCL','ADBE','CSCO','IBM','QCOM','TXN','AVGO',
        # US Finance
        'JPM','BAC','WFC','GS','MS','C','BLK','AXP','V','MA',
        # US Healthcare
        'JNJ','UNH','PFE','MRK','LLY','ABBV','TMO','ABT','BMY','CVS',
        # US Consumer
        'WMT','HD','PG','KO','PEP','MCD','NKE','SBUX','DIS','COST',
        # US Energy / Industrial
        'XOM','CVX','BA','CAT','GE','HON','UPS','LMT','RTX','DE',
        # German
        'SAP','SIE','ALV','DTE','BMW','MBG','VOW3','BAS','DBK','ADS','MUV2',
        # Other notable
        'BRK.A','BRK.B','BABA','TSM','ASML','NVO','TM','NESN','RHHBY',
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{AV_BASE}?function=EARNINGS_CALENDAR&horizon=3month&apikey={AV_KEY}"
            )
        text = r.text
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return {"earnings": []}
        headers = [h.strip() for h in lines[0].split(",")]
        this_week = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        upcoming = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, [p.strip() for p in parts]))
            symbol = row.get("symbol", "").upper()
            report_date = row.get("reportDate", "")
            if symbol not in KNOWN_TICKERS:
                continue
            if report_date not in this_week:
                continue
            upcoming.append({
                "symbol": symbol,
                "name": row.get("name", ""),
                "date": report_date,
                "estimate": row.get("estimate", "") or "—",
                "currency": row.get("currency", "USD"),
            })
        result = {"earnings": upcoming[:8]}
        fundamentals_cache[cache_key] = {
            "data": result,
            "expires": datetime.now() + timedelta(hours=6),
        }
        return result
    except Exception as e:
        return {"earnings": [], "error": str(e)}


# ─── Economic Calendar ────────────────────────────────────────────────────────

@app.get("/economic-calendar")
async def economic_calendar():
    today = datetime.now()
    events = []

    fomc_2026 = [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ]
    for date_str in fomc_2026:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_until = (event_date - today).days
        if 0 <= days_until <= 30:
            events.append({
                "name": "FOMC Zinsentscheidung",
                "date": date_str,
                "days_until": days_until,
                "importance": "high",
                "category": "Fed",
            })

    for month_offset in range(2):
        try:
            cpi_date = (today.replace(day=13) + timedelta(days=30 * month_offset))
            days_until = (cpi_date - today).days
            if 0 <= days_until <= 30:
                events.append({
                    "name": "CPI Inflation",
                    "date": cpi_date.strftime("%Y-%m-%d"),
                    "days_until": days_until,
                    "importance": "high",
                    "category": "Inflation",
                })
        except Exception:
            pass

    for month_offset in range(2):
        try:
            first_day = (today.replace(day=1) + timedelta(days=30 * month_offset)).replace(day=1)
            days_to_friday = (4 - first_day.weekday()) % 7
            nfp_date = first_day + timedelta(days=days_to_friday)
            days_until = (nfp_date - today).days
            if 0 <= days_until <= 30:
                events.append({
                    "name": "Nonfarm Payrolls",
                    "date": nfp_date.strftime("%Y-%m-%d"),
                    "days_until": days_until,
                    "importance": "high",
                    "category": "Arbeitsmarkt",
                })
        except Exception:
            pass

    events.sort(key=lambda x: x["days_until"])
    return {"events": events[:8]}


# ─── Family Office ────────────────────────────────────────────────────────────

def generate_invite_code():
    prefix = ''.join(random.choices(string.ascii_uppercase, k=3))
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}-{suffix}"


@app.post("/fo/groups")
async def create_fo_group(request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    code = generate_invite_code()
    for _ in range(5):
        check = sb_client.table("fo_groups").select("id").eq("invite_code", code).execute()
        if not check.data:
            break
        code = generate_invite_code()

    result = sb_client.table("fo_groups").insert({
        "name":        data.get("name"),
        "description": data.get("description", ""),
        "invite_code": code,
        "owner_id":    user_id,
    }).execute()
    group = result.data[0]

    sb_client.table("fo_members").insert({
        "group_id":     group["id"],
        "user_id":      user_id,
        "display_name": data.get("display_name", "Owner"),
        "avatar_emoji": data.get("avatar_emoji", "👤"),
        "role":         "owner",
    }).execute()

    return {"group": group}


@app.get("/fo/my-groups")
def my_fo_groups(user_id: str = Depends(verify_jwt)):
    memberships = sb_client.table("fo_members").select(
        "group_id, role, display_name, avatar_emoji"
    ).eq("user_id", user_id).execute()

    group_ids = [m["group_id"] for m in memberships.data]
    if not group_ids:
        return {"groups": []}

    groups = sb_client.table("fo_groups").select("*").in_("id", group_ids).execute()
    enriched = []
    for g in groups.data:
        mc = sb_client.table("fo_members").select("id").eq("group_id", g["id"]).execute()
        my = next((m for m in memberships.data if m["group_id"] == g["id"]), {})
        enriched.append({
            **g,
            "member_count":    len(mc.data),
            "my_role":         my.get("role"),
            "my_display_name": my.get("display_name"),
            "my_avatar":       my.get("avatar_emoji"),
        })
    return {"groups": enriched}


@app.post("/fo/join")
async def join_fo_group(request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    code = data.get("invite_code", "").strip().upper()
    group = sb_client.table("fo_groups").select("*").eq("invite_code", code).execute()
    if not group.data:
        raise HTTPException(status_code=404, detail="Ungültiger Code")
    group_id = group.data[0]["id"]
    existing = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Bereits Mitglied")
    sb_client.table("fo_members").insert({
        "group_id":     group_id,
        "user_id":      user_id,
        "display_name": data.get("display_name", "Mitglied"),
        "avatar_emoji": data.get("avatar_emoji", "👤"),
        "role":         "member",
    }).execute()
    emit_activity(group_id, user_id, "join", {})
    # Achievement: early_bird if <=3 members
    mc = sb_client.table("fo_members").select("id", count="exact").eq("group_id", group_id).execute()
    if (mc.count or 0) <= 3:
        check_and_unlock(group_id, user_id, "early_bird")
    return {"group": group.data[0]}


@app.delete("/fo/groups/{group_id}/leave")
def leave_fo_group(group_id: str, user_id: str = Depends(verify_jwt)):
    sb_client.table("fo_members").delete().eq("group_id", group_id).eq("user_id", user_id).execute()
    sb_client.table("fo_shared_portfolios").delete().eq("group_id", group_id).eq("user_id", user_id).execute()
    return {"success": True}


@app.get("/fo/groups/{group_id}/members")
def fo_group_members(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(status_code=403, detail="Nicht Mitglied")
    members = sb_client.table("fo_members").select("*").eq("group_id", group_id).execute()
    return {"members": members.data}


@app.post("/fo/groups/{group_id}/share-portfolio")
async def share_fo_portfolio(group_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    positions = data.get("positions", [])
    # Dedupe by symbol — sum weight_pct if ticker appears multiple times
    deduped: dict = {}
    for p in positions:
        if "symbol" not in p or "weight_pct" not in p:
            continue
        sym = str(p["symbol"]).upper()
        deduped[sym] = deduped.get(sym, 0) + float(p["weight_pct"])
    validated = [
        {"symbol": sym, "weight_pct": round(w, 2), "sector": "Sonstige"}
        for sym, w in deduped.items()
        if w > 0
    ]
    cash_pct = float(data.get("cash_pct", 0))
    existing = sb_client.table("fo_shared_portfolios").select("id").eq("user_id", user_id).eq("group_id", group_id).execute()
    payload = {
        "user_id":         user_id,
        "group_id":        group_id,
        "positions":       validated,
        "cash_pct":        cash_pct,
        "total_positions": len(validated),
        "risk_profile":    data.get("risk_profile", "balanced"),
        "updated_at":      datetime.now().isoformat(),
    }
    if existing.data:
        sb_client.table("fo_shared_portfolios").update(payload).eq("id", existing.data[0]["id"]).execute()
    else:
        sb_client.table("fo_shared_portfolios").insert(payload).execute()
    emit_activity(group_id, user_id, "portfolio_share", {"positions": len(validated)})
    return {"success": True}


def _aggregate_positions(positions) -> list:
    if isinstance(positions, str):
        positions = json.loads(positions)
    agg: dict = {}
    for p in (positions or []):
        t = p.get("ticker") or p.get("symbol", "")
        if t in agg:
            agg[t]["weight_pct"] = agg[t].get("weight_pct", 0) + p.get("weight_pct", 0)
        else:
            agg[t] = {**p}
    return list(agg.values())


@app.get("/fo/groups/{group_id}/portfolios")
def fo_group_portfolios(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(status_code=403, detail="Nicht Mitglied")
    portfolios = sb_client.table("fo_shared_portfolios").select("*").eq("group_id", group_id).execute()
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    member_map = {m["user_id"]: m for m in members.data}
    enriched = []
    for p in portfolios.data:
        m = member_map.get(p["user_id"], {})
        enriched.append({
            **p,
            "positions":    _aggregate_positions(p.get("positions") or []),
            "display_name": m.get("display_name", "Unbekannt"),
            "avatar_emoji": m.get("avatar_emoji", "👤"),
            "is_me":        p["user_id"] == user_id,
        })
    return {"portfolios": enriched}


@app.get("/fo/groups/{group_id}/aggregate")
def fo_group_aggregate(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(status_code=403, detail="Nicht Mitglied")

    portfolios = sb_client.table("fo_shared_portfolios").select("*").eq("group_id", group_id).execute()
    sector_weights: dict = {}
    symbol_holders: dict = {}
    n = len(portfolios.data) or 1

    for p in portfolios.data:
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        for pos in raw:
            sym = pos.get("symbol", "")
            sec = pos.get("sector", "Sonstige")
            w   = float(pos.get("weight_pct", 0))
            sector_weights[sec] = sector_weights.get(sec, 0) + (w / n)
            symbol_holders[sym] = symbol_holders.get(sym, 0) + 1

    return {
        "sectors": sorted(
            [{"sector": k, "weight_pct": round(v, 1)} for k, v in sector_weights.items()],
            key=lambda x: x["weight_pct"], reverse=True,
        ),
        "overlap_symbols":        {sym: cnt for sym, cnt in symbol_holders.items() if cnt > 1},
        "total_unique_symbols":   len(symbol_holders),
        "members_with_portfolio": len(portfolios.data),
    }


# ─── FO Chat & Posts ──────────────────────────────────────────────────────────

REACTION_EMOJIS_ALLOWED = {'👍', '🚀', '💎', '🐻', '🔥', '🤔'}


@app.get("/fo/groups/{group_id}/messages")
def fo_messages(group_id: str, limit: int = 100, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    msgs = sb_client.table("fo_messages").select("*").eq("group_id", group_id).order("created_at").limit(limit).execute()
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    mmap = {m["user_id"]: m for m in members.data}
    return {"messages": [{
        **msg,
        "display_name": mmap.get(msg["user_id"], {}).get("display_name", "Unbekannt"),
        "avatar_emoji": mmap.get(msg["user_id"], {}).get("avatar_emoji", "👤"),
        "is_me": msg["user_id"] == user_id,
    } for msg in msgs.data]}


@app.post("/fo/groups/{group_id}/messages")
async def create_fo_message(group_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "Nachricht leer")
    result = sb_client.table("fo_messages").insert({
        "group_id": group_id, "user_id": user_id, "content": content[:2000],
    }).execute()
    return {"message": result.data[0]}


@app.delete("/fo/messages/{message_id}")
def delete_fo_message(message_id: str, user_id: str = Depends(verify_jwt)):
    msg = sb_client.table("fo_messages").select("user_id").eq("id", message_id).execute()
    if not msg.data:
        raise HTTPException(404, "Nicht gefunden")
    if msg.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Keine Berechtigung")
    sb_client.table("fo_messages").delete().eq("id", message_id).execute()
    return {"success": True}


@app.get("/fo/groups/{group_id}/posts")
def fo_posts(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")

    posts = sb_client.table("fo_posts").select("*").eq("group_id", group_id).order("created_at", desc=True).execute()
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    mmap = {m["user_id"]: m for m in members.data}

    post_ids = [p["id"] for p in posts.data] or ["00000000-0000-0000-0000-000000000000"]
    reactions_all = sb_client.table("fo_reactions").select("post_id, user_id, emoji").in_("post_id", post_ids).execute()
    comments_all  = sb_client.table("fo_comments").select("post_id").in_("post_id", post_ids).execute()

    react_map: dict  = {}
    my_react_map: dict = {}
    for r in reactions_all.data:
        pid = r["post_id"]
        react_map.setdefault(pid, {})
        react_map[pid][r["emoji"]] = react_map[pid].get(r["emoji"], 0) + 1
        if r["user_id"] == user_id:
            my_react_map.setdefault(pid, [])
            my_react_map[pid].append(r["emoji"])

    comment_count: dict = {}
    for c in comments_all.data:
        comment_count[c["post_id"]] = comment_count.get(c["post_id"], 0) + 1

    return {"posts": [{
        **p,
        "display_name":  mmap.get(p["user_id"], {}).get("display_name", "Unbekannt"),
        "avatar_emoji":  mmap.get(p["user_id"], {}).get("avatar_emoji", "👤"),
        "is_me":         p["user_id"] == user_id,
        "reactions":     react_map.get(p["id"], {}),
        "my_reactions":  my_react_map.get(p["id"], []),
        "comment_count": comment_count.get(p["id"], 0),
    } for p in posts.data]}


@app.post("/fo/groups/{group_id}/posts")
async def create_fo_post(group_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    post_type = data.get("post_type", "thesis")
    if post_type not in ("buy", "sell", "thesis", "question"):
        raise HTTPException(400, "Ungültiger post_type")
    result = sb_client.table("fo_posts").insert({
        "group_id":  group_id,
        "user_id":   user_id,
        "post_type": post_type,
        "symbol":    (data.get("symbol") or "").strip().upper()[:10] or None,
        "title":     (data.get("title") or "").strip()[:200],
        "thesis":    (data.get("thesis") or "").strip()[:2000],
    }).execute()
    emit_activity(group_id, user_id, "post", {"title": (data.get("title") or "").strip()[:80]})
    # Achievements
    post_count = sb_client.table("fo_posts").select("id", count="exact").eq("group_id", group_id).eq("user_id", user_id).execute()
    if (post_count.count or 0) == 1:
        check_and_unlock(group_id, user_id, "first_post")
    thesis_count = sb_client.table("fo_posts").select("id", count="exact").eq("group_id", group_id).eq("user_id", user_id).neq("thesis", "").execute()
    if (thesis_count.count or 0) >= 5:
        check_and_unlock(group_id, user_id, "thesis_master")
    return {"post": result.data[0]}


@app.delete("/fo/posts/{post_id}")
def delete_fo_post(post_id: str, user_id: str = Depends(verify_jwt)):
    post = sb_client.table("fo_posts").select("user_id").eq("id", post_id).execute()
    if not post.data:
        raise HTTPException(404, "Nicht gefunden")
    if post.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Keine Berechtigung")
    sb_client.table("fo_comments").delete().eq("post_id", post_id).execute()
    sb_client.table("fo_reactions").delete().eq("post_id", post_id).execute()
    sb_client.table("fo_posts").delete().eq("id", post_id).execute()
    return {"success": True}


@app.post("/fo/posts/{post_id}/react")
async def toggle_fo_reaction(post_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    emoji = data.get("emoji", "")
    if emoji not in REACTION_EMOJIS_ALLOWED:
        raise HTTPException(400, "Ungültiges Emoji")
    existing = sb_client.table("fo_reactions").select("id").eq("post_id", post_id).eq("user_id", user_id).eq("emoji", emoji).execute()
    if existing.data:
        sb_client.table("fo_reactions").delete().eq("id", existing.data[0]["id"]).execute()
        return {"action": "removed"}
    sb_client.table("fo_reactions").insert({"post_id": post_id, "user_id": user_id, "emoji": emoji}).execute()
    return {"action": "added"}


@app.get("/fo/posts/{post_id}/comments")
def fo_comments(post_id: str, user_id: str = Depends(verify_jwt)):
    post = sb_client.table("fo_posts").select("group_id").eq("id", post_id).execute()
    if not post.data:
        raise HTTPException(404, "Post nicht gefunden")
    group_id = post.data[0]["group_id"]
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    comments = sb_client.table("fo_comments").select("*").eq("post_id", post_id).order("created_at").execute()
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    mmap = {m["user_id"]: m for m in members.data}
    return {"comments": [{
        **c,
        "display_name": mmap.get(c["user_id"], {}).get("display_name", "Unbekannt"),
        "avatar_emoji": mmap.get(c["user_id"], {}).get("avatar_emoji", "👤"),
        "is_me":        c["user_id"] == user_id,
    } for c in comments.data]}


@app.post("/fo/posts/{post_id}/comments")
async def create_fo_comment(post_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    post = sb_client.table("fo_posts").select("group_id").eq("id", post_id).execute()
    if not post.data:
        raise HTTPException(404, "Post nicht gefunden")
    group_id = post.data[0]["group_id"]
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "Kommentar leer")
    result = sb_client.table("fo_comments").insert({
        "post_id": post_id, "user_id": user_id, "content": content[:1000],
    }).execute()
    return {"comment": result.data[0]}


# ─── FO Notifications, Leaderboard, Polls ─────────────────────────────────────

@app.post("/fo/groups/{group_id}/mark-read")
def fo_mark_read(group_id: str, tab: str = Query(...), user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    if tab not in ("messages", "posts", "polls"):
        raise HTTPException(400, "Ungültiger tab")
    field = f"last_read_{tab}_at"
    now_str = datetime.utcnow().isoformat() + "Z"
    existing = sb_client.table("fo_last_read").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if existing.data:
        sb_client.table("fo_last_read").update({field: now_str}).eq("id", existing.data[0]["id"]).execute()
    else:
        sb_client.table("fo_last_read").insert({"user_id": user_id, "group_id": group_id, field: now_str}).execute()
    return {"success": True}


@app.get("/fo/groups/{group_id}/unread-counts")
def fo_unread_counts(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    lr = sb_client.table("fo_last_read").select("*").eq("group_id", group_id).eq("user_id", user_id).execute()
    last_read = lr.data[0] if lr.data else {}
    epoch = "1970-01-01T00:00:00"
    lr_msgs  = last_read.get("last_read_messages_at", epoch) or epoch
    lr_posts = last_read.get("last_read_posts_at",    epoch) or epoch
    lr_polls = last_read.get("last_read_polls_at",    epoch) or epoch
    msgs  = sb_client.table("fo_messages").select("id", count="exact").eq("group_id", group_id).neq("user_id", user_id).gt("created_at", lr_msgs).execute()
    posts = sb_client.table("fo_posts").select("id", count="exact").eq("group_id", group_id).neq("user_id", user_id).gt("created_at", lr_posts).execute()
    polls = sb_client.table("fo_polls").select("id", count="exact").eq("group_id", group_id).neq("user_id", user_id).gt("created_at", lr_polls).execute()
    return {
        "messages": msgs.count or 0,
        "posts":    posts.count or 0,
        "polls":    polls.count or 0,
    }


@app.get("/fo/groups/{group_id}/leaderboard")
def fo_leaderboard(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    portfolios = sb_client.table("fo_shared_portfolios").select("*").eq("group_id", group_id).execute()
    if not portfolios.data:
        return {"leaderboard": [], "sp500_ytd": 0.0}
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    mmap = {m["user_id"]: m for m in members.data}

    all_symbols: set = set()
    for p in portfolios.data:
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        for pos in raw:
            sym = pos.get("symbol", "")
            if sym:
                all_symbols.add(sym)
    all_symbols.add("SPY")

    ytd_year = datetime.now().year
    ytd_start = f"{ytd_year}-01-01"
    ytd_start_end = f"{ytd_year}-01-10"
    price_start: dict = {}
    price_now: dict = {}

    for sym in all_symbols:
        try:
            hist = requests.get(
                f"{EODHD_BASE}/eod/{sym}.US",
                params={"from": ytd_start, "to": ytd_start_end, "api_token": EODHD_API_KEY, "fmt": "json"},
                timeout=8,
            ).json()
            if isinstance(hist, list) and hist:
                price_start[sym] = float(hist[0]["close"])
            rt = requests.get(
                f"{EODHD_BASE}/real-time/{sym}.US",
                params={"api_token": EODHD_API_KEY, "fmt": "json"},
                timeout=8,
            ).json()
            if isinstance(rt, dict):
                price_now[sym] = float(rt.get("close") or rt.get("adjusted_close") or 0)
        except Exception:
            pass

    sp500_ytd = 0.0
    if price_start.get("SPY", 0) > 0 and price_now.get("SPY", 0) > 0:
        sp500_ytd = (price_now["SPY"] - price_start["SPY"]) / price_start["SPY"] * 100

    entries = []
    for p in portfolios.data:
        m = mmap.get(p["user_id"], {})
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        ytd = 0.0
        total_w = 0.0
        for pos in raw:
            sym = pos.get("symbol", "")
            w = float(pos.get("weight_pct", 0)) / 100
            ps = price_start.get(sym, 0)
            pn = price_now.get(sym, 0)
            if ps > 0 and pn > 0:
                ytd += ((pn - ps) / ps) * w
                total_w += w
        ytd_pct = (ytd / total_w * 100) if total_w > 0 else 0.0
        entries.append({
            "user_id":      p["user_id"],
            "display_name": m.get("display_name", "Unbekannt"),
            "avatar":       m.get("avatar_emoji", "👤"),
            "ytd_pct":      round(ytd_pct, 2),
            "vs_sp500_pct": round(ytd_pct - sp500_ytd, 2),
            "is_me":        p["user_id"] == user_id,
        })

    entries.sort(key=lambda x: x["ytd_pct"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return {"leaderboard": entries, "sp500_ytd": round(sp500_ytd, 2)}


@app.post("/fo/groups/{group_id}/polls")
async def create_fo_poll(group_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    question = (data.get("question") or "").strip()
    options_raw = data.get("options", [])
    if not question or len([o for o in options_raw if str(o).strip()]) < 2:
        raise HTTPException(400, "Frage und mindestens 2 Optionen erforderlich")
    options = [{"id": str(i), "label": str(o).strip()[:100]} for i, o in enumerate(options_raw[:6]) if str(o).strip()]
    closes_at = None
    try:
        hours = int(data.get("closes_in_hours") or 0)
        if hours > 0:
            closes_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat() + "Z"
    except (ValueError, TypeError):
        pass
    result = sb_client.table("fo_polls").insert({
        "group_id": group_id, "user_id": user_id,
        "question": question[:500], "options": options, "closes_at": closes_at,
    }).execute()
    emit_activity(group_id, user_id, "poll_create", {"title": question[:80]})
    # Achievements
    poll_count = sb_client.table("fo_polls").select("id", count="exact").eq("group_id", group_id).eq("user_id", user_id).execute()
    if (poll_count.count or 0) == 1:
        check_and_unlock(group_id, user_id, "debate_starter")
    if (poll_count.count or 0) >= 3:
        check_and_unlock(group_id, user_id, "consensus_builder")
    return {"poll": result.data[0]}


@app.get("/fo/groups/{group_id}/polls")
def fo_polls_list(group_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    polls = sb_client.table("fo_polls").select("*").eq("group_id", group_id).order("created_at", desc=True).execute()
    if not polls.data:
        return {"polls": []}
    poll_ids = [p["id"] for p in polls.data]
    votes_all = sb_client.table("fo_poll_votes").select("poll_id, user_id, option_id").in_("poll_id", poll_ids).execute()
    members = sb_client.table("fo_members").select("user_id, display_name, avatar_emoji").eq("group_id", group_id).execute()
    mmap = {m["user_id"]: m for m in members.data}
    vote_counts: dict = {}
    my_vote: dict = {}
    for v in votes_all.data:
        pid = v["poll_id"]
        vote_counts.setdefault(pid, {})
        vote_counts[pid][v["option_id"]] = vote_counts[pid].get(v["option_id"], 0) + 1
        if v["user_id"] == user_id:
            my_vote[pid] = v["option_id"]
    result = []
    for p in polls.data:
        m = mmap.get(p["user_id"], {})
        is_closed = p.get("is_closed", False)
        if p.get("closes_at") and not is_closed:
            try:
                if datetime.fromisoformat(p["closes_at"].replace("Z", "")) < datetime.utcnow():
                    is_closed = True
            except Exception:
                pass
        counts = vote_counts.get(p["id"], {})
        result.append({
            **p,
            "display_name": m.get("display_name", "Unbekannt"),
            "avatar_emoji": m.get("avatar_emoji", "👤"),
            "is_mine":      p["user_id"] == user_id,
            "is_closed":    is_closed,
            "vote_counts":  counts,
            "my_vote":      my_vote.get(p["id"]),
            "total_votes":  sum(counts.values()),
        })
    return {"polls": result}


@app.post("/fo/polls/{poll_id}/vote")
async def fo_vote(poll_id: str, request: Request, user_id: str = Depends(verify_jwt)):
    data = await request.json()
    poll = sb_client.table("fo_polls").select("*").eq("id", poll_id).execute()
    if not poll.data:
        raise HTTPException(404, "Umfrage nicht gefunden")
    p = poll.data[0]
    if p.get("is_closed"):
        raise HTTPException(400, "Umfrage ist geschlossen")
    membership = sb_client.table("fo_members").select("id").eq("group_id", p["group_id"]).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    option_id = str(data.get("option_id", ""))
    valid_ids = [str(o["id"]) for o in (p.get("options") or [])]
    if option_id not in valid_ids:
        raise HTTPException(400, "Ungültige Option")
    existing = sb_client.table("fo_poll_votes").select("id").eq("poll_id", poll_id).eq("user_id", user_id).execute()
    if existing.data:
        sb_client.table("fo_poll_votes").update({"option_id": option_id}).eq("id", existing.data[0]["id"]).execute()
    else:
        sb_client.table("fo_poll_votes").insert({"poll_id": poll_id, "user_id": user_id, "option_id": option_id}).execute()
    return {"success": True}


@app.post("/fo/polls/{poll_id}/close")
def fo_close_poll(poll_id: str, user_id: str = Depends(verify_jwt)):
    poll = sb_client.table("fo_polls").select("user_id").eq("id", poll_id).execute()
    if not poll.data:
        raise HTTPException(404, "Umfrage nicht gefunden")
    if poll.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Nur der Ersteller kann die Umfrage schließen")
    sb_client.table("fo_polls").update({"is_closed": True}).eq("id", poll_id).execute()
    return {"success": True}


@app.delete("/fo/polls/{poll_id}")
def fo_delete_poll(poll_id: str, user_id: str = Depends(verify_jwt)):
    poll = sb_client.table("fo_polls").select("user_id").eq("id", poll_id).execute()
    if not poll.data:
        raise HTTPException(404, "Umfrage nicht gefunden")
    if poll.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Keine Berechtigung")
    sb_client.table("fo_poll_votes").delete().eq("poll_id", poll_id).execute()
    sb_client.table("fo_polls").delete().eq("id", poll_id).execute()
    return {"success": True}


@app.get("/fo/groups/{group_id}/member-profile/{target_user_id}")
def fo_member_profile(group_id: str, target_user_id: str, user_id: str = Depends(verify_jwt)):
    membership = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not membership.data:
        raise HTTPException(403, "Nicht Mitglied")
    target = sb_client.table("fo_members").select("*").eq("group_id", group_id).eq("user_id", target_user_id).execute()
    if not target.data:
        raise HTTPException(404, "Mitglied nicht gefunden")
    mem = target.data[0]

    league = None
    xp = None
    try:
        prog = sb_client.table("user_progress").select("league, xp_total").eq("user_id", target_user_id).execute()
        if prog.data:
            league = prog.data[0].get("league")
            xp = prog.data[0].get("xp_total")
    except Exception:
        pass

    port = sb_client.table("fo_shared_portfolios").select("positions").eq("group_id", group_id).eq("user_id", target_user_id).execute()
    positions = []
    ytd_pct = None
    if port.data:
        raw = port.data[0].get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        positions = raw
        try:
            ytd_year = datetime.now().year
            ytd_start = f"{ytd_year}-01-01"
            ytd_end = f"{ytd_year}-01-10"
            price_start: dict = {}
            price_now: dict = {}
            syms = list({pos.get("symbol") for pos in positions if pos.get("symbol")})
            for sym in syms:
                hist = requests.get(
                    f"{EODHD_BASE}/eod/{sym}.US",
                    params={"from": ytd_start, "to": ytd_end, "api_token": EODHD_API_KEY, "fmt": "json"},
                    timeout=5,
                ).json()
                if isinstance(hist, list) and hist:
                    price_start[sym] = float(hist[0]["close"])
                rt = requests.get(
                    f"{EODHD_BASE}/real-time/{sym}.US",
                    params={"api_token": EODHD_API_KEY, "fmt": "json"},
                    timeout=5,
                ).json()
                if isinstance(rt, dict):
                    price_now[sym] = float(rt.get("close") or rt.get("adjusted_close") or 0)
            ytd = 0.0
            total_w = 0.0
            for pos in positions:
                sym = pos.get("symbol", "")
                w = float(pos.get("weight_pct", 0)) / 100
                ps = price_start.get(sym, 0)
                pn = price_now.get(sym, 0)
                if ps > 0 and pn > 0:
                    ytd += ((pn - ps) / ps) * w
                    total_w += w
            if total_w > 0:
                ytd_pct = round((ytd / total_w) * 100, 2)
        except Exception:
            ytd_pct = None

    return {
        "display_name": mem.get("display_name", "Unbekannt"),
        "avatar_emoji": mem.get("avatar_emoji", "👤"),
        "joined_at":    mem.get("joined_at"),
        "role":         mem.get("role", "member"),
        "ytd_pct":      ytd_pct,
        "league":       league,
        "xp":           xp,
        "positions":    positions,
    }


# ─── Gruppen (Sprint 34) ───────────────────────────────────────────────────────

ACHIEVEMENTS = {
    "first_post":        {"name": "Erster Post",        "icon": "✍️"},
    "consensus_builder": {"name": "Konsens-Baumeister", "icon": "🤝"},
    "first_idea":        {"name": "Ideen-Starter",      "icon": "💡"},
    "thesis_master":     {"name": "These-Meister",      "icon": "📝"},
    "early_bird":        {"name": "Early Bird",         "icon": "🐦"},
    "top_performer":     {"name": "Top Performer",      "icon": "🏆"},
    "debate_starter":    {"name": "Debate-Starter",     "icon": "🔥"},
    "watchlist_curator": {"name": "Watchlist-Kurator",  "icon": "👁"},
}

SP500_SECTOR_WEIGHTS = {
    "Technologie": 28, "Gesundheit": 13, "Finanzen": 13, "Konsum": 10,
    "Industrie": 9, "Kommunikation": 9, "Energie": 4, "Materialien": 3,
    "Immobilien": 3, "Versorger": 3, "Sonstige": 5,
}

def emit_activity(group_id: str, user_id: str, activity_type: str, payload: dict):
    try:
        mem = sb_client.table("fo_members").select("display_name,avatar_emoji").eq("group_id", group_id).eq("user_id", user_id).execute()
        display_name = mem.data[0].get("display_name", "?") if mem.data else "?"
        avatar_emoji = mem.data[0].get("avatar_emoji", "👤") if mem.data else "👤"
        sb_client.table("group_activity").insert({
            "group_id": group_id, "user_id": user_id,
            "activity_type": activity_type,
            "payload": {**payload, "display_name": display_name, "avatar_emoji": avatar_emoji},
        }).execute()
    except Exception:
        pass

def check_and_unlock(group_id: str, user_id: str, achievement_id: str):
    try:
        sb_client.table("group_achievements").insert({
            "group_id": group_id, "user_id": user_id, "achievement_id": achievement_id,
        }).execute()
        meta = ACHIEVEMENTS.get(achievement_id, {})
        emit_activity(group_id, user_id, "achievement", {"achievement_id": achievement_id, "achievement_name": meta.get("name", achievement_id)})
    except Exception:
        pass  # unique constraint = already unlocked, ignore


@app.get("/groups/{group_id}/activity")
def groups_activity(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    rows = sb_client.table("group_activity").select("*").eq("group_id", group_id).order("created_at", desc=True).limit(60).execute()
    activities = []
    for r in (rows.data or []):
        p = r.get("payload") or {}
        activities.append({
            "id":            r["id"],
            "user_id":       r["user_id"],
            "activity_type": r["activity_type"],
            "display_name":  p.get("display_name", "?"),
            "avatar_emoji":  p.get("avatar_emoji", "👤"),
            "payload":       {k: v for k, v in p.items() if k not in ("display_name", "avatar_emoji")},
            "created_at":    r["created_at"],
        })
    return {"activities": activities}


@app.get("/groups/{group_id}/performance-history")
def groups_performance_history(group_id: str, period: str = "1m", user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")

    cache_key = f"perf-history:{group_id}:{period}"
    cached = cache.get(cache_key, 21600)
    if cached:
        return cached

    # Date range
    today = datetime.now().date()
    if period == "1m":
        from_date = today - timedelta(days=31)
    elif period == "3m":
        from_date = today - timedelta(days=93)
    else:  # ytd
        from_date = datetime(today.year, 1, 1).date()

    portfolios = sb_client.table("fo_shared_portfolios").select("user_id,positions").eq("group_id", group_id).execute()
    if not portfolios.data:
        return {"chartData": [], "members": {}}

    member_rows = sb_client.table("fo_members").select("user_id,display_name,avatar_emoji").eq("group_id", group_id).execute()
    member_map  = {m["user_id"]: m for m in (member_rows.data or [])}

    # Collect all unique symbols
    all_symbols = set()
    port_map    = {}
    for p in portfolios.data:
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        positions = [pos for pos in raw if pos.get("symbol") and pos.get("weight_pct", 0) > 0]
        port_map[p["user_id"]] = positions
        all_symbols.update(pos["symbol"] for pos in positions)

    # Fetch EOD history for each symbol
    price_history = {}
    for sym in all_symbols:
        try:
            hist = requests.get(
                f"{EODHD_BASE}/eod/{sym}.US",
                params={"from": str(from_date), "to": str(today), "api_token": EODHD_API_KEY, "fmt": "json"},
                timeout=8,
            ).json()
            if isinstance(hist, list):
                price_history[sym] = {h["date"]: float(h["adjusted_close"] or h["close"]) for h in hist}
        except Exception:
            pass

    # Get all dates
    all_dates = sorted({d for ph in price_history.values() for d in ph.keys()})
    if not all_dates:
        return {"chartData": [], "members": {}}

    chart_data = []
    for date in all_dates:
        row = {"date": date}
        for uid, positions in port_map.items():
            total_w = sum(float(pos.get("weight_pct", 0)) / 100 for pos in positions)
            if total_w <= 0:
                continue
            pct = 0.0
            for pos in positions:
                sym  = pos["symbol"]
                w    = float(pos.get("weight_pct", 0)) / 100
                ph   = price_history.get(sym, {})
                # Base price = first available date
                dates_sorted = sorted(ph.keys())
                base = ph.get(dates_sorted[0]) if dates_sorted else None
                curr = ph.get(date)
                if base and curr and base > 0:
                    pct += ((curr - base) / base) * (w / total_w)
            row[uid] = round(pct * 100, 2)
        chart_data.append(row)

    members_out = {uid: {"display_name": member_map.get(uid, {}).get("display_name", "?"),
                         "avatar_emoji": member_map.get(uid, {}).get("avatar_emoji", "👤")}
                   for uid in port_map.keys()}
    result = {"chartData": chart_data, "members": members_out}
    cache.set(cache_key, result)
    return result


# ── Group Watchlist ────────────────────────────────────────────────────────────

class WatchlistAddBody(BaseModel):
    ticker: str

@app.post("/groups/{group_id}/watchlist")
def groups_watchlist_add(group_id: str, body: WatchlistAddBody, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    ticker = body.ticker.upper().split(".")[0]
    try:
        sb_client.table("group_watchlist").insert({"group_id": group_id, "user_id": user_id, "ticker": ticker}).execute()
    except Exception:
        raise HTTPException(409, "Ticker bereits in Watchlist")

    emit_activity(group_id, user_id, "watchlist_add", {"ticker": ticker})
    # Achievement: watchlist_curator after 5 entries
    count = sb_client.table("group_watchlist").select("id", count="exact").eq("group_id", group_id).eq("user_id", user_id).execute()
    if (count.count or 0) >= 5:
        check_and_unlock(group_id, user_id, "watchlist_curator")
    return {"success": True}


@app.delete("/groups/{group_id}/watchlist/{ticker}")
def groups_watchlist_remove(group_id: str, ticker: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    sb_client.table("group_watchlist").delete().eq("group_id", group_id).eq("user_id", user_id).eq("ticker", ticker.upper()).execute()
    return {"success": True}


@app.get("/groups/{group_id}/watchlist")
def groups_watchlist_list(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")

    rows     = sb_client.table("group_watchlist").select("*").eq("group_id", group_id).order("created_at").execute()
    mem_rows = sb_client.table("fo_members").select("user_id,display_name,avatar_emoji").eq("group_id", group_id).execute()
    mem_map  = {m["user_id"]: m for m in (mem_rows.data or [])}

    items = []
    for r in (rows.data or []):
        ticker = r["ticker"]
        price  = None
        chg    = None
        try:
            rt = requests.get(
                f"{EODHD_BASE}/real-time/{ticker}.US",
                params={"api_token": EODHD_API_KEY, "fmt": "json"},
                timeout=4,
            ).json()
            if isinstance(rt, dict):
                price = float(rt.get("close") or rt.get("adjusted_close") or 0) or None
                chg   = float(rt.get("change_p", 0) or 0)
        except Exception:
            pass
        mem = mem_map.get(r["user_id"], {})
        items.append({
            "ticker":       ticker,
            "user_id":      r["user_id"],
            "display_name": mem.get("display_name", "?"),
            "avatar_emoji": mem.get("avatar_emoji", "👤"),
            "is_mine":      r["user_id"] == user_id,
            "created_at":   r["created_at"],
            "price":        price,
            "change_pct":   chg,
        })
    return {"items": items}


# ── Group Trade Ideas ──────────────────────────────────────────────────────────

class TradeIdeaBody(BaseModel):
    ticker:    str
    direction: str
    title:     str
    rationale: str = ""

class IdeaVoteBody(BaseModel):
    vote: str  # "up", "down", "none"

@app.post("/groups/{group_id}/trade-ideas")
def groups_ideas_create(group_id: str, body: TradeIdeaBody, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    if body.direction not in ("long", "short"):
        raise HTTPException(400, "direction muss long oder short sein")

    sb_client.table("group_trade_ideas").insert({
        "group_id": group_id, "user_id": user_id,
        "ticker": body.ticker.upper(), "direction": body.direction,
        "title": body.title, "rationale": body.rationale,
    }).execute()

    emit_activity(group_id, user_id, "trade_idea", {"ticker": body.ticker.upper(), "title": body.title})
    # Check first_idea achievement
    count = sb_client.table("group_trade_ideas").select("id", count="exact").eq("group_id", group_id).eq("user_id", user_id).execute()
    if (count.count or 0) == 1:
        check_and_unlock(group_id, user_id, "first_idea")
    return {"success": True}


@app.get("/groups/{group_id}/trade-ideas")
def groups_ideas_list(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")

    rows     = sb_client.table("group_trade_ideas").select("*").eq("group_id", group_id).order("created_at", desc=True).execute()
    mem_rows = sb_client.table("fo_members").select("user_id,display_name,avatar_emoji").eq("group_id", group_id).execute()
    mem_map  = {m["user_id"]: m for m in (mem_rows.data or [])}

    ideas = []
    for r in (rows.data or []):
        votes = sb_client.table("group_trade_idea_votes").select("user_id,vote").eq("idea_id", r["id"]).execute()
        upvotes   = sum(1 for v in (votes.data or []) if v["vote"] == "up")
        downvotes = sum(1 for v in (votes.data or []) if v["vote"] == "down")
        my_vote   = next((v["vote"] for v in (votes.data or []) if v["user_id"] == user_id), None)
        mem = mem_map.get(r["user_id"], {})
        ideas.append({
            "id":           r["id"],
            "user_id":      r["user_id"],
            "display_name": mem.get("display_name", "?"),
            "avatar_emoji": mem.get("avatar_emoji", "👤"),
            "ticker":       r["ticker"],
            "direction":    r["direction"],
            "title":        r["title"],
            "rationale":    r.get("rationale", ""),
            "is_closed":    r.get("is_closed", False),
            "created_at":   r["created_at"],
            "upvotes":      upvotes,
            "downvotes":    downvotes,
            "my_vote":      my_vote,
            "is_mine":      r["user_id"] == user_id,
        })
    return {"ideas": ideas}


@app.post("/groups/{group_id}/trade-ideas/{idea_id}/vote")
def groups_ideas_vote(group_id: str, idea_id: str, body: IdeaVoteBody, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    if body.vote == "none":
        sb_client.table("group_trade_idea_votes").delete().eq("idea_id", idea_id).eq("user_id", user_id).execute()
    else:
        existing = sb_client.table("group_trade_idea_votes").select("id").eq("idea_id", idea_id).eq("user_id", user_id).execute()
        if existing.data:
            sb_client.table("group_trade_idea_votes").update({"vote": body.vote}).eq("id", existing.data[0]["id"]).execute()
        else:
            sb_client.table("group_trade_idea_votes").insert({"idea_id": idea_id, "user_id": user_id, "vote": body.vote}).execute()
    return {"success": True}


@app.post("/groups/{group_id}/trade-ideas/{idea_id}/close")
def groups_ideas_close(group_id: str, idea_id: str, user_id: str = Depends(verify_jwt)):
    idea = sb_client.table("group_trade_ideas").select("user_id").eq("id", idea_id).execute()
    if not idea.data:
        raise HTTPException(404, "Nicht gefunden")
    if idea.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Keine Berechtigung")
    sb_client.table("group_trade_ideas").update({"is_closed": True}).eq("id", idea_id).execute()
    return {"success": True}


@app.delete("/groups/{group_id}/trade-ideas/{idea_id}")
def groups_ideas_delete(group_id: str, idea_id: str, user_id: str = Depends(verify_jwt)):
    idea = sb_client.table("group_trade_ideas").select("user_id").eq("id", idea_id).execute()
    if not idea.data:
        raise HTTPException(404, "Nicht gefunden")
    if idea.data[0]["user_id"] != user_id:
        raise HTTPException(403, "Keine Berechtigung")
    sb_client.table("group_trade_idea_votes").delete().eq("idea_id", idea_id).execute()
    sb_client.table("group_trade_ideas").delete().eq("id", idea_id).execute()
    return {"success": True}


# ── Group Sector Allocation ────────────────────────────────────────────────────

def lookup_sector(ticker: str) -> str:
    base = ticker.split(".")[0].upper()
    return TICKER_SECTOR_MAP.get(base, "Sonstige")


@app.get("/groups/{group_id}/sector-allocation")
def groups_sector_allocation(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")

    portfolios = sb_client.table("fo_shared_portfolios").select("positions").eq("group_id", group_id).execute()
    sector_totals: dict = {}
    member_count = 0

    for p in (portfolios.data or []):
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not raw:
            continue
        member_count += 1
        for pos in raw:
            ticker = pos.get("symbol", "")
            sector = lookup_sector(ticker) if ticker else "Sonstige"
            sector_totals[sector] = sector_totals.get(sector, 0) + float(pos.get("weight_pct", 0))

    if member_count == 0:
        return {"group_allocation": {}, "sp500_weights": SP500_SECTOR_WEIGHTS}

    group_allocation = {k: round(v / member_count, 1) for k, v in sector_totals.items()}
    return {"group_allocation": group_allocation, "sp500_weights": SP500_SECTOR_WEIGHTS}


# ── Group KI Briefing ──────────────────────────────────────────────────────────

@app.get("/groups/{group_id}/briefing/latest")
def groups_briefing_latest(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    row = sb_client.table("group_briefings").select("*").eq("group_id", group_id).order("generated_at", desc=True).limit(1).execute()
    if not row.data:
        return {"briefing": None}
    b = row.data[0]
    return {"briefing": {"content": b["content"], "generated_at": b["generated_at"]}}


@app.post("/groups/{group_id}/briefing/generate")
def groups_briefing_generate(group_id: str, user_id: str = Depends(verify_jwt)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")

    # Gather context
    portfolios = sb_client.table("fo_shared_portfolios").select("positions").eq("group_id", group_id).execute()
    ideas_rows = sb_client.table("group_trade_ideas").select("ticker,direction,title").eq("group_id", group_id).eq("is_closed", False).limit(10).execute()
    polls_rows = sb_client.table("fo_polls").select("question").eq("group_id", group_id).eq("is_closed", False).limit(5).execute()

    all_tickers = set()
    for p in (portfolios.data or []):
        raw = p.get("positions") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        all_tickers.update(pos.get("symbol", "") for pos in raw if pos.get("symbol"))

    ideas_text = "\n".join(f"- {i['ticker']} ({i['direction']}): {i['title']}" for i in (ideas_rows.data or []))
    polls_text = "\n".join(f"- {q['question']}" for q in (polls_rows.data or []))
    tickers_text = ", ".join(sorted(all_tickers)[:20])

    prompt = f"""Du bist ein kompakter KI-Marktanalyst für eine private Investment-Gruppe.

Portfolio-Positionen der Gruppe: {tickers_text or 'keine'}
Offene Trade-Ideen:
{ideas_text or 'keine'}
Aktive Diskussions-Fragen:
{polls_text or 'keine'}

Erstelle ein kurzes, prägnantes Gruppen-Briefing (max. 200 Wörter) auf Deutsch:
1. Kurze Markteinschätzung zu den gehaltenen Titeln
2. Kommentar zu den Trade-Ideen
3. Ein konkreter Ausblick / Handlungshinweis für die Gruppe
Keine Anlageberatung — nur Marktbeobachtung."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        content = resp.json()["content"][0]["text"]
    except Exception:
        raise HTTPException(500, "KI-Briefing konnte nicht generiert werden")

    now_str = datetime.utcnow().isoformat()
    sb_client.table("group_briefings").insert({"group_id": group_id, "content": content, "generated_at": now_str}).execute()
    return {"content": content, "generated_at": now_str}


# ── Group Achievements ─────────────────────────────────────────────────────────

@app.get("/groups/{group_id}/achievements")
def groups_achievements(group_id: str, user_id: str = Depends(verify_jwt), target_user_id: str = Query(None)):
    mb = sb_client.table("fo_members").select("id").eq("group_id", group_id).eq("user_id", user_id).execute()
    if not mb.data:
        raise HTTPException(403, "Nicht Mitglied")
    uid = target_user_id or user_id
    rows = sb_client.table("group_achievements").select("*").eq("group_id", group_id).eq("user_id", uid).execute()
    achievements = [{"achievement_id": r["achievement_id"], "unlocked_at": r["unlocked_at"]} for r in (rows.data or [])]
    return {"achievements": achievements}


# ─── 13F Filing Tracker ───────────────────────────────────────────────────────

INVESTORS = {
    "berkshire": {
        "id": "berkshire",
        "name": "Berkshire Hathaway",
        "manager": "Warren Buffett",
        "aum": "354B",
        "strategy": "Value Investing — Quality Compounders",
        "quarter": "Q4 2024",
        "holdings": [
            {"ticker": "AAPL",  "name": "Apple Inc.",           "pct": 28.1, "value_m": 75217, "change": "sold"},
            {"ticker": "AXP",   "name": "American Express",     "pct": 16.8, "value_m": 44975, "change": "held"},
            {"ticker": "BAC",   "name": "Bank of America",      "pct": 11.4, "value_m": 30511, "change": "sold"},
            {"ticker": "KO",    "name": "Coca-Cola",            "pct": 9.2,  "value_m": 24611, "change": "held"},
            {"ticker": "CVX",   "name": "Chevron",              "pct": 6.4,  "value_m": 17131, "change": "sold"},
            {"ticker": "OXY",   "name": "Occidental Petroleum", "pct": 5.3,  "value_m": 14181, "change": "held"},
            {"ticker": "MCO",   "name": "Moody's Corp.",        "pct": 4.9,  "value_m": 13111, "change": "held"},
            {"ticker": "KHC",   "name": "Kraft Heinz",          "pct": 2.1,  "value_m": 5616,  "change": "held"},
        ],
    },
    "bridgewater": {
        "id": "bridgewater",
        "name": "Bridgewater Associates",
        "manager": "Ray Dalio",
        "aum": "124B",
        "strategy": "All Weather / Risk Parity — Macro Diversification",
        "quarter": "Q4 2024",
        "holdings": [
            {"ticker": "SPY",   "name": "S&P 500 ETF (SPDR)",   "pct": 22.4, "value_m": 2150, "change": "bought"},
            {"ticker": "IVV",   "name": "iShares S&P 500 ETF",  "pct": 18.1, "value_m": 1738, "change": "held"},
            {"ticker": "EEM",   "name": "Emerging Markets ETF", "pct": 12.6, "value_m": 1210, "change": "sold"},
            {"ticker": "GLD",   "name": "SPDR Gold Shares",     "pct": 9.4,  "value_m": 902,  "change": "bought"},
            {"ticker": "VWO",   "name": "Vanguard EM ETF",      "pct": 7.8,  "value_m": 749,  "change": "held"},
            {"ticker": "VOO",   "name": "Vanguard S&P 500 ETF", "pct": 7.2,  "value_m": 691,  "change": "held"},
            {"ticker": "IAU",   "name": "iShares Gold Trust",   "pct": 5.4,  "value_m": 518,  "change": "bought"},
            {"ticker": "BRK.B", "name": "Berkshire Hathaway B", "pct": 4.1,  "value_m": 394,  "change": "held"},
        ],
    },
    "pershing": {
        "id": "pershing",
        "name": "Pershing Square Capital",
        "manager": "Bill Ackman",
        "aum": "18B",
        "strategy": "Concentrated Value — Activist Investing",
        "quarter": "Q4 2024",
        "holdings": [
            {"ticker": "GOOG",  "name": "Alphabet Inc. (C)",         "pct": 22.3, "value_m": 2167, "change": "bought"},
            {"ticker": "HLT",   "name": "Hilton Worldwide",          "pct": 18.7, "value_m": 1817, "change": "held"},
            {"ticker": "CMG",   "name": "Chipotle Mexican Grill",    "pct": 17.1, "value_m": 1661, "change": "held"},
            {"ticker": "QSR",   "name": "Restaurant Brands Intl.",   "pct": 13.4, "value_m": 1301, "change": "held"},
            {"ticker": "CP",    "name": "Canadian Pacific Kansas",   "pct": 12.8, "value_m": 1243, "change": "held"},
            {"ticker": "NFLX",  "name": "Netflix",                   "pct": 9.6,  "value_m": 932,  "change": "bought"},
            {"ticker": "NKE",   "name": "Nike Inc.",                 "pct": 6.1,  "value_m": 592,  "change": "bought"},
        ],
    },
    "scion": {
        "id": "scion",
        "name": "Scion Asset Management",
        "manager": "Michael Burry",
        "aum": "0.3B",
        "strategy": "Deep Value / Contrarian — Short Seller",
        "quarter": "Q3 2024",
        "holdings": [
            {"ticker": "JD",    "name": "JD.com",          "pct": 28.1, "value_m": 51,  "change": "bought"},
            {"ticker": "BABA",  "name": "Alibaba Group",   "pct": 22.4, "value_m": 41,  "change": "held"},
            {"ticker": "BIDU",  "name": "Baidu Inc.",      "pct": 16.8, "value_m": 30,  "change": "held"},
            {"ticker": "PDD",   "name": "PDD Holdings",    "pct": 13.2, "value_m": 24,  "change": "bought"},
            {"ticker": "HCA",   "name": "HCA Healthcare",  "pct": 7.4,  "value_m": 13,  "change": "sold"},
            {"ticker": "GEO",   "name": "GEO Group",       "pct": 6.1,  "value_m": 11,  "change": "bought"},
            {"ticker": "GOOGL", "name": "Alphabet (A)",    "pct": 4.3,  "value_m": 8,   "change": "sold"},
            {"ticker": "CVS",   "name": "CVS Health",      "pct": 1.7,  "value_m": 3,   "change": "bought"},
        ],
    },
    "tepper": {
        "id": "tepper",
        "name": "Appaloosa Management",
        "manager": "David Tepper",
        "aum": "22B",
        "strategy": "Distressed Assets / Growth at Reasonable Price",
        "quarter": "Q4 2024",
        "holdings": [
            {"ticker": "AMZN",  "name": "Amazon.com",     "pct": 18.4, "value_m": 1247, "change": "bought"},
            {"ticker": "META",  "name": "Meta Platforms", "pct": 16.2, "value_m": 1098, "change": "held"},
            {"ticker": "NVDA",  "name": "NVIDIA Corp.",   "pct": 14.1, "value_m": 955,  "change": "sold"},
            {"ticker": "GOOGL", "name": "Alphabet (A)",   "pct": 13.3, "value_m": 901,  "change": "held"},
            {"ticker": "AAPL",  "name": "Apple Inc.",     "pct": 10.8, "value_m": 731,  "change": "bought"},
            {"ticker": "MSFT",  "name": "Microsoft",      "pct": 9.2,  "value_m": 623,  "change": "held"},
            {"ticker": "MGM",   "name": "MGM Resorts",    "pct": 5.4,  "value_m": 366,  "change": "bought"},
            {"ticker": "JD",    "name": "JD.com",         "pct": 4.1,  "value_m": 278,  "change": "held"},
        ],
    },
}


@app.get("/13f")
async def list_13f():
    return {"funds": [
        {
            "id":            v["id"],
            "name":          v["name"],
            "manager":       v["manager"],
            "aum":           v["aum"],
            "strategy":      v["strategy"],
            "quarter":       v["quarter"],
            "top_holding":   v["holdings"][0]["ticker"] if v["holdings"] else None,
            "holding_count": len(v["holdings"]),
        }
        for v in INVESTORS.values()
    ]}


@app.get("/13f/{fund_id}")
async def get_13f(fund_id: str):
    fund = INVESTORS.get(fund_id)
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    return fund


# ─── Super-Investor Screener ──────────────────────────────────────────────────

BUFFETT_UNIVERSE = ["KO", "JNJ", "WMT", "PG", "JPM", "BAC", "AXP", "V"]
BURRY_UNIVERSE   = ["NVDA", "TSLA", "AMZN", "CRM", "SHOP", "PLTR", "NET", "SNOW"]


@app.get("/screener/buffett")
async def screener_buffett():
    cache_key = "screener_buffett"
    if cache_key in fundamentals_cache:
        cached = fundamentals_cache[cache_key]
        if datetime.now() < cached["expires"]:
            return cached["data"]

    results = []
    async with httpx.AsyncClient(timeout=20) as client:
        for i, ticker in enumerate(BUFFETT_UNIVERSE):
            if i > 0:
                await asyncio.sleep(13)
            try:
                r = await client.get(f"{AV_BASE}?function=OVERVIEW&symbol={ticker}&apikey={AV_KEY}")
                d = r.json()
                if "Symbol" not in d:
                    continue
                def safe_float(key):
                    try: return float(d.get(key) or 0)
                    except: return 0.0
                roe        = safe_float("ReturnOnEquityTTM") * 100
                pe         = safe_float("PERatio")
                pb         = safe_float("PriceToBookRatio")
                debt_eq    = safe_float("DebtToEquityRatio")
                net_margin = safe_float("ProfitMargin") * 100
                score = 0
                if roe > 15:        score += 20
                if 0 < pe < 20:     score += 20
                if 0 < pb < 5:      score += 20
                if 0 < debt_eq < 1: score += 20
                if net_margin > 10: score += 20
                results.append({
                    "ticker":     ticker,
                    "name":       d.get("Name", ticker),
                    "sector":     d.get("Sector", ""),
                    "roe":        round(roe, 1),
                    "pe":         round(pe, 1),
                    "pb":         round(pb, 2),
                    "debt_eq":    round(debt_eq, 2),
                    "net_margin": round(net_margin, 1),
                    "score":      score,
                    "verdict":    "Value-Kandidat" if score >= 60 else "Kein Fit",
                })
            except Exception as e:
                print(f"[buffett screener] {ticker}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    data = {"results": results, "timestamp": datetime.now().isoformat()}
    fundamentals_cache[cache_key] = {"data": data, "expires": datetime.now() + timedelta(hours=24)}
    return data


@app.get("/screener/burry")
async def screener_burry():
    cache_key = "screener_burry"
    if cache_key in fundamentals_cache:
        cached = fundamentals_cache[cache_key]
        if datetime.now() < cached["expires"]:
            return cached["data"]

    results = []
    async with httpx.AsyncClient(timeout=20) as client:
        for i, ticker in enumerate(BURRY_UNIVERSE):
            if i > 0:
                await asyncio.sleep(13)
            try:
                r = await client.get(f"{AV_BASE}?function=OVERVIEW&symbol={ticker}&apikey={AV_KEY}")
                d = r.json()
                if "Symbol" not in d:
                    continue
                def safe_float(key):
                    try: return float(d.get(key) or 0)
                    except: return 0.0
                pe  = safe_float("PERatio")
                pb  = safe_float("PriceToBookRatio")
                ps  = safe_float("PriceToSalesRatioTTM")
                peg = safe_float("PEGRatio")
                score = 0
                if pe  > 50: score += 25
                if pb  > 10: score += 25
                if ps  > 10: score += 25
                if peg > 2:  score += 25
                results.append({
                    "ticker":  ticker,
                    "name":    d.get("Name", ticker),
                    "sector":  d.get("Sector", ""),
                    "pe":      round(pe, 1),
                    "pb":      round(pb, 2),
                    "ps":      round(ps, 1),
                    "peg":     round(peg, 2),
                    "score":   score,
                    "verdict": "Überbewertungs-Signal" if score >= 50 else "Moderat bewertet",
                })
            except Exception as e:
                print(f"[burry screener] {ticker}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    data = {"results": results, "timestamp": datetime.now().isoformat()}
    fundamentals_cache[cache_key] = {"data": data, "expires": datetime.now() + timedelta(hours=24)}
    return data


# ─── Debug ────────────────────────────────────────────────────────────────────

@app.get("/debug-av/{ticker}")
async def debug_av(ticker: str):
    t = ticker.replace(".US", "").replace(".XETRA", "")
    async with httpx.AsyncClient(timeout=20) as client:
        bs_r = await client.get(
            f"{AV_BASE}?function=BALANCE_SHEET&symbol={t}&apikey={AV_KEY}"
        )
    data   = bs_r.json()
    annual = data.get("annualReports", [])
    if annual:
        return {
            "count":             len(annual),
            "first_report_keys": list(annual[0].keys()),
            "sample":            annual[0],
        }
    return {"raw": data}


@app.get("/test-av")
async def test_av():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{AV_BASE}?function=OVERVIEW&symbol=AAPL&apikey={AV_KEY}"
        )
        return {"status": r.status_code, "keys": list(r.json().keys())[:10]}


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

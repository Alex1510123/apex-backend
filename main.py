"""
APEX Markets — Backend API v6 (EODHD + Yahoo Finance + FRED edition)
"""
import os
import re
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from auth import verify_jwt
from supabase_client import supabase as sb_client
from yahoo_finance import fetch_yahoo_quote   # TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
from fred_api import fetch_fred_series, fetch_fred_cpi_yoy, fetch_fred_indpro_yoy, fetch_fred_yield_history
from ecb_api import fetch_ecb_series, fetch_ecb_yield_history

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
        "https://finscope-phi.vercel.app",
        "https://*.vercel.app",
        "*",
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

ACADEMY_TRACKS = [
    {
        "id": "foundation",
        "title": "Foundation",
        "subtitle": "Grundlagen des Investierens",
        "description": "Verstehen Sie die Bausteine der Kapitalmärkte: Aktien, Börsen, Indizes und Bewertungsmethoden.",
        "icon": "📚",
        "color": "#4C2EE5",
        "total_lessons": 10,
        "status": "available",
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
    },
    {
        "id": "macro",
        "title": "Makroökonomie",
        "subtitle": "Wirtschaft und Märkte verstehen",
        "description": "Zinsen, Inflation, Konjunkturzyklen und globale Zusammenhänge.",
        "icon": "🌍",
        "color": "#d29922",
        "total_lessons": 8,
        "status": "coming_soon",
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
    """Load lesson JSON files from academy_content/ and merge into ACADEMY_LESSONS_DATA."""
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
                lesson = json.load(f)
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


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

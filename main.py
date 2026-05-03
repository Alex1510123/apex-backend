"""
APEX Markets — Backend API v5 (EODHD edition)
"""
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

EODHD_API_KEY = "69ee10907be601.18560848"
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
MACRO_TICKERS = {
    "S&P 500":         "GSPC.INDX",
    "Gold":            "GLD",
    "WTI_Crude":       "UKOIL.COMMODITY",
    "10Y_Treasury":    "TNX.INDX",
    "Volatility":      "VIXY",
    "US_Dollar":       "UUP",
    "Bitcoin":         "BTC-USD.CC",
    "EUR/USD":         "EURUSD.FOREX",
    "DAX":             "GDAXI.INDX",
    "Euro Stoxx 50":   "STOXX50E.INDX",
    "ATX":             "ATX.INDX",
}

SCREEN_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "JPM",  "LLY",  "XOM",
    "V",    "UNH",  "AVGO", "HD",   "CAT",
    "ORCL", "CRM",  "AMD",  "GS",   "MA",
]

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

    # Per-ticker fallback for anything still absent after the batch call
    for t in missing:
        if t in found:
            continue
        for fmt in _eodhd_ticker_formats(t):
            if fmt == t:
                continue  # already attempted in the batch call
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


@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$"),
):
    period_days = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740, "5y": 1830}
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
def screener_top(limit: int = Query(10, ge=1, le=20)):
    """
    Top stocks by composite score (momentum + RS vs SPY).
    Batch real-time (1 call) for quotes; EOD per ticker for scoring (cached 24h).
    """
    cache_key = f"screener:top:{limit}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

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

            results.append({
                "ticker":     ticker,
                "price":      snap.get("price", round(float(close.iloc[-1]), 2)),
                "change_pct": snap.get("change_pct", 0.0),
                "score":      scores["composite"],
                "momentum":   scores["momentum"],
                "rs_vs_spy":  scores["rs"],
                "perf_1M":    returns["1M"],
                "perf_3M":    returns["3M"],
                "perf_6M":    returns["6M"],
                "perf_1Y":    returns["1Y"],
                "perf_YTD":   returns["YTD"],
                "signal": (
                    "Strong Buy" if scores["composite"] >= 80 else
                    "Buy"        if scores["composite"] >= 65 else
                    "Hold"       if scores["composite"] >= 45 else
                    "Sell"
                ),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    output = {"timestamp": datetime.utcnow().isoformat(), "results": results[:limit]}
    cache.set(cache_key, output)
    return output


@app.get("/macro")
def macro():
    """
    Macro indicators — single batch real-time call covers all tickers including
    crypto (BTC-USD.CC), forex (EURUSD.FOREX), and indices (GDAXI.INDX, etc.).
    """
    tickers = list(MACRO_TICKERS.values())
    snaps   = fetch_realtime(tickers)

    out = []
    for label, ticker in MACRO_TICKERS.items():
        snap = snaps.get(ticker)
        if snap:
            value = snap["price"]
            # EODHD reports TNX.INDX scaled x10 (e.g. 43 instead of 4.3)
            if ticker == "TNX.INDX" and value > 20:
                value = round(value / 10, 4)
            out.append({
                "label":      label,
                "ticker":     ticker,
                "value":      value,
                "change":     snap["change"],
                "change_pct": snap["change_pct"],
            })

    return {"timestamp": datetime.utcnow().isoformat(), "indicators": out}


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

# CBOE INDX tickers — available on current plan; values are ×10 and need /10 scaling
YIELD_CURVE_TICKERS = {
    "3M":  "IRX.INDX",
    "5Y":  "FVX.INDX",
    "10Y": "TNX.INDX",
    "30Y": "TYX.INDX",
}

# FRED tickers are not available on the current EODHD plan.
# ref_value/ref_change/ref_change_pct/ref_trend are returned as reference fallbacks.
MACRO_INDICATORS_CFG = [
    {"label": "Fed Funds Rate",   "ticker": "FEDFUNDS.FRED", "unit": "%",       "desc": "US-Leitzins (Federal Reserve)",
     "ref_value": 5.33,  "ref_change": 0.0,  "ref_change_pct": 0.0,   "ref_trend": "neutral"},
    {"label": "CPI YoY",          "ticker": "CPIAUCSL.FRED", "unit": "%",       "desc": "Inflationsrate USA (YoY)",
     "ref_value": 3.2,   "ref_change": -0.1, "ref_change_pct": -0.1,  "ref_trend": "down"},
    {"label": "Arbeitslosigkeit", "ticker": "UNRATE.FRED",   "unit": "%",       "desc": "US-Arbeitslosenquote",
     "ref_value": 3.9,   "ref_change": 0.1,  "ref_change_pct": 2.56,  "ref_trend": "up"},
    {"label": "ISM PMI",          "ticker": "NAPM.FRED",     "unit": "Punkte",  "desc": "ISM Einkaufsmanagerindex Verarb.",
     "ref_value": 48.7,  "ref_change": -0.4, "ref_change_pct": -0.81, "ref_trend": "down"},
    {"label": "M2 Geldmenge",     "ticker": "M2SL.FRED",     "unit": "Mrd. $",  "desc": "US-Geldmenge M2",
     "ref_value": 20985, "ref_change": 32.0, "ref_change_pct": 0.15,  "ref_trend": "up"},
    # UUP tracks DXY; scale_factor converts UUP price (~27) to approximate DXY (~104)
    {"label": "US Dollar Index",  "ticker": "UUP",           "unit": "Punkte",  "desc": "US Dollar Index (DXY)",
     "scale_factor": 3.82},
]


@app.get("/yield-curve")
def yield_curve():
    cache_key = "yield_curve:full"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    frm, to = _date_range(420)
    cur, m3, y1 = {}, {}, {}
    fetched = 0

    for mat, ticker in YIELD_CURVE_TICKERS.items():
        try:
            df     = fetch_eod(ticker, frm, to)
            series = df["Close"].dropna()
            if len(series) == 0:
                continue
            # CBOE INDX tickers report values ×10; divide to get actual yield %
            cur[mat] = round(float(series.iloc[-1]) / 10, 3)
            fetched += 1
            if len(series) >= 63:
                m3[mat] = round(float(series.iloc[-63]) / 10, 3)
            if len(series) >= 252:
                y1[mat] = round(float(series.iloc[-252]) / 10, 3)
        except Exception:
            pass

    if fetched == 0:
        return None  # frontend falls back to DEMO_YIELD

    y10    = cur.get("10Y")
    y5     = cur.get("5Y")
    spread = round(y10 - y5, 3) if y10 is not None and y5 is not None else None
    status = ("Invers"    if spread is not None and spread < -0.10
              else "Flach"   if spread is not None and spread < 0.25
              else "Normal"  if spread is not None
              else "Unbekannt")

    result = {
        "maturities":   ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"],
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
    cache_key = "macro_indicators:full"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    frm, to = _date_range(420)
    results = []

    for cfg in MACRO_INDICATORS_CFG:
        label  = cfg["label"]
        ticker = cfg["ticker"]
        unit   = cfg["unit"]
        desc   = cfg["desc"]

        # FRED tickers unavailable on current EODHD plan — return reference fallback immediately
        if ticker.endswith(".FRED"):
            results.append({
                "label": label, "ticker": ticker, "unit": unit, "desc": desc,
                "value": cfg["ref_value"], "change": cfg["ref_change"],
                "change_pct": cfg["ref_change_pct"], "trend": cfg["ref_trend"],
                "is_reference": True,
            })
            continue

        # Live fetch for non-FRED tickers (UUP etc.)
        scale = cfg.get("scale_factor", 1.0)
        try:
            series = pd.Series(dtype=float)
            snap   = _fetch_rt_one(ticker)
            if snap and snap.get("price"):
                try:
                    df     = fetch_eod(ticker, frm, to)
                    series = df["Close"].dropna() if not df.empty else pd.Series(dtype=float)
                except Exception:
                    pass
                trend = "neutral"
                if len(series) >= 90:
                    ra = float(series.iloc[-30:].mean())
                    pa = float(series.iloc[-90:-60].mean())
                    d  = (ra - pa) / pa if pa else 0
                    trend = "up" if d > 0.005 else "down" if d < -0.005 else "neutral"
                results.append({
                    "label": label, "ticker": ticker, "unit": unit, "desc": desc,
                    "value": round(snap["price"] * scale, 2),
                    "change": round(snap["change"] * scale, 4),
                    "change_pct": snap["change_pct"],  # % stays unchanged
                    "trend": trend,
                })
                continue

            df = fetch_eod(ticker, frm, to)
            if df.empty:
                continue
            series = df["Close"].dropna()
            if len(series) < 2:
                continue

            current  = float(series.iloc[-1])
            previous = float(series.iloc[-2])
            change_abs = round(current - previous, 4)
            change_pct = round((change_abs / previous * 100), 4) if previous else 0

            trend = "neutral"
            if len(series) >= 90:
                ra   = float(series.iloc[-30:].mean())
                pa   = float(series.iloc[-90:-60].mean())
                diff = (ra - pa) / pa if pa else 0
                trend = "up" if diff > 0.005 else "down" if diff < -0.005 else "neutral"

            results.append({
                "label": label, "ticker": ticker, "unit": unit, "desc": desc,
                "value": round(current, 3), "change": change_abs, "change_pct": change_pct,
                "trend": trend,
            })
        except Exception as e:
            results.append({"label": label, "ticker": ticker, "unit": unit, "desc": desc, "error": str(e)})

    output = {"timestamp": datetime.utcnow().isoformat(), "indicators": results}
    cache.set(cache_key, output)
    return output


# ─── Sector Holdings ─────────────────────────────────────────────────────────

@app.get("/sector-holdings/{ticker}")
def sector_holdings(ticker: str):
    t         = ticker.upper()
    cache_key = f"sector_holdings:{t}"
    cached    = cache.get(cache_key, ttl_seconds=86400)  # holdings change rarely
    if cached is not None:
        return cached

    try:
        data = eodhd_get(f"/fundamentals/{t}.US", {"filter": "Components"})
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not isinstance(data, dict) or not data:
        raise HTTPException(status_code=404, detail=f"No holdings data for {t}")

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


# ─── Ticker Fundamentals ──────────────────────────────────────────────────────

@app.get("/ticker-fundamentals/{ticker}")
def ticker_fundamentals(ticker: str):
    t         = ticker.upper()
    cache_key = f"ticker_fundamentals:{t}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    eodhd_t = t if "." in t else f"{t}.US"

    try:
        highlights = eodhd_get(f"/fundamentals/{eodhd_t}", {"filter": "Highlights"})
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not isinstance(highlights, dict):
        raise HTTPException(status_code=404, detail=f"No fundamentals for {t}")

    quote = _fetch_rt_one(eodhd_t)

    def _safe(key):
        v = highlights.get(key)
        try: return float(v) if v not in (None, "", "None") else None
        except (ValueError, TypeError): return None

    result = {
        "ticker":           t,
        "price":            quote["price"]      if quote else None,
        "change":           quote["change"]     if quote else None,
        "change_pct":       quote["change_pct"] if quote else None,
        "market_cap":       _safe("MarketCapitalization"),
        "pe_ratio":         _safe("PERatio"),
        "eps":              _safe("EarningsShare"),
        "high_52w":         _safe("52WeekHigh"),
        "low_52w":          _safe("52WeekLow"),
        "revenue_ttm":      _safe("RevenueTTM"),
        "target_price":     _safe("WallStreetTargetPrice"),
        "profit_margin":    _safe("ProfitMargin"),
    }
    cache.set(cache_key, result)
    return result


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

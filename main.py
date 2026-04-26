"""
APEX Markets — Backend API v4 (Financial Modeling Prep edition)
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

FMP_API_KEY = "M94snYNBq8LP36YITTdkbuicwAykE4U5"
FMP_BASE    = "https://financialmodelingprep.com/api/v3"

BENCHMARK = "SPY"

# FMP has a built-in /sector-performance endpoint — no ETF proxies needed for sectors.
# These ETFs are kept for multi-timeframe scoring (1M, 3M, YTD, 1Y) via historical data.
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

# FMP sector-performance label → SECTOR_ETFS key
FMP_SECTOR_MAP = {
    "Technology":             "Technologie",
    "Communication Services": "Kommunikation",
    "Financial Services":     "Finanzen",
    "Healthcare":             "Gesundheit",
    "Industrials":            "Industrie",
    "Consumer Cyclical":      "Konsum_Zyklisch",
    "Consumer Defensive":     "Konsum_Basis",
    "Energy":                 "Energie",
    "Basic Materials":        "Materialien",
    "Real Estate":            "Immobilien",
    "Utilities":              "Versorger",
}

# Macro tickers as requested — FMP handles ETFs, crypto, forex, and indices
MACRO_TICKERS = {
    "S&P 500":      "SPY",
    "Gold":         "GLD",
    "WTI_Crude":    "USO",
    "10Y_Treasury": "TLT",
    "Bitcoin":      "BTC-USD",
    "US_Dollar":    "DX-Y.NYB",
    "Volatility":   "^VIX",
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
    description="Marktanalyse-Backend mit Financial Modeling Prep Daten",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ─── FMP request layer ────────────────────────────────────────────────────────

_last_fmp_call: float = 0.0
_FMP_MIN_INTERVAL = 2.0  # FMP free tier: 250 req/day, ~10 req/min — 2s is conservative

def fmp_get(path: str, params: dict | None = None):
    """Rate-limited GET to FMP REST API. Returns parsed JSON (list or dict)."""
    global _last_fmp_call

    elapsed = time.time() - _last_fmp_call
    if elapsed < _FMP_MIN_INTERVAL:
        time.sleep(_FMP_MIN_INTERVAL - elapsed)

    p = dict(params or {})
    p["apikey"] = FMP_API_KEY

    try:
        resp = requests.get(f"{FMP_BASE}{path}", params=p, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"FMP network error: {e}")
    finally:
        _last_fmp_call = time.time()

    data = resp.json()

    # FMP returns {"Error Message": "..."} on bad key / invalid ticker
    if isinstance(data, dict) and "Error Message" in data:
        raise HTTPException(status_code=502, detail=data["Error Message"])

    return data

# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """
    Batch quote fetch — FMP accepts comma-separated tickers in one call.
    Returns {TICKER: {price, change, change_pct, ...}}.
    Cached 15 min per individual ticker; only uncached tickers are fetched.
    """
    missing = [t for t in tickers if cache.get(f"quote:{t}", 900) is None]
    result  = {t: cache.get(f"quote:{t}", 900) for t in tickers if t not in missing}

    if not missing:
        return result

    data = fmp_get(f"/quote/{','.join(missing)}")

    for q in (data if isinstance(data, list) else []):
        sym = q.get("symbol", "")
        if not sym:
            continue
        entry = {
            "ticker":         sym,
            "name":           q.get("name", ""),
            "price":          round(float(q.get("price") or 0), 2),
            "change":         round(float(q.get("change") or 0), 2),
            "change_pct":     round(float(q.get("changesPercentage") or 0), 2),
            "previous_close": round(float(q.get("previousClose") or 0), 2),
            "volume":         int(q.get("volume") or 0),
            "market_cap":     q.get("marketCap"),
            "pe":             q.get("pe"),
        }
        cache.set(f"quote:{sym}", entry)
        result[sym] = entry

    return result


def fetch_historical(ticker: str, timeseries: int = 365) -> pd.DataFrame:
    """
    FMP /historical-price-full — daily OHLCV, adjusted.
    FMP returns newest-first; we sort ascending for scoring functions.
    Cached 24h.
    """
    cache_key = f"hist:{ticker}:{timeseries}"
    cached    = cache.get(cache_key, ttl_seconds=86400)
    if cached is not None:
        return cached

    data       = fmp_get(f"/historical-price-full/{ticker}", {"timeseries": timeseries})
    historical = data.get("historical", []) if isinstance(data, dict) else []

    if not historical:
        raise HTTPException(status_code=502, detail=f"No historical data for {ticker}")

    df = pd.DataFrame(historical)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)  # ascending — oldest first
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }, inplace=True)
    df["Close"]  = df["Close"].astype(float)
    df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0).astype(int)

    cache.set(cache_key, df)
    return df


def fetch_sector_performance() -> list[dict]:
    """
    FMP /sector-performance — 1d % change for all 11 sectors in one call.
    Cached 1h.
    """
    cache_key = "fmp:sector_perf"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    data   = fmp_get("/sector-performance")
    result = data if isinstance(data, list) else []
    cache.set(cache_key, result)
    return result

# ─── Scoring (unchanged logic) ────────────────────────────────────────────────

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
        score = 50 + np.tanh(weighted * 5) * 50
        return float(np.clip(score, 0, 100))
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

    rs_score  = 50 + np.tanh(rs / 10) * 50
    composite = 0.5 * momentum + 0.5 * rs_score
    return {
        "composite": round(float(composite), 1),
        "momentum":  round(float(momentum), 1),
        "rs":        round(float(rs), 2),
    }


def calc_returns(prices: pd.Series) -> dict:
    if len(prices) == 0:
        return {"1M": 0.0, "3M": 0.0, "YTD": 0.0, "1Y": 0.0}
    last = prices.iloc[-1]
    out  = {
        "1M":  round(float((last / prices.iloc[-21]  - 1) * 100), 2) if len(prices) >= 21  else 0.0,
        "3M":  round(float((last / prices.iloc[-63]  - 1) * 100), 2) if len(prices) >= 63  else 0.0,
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
        "version":   "4.0.0 (Financial Modeling Prep)",
        "status":    "online",
        "endpoints": [
            "/health", "/quote/{ticker}", "/history/{ticker}",
            "/sectors", "/screener/top", "/macro", "/portfolio/analyze",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/quote/{ticker}")
def quote(ticker: str):
    quotes = fetch_quotes([ticker.upper()])
    data   = quotes.get(ticker.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"No quote for {ticker}")
    return data


@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$"),
):
    """Daily close prices. FMP timeseries = number of trading days."""
    period_ts = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 365, "2y": 730, "5y": 1825}
    df = fetch_historical(ticker.upper(), timeseries=period_ts[period])

    out = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "period": period, "data": out}


@app.get("/sectors")
def sectors():
    """
    Sector performance via FMP /sector-performance (1 call → all 11 sectors, 1d change)
    enriched with multi-timeframe returns and composite scores from ETF history (cached 24h).
    """
    cache_key = "sectors:full"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    # 1d performance for all sectors — 1 API call
    sector_perf = fetch_sector_performance()
    perf_by_name: dict[str, float] = {}
    for item in sector_perf:
        raw_name = item.get("sector", "")
        de_name  = FMP_SECTOR_MAP.get(raw_name, raw_name)
        pct_str  = str(item.get("changesPercentage", "0")).replace("%", "").strip()
        try:
            perf_by_name[de_name] = round(float(pct_str), 2)
        except ValueError:
            perf_by_name[de_name] = 0.0

    benchmark_df    = fetch_historical(BENCHMARK, timeseries=365)
    benchmark_close = benchmark_df["Close"]

    results = []
    for name, ticker in SECTOR_ETFS.items():
        try:
            df      = fetch_historical(ticker, timeseries=365)
            close   = df["Close"]
            returns = calc_returns(close)
            scores  = calc_composite_score(close, benchmark_close)
            trend   = trend_indicator(close)

            results.append({
                "name":           name,
                "ticker":         ticker,
                "performance":    perf_by_name.get(name, returns["1M"]),  # 1d % for bar chart
                "perf_1d":        perf_by_name.get(name, 0.0),
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
                "performance": perf_by_name.get(name, 0.0),
                "perf_1d":     perf_by_name.get(name, 0.0),
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
    Batch quote (1 call) + historical per ticker for scoring (cached 24h).
    """
    cache_key = f"screener:top:{limit}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    # All quotes in one batch call
    quotes = fetch_quotes(SCREEN_UNIVERSE + [BENCHMARK])

    benchmark_df    = fetch_historical(BENCHMARK, timeseries=365)
    benchmark_close = benchmark_df["Close"]

    results = []
    for ticker in SCREEN_UNIVERSE:
        try:
            q       = quotes.get(ticker, {})
            df      = fetch_historical(ticker, timeseries=365)
            close   = df["Close"]
            scores  = calc_composite_score(close, benchmark_close)
            returns = calc_returns(close)

            results.append({
                "ticker":     ticker,
                "name":       q.get("name", ""),
                "price":      q.get("price", round(float(close.iloc[-1]), 2)),
                "change_pct": q.get("change_pct", 0.0),
                "score":      scores["composite"],
                "momentum":   scores["momentum"],
                "rs_vs_spy":  scores["rs"],
                "perf_1M":    returns["1M"],
                "perf_3M":    returns["3M"],
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
    Macro indicators — all fetched in a single batch quote call.
    Tickers: SPY, GLD, USO, TLT, BTC-USD, DX-Y.NYB, ^VIX.
    Cached 15 min per ticker inside fetch_quotes.
    """
    tickers = list(MACRO_TICKERS.values())
    quotes  = fetch_quotes(tickers)

    out = []
    for label, ticker in MACRO_TICKERS.items():
        q = quotes.get(ticker)
        if q:
            out.append({
                "label":      label,
                "ticker":     ticker,
                "value":      q["price"],
                "change":     q["change"],
                "change_pct": q["change_pct"],
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
    quotes            = fetch_quotes(tickers + [BENCHMARK])
    benchmark_df      = fetch_historical(BENCHMARK, timeseries=365)
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

    enriched    = []
    total_value = 0.0
    total_cost  = 0.0

    for pos in req.positions:
        ticker = pos.ticker.upper()
        try:
            q       = quotes.get(ticker, {})
            current = q.get("price") or 0.0
            if not current:
                raise ValueError("No price available")

            value   = current * pos.shares
            cost    = pos.avg_cost * pos.shares
            pnl     = value - cost
            pnl_pct = (pnl / cost * 100) if cost else 0

            df  = fetch_historical(ticker, timeseries=365)
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


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

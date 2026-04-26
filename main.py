"""
APEX Markets — Backend API v2 (Alpha Vantage edition)
"""
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

AV_KEY  = "Q3CSQO9RMTJ9VULY"
AV_BASE = "https://www.alphavantage.co/query"

BENCHMARK = "SPY"

# ETF proxies — all fetchable via GLOBAL_QUOTE, no index tickers needed
MACRO_TICKERS = {
    "10Y_Treasury": "TLT",
    "Volatility":   "VIXY",
    "Gold":         "GLD",
    "WTI_Crude":    "USO",
    "Bitcoin":      "GBTC",
    "US_Dollar":    "UUP",
}

# Kept to 15 tickers to stay within AV free tier (25 req/day).
# Cold-start loads all at once; every subsequent call within 24h is instant.
SCREEN_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "JPM",  "LLY",  "XOM",
    "V",    "UNH",  "AVGO", "HD",   "CAT",
]

# Alpha Vantage English sector names → German display names
SECTOR_NAME_MAP = {
    "Information Technology": "Technologie",
    "Health Care":            "Gesundheit",
    "Financials":             "Finanzen",
    "Consumer Discretionary": "Konsum Zyklisch",
    "Communication Services": "Kommunikation",
    "Industrials":            "Industrie",
    "Consumer Staples":       "Konsum Basis",
    "Energy":                 "Energie",
    "Utilities":              "Versorger",
    "Real Estate":            "Immobilien",
    "Materials":              "Materialien",
}

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="APEX Markets API",
    description="Marktanalyse-Backend mit Alpha Vantage Daten",
    version="2.0.0",
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
        self._store = {}

    def get(self, key, ttl_seconds: int):
        if key in self._store:
            value, ts = self._store[key]
            if time.time() - ts < ttl_seconds:
                return value
        return None

    def set(self, key, value):
        self._store[key] = (value, time.time())

cache = TimedCache()

# ─── Alpha Vantage request layer ──────────────────────────────────────────────

_last_av_call: float = 0.0
_AV_MIN_INTERVAL = 13.0  # seconds between calls (free tier: 5/min)

def av_request(params: dict) -> dict:
    """Make a rate-limited request to Alpha Vantage."""
    global _last_av_call
    elapsed = time.time() - _last_av_call
    if elapsed < _AV_MIN_INTERVAL:
        time.sleep(_AV_MIN_INTERVAL - elapsed)

    params["apikey"] = AV_KEY
    try:
        resp = requests.get(AV_BASE, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"AV network error: {e}")
    finally:
        _last_av_call = time.time()

    data = resp.json()

    # AV encodes rate-limit errors as normal 200 responses
    if "Note" in data or "Information" in data:
        msg = data.get("Note") or data.get("Information", "Rate limit exceeded")
        raise HTTPException(status_code=429, detail=msg)
    if "Error Message" in data:
        raise HTTPException(status_code=502, detail=data["Error Message"])

    return data

# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_daily(ticker: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    TIME_SERIES_DAILY.
    compact  = last 100 trading days  (screener, scoring, quotes)
    full     = full history up to 20y (history endpoint)
    Cached 24h — conserves the 25 req/day free-tier budget.
    """
    cache_key = f"daily:{outputsize}:{ticker}"
    cached = cache.get(cache_key, ttl_seconds=86400)
    if cached is not None:
        return cached

    data = av_request({
        "function":  "TIME_SERIES_DAILY",
        "symbol":    ticker,
        "outputsize": outputsize,
    })

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise HTTPException(status_code=502, detail=f"No daily data returned for {ticker}")

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={
        "1. open":   "Open",
        "2. high":   "High",
        "3. low":    "Low",
        "4. close":  "Close",
        "5. volume": "Volume",
    }, inplace=True)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].astype(float)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

    cache.set(cache_key, df)
    return df


def fetch_quote(ticker: str) -> dict:
    """
    GLOBAL_QUOTE — current price + daily change.
    Cached 15 min.
    """
    cache_key = f"quote:{ticker}"
    cached = cache.get(cache_key, ttl_seconds=900)
    if cached is not None:
        return cached

    data = av_request({"function": "GLOBAL_QUOTE", "symbol": ticker})
    q = data.get("Global Quote", {})
    if not q or not q.get("05. price"):
        raise HTTPException(status_code=502, detail=f"No quote data for {ticker}")

    price      = float(q["05. price"])
    prev_close = float(q.get("08. previous close", price))
    change     = float(q.get("09. change", 0))
    chg_str    = q.get("10. change percent", "0%").replace("%", "").strip()
    change_pct = float(chg_str) if chg_str else 0.0

    result = {
        "ticker":         ticker,
        "price":          round(price, 2),
        "previous_close": round(prev_close, 2),
        "change":         round(change, 2),
        "change_pct":     round(change_pct, 2),
    }
    cache.set(cache_key, result)
    return result


def fetch_av_sectors() -> dict:
    """
    SECTOR — Alpha Vantage built-in sector performance across multiple timeframes.
    Cached 1h (1 API call covers all 11 sectors).
    """
    cache_key = "av:sectors"
    cached = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    data = av_request({"function": "SECTOR"})
    cache.set(cache_key, data)
    return data

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

    momentum = calc_momentum_score(prices)
    common_idx = prices.index.intersection(benchmark_prices.index)
    if len(common_idx) < 21:
        return {"composite": momentum, "momentum": momentum, "rs": 0.0}

    p = prices.loc[common_idx].iloc[-63:]         if len(common_idx) >= 63 else prices.loc[common_idx]
    b = benchmark_prices.loc[common_idx].iloc[-63:] if len(common_idx) >= 63 else benchmark_prices.loc[common_idx]

    rs       = calc_relative_strength(p.pct_change().dropna(), b.pct_change().dropna())
    rs_score = 50 + np.tanh(rs / 10) * 50
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
    out = {
        "1M":  round(float((last / prices.iloc[-21]  - 1) * 100), 2) if len(prices) >= 21  else 0.0,
        "3M":  round(float((last / prices.iloc[-63]  - 1) * 100), 2) if len(prices) >= 63  else 0.0,
        "1Y":  round(float((last / prices.iloc[-252] - 1) * 100), 2) if len(prices) >= 252 else 0.0,
    }
    ytd = prices[prices.index.year == prices.index[-1].year]
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
        "version":   "2.0.0 (Alpha Vantage)",
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
    return fetch_quote(ticker.upper())


@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$"),
):
    outputsize = "compact" if period in ("1mo", "3mo") else "full"
    df = fetch_daily(ticker.upper(), outputsize=outputsize)

    cutoff_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=cutoff_days[period])
    df = df[df.index >= cutoff]

    out = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "period": period, "data": out}


@app.get("/sectors")
def sectors():
    """
    Sector performance from AV SECTOR endpoint — 1 API call covers all 11 sectors
    across real-time, 1d, 1M, 3M, YTD, and 1Y timeframes.
    """
    data = fetch_av_sectors()

    period_map = {
        "Rank A: Real-Time Performance":          "perf_realtime",
        "Rank B: 1 Day Performance":              "perf_1d",
        "Rank D: 1 Month Performance":            "perf_1M",
        "Rank E: 3 Month Performance":            "perf_3M",
        "Rank F: Year-to-Date (YTD) Performance": "perf_YTD",
        "Rank G: 1 Year Performance":             "perf_1Y",
    }

    sectors_by_name: dict = {}
    for av_key, field in period_map.items():
        for av_sector, pct_str in data.get(av_key, {}).items():
            de_name = SECTOR_NAME_MAP.get(av_sector, av_sector)
            if de_name not in sectors_by_name:
                sectors_by_name[de_name] = {"name": de_name}
            try:
                pct = float(str(pct_str).replace("%", "").strip())
            except (ValueError, AttributeError):
                pct = 0.0
            sectors_by_name[de_name][field] = round(pct, 2)

    results = list(sectors_by_name.values())
    for s in results:
        # "performance" alias used by frontend bar charts
        s["performance"] = s.get("perf_1d") or s.get("perf_realtime") or 0.0

    results.sort(key=lambda x: x.get("perf_1d", 0), reverse=True)
    return {"timestamp": datetime.utcnow().isoformat(), "sectors": results}


@app.get("/screener/top")
def screener_top(limit: int = Query(10, ge=1, le=15)):
    """
    Top stocks by composite score (momentum + relative strength vs SPY).
    Universe: 15 curated large-caps. Results cached 24h to stay within AV free tier.
    Note: cold-start fetch takes ~3 min due to AV rate limits; all subsequent calls instant.
    """
    cache_key = f"screener:top:{limit}"
    cached = cache.get(cache_key, ttl_seconds=86400)
    if cached is not None:
        return cached

    benchmark_df    = fetch_daily(BENCHMARK, outputsize="compact")
    benchmark_close = benchmark_df["Close"]

    results = []
    for ticker in SCREEN_UNIVERSE:
        try:
            df    = fetch_daily(ticker, outputsize="compact")
            close = df["Close"]

            scores  = calc_composite_score(close, benchmark_close)
            returns = calc_returns(close)

            last_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2]) if len(close) >= 2 else last_price
            change_pct = round((last_price / prev_price - 1) * 100, 2)

            results.append({
                "ticker":     ticker,
                "price":      round(last_price, 2),
                "change_pct": change_pct,
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
    """Macro indicators via GLOBAL_QUOTE on ETF proxies. Cached 15 min per ticker."""
    out = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            q = fetch_quote(ticker)
            out.append({
                "label":      label,
                "ticker":     ticker,
                "value":      q["price"],
                "change":     q["change"],
                "change_pct": q["change_pct"],
            })
        except Exception:
            continue
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

    benchmark_df      = fetch_daily(BENCHMARK, outputsize="compact")
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

    enriched     = []
    total_value  = 0.0
    total_cost   = 0.0
    port_returns = None

    for pos in req.positions:
        ticker = pos.ticker.upper()
        try:
            q       = fetch_quote(ticker)
            current = q["price"]
            value   = current * pos.shares
            cost    = pos.avg_cost * pos.shares
            pnl     = value - cost
            pnl_pct = (pnl / cost * 100) if cost else 0

            df  = fetch_daily(ticker, outputsize="compact")
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

        cum        = (1 + p_ret).cumprod()
        max_dd     = ((cum - cum.cummax()) / cum.cummax()).min() * 100

        risk = {
            "beta":                    round(float(beta), 2),
            "sharpe_ratio":            round(float(sharpe), 2),
            "volatility_annualized":   round(float(vol), 2),
            "max_drawdown":            round(float(max_dd), 2),
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

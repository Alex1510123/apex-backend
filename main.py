"""
APEX Markets — Backend API v3 (Polygon.io edition)
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

POLYGON_API_KEY = "mrjT33BPvQVbSrYsdUU1k94NW9Ta0MJe"
POLYGON_BASE    = "https://api.polygon.io"

BENCHMARK = "SPY"

SECTOR_ETFS = {
    "Technologie":    "XLK",
    "Kommunikation":  "XLC",
    "Finanzen":       "XLF",
    "Gesundheit":     "XLV",
    "Industrie":      "XLI",
    "Konsum_Zyklisch":"XLY",
    "Konsum_Basis":   "XLP",
    "Energie":        "XLE",
    "Materialien":    "XLB",
    "Immobilien":     "XLRE",
    "Versorger":      "XLU",
}

# ETF proxies for macro indicators — all available via Polygon snapshot
MACRO_TICKERS = {
    "10Y_Treasury": "TLT",
    "Volatility":   "VIXY",
    "Gold":         "GLD",
    "WTI_Crude":    "USO",
    "Bitcoin":      "GBTC",
    "US_Dollar":    "UUP",
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
    description="Marktanalyse-Backend mit Polygon.io Daten",
    version="3.0.0",
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

# ─── Polygon request layer ────────────────────────────────────────────────────

_last_poly_call: float = 0.0
_POLY_MIN_INTERVAL = 13.0  # 5 req/min on free tier → 12s apart; 13s adds buffer

def poly_get(path: str, params: dict | None = None) -> dict:
    """Rate-limited GET to Polygon REST API."""
    global _last_poly_call

    elapsed = time.time() - _last_poly_call
    if elapsed < _POLY_MIN_INTERVAL:
        time.sleep(_POLY_MIN_INTERVAL - elapsed)

    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY

    try:
        resp = requests.get(f"{POLYGON_BASE}{path}", params=p, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Polygon network error: {e}")
    finally:
        _last_poly_call = time.time()

    data = resp.json()
    status = data.get("status", "")

    # "DELAYED" is valid on free tier (15-min lag); treat as OK
    if status not in ("OK", "DELAYED", "ok"):
        raise HTTPException(
            status_code=502,
            detail=data.get("message") or data.get("error") or f"Polygon error: {status}",
        )

    return data

# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_snapshots(tickers: list[str]) -> dict[str, dict]:
    """
    Batch snapshot for N tickers — 1 API call regardless of count.
    Returns {TICKER: {price, change, change_pct, prev_close, volume}}.
    Cached 15 min.
    """
    # Return from cache for any ticker already cached; only fetch the rest
    missing   = [t for t in tickers if cache.get(f"snap:{t}", 900) is None]
    result    = {t: cache.get(f"snap:{t}", 900) for t in tickers if t not in missing}

    if not missing:
        return result

    # Polygon allows up to ~200 tickers per snapshot call
    tickers_param = ",".join(missing)
    data = poly_get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        {"tickers": tickers_param},
    )

    for snap in data.get("tickers", []):
        t        = snap["ticker"]
        day      = snap.get("day", {})
        prev_day = snap.get("prevDay", {})

        # Price: prefer lastTrade.p → day.c → prevDay.c
        price = (
            (snap.get("lastTrade") or {}).get("p")
            or day.get("c")
            or prev_day.get("c")
            or 0.0
        )
        prev_close  = prev_day.get("c") or price
        change      = snap.get("todaysChange", price - prev_close)
        change_pct  = snap.get("todaysChangePerc", 0.0)

        entry = {
            "ticker":         t,
            "price":          round(float(price), 2),
            "previous_close": round(float(prev_close), 2),
            "change":         round(float(change), 2),
            "change_pct":     round(float(change_pct), 2),
            "volume":         int(day.get("v", 0)),
        }
        cache.set(f"snap:{t}", entry)
        result[t] = entry

    return result


def fetch_aggs(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Daily OHLCV bars from Polygon aggs endpoint.
    from_date / to_date: "YYYY-MM-DD".
    Cached 24h — daily bars are immutable once the session closes.
    """
    cache_key = f"aggs:{ticker}:{from_date}:{to_date}"
    cached = cache.get(cache_key, ttl_seconds=86400)
    if cached is not None:
        return cached

    data = poly_get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )

    results = data.get("results", [])
    if not results:
        raise HTTPException(status_code=502, detail=f"No agg data for {ticker} ({from_date}→{to_date})")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype({"Close": float, "Volume": float})

    cache.set(cache_key, df)
    return df


def _date_range(days: int) -> tuple[str, str]:
    """Returns (from_date, to_date) as "YYYY-MM-DD" strings."""
    to  = datetime.utcnow()
    frm = to - timedelta(days=days)
    return frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d")


def fetch_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """Convenience wrapper around fetch_aggs with a day count."""
    frm, to = _date_range(days)
    return fetch_aggs(ticker, frm, to)

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

    momentum    = calc_momentum_score(prices)
    common_idx  = prices.index.intersection(benchmark_prices.index)
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
    ytd       = prices[prices.index.year == prices.index[-1].year]
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
        "version":   "3.0.0 (Polygon.io)",
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
    """Current quote for a single ticker."""
    snaps = fetch_snapshots([ticker.upper()])
    data  = snaps.get(ticker.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"No snapshot for {ticker}")
    return data


@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$"),
):
    """Daily close prices for the requested period."""
    period_days = {"1mo": 35, "3mo": 95, "6mo": 185, "1y": 370, "2y": 740, "5y": 1830}
    frm, to = _date_range(period_days[period])
    df = fetch_aggs(ticker.upper(), frm, to)

    out = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "period": period, "data": out}


@app.get("/sectors")
def sectors():
    """
    Sector ETF performance.
    Snapshot (1 call) → 1d change.
    Aggs per ETF (cached 24h) → 1M, 3M, YTD, 1Y + composite score.
    """
    cache_key = "sectors:full"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    sector_tickers = list(SECTOR_ETFS.values())

    # 1 batch call for current-day quotes
    snaps           = fetch_snapshots(sector_tickers + [BENCHMARK])
    benchmark_close = fetch_history(BENCHMARK, days=370)["Close"]

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
                "name":         name,
                "ticker":       ticker,
                "price":        snap.get("price", round(float(close.iloc[-1]), 2)),
                "performance":  snap.get("change_pct", returns["1M"]),  # 1d % for bar chart
                "perf_1d":      snap.get("change_pct", 0.0),
                "perf_1M":      returns["1M"],
                "perf_3M":      returns["3M"],
                "perf_YTD":     returns["YTD"],
                "perf_1Y":      returns["1Y"],
                "score":        scores["composite"],
                "momentum_score": scores["momentum"],
                "rs_vs_spy":    scores["rs"],
                "trend":        trend,
            })
        except Exception as e:
            results.append({"name": name, "ticker": ticker, "error": str(e)})

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    output = {"timestamp": datetime.utcnow().isoformat(), "sectors": results}
    cache.set(cache_key, output)
    return output


@app.get("/screener/top")
def screener_top(limit: int = Query(10, ge=1, le=20)):
    """
    Top stocks by composite score (momentum + RS vs SPY).
    Snapshot batch (1 call) + aggs per ticker for scoring (cached 24h).
    """
    cache_key = f"screener:top:{limit}"
    cached    = cache.get(cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    # Fetch all quotes in one batch call
    all_tickers = SCREEN_UNIVERSE + [BENCHMARK]
    snaps       = fetch_snapshots(all_tickers)

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
    Macro indicators — all fetched in a single batch snapshot call.
    Cached 15 min per ticker inside fetch_snapshots.
    """
    tickers = list(MACRO_TICKERS.values())
    snaps   = fetch_snapshots(tickers)

    out = []
    for label, ticker in MACRO_TICKERS.items():
        snap = snaps.get(ticker)
        if snap:
            out.append({
                "label":      label,
                "ticker":     ticker,
                "value":      snap["price"],
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

    tickers         = [p.ticker.upper() for p in req.positions]
    snaps           = fetch_snapshots(tickers + [BENCHMARK])
    benchmark_df    = fetch_history(BENCHMARK, days=370)
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


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

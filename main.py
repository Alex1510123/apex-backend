"""
APEX Markets — Backend API
Marktanalyse-Backend mit Yahoo Finance Daten, Caching und Composite Scoring.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
import time

app = FastAPI(
    title="APEX Markets API",
    description="Marktanalyse-Backend mit Sektor-Scoring und Portfolio-Analyse",
    version="1.0.0",
)

# CORS — Frontend darf zugreifen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion: konkrete Domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  KONFIGURATION
# ============================================================

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

BENCHMARK = "^GSPC"  # S&P 500

MACRO_TICKERS = {
    "10Y_Treasury": "^TNX",
    "DXY":          "DX-Y.NYB",
    "VIX":          "^VIX",
    "Gold":         "GC=F",
    "WTI_Crude":    "CL=F",
    "BTC":          "BTC-USD",
}

# Universum für Top Outperformer Screening
SCREEN_UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO","ORCL","CRM",
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","AXP",
    "UNH","LLY","JNJ","PFE","MRK","ABBV","TMO","DHR",
    "XOM","CVX","COP","SLB",
    "HD","MCD","NKE","SBUX","LOW",
    "BA","CAT","GE","HON","UNP",
    "WMT","PG","KO","PEP","COST",
]

# ============================================================
#  CACHE — vermeidet API-Spam
# ============================================================

class TimedCache:
    """Simpler Cache mit TTL pro Eintrag."""
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

# ============================================================
#  SCORING — Das Herzstück
# ============================================================

def calc_relative_strength(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Relative Strength: kumulierte Outperformance vs. Benchmark in Prozent.
    Positiv = outperformend, Negativ = underperformend.
    """
    if len(ticker_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    t_cum = (1 + ticker_returns).prod() - 1
    b_cum = (1 + benchmark_returns).prod() - 1
    return float((t_cum - b_cum) * 100)

def calc_momentum_score(prices: pd.Series) -> float:
    """
    Momentum-Score 0-100. Gewichtet 1M/3M/6M/12M Returns.
    """
    if len(prices) < 21:
        return 50.0
    try:
        r_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0
        r_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) >= 63 else r_1m
        r_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else r_3m
        r_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) if len(prices) >= 252 else r_6m

        # Gewichteter Composite
        weighted = 0.4 * r_1m + 0.3 * r_3m + 0.2 * r_6m + 0.1 * r_12m
        # Skalieren auf 0-100 (sigmoid-artig)
        score = 50 + np.tanh(weighted * 5) * 50
        return float(np.clip(score, 0, 100))
    except Exception:
        return 50.0

def calc_composite_score(prices: pd.Series, benchmark_prices: pd.Series) -> dict:
    """
    Composite Score = 0.5 * Momentum + 0.5 * Relative-Strength-Score.
    """
    if len(prices) < 21:
        return {"composite": 50.0, "momentum": 50.0, "rs": 0.0}

    momentum = calc_momentum_score(prices)

    # RS über letzte ~63 Handelstage (3M)
    common_idx = prices.index.intersection(benchmark_prices.index)
    if len(common_idx) < 21:
        return {"composite": momentum, "momentum": momentum, "rs": 0.0}

    p = prices.loc[common_idx].iloc[-63:] if len(common_idx) >= 63 else prices.loc[common_idx]
    b = benchmark_prices.loc[common_idx].iloc[-63:] if len(common_idx) >= 63 else benchmark_prices.loc[common_idx]

    t_ret = p.pct_change().dropna()
    b_ret = b.pct_change().dropna()

    rs = calc_relative_strength(t_ret, b_ret)
    # RS-Wert auf 0-100 skalieren
    rs_score = 50 + np.tanh(rs / 10) * 50

    composite = 0.5 * momentum + 0.5 * rs_score
    return {
        "composite": round(float(composite), 1),
        "momentum": round(float(momentum), 1),
        "rs": round(float(rs), 2),
    }

def calc_returns(prices: pd.Series) -> dict:
    """Returns über verschiedene Zeiträume."""
    if len(prices) == 0:
        return {"1M": 0.0, "3M": 0.0, "YTD": 0.0, "1Y": 0.0}

    last = prices.iloc[-1]
    out = {}

    # 1M ≈ 21 Handelstage, 3M ≈ 63
    if len(prices) >= 21:
        out["1M"] = round(float((last / prices.iloc[-21] - 1) * 100), 2)
    else:
        out["1M"] = 0.0

    if len(prices) >= 63:
        out["3M"] = round(float((last / prices.iloc[-63] - 1) * 100), 2)
    else:
        out["3M"] = 0.0

    if len(prices) >= 252:
        out["1Y"] = round(float((last / prices.iloc[-252] - 1) * 100), 2)
    else:
        out["1Y"] = 0.0

    # YTD
    current_year = prices.index[-1].year
    ytd_data = prices[prices.index.year == current_year]
    if len(ytd_data) > 1:
        out["YTD"] = round(float((ytd_data.iloc[-1] / ytd_data.iloc[0] - 1) * 100), 2)
    else:
        out["YTD"] = 0.0

    return out

def trend_indicator(prices: pd.Series) -> str:
    """Einfacher SMA-basierter Trend."""
    if len(prices) < 50:
        return "flat"
    sma_20 = prices.iloc[-20:].mean()
    sma_50 = prices.iloc[-50:].mean()
    if sma_20 > sma_50 * 1.01:
        return "up"
    if sma_20 < sma_50 * 0.99:
        return "down"
    return "flat"

# ============================================================
#  DATA FETCHING
# ============================================================

def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Holt Kurshistorie mit Cache (TTL: 5 Min während Marktzeit)."""
    cache_key = f"hist:{ticker}:{period}"
    cached = cache.get(cache_key, ttl_seconds=300)
    if cached is not None:
        return cached

    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Keine Daten für {ticker}")
        cache.set(cache_key, df)
        return df
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Datenfehler {ticker}: {str(e)}")

def fetch_quote(ticker: str) -> dict:
    """Aktueller Kurs + Tagesänderung."""
    cache_key = f"quote:{ticker}"
    cached = cache.get(cache_key, ttl_seconds=60)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.get("last_price") or info.get("lastPrice") or 0)
        prev_close = float(info.get("previous_close") or info.get("previousClose") or price)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        result = {
            "ticker": ticker,
            "price": round(price, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
        cache.set(cache_key, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quote-Fehler {ticker}: {str(e)}")

# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": "APEX Markets API",
        "status": "online",
        "endpoints": [
            "/health",
            "/quote/{ticker}",
            "/history/{ticker}",
            "/sectors",
            "/screener/top",
            "/macro",
            "/portfolio/analyze",
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/quote/{ticker}")
def quote(ticker: str):
    """Aktueller Kurs eines Tickers."""
    return fetch_quote(ticker.upper())

@app.get("/history/{ticker}")
def history(ticker: str, period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$")):
    """Kurshistorie als Liste {date, close}."""
    df = fetch_history(ticker.upper(), period)
    out = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]
    return {"ticker": ticker.upper(), "period": period, "data": out}

@app.get("/sectors")
def sectors():
    """
    Alle 11 SPDR Sektor-ETFs mit Returns + Composite Score.
    Das ist das Herz der Outperformance-Analyse.
    """
    benchmark_df = fetch_history(BENCHMARK, "1y")
    benchmark_close = benchmark_df["Close"]

    results = []
    for name, ticker in SECTOR_ETFS.items():
        try:
            df = fetch_history(ticker, "1y")
            close = df["Close"]
            returns = calc_returns(close)
            scores = calc_composite_score(close, benchmark_close)
            trend = trend_indicator(close)

            results.append({
                "name": name,
                "ticker": ticker,
                "perf_1M": returns["1M"],
                "perf_3M": returns["3M"],
                "perf_YTD": returns["YTD"],
                "perf_1Y": returns["1Y"],
                "score": scores["composite"],
                "momentum_score": scores["momentum"],
                "rs_vs_spy": scores["rs"],
                "trend": trend,
            })
        except Exception as e:
            results.append({"name": name, "ticker": ticker, "error": str(e)})

    # Sortiert nach Score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"timestamp": datetime.utcnow().isoformat(), "sectors": results}

@app.get("/screener/top")
def screener_top(limit: int = Query(10, ge=1, le=30)):
    """
    Top Outperformer aus dem Universum.
    Ranking nach Composite Score (Momentum + RS).
    """
    cache_key = f"screener:top:{limit}"
    cached = cache.get(cache_key, ttl_seconds=600)
    if cached is not None:
        return cached

    benchmark_df = fetch_history(BENCHMARK, "1y")
    benchmark_close = benchmark_df["Close"]

    results = []
    for ticker in SCREEN_UNIVERSE:
        try:
            df = fetch_history(ticker, "1y")
            close = df["Close"]
            scores = calc_composite_score(close, benchmark_close)
            returns = calc_returns(close)

            # Aktuellen Quote zusätzlich
            last_price = float(close.iloc[-1])

            results.append({
                "ticker": ticker,
                "price": round(last_price, 2),
                "score": scores["composite"],
                "momentum": scores["momentum"],
                "rs_vs_spy": scores["rs"],
                "perf_1M": returns["1M"],
                "perf_3M": returns["3M"],
                "perf_YTD": returns["YTD"],
                "signal": "Strong Buy" if scores["composite"] >= 80 else "Buy" if scores["composite"] >= 65 else "Hold" if scores["composite"] >= 45 else "Sell",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    output = {"timestamp": datetime.utcnow().isoformat(), "results": results[:limit]}
    cache.set(cache_key, output)
    return output

@app.get("/macro")
def macro():
    """Makroindikatoren: Zinsen, USD, VIX, Gold, Öl, BTC."""
    out = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            q = fetch_quote(ticker)
            out.append({
                "label": label,
                "ticker": ticker,
                "value": q["price"],
                "change": q["change"],
                "change_pct": q["change_pct"],
            })
        except Exception:
            continue
    return {"timestamp": datetime.utcnow().isoformat(), "indicators": out}

# ---------- Portfolio ----------

class Position(BaseModel):
    ticker: str
    shares: float
    avg_cost: float

class PortfolioRequest(BaseModel):
    positions: list[Position]

@app.post("/portfolio/analyze")
def portfolio_analyze(req: PortfolioRequest):
    """
    Analysiert ein Portfolio: P&L, Gewichtungen, Risiko-Metriken.
    """
    if not req.positions:
        raise HTTPException(status_code=400, detail="Keine Positionen übergeben")

    benchmark_df = fetch_history(BENCHMARK, "1y")
    benchmark_returns = benchmark_df["Close"].pct_change().dropna()

    enriched = []
    total_value = 0.0
    total_cost = 0.0
    portfolio_returns_series = None

    for pos in req.positions:
        ticker = pos.ticker.upper()
        try:
            q = fetch_quote(ticker)
            current = q["price"]
            value = current * pos.shares
            cost = pos.avg_cost * pos.shares
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost else 0

            # Returns für Risiko-Berechnung
            df = fetch_history(ticker, "1y")
            ret = df["Close"].pct_change().dropna()

            enriched.append({
                "ticker": ticker,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": current,
                "value": round(value, 2),
                "cost_basis": round(cost, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "weight": 0,  # gleich berechnet
                "returns_series": ret,
            })
            total_value += value
            total_cost += cost
        except Exception as e:
            enriched.append({"ticker": ticker, "error": str(e)})

    # Gewichtungen + gewichtete Portfolio-Returns
    if total_value > 0:
        for p in enriched:
            if "error" not in p:
                p["weight"] = round(p["value"] / total_value * 100, 2)

        # Portfolio-Returns gewichtet
        weighted_returns = pd.Series(0.0, index=benchmark_returns.index)
        for p in enriched:
            if "error" in p:
                continue
            w = p["weight"] / 100
            r = p["returns_series"]
            common = r.index.intersection(weighted_returns.index)
            weighted_returns.loc[common] += r.loc[common] * w
        portfolio_returns_series = weighted_returns.dropna()

    # Risiko-Metriken
    risk = {}
    if portfolio_returns_series is not None and len(portfolio_returns_series) > 30:
        common_idx = portfolio_returns_series.index.intersection(benchmark_returns.index)
        p_ret = portfolio_returns_series.loc[common_idx]
        b_ret = benchmark_returns.loc[common_idx]

        # Beta
        cov = np.cov(p_ret, b_ret)[0, 1]
        var_b = np.var(b_ret)
        beta = cov / var_b if var_b else 1.0

        # Sharpe (annualisiert, RFR = 4.4%)
        rfr_daily = 0.044 / 252
        excess = p_ret - rfr_daily
        sharpe = (excess.mean() / p_ret.std() * np.sqrt(252)) if p_ret.std() else 0

        # Volatilität (annualisiert)
        vol = p_ret.std() * np.sqrt(252) * 100

        # Max Drawdown
        cum = (1 + p_ret).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        risk = {
            "beta": round(float(beta), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "volatility_annualized": round(float(vol), 2),
            "max_drawdown": round(float(max_dd), 2),
        }

    # Cleanup: returns_series rauswerfen aus Output
    for p in enriched:
        p.pop("returns_series", None)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_value - total_cost, 2),
            "total_pnl_pct": round((total_value - total_cost) / total_cost * 100, 2) if total_cost else 0,
            "position_count": len([p for p in enriched if "error" not in p]),
        },
        "positions": enriched,
        "risk_metrics": risk,
    }

# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

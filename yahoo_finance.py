# ==============================================================================
# TEMPORÄR — VOR COMMERCIAL LAUNCH ERSETZEN
# Yahoo Finance wird als temporäre Datenquelle für Rohstoffe und DXY genutzt.
# TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
# ==============================================================================

import time
import logging
import requests

logger = logging.getLogger(__name__)

# TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Finscope/1.0)",
    "Accept": "application/json",
}

_CACHE: dict = {}
_CACHE_TTL = 300  # 5 minutes


def fetch_yahoo_quote(symbol: str) -> dict | None:
    """
    TODO PRE-LAUNCH: Replace Yahoo Finance with licensed source (EODHD upgrade or Tiingo) before commercial launch

    Real-time quote from Yahoo Finance. Returns {price, change, change_pct} or None.
    5-minute in-memory cache. Timeout 5 seconds. Never raises — returns None on any error.

    Key symbols used by this project:
      GC=F      — Gold Spot Futures (~$4500)
      BZ=F      — Brent Crude Futures (~$109)
      CL=F      — WTI Crude Futures (~$107)
      DX-Y.NYB  — US Dollar Index (DXY, ~99)
    """
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    url = f"{YAHOO_BASE}/{symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Yahoo Finance fetch failed for %s: %s", symbol, exc)
        return None

    try:
        result_list = data.get("chart", {}).get("result") or []
        if not result_list:
            logger.warning("Yahoo Finance empty result for %s", symbol)
            return None

        meta = result_list[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            logger.warning("Yahoo Finance no regularMarketPrice for %s", symbol)
            return None

        price = float(price)
        prev_close = meta.get("previousClose") or meta.get("chartPreviousClose") or price
        prev_close = float(prev_close)

        change = round(price - prev_close, 4)
        change_pct = round((change / prev_close * 100) if prev_close else 0.0, 4)

        result = {"price": round(price, 4), "change": change, "change_pct": change_pct}
        _CACHE[symbol] = {**result, "_ts": now}
        return result

    except Exception as exc:
        logger.warning("Yahoo Finance parse error for %s: %s", symbol, exc)
        return None

# ==============================================================================
# FRED API Adapter — Federal Reserve Economic Data
# FRED_API_KEY muss als Environment Variable gesetzt sein (Railway: Settings > Variables)
# ==============================================================================

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_CACHE: dict = {}
_CACHE_TTL = 3600  # 1 hour


def _get_api_key() -> str | None:
    return os.environ.get("FRED_API_KEY")


def fetch_fred_series(series_id: str) -> dict | None:
    """
    Fetch the two most recent valid observations for a FRED series.
    Returns {value, prev_value, date, change, change_pct} or None.
    FRED marks missing data with '.'; those observations are skipped.
    Cached 1 hour.
    """
    now = time.time()
    cached = _CACHE.get(series_id)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    api_key = _get_api_key()
    if not api_key:
        logger.error("FRED_API_KEY not set — cannot fetch %s", series_id)
        return None

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 6,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("FRED fetch failed for %s: %s", series_id, exc)
        return None

    try:
        valid = [
            o for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]
        if len(valid) == 0:
            logger.warning("FRED no valid observations for %s", series_id)
            return None

        current_val = float(valid[0]["value"])
        prev_val = float(valid[1]["value"]) if len(valid) > 1 else current_val
        change = round(current_val - prev_val, 4)
        change_pct = round((change / prev_val * 100) if prev_val else 0.0, 4)

        result = {
            "value":      round(current_val, 4),
            "prev_value": round(prev_val, 4),
            "date":       valid[0].get("date", ""),
            "change":     change,
            "change_pct": change_pct,
        }
        _CACHE[series_id] = {**result, "_ts": now}
        return result

    except Exception as exc:
        logger.error("FRED parse error for %s: %s", series_id, exc)
        return None


def fetch_fred_indpro_yoy() -> dict | None:
    """
    Industrial Production YoY %: (current - 12_months_ago) / 12_months_ago * 100.
    Replaces ISM PMI (NAPM) which was removed from FRED in 2024 due to ISM license withdrawal.
    Returns same structure as fetch_fred_series (value = YoY %, change = pp MoM delta).
    Cached 1 hour.
    """
    cache_key = "INDPRO_YOY"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    api_key = _get_api_key()
    if not api_key:
        logger.error("FRED_API_KEY not set — cannot fetch INDPRO YoY")
        return None

    params = {
        "series_id": "INDPRO",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 16,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("FRED INDPRO fetch failed: %s", exc)
        return None

    try:
        valid = [
            float(o["value"])
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]
        dates = [
            o["date"]
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]

        if len(valid) < 13:
            logger.warning("FRED INDPRO insufficient data for YoY (got %d observations)", len(valid))
            return None

        yoy_current = round((valid[0] - valid[12]) / valid[12] * 100, 2)
        yoy_prev = round((valid[1] - valid[13]) / valid[13] * 100, 2) if len(valid) > 13 else yoy_current

        change = round(yoy_current - yoy_prev, 4)
        result = {
            "value":      yoy_current,
            "prev_value": yoy_prev,
            "date":       dates[0] if dates else "",
            "change":     change,
            "change_pct": change,
        }
        _CACHE[cache_key] = {**result, "_ts": now}
        return result

    except Exception as exc:
        logger.error("FRED INDPRO YoY parse error: %s", exc)
        return None


def fetch_fred_cpi_yoy() -> dict | None:
    """
    CPI YoY %: (current_month - 12_months_ago) / 12_months_ago * 100.
    Returns same structure as fetch_fred_series (value = YoY %, change = pp MoM delta).
    Cached 1 hour.
    """
    cache_key = "CPIAUCSL_YOY"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    api_key = _get_api_key()
    if not api_key:
        logger.error("FRED_API_KEY not set — cannot fetch CPI YoY")
        return None

    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 16,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("FRED CPI fetch failed: %s", exc)
        return None

    try:
        valid = [
            float(o["value"])
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]
        dates = [
            o["date"]
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]

        if len(valid) < 13:
            logger.warning("FRED CPI insufficient data for YoY (got %d observations)", len(valid))
            return None

        yoy_current = round((valid[0] - valid[12]) / valid[12] * 100, 2)
        yoy_prev = round((valid[1] - valid[13]) / valid[13] * 100, 2) if len(valid) > 13 else yoy_current

        change = round(yoy_current - yoy_prev, 4)
        result = {
            "value":      yoy_current,
            "prev_value": yoy_prev,
            "date":       dates[0] if dates else "",
            "change":     change,
            "change_pct": change,  # for YoY%, change in pp == change_pct
        }
        _CACHE[cache_key] = {**result, "_ts": now}
        return result

    except Exception as exc:
        logger.error("FRED CPI YoY parse error: %s", exc)
        return None


def fetch_fred_yield_history(series_id: str, limit: int = 400) -> list[dict] | None:
    """
    Fetch up to `limit` recent daily observations for a FRED Treasury yield series (DGS*).
    Returns list of {date, value} dicts newest-first, or None on failure.
    Cached 1 hour per series.

    Usage for yield curve:
      history[0]   = current value
      history[63]  = ~3 months ago (63 trading days ≈ 90 calendar days)
      history[252] = ~1 year ago   (252 trading days ≈ 365 calendar days)
    """
    cache_key = f"yield_hist:{series_id}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["data"]

    api_key = _get_api_key()
    if not api_key:
        logger.error("FRED_API_KEY not set — cannot fetch yield history for %s", series_id)
        return None

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = [
            {"date": o["date"], "value": round(float(o["value"]), 3)}
            for o in data.get("observations", [])
            if o.get("value") not in (".", None, "")
        ]
        if not result:
            logger.warning("FRED yield history empty for %s", series_id)
            return None
        _CACHE[cache_key] = {"data": result, "_ts": now}
        return result

    except Exception as exc:
        logger.error("FRED yield history fetch failed for %s: %s", series_id, exc)
        return None

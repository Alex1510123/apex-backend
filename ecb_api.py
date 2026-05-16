# ==============================================================================
# ECB Statistical Data Warehouse Adapter
# Kostenlos, kein API-Key nötig.
# Endpoint: https://data-api.ecb.europa.eu/service/data/{flowRef}/{key}?format=jsondata
# ==============================================================================

import time
import logging
import requests

logger = logging.getLogger(__name__)

ECB_BASE    = "https://data-api.ecb.europa.eu/service/data"
ECB_HEADERS = {"User-Agent": "Finscope/1.0", "Accept": "application/json"}

_CACHE: dict = {}
_CACHE_TTL   = 3600  # 1 hour — ECB publishes daily/monthly


def _parse_ecb_jsondata(data: dict) -> list[dict]:
    """
    Parse ECB JSON-stat (format=jsondata) into list of {date, value} newest-first.
    ECB stores observations as {"0": [value, status], "1": [value, status], ...}
    where keys are positional indices into the TIME_PERIOD dimension.
    Returns [] on any parse failure.
    """
    try:
        datasets = data.get("dataSets", [])
        if not datasets:
            return []

        series_map = datasets[0].get("series", {})
        if not series_map:
            return []

        first_series = next(iter(series_map.values()))
        observations  = first_series.get("observations", {})
        if not observations:
            return []

        # Extract ordered time labels from structure
        structure    = data.get("structure", {})
        time_periods: list[str] = []
        for dim in structure.get("dimensions", {}).get("observation", []):
            if dim.get("id") == "TIME_PERIOD":
                time_periods = [v["id"] for v in dim.get("values", [])]

        result = []
        for idx_str, vals in sorted(
            observations.items(), key=lambda x: int(x[0]), reverse=True
        ):
            idx = int(idx_str)
            if vals and vals[0] is not None:
                date = time_periods[idx] if idx < len(time_periods) else ""
                result.append({"date": date, "value": float(vals[0])})

        return result

    except Exception as exc:
        logger.error("ECB JSON parse error: %s", exc)
        return []


def fetch_ecb_series(flow_ref: str, series_key: str) -> dict | None:
    """
    Fetch the two most recent valid observations for an ECB SDW series.
    Returns {value, prev_value, date, change, change_pct} or None on failure.
    Cached 1 hour.

    Key ECB series used by Finscope:
      ECB MRR (Leitzins):     flow_ref="FM",   series_key="D.U2.EUR.4F.KR.MRR_FR.LEV"
      Eurozone HICP (Inflation):flow_ref="ICP",  series_key="M.U2.N.000000.4.ANR"
      Eurozone Unemployment:  flow_ref="LFSI",  series_key="M.I9.S.UNEHRT.TOTAL0.15_74.T"
    """
    cache_key = f"ecb:{flow_ref}:{series_key}"
    now       = time.time()
    cached    = _CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    url    = f"{ECB_BASE}/{flow_ref}/{series_key}"
    params = {"format": "jsondata", "detail": "dataonly", "lastNObservations": "3"}

    try:
        resp = requests.get(url, params=params, timeout=15, headers=ECB_HEADERS)
        resp.raise_for_status()
        raw  = resp.json()
    except requests.HTTPError as exc:
        logger.error("ECB HTTP error for %s/%s: %s (status %s)", flow_ref, series_key, exc,
                     getattr(exc.response, "status_code", "?"))
        return None
    except Exception as exc:
        logger.error("ECB fetch failed for %s/%s: %s", flow_ref, series_key, exc)
        return None

    observations = _parse_ecb_jsondata(raw)

    if not observations:
        logger.warning("ECB no valid observations for %s/%s", flow_ref, series_key)
        return None

    current_val  = observations[0]["value"]
    current_date = observations[0]["date"]
    prev_val     = observations[1]["value"] if len(observations) > 1 else current_val

    change     = round(current_val - prev_val, 4)
    change_pct = round((change / prev_val * 100) if prev_val else 0.0, 4)

    result = {
        "value":      round(current_val, 4),
        "prev_value": round(prev_val, 4),
        "date":       current_date,
        "change":     change,
        "change_pct": change_pct,
    }
    _CACHE[cache_key] = {**result, "_ts": now}
    logger.info("ECB OK %s/%s: %.4f (date: %s)", flow_ref, series_key, current_val, current_date)
    return result


def fetch_ecb_yield_history(maturity_key: str, limit: int = 300) -> list[dict] | None:
    """
    Fetch ECB Euro Area AAA-rated government bond spot rate history.
    Uses the ECB Yield Curve (YC) dataset — daily business day series.

    maturity_key: 'SR_2Y' | 'SR_5Y' | 'SR_10Y' | 'SR_30Y'
    Full series key: YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity_key}

    Returns list of {date, value} newest-first, or None on failure.
    Cached 1 hour.
    index[0]   = current value
    index[63]  ≈ 3 months ago (63 business days)
    index[252] ≈ 1 year ago  (252 business days)
    """
    flow_ref   = "YC"
    series_key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity_key}"
    cache_key  = f"ecb_yield_hist:{maturity_key}"

    now    = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["data"]

    url    = f"{ECB_BASE}/{flow_ref}/{series_key}"
    params = {"format": "jsondata", "detail": "dataonly", "lastNObservations": str(limit)}

    try:
        resp = requests.get(url, params=params, timeout=15, headers=ECB_HEADERS)
        resp.raise_for_status()
        raw  = resp.json()
    except requests.HTTPError as exc:
        logger.error("ECB yield history HTTP error for %s: %s", maturity_key, exc)
        return None
    except Exception as exc:
        logger.error("ECB yield history failed for %s: %s", maturity_key, exc)
        return None

    observations = _parse_ecb_jsondata(raw)

    if not observations:
        logger.warning("ECB yield history empty for %s (series: %s)", maturity_key, series_key)
        return None

    _CACHE[cache_key] = {"data": observations, "_ts": now}
    logger.info(
        "ECB yield history OK %s: %d obs, latest %.3f%% (%s)",
        maturity_key, len(observations), observations[0]["value"], observations[0]["date"],
    )
    return observations

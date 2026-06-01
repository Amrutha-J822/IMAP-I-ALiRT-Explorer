"""Data access for the public IMAP I-ALiRT API.

Primary endpoint: ``https://ialirt.imap-mission.com``

Schema notes (probed against the live API):

- ``GET /space-weather?instrument=<name>`` returns
  ``{"meta": {"count": N, ...}, "data": [<record>, ...]}``.
- Variable names are prefixed by instrument and vector fields are encoded as
  3- or 4-element lists (e.g. ``mag_B_GSE: [bx, by, bz]``,
  ``codice_hi_h: [e0, e1, e2, e3]``).
- ``time_utc`` is an ISO-8601 string per record.
- Each record may have non-trivial nulls (e.g. CoDICE composition ratios can
  be ``None`` when not yet computed).
- Open-ended queries return the most recent ~5 minutes of samples. Queries
  with explicit ``time_utc_start`` / ``time_utc_end`` longer than that
  return a 400 ``"too much data"`` response.

- ``GET /ialirt-archive-query`` returns ``{"files": [<filename>, ...]}`` -
  a list of bare CDF filenames, not records.
- ``GET /ialirt-download/archive/<filename>`` streams the CDF body.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_API_URL = os.environ.get(
    "IALIRT_DATA_ACCESS_URL", "https://ialirt.imap-mission.com"
).rstrip("/")
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class InstrumentSpec:
    """Normalized I-ALiRT schema for one instrument."""

    instrument: str
    cadence_seconds: int
    columns: tuple[str, ...]


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _vector_component(record: dict[str, Any], key: str, index: int) -> float:
    value = record.get(key)
    if isinstance(value, list) and len(value) > index:
        return _coerce_float(value[index])
    return float("nan")


def _extract_mag(record: dict[str, Any]) -> dict[str, float]:
    return {
        "Bx_nT": _vector_component(record, "mag_B_GSE", 0),
        "By_nT": _vector_component(record, "mag_B_GSE", 1),
        "Bz_nT": _vector_component(record, "mag_B_GSE", 2),
        "B_total_nT": _coerce_float(record.get("mag_B_magnitude")),
    }


def _extract_swapi(record: dict[str, Any]) -> dict[str, float]:
    return {
        "proton_speed_km_s": _coerce_float(record.get("swapi_pseudo_proton_speed")),
        "proton_density_cc": _coerce_float(record.get("swapi_pseudo_proton_density")),
        "proton_temp_K": _coerce_float(record.get("swapi_pseudo_proton_temperature")),
    }


def _extract_swe(record: dict[str, Any]) -> dict[str, float]:
    counts = record.get("swe_normalized_counts")
    if isinstance(counts, list) and counts:
        valid = [_coerce_float(value) for value in counts]
        valid = [value for value in valid if np.isfinite(value)]
        mean_val = float(np.mean(valid)) if valid else float("nan")
        max_val = float(np.max(valid)) if valid else float("nan")
    else:
        mean_val = float("nan")
        max_val = float("nan")
    return {
        "electron_counts_mean": mean_val,
        "electron_counts_max": max_val,
        "counterstreaming_flag": _coerce_float(record.get("swe_counterstreaming_electrons")),
    }


def _extract_hit(record: dict[str, Any]) -> dict[str, float]:
    return {
        "h_low_en": _coerce_float(record.get("hit_h_omni_low_en")),
        "h_med_en": _coerce_float(record.get("hit_h_omni_med_en")),
        "he_low_en": _coerce_float(record.get("hit_he_omni_low_en")),
        "he_high_en": _coerce_float(record.get("hit_he_omni_high_en")),
        "e_a_med_en": _coerce_float(record.get("hit_e_a_side_med_en")),
        "e_b_med_en": _coerce_float(record.get("hit_e_b_side_med_en")),
    }


def _extract_codice_lo(record: dict[str, Any]) -> dict[str, float]:
    return {
        "c_over_o": _coerce_float(record.get("codice_lo_c_over_o_abundance")),
        "fe_over_o": _coerce_float(record.get("codice_lo_fe_over_o_abundance")),
        "mg_over_o": _coerce_float(record.get("codice_lo_mg_over_o_abundance")),
        "o7_over_o6": _coerce_float(record.get("codice_lo_o_plus_7_over_o_plus_6")),
        "c6_over_c5": _coerce_float(record.get("codice_lo_c_plus_6_over_c_plus_5")),
        "fe_low_over_fe_high": _coerce_float(record.get("codice_lo_fe_low_over_fe_high")),
    }


def _extract_codice_hi(record: dict[str, Any]) -> dict[str, float]:
    return {
        "h_e0": _vector_component(record, "codice_hi_h", 0),
        "h_e1": _vector_component(record, "codice_hi_h", 1),
        "h_e2": _vector_component(record, "codice_hi_h", 2),
        "h_e3": _vector_component(record, "codice_hi_h", 3),
    }


IALIRT_INSTRUMENTS: dict[str, InstrumentSpec] = {
    "mag": InstrumentSpec("mag", 4, ("Bx_nT", "By_nT", "Bz_nT", "B_total_nT")),
    "swapi": InstrumentSpec(
        "swapi", 30, ("proton_speed_km_s", "proton_density_cc", "proton_temp_K")
    ),
    "swe": InstrumentSpec(
        "swe", 12, ("electron_counts_mean", "electron_counts_max", "counterstreaming_flag")
    ),
    "hit": InstrumentSpec(
        "hit",
        60,
        ("h_low_en", "h_med_en", "he_low_en", "he_high_en", "e_a_med_en", "e_b_med_en"),
    ),
    "codice_lo": InstrumentSpec(
        "codice_lo",
        60,
        ("c_over_o", "fe_over_o", "mg_over_o", "o7_over_o6", "c6_over_c5", "fe_low_over_fe_high"),
    ),
    "codice_hi": InstrumentSpec(
        "codice_hi", 60, ("h_e0", "h_e1", "h_e2", "h_e3")
    ),
}

_EXTRACTORS: dict[str, Callable[[dict[str, Any]], dict[str, float]]] = {
    "mag": _extract_mag,
    "swapi": _extract_swapi,
    "swe": _extract_swe,
    "hit": _extract_hit,
    "codice_lo": _extract_codice_lo,
    "codice_hi": _extract_codice_hi,
}


def _validate_instrument(instrument: str) -> InstrumentSpec:
    key = instrument.lower()
    if key not in IALIRT_INSTRUMENTS:
        known = ", ".join(sorted(IALIRT_INSTRUMENTS))
        raise ValueError(f"Unknown instrument {instrument!r}. Expected one of: {known}")
    return IALIRT_INSTRUMENTS[key]


def _iso_utc(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _yyyymmdd(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("-", "")
    return value.strftime("%Y%m%d")


def _api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("IMAP_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _records_from_space_weather(payload: Any) -> list[dict[str, Any]]:
    """Extract the per-sample record list from a /space-weather response."""

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        for key in ("results", "items", "records", "body"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _records_to_frame(
    records: list[dict[str, Any]], spec: InstrumentSpec
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    extractor = _EXTRACTORS[spec.instrument]
    times: list[Any] = []
    rows: list[dict[str, float]] = []
    for record in records:
        times.append(record.get("time_utc") or record.get("epoch") or record.get("timestamp"))
        rows.append(extractor(record))

    try:
        index = pd.to_datetime(times, utc=True, errors="coerce")
    except (TypeError, ValueError):
        index = pd.DatetimeIndex(
            pd.date_range(
                end=pd.Timestamp.now(tz="UTC").floor("s"),
                periods=len(records),
                freq=f"{max(spec.cadence_seconds, 1)}s",
            )
        )

    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="time"))
    frame = frame[~frame.index.isna()]
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    # Ensure schema columns are always present in stable order, even if a
    # particular record omitted a field (e.g. CoDICE composition nulls).
    for column in spec.columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[list(spec.columns)]

    if spec.instrument == "mag" and "B_total_nT" in frame.columns:
        derived_total = np.sqrt(
            frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2
        )
        # If the API didn't include the magnitude (or it's NaN), fall back to
        # the derived value; otherwise trust the mission-provided value.
        frame["B_total_nT"] = frame["B_total_nT"].where(
            frame["B_total_nT"].notna(), derived_total
        )

    return frame.dropna(how="all")


def _query_space_weather_package(
    instrument: str,
    time_utc_start: str | None,
    time_utc_end: str | None,
) -> list[dict[str, Any]]:
    """Call the official ialirt-data-access package if it exposes a function."""

    try:
        import ialirt_data_access
    except ImportError:
        return []

    for attr in ("space_weather", "query", "space_weather_query", "query_space_weather"):
        func = getattr(ialirt_data_access, attr, None)
        if callable(func):
            try:
                kwargs: dict[str, Any] = {"instrument": instrument}
                if time_utc_start is not None:
                    kwargs["time_utc_start"] = time_utc_start
                if time_utc_end is not None:
                    kwargs["time_utc_end"] = time_utc_end
                result = func(**kwargs)
                normalized = _records_from_space_weather(result)
                if normalized:
                    return normalized
            except Exception as exc:  # pragma: no cover - depends on remote state
                log.info("ialirt_data_access.%s failed: %s", attr, exc)
                continue
    return []


def _query_space_weather_rest(
    api_url: str,
    instrument: str,
    time_utc_start: str | None,
    time_utc_end: str | None,
) -> list[dict[str, Any]]:
    """Direct REST call against /space-weather.

    Open-ended queries (no time params) return the most recent samples; the
    endpoint enforces a small window and returns HTTP 400 if a longer window
    is requested. Callers can pass a narrow window when they care about a
    specific interval.
    """

    params: dict[str, str] = {"instrument": instrument}
    if time_utc_start is not None:
        params["time_utc_start"] = time_utc_start
    if time_utc_end is not None:
        params["time_utc_end"] = time_utc_end

    try:
        response = requests.get(
            f"{api_url}/space-weather",
            params=params,
            headers=_api_headers(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.info("REST /space-weather failed: %s", exc)
        return []

    try:
        payload = response.json()
    except ValueError:
        return []
    return _records_from_space_weather(payload)


def fetch_space_weather(
    instrument: str,
    *,
    time_utc_start: date | datetime | str | None = None,
    time_utc_end: date | datetime | str | None = None,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch normalized live I-ALiRT samples for one instrument.

    Tries the ``ialirt-data-access`` Python package first, falls back to the
    documented public REST endpoint, and finally to deterministic synthetic
    data if both fail and ``fallback=True``. When no time window is given
    the endpoint returns its native recent-samples window (~5 minutes).
    """

    spec = _validate_instrument(instrument)
    start = _iso_utc(time_utc_start)
    end = _iso_utc(time_utc_end)

    records = _query_space_weather_package(spec.instrument, start, end)
    if not records:
        records = _query_space_weather_rest(api_url, spec.instrument, start, end)

    frame = _records_to_frame(records, spec)
    if not frame.empty:
        frame.attrs["source"] = "ialirt-sdc"
        frame.attrs["instrument"] = spec.instrument
        return frame

    if not fallback:
        raise RuntimeError(
            f"No I-ALiRT samples available for {spec.instrument!r} between "
            f"{start} and {end}."
        )

    log.info("Using synthetic fallback for %s", spec.instrument)
    return _synthetic_data(spec.instrument, days=1)


async def fetch_space_weather_async(
    client: httpx.AsyncClient,
    instrument: str,
    *,
    time_utc_start: str | None = None,
    time_utc_end: str | None = None,
) -> pd.DataFrame:
    """Async variant of :func:`fetch_space_weather` for the live poller."""

    spec = _validate_instrument(instrument)
    params: dict[str, str] = {"instrument": spec.instrument}
    if time_utc_start is not None:
        params["time_utc_start"] = time_utc_start
    if time_utc_end is not None:
        params["time_utc_end"] = time_utc_end

    try:
        response = await client.get(
            "/space-weather", params=params, headers=_api_headers()
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.info("async /space-weather failed for %s: %s", spec.instrument, exc)
        return pd.DataFrame()

    frame = _records_to_frame(_records_from_space_weather(payload), spec)
    if not frame.empty:
        frame.attrs["source"] = "ialirt-sdc"
        frame.attrs["instrument"] = spec.instrument
    return frame


def _records_from_archive(payload: Any) -> list[dict[str, Any]]:
    """Adapt /ialirt-archive-query response to a list of record dicts."""

    if isinstance(payload, dict):
        files = payload.get("files")
        if isinstance(files, list):
            return [{"filename": str(name)} for name in files if name]
        for key in ("data", "results", "items", "records", "body"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [
                    item if isinstance(item, dict) else {"filename": str(item)}
                    for item in inner
                ]
    if isinstance(payload, list):
        return [
            item if isinstance(item, dict) else {"filename": str(item)}
            for item in payload
        ]
    return []


def list_available(
    instrument: str | None = None,
    *,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    since: date | datetime | str | None = None,
    version: int = 1,
    api_url: str = DEFAULT_API_URL,
) -> list[dict[str, Any]]:
    """List archived I-ALiRT CDF files using ``/ialirt-archive-query``.

    The response is normalized to ``[{"filename": str}, ...]`` so callers
    don't have to special-case the bare-string list shape returned by the
    server.
    """

    params: dict[str, str | int] = {"version": version}
    if since is not None:
        params["since"] = _yyyymmdd(since) or ""
    else:
        if year is not None:
            params["year"] = year
        if month is not None:
            params["month"] = month
        if day is not None:
            params["day"] = day

    try:
        response = requests.get(
            f"{api_url}/ialirt-archive-query",
            params=params,
            headers=_api_headers(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        records = _records_from_archive(response.json())
    except (requests.RequestException, ValueError) as exc:
        log.info("REST /ialirt-archive-query failed: %s", exc)
        return []

    if instrument is None:
        return records
    instrument_lower = instrument.lower()
    return [
        record
        for record in records
        if instrument_lower in record.get("filename", "").lower()
    ]


def _download_archive_with_package(filename: str, target_dir: Path) -> Path | None:
    try:
        import ialirt_data_access
    except ImportError:
        return None
    download = getattr(ialirt_data_access, "download", None)
    if not callable(download):
        return None
    try:
        result = download(
            filetype="archive", filename=filename, downloads_dir=str(target_dir)
        )
    except Exception as exc:  # pragma: no cover - depends on remote state
        log.info("ialirt_data_access.download failed for %s: %s", filename, exc)
        return None
    if result is None:
        candidate = target_dir / filename
    else:
        candidate = Path(result)
    return candidate if candidate.exists() else None


def _download_archive_with_rest(
    filename: str, target_dir: Path, api_url: str
) -> Path | None:
    try:
        response = requests.get(
            f"{api_url}/ialirt-download/archive/{filename}",
            headers=_api_headers(),
            timeout=60,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.info("REST archive download failed for %s: %s", filename, exc)
        return None
    target = target_dir / Path(filename).name
    target.write_bytes(response.content)
    return target


def _cdf_time_index(cdf: Any, n_rows: int) -> pd.DatetimeIndex:
    info = cdf.cdf_info()
    variables = list(getattr(info, "zVariables", []) or []) + list(
        getattr(info, "rVariables", []) or []
    )
    candidates = [
        var for var in variables if "epoch" in var.lower() or "time" in var.lower()
    ]
    for var in candidates:
        try:
            raw = cdf.varget(var)
            if len(raw) != n_rows:
                continue
            try:
                from cdflib import cdfepoch

                converted = cdfepoch.to_datetime(raw)
                return pd.DatetimeIndex(converted, tz="UTC", name="time")
            except Exception:
                numeric = np.asarray(raw, dtype=float)
                if np.nanmedian(numeric) > 1e12:
                    unit = "ns"
                elif np.nanmedian(numeric) > 1e9:
                    unit = "s"
                else:
                    continue
                return pd.to_datetime(numeric, unit=unit, utc=True).rename("time")
        except Exception:
            continue
    return pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("s"),
        periods=n_rows,
        freq="1min",
        name="time",
    )


def _read_cdf(path: Path, instrument: str) -> pd.DataFrame:
    """Read a CDF file. CDF parsing is best-effort - returns whatever
    numeric variables it can recognize and aligns them on a common index."""

    try:
        from cdflib import CDF
    except ImportError as exc:
        raise RuntimeError("cdflib is required to parse CDF files") from exc

    _validate_instrument(instrument)
    with CDF(str(path)) as cdf:
        info = cdf.cdf_info()
        variables = list(getattr(info, "zVariables", []) or []) + list(
            getattr(info, "rVariables", []) or []
        )
        columns: dict[str, np.ndarray] = {}
        for var in variables:
            if "epoch" in var.lower() or "time" in var.lower():
                continue
            try:
                values = np.asarray(cdf.varget(var), dtype=float)
            except (TypeError, ValueError):
                continue
            if values.ndim == 1:
                columns[var] = values
        if not columns:
            raise ValueError(f"No numeric variables in {path}")
        n_rows = min(len(values) for values in columns.values())
        index = _cdf_time_index(cdf, n_rows)
        frame = pd.DataFrame(
            {key: value[:n_rows] for key, value in columns.items()},
            index=index[:n_rows],
        )
    frame.index.name = "time"
    return frame.sort_index()


def fetch_archive(
    instrument: str,
    *,
    since: date | datetime | str | None = None,
    max_files: int = 4,
    parallel: bool = True,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch archived CDF data for an instrument."""

    spec = _validate_instrument(instrument)
    records = list_available(
        spec.instrument, since=since, api_url=api_url
    )[-max_files:]

    if not records:
        if not fallback:
            raise RuntimeError(f"No archive files found for {spec.instrument!r}")
        log.info("No archive files for %s; using synthetic fallback", spec.instrument)
        return _synthetic_data(spec.instrument, days=3)

    frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="ialirt_") as tmp:
        tmp_path = Path(tmp)

        def _fetch_one(record: dict[str, Any]) -> pd.DataFrame | None:
            filename = (
                record.get("filename")
                or record.get("file_name")
                or record.get("file_path")
            )
            if not filename:
                return None
            filename = Path(str(filename)).name
            local = _download_archive_with_package(filename, tmp_path)
            if local is None:
                local = _download_archive_with_rest(filename, tmp_path, api_url)
            if local is None:
                return None
            try:
                return _read_cdf(local, spec.instrument)
            except Exception as exc:
                log.info("Could not parse %s: %s", filename, exc)
                return None

        if parallel and len(records) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(records))) as executor:
                for future in as_completed(
                    [executor.submit(_fetch_one, record) for record in records]
                ):
                    frame = future.result()
                    if frame is not None and not frame.empty:
                        frames.append(frame)
        else:
            for record in records:
                frame = _fetch_one(record)
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if not frames:
        if not fallback:
            raise RuntimeError(f"No usable archive files for {spec.instrument!r}")
        return _synthetic_data(spec.instrument, days=3)

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.attrs["source"] = "ialirt-archive"
    combined.attrs["instrument"] = spec.instrument
    return combined


def fetch_latest(
    instrument: str,
    *,
    days: int = 1,  # noqa: ARG001 - retained for backward compatibility
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch the most recent live I-ALiRT samples for an instrument.

    The ``/space-weather`` endpoint serves a fixed recent-samples window
    (open-ended queries return the latest available data and explicit long
    windows are rejected). The ``days`` argument is kept for API
    compatibility but does not widen the window beyond what the endpoint
    allows; the archive endpoint covers longer ranges.
    """

    try:
        frame = fetch_space_weather(instrument, api_url=api_url, fallback=False)
    except RuntimeError:
        frame = pd.DataFrame()

    if not frame.empty:
        return frame

    if not fallback:
        raise RuntimeError(f"No I-ALiRT data available for {instrument!r}")

    return _synthetic_data(instrument, days=1)


def fetch_range(
    instrument: str,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch I-ALiRT data over an explicit date range.

    Combines archive CDF products (best for historical windows) with the
    live ``/space-weather`` tail (the most recent few minutes).
    """

    spec = _validate_instrument(instrument)

    try:
        live = fetch_space_weather(
            spec.instrument,
            api_url=api_url,
            fallback=False,
        )
    except RuntimeError:
        live = pd.DataFrame()

    try:
        archive = fetch_archive(
            spec.instrument,
            since=start_date,
            api_url=api_url,
            fallback=False,
        )
    except RuntimeError:
        archive = pd.DataFrame()

    # Optional client-side trim
    if end_date is not None:
        end_ts = pd.Timestamp(_iso_utc(end_date) or end_date, tz="UTC")
        live = live[live.index <= end_ts] if not live.empty else live
        archive = archive[archive.index <= end_ts] if not archive.empty else archive

    frames = [df for df in (archive, live) if not df.empty]
    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.attrs["source"] = "ialirt-sdc"
        combined.attrs["instrument"] = spec.instrument
        return combined

    if not fallback:
        raise RuntimeError(f"No usable public IMAP files found for {spec.instrument}")

    log.info("Using synthetic fallback data for %s", spec.instrument)
    return _synthetic_data(spec.instrument, days=7)


def _synthetic_data(
    instrument: str,
    *,
    n_points: int | None = None,
    days: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, physically plausible I-ALiRT-like data.

    Used when the upstream API is unreachable, so dashboards, tests, and the
    pub/sub service remain demonstrable offline. The generated frames carry
    the same column schema that the real API returns, so downstream code
    does not need to special-case the fallback path.
    """

    spec = _validate_instrument(instrument)
    if n_points is None:
        cadence = max(spec.cadence_seconds, 60)
        n_points = max(12, int(days * 24 * 3600 / cadence))

    rng = np.random.default_rng(seed + sum(ord(char) for char in spec.instrument))
    freq = f"{max(spec.cadence_seconds, 60)}s"
    index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").floor("min"),
        periods=n_points,
        freq=freq,
        name="time",
    )
    phase = np.linspace(0, 8 * np.pi, n_points, dtype=np.float64)
    slow = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)

    if spec.instrument == "mag":
        bx = 2.3 + 0.5 * np.sin(phase) + 0.4 * rng.normal(size=n_points) + 0.2 * slow
        by = -4.2 + 0.4 * np.cos(phase / 2.0) + 0.5 * rng.normal(size=n_points)
        bz = -4.7 + 0.3 * np.sin(phase / 3.0) + 0.5 * rng.normal(size=n_points)
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 40)))
            bz[event] -= 7.0
            bx[event] += 3.0
        data: dict[str, np.ndarray] = {
            "Bx_nT": bx,
            "By_nT": by,
            "Bz_nT": bz,
            "B_total_nT": np.sqrt(bx**2 + by**2 + bz**2),
        }
    elif spec.instrument == "swapi":
        speed = 425 + 45 * np.sin(phase / 3.0) + rng.normal(0, 20, n_points)
        density = 6 + 1.3 * np.cos(phase / 2.0) + rng.normal(0, 0.45, n_points)
        temp = 95_000 + 18_000 * np.sin(phase / 2.5) + rng.normal(0, 5_000, n_points)
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 35)))
            speed[event] += 180
            density[event] += 4
            temp[event] += 60_000
        data = {
            "proton_speed_km_s": np.clip(speed, 250, 950),
            "proton_density_cc": np.clip(density, 0.2, None),
            "proton_temp_K": np.clip(temp, 10_000, None),
        }
    elif spec.instrument == "swe":
        mean_counts = 50 + 15 * np.sin(phase / 2.0) + rng.normal(0, 4, n_points)
        max_counts = mean_counts * (1.4 + 0.2 * np.cos(phase / 3.0))
        flag = (rng.random(n_points) > 0.7).astype(float)
        data = {
            "electron_counts_mean": np.clip(mean_counts, 1.0, None),
            "electron_counts_max": np.clip(max_counts, 1.0, None),
            "counterstreaming_flag": flag,
        }
    elif spec.instrument == "hit":
        h_low = rng.poisson(lam=3, size=n_points).astype(float)
        h_med = rng.poisson(lam=1, size=n_points).astype(float)
        he_low = rng.poisson(lam=0.5, size=n_points).astype(float)
        he_high = rng.poisson(lam=0.2, size=n_points).astype(float)
        e_a = rng.poisson(lam=18, size=n_points).astype(float)
        e_b = rng.poisson(lam=22, size=n_points).astype(float)
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 30)))
            h_low[event] *= 8
            he_low[event] *= 5
        data = {
            "h_low_en": h_low,
            "h_med_en": h_med,
            "he_low_en": he_low,
            "he_high_en": he_high,
            "e_a_med_en": e_a,
            "e_b_med_en": e_b,
        }
    elif spec.instrument == "codice_lo":
        data = {
            "c_over_o": 0.7 + 0.05 * np.sin(phase / 2) + rng.normal(0, 0.02, n_points),
            "fe_over_o": 0.13 + 0.02 * np.cos(phase / 3) + rng.normal(0, 0.01, n_points),
            "mg_over_o": 0.15 + 0.02 * np.sin(phase / 4) + rng.normal(0, 0.01, n_points),
            "o7_over_o6": 0.20 + 0.05 * np.sin(phase / 2.5) + rng.normal(0, 0.02, n_points),
            "c6_over_c5": 0.50 + 0.10 * np.cos(phase / 3) + rng.normal(0, 0.03, n_points),
            "fe_low_over_fe_high": 1.0 + 0.2 * np.sin(phase / 4) + rng.normal(0, 0.05, n_points),
        }
    else:  # codice_hi
        h_e0 = rng.poisson(lam=5, size=n_points).astype(float)
        h_e1 = rng.poisson(lam=3, size=n_points).astype(float)
        h_e2 = rng.poisson(lam=1.5, size=n_points).astype(float)
        h_e3 = rng.poisson(lam=0.8, size=n_points).astype(float)
        data = {"h_e0": h_e0, "h_e1": h_e1, "h_e2": h_e2, "h_e3": h_e3}

    frame = pd.DataFrame(data, index=index)
    frame.attrs["source"] = "synthetic-fallback"
    frame.attrs["instrument"] = spec.instrument
    return frame.astype("float64")

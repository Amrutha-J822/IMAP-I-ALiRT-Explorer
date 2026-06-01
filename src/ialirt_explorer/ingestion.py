"""Data access for the public IMAP I-ALiRT API.

Primary endpoint: ``https://ialirt.imap-mission.com``

The module prefers the official ``ialirt-data-access`` Python package when it is
installed, then falls back to direct REST calls against the documented public
endpoints, and finally to a deterministic synthetic generator so demos, tests,
and CI keep working without network.

Public endpoints (per the IMAP SOC documentation):

- ``GET /space-weather`` - live per-instrument data (the primary feed)
- ``GET /ialirt-archive-query`` - list archived CDF files
- ``GET /ialirt-download/<filetype>/<filename>`` - retrieve archived files
"""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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


IALIRT_INSTRUMENTS: dict[str, InstrumentSpec] = {
    "mag": InstrumentSpec("mag", 1, ("Bx_nT", "By_nT", "Bz_nT", "B_total_nT")),
    "swe": InstrumentSpec(
        "swe", 12, ("electron_density_cc", "electron_temp_K", "heat_flux")
    ),
    "swapi": InstrumentSpec(
        "swapi", 30, ("proton_speed_km_s", "proton_density_cc", "proton_temp_K")
    ),
    "hit": InstrumentSpec("hit", 60, ("h_flux", "he_flux", "heavy_ion_flux")),
    "codice_lo": InstrumentSpec(
        "codice_lo", 60, ("ion_flux_low_energy", "ion_temp_K")
    ),
    "codice_hi": InstrumentSpec(
        "codice_hi", 60, ("ion_flux_high_energy", "energetic_ion_temp_K")
    ),
}

VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "Bx_nT": ("bx_gse", "bx", "b_gse_x", "mag_x", "x", "Bx"),
    "By_nT": ("by_gse", "by", "b_gse_y", "mag_y", "y", "By"),
    "Bz_nT": ("bz_gse", "bz", "b_gse_z", "mag_z", "z", "Bz"),
    "B_total_nT": ("bt", "btotal", "b_total", "mag_total", "magnitude", "|B|"),
    "proton_speed_km_s": ("proton_speed", "speed", "v_sw", "velocity", "vp"),
    "proton_density_cc": ("proton_density", "density", "n_p", "np"),
    "proton_temp_K": ("proton_temperature", "temperature", "t_p", "tp"),
    "electron_density_cc": ("electron_density", "n_e", "ne"),
    "electron_temp_K": ("electron_temperature", "t_e", "te"),
    "heat_flux": ("heat_flux", "q_e", "strahl"),
    "h_flux": ("h_flux", "proton_flux", "hydrogen_flux"),
    "he_flux": ("he_flux", "helium_flux", "alpha_flux"),
    "heavy_ion_flux": ("heavy_ion_flux", "ion_flux", "z_gt_2_flux"),
    "ion_flux_low_energy": ("low_energy_flux", "lo_flux", "codice_lo_flux"),
    "ion_temp_K": ("ion_temperature", "t_i", "ti"),
    "ion_flux_high_energy": ("high_energy_flux", "hi_flux", "codice_hi_flux"),
    "energetic_ion_temp_K": ("energetic_temp", "t_ei"),
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


def _normalize_records(payload: Any) -> list[dict[str, Any]]:
    """Coerce a heterogeneous API response into a list of record dicts."""

    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "records", "body"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        if all(isinstance(value, list) for value in payload.values()) and payload:
            length = min(len(v) for v in payload.values())
            return [
                {column: payload[column][i] for column in payload}
                for i in range(length)
            ]
    return []


def _find_first(record: dict[str, Any], options: tuple[str, ...]) -> Any:
    lower_map = {key.lower(): key for key in record}
    for option in options:
        if option.lower() in lower_map:
            return record[lower_map[option.lower()]]
    for option in options:
        option_lower = option.lower()
        for lower_key, original in lower_map.items():
            if option_lower in lower_key:
                return record[original]
    return None


def _records_to_frame(
    records: list[dict[str, Any]], spec: InstrumentSpec
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    time_keys = ("time_utc", "epoch", "timestamp", "met_in_utc", "time", "date")
    times: list[Any] = []
    for record in records:
        raw_time = _find_first(record, time_keys)
        times.append(raw_time)

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
    if index.isna().all():
        index = pd.DatetimeIndex(
            pd.date_range(
                end=pd.Timestamp.now(tz="UTC").floor("s"),
                periods=len(records),
                freq=f"{max(spec.cadence_seconds, 1)}s",
            )
        )

    columns: dict[str, list[float]] = {column: [] for column in spec.columns}
    for record in records:
        for column in spec.columns:
            aliases = VARIABLE_ALIASES.get(column, (column,))
            value = _find_first(record, aliases)
            try:
                columns[column].append(float(value))
            except (TypeError, ValueError):
                columns[column].append(np.nan)

    frame = pd.DataFrame(columns, index=pd.DatetimeIndex(index, name="time"))
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    if spec.instrument == "mag" and {"Bx_nT", "By_nT", "Bz_nT"}.issubset(frame.columns):
        frame["B_total_nT"] = np.sqrt(
            frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2
        )

    frame = frame.dropna(how="all")
    return frame


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
                normalized = _normalize_records(result)
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
    params = {"instrument": instrument}
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
    return _normalize_records(payload)


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
    data if both fail and ``fallback=True``.
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
    """Async variant of :func:`fetch_space_weather` for the live poller.

    Uses the provided ``httpx.AsyncClient`` so callers can pool connections
    across many polling cycles.
    """

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

    frame = _records_to_frame(_normalize_records(payload), spec)
    if not frame.empty:
        frame.attrs["source"] = "ialirt-sdc"
        frame.attrs["instrument"] = spec.instrument
    return frame


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

    The optional ``instrument`` filter is applied client-side since the
    public endpoint groups archive products together.
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
        records = _normalize_records(response.json())
    except (requests.RequestException, ValueError) as exc:
        log.info("REST /ialirt-archive-query failed: %s", exc)
        return []

    if instrument is None:
        return records
    instrument_lower = instrument.lower()
    return [
        record
        for record in records
        if instrument_lower in str(record).lower()
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
        result = download(filetype="archive", filename=filename, downloads_dir=str(target_dir))
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


def _first_matching_variable(cdf: Any, options: tuple[str, ...]) -> np.ndarray | None:
    info = cdf.cdf_info()
    variables = list(getattr(info, "zVariables", []) or []) + list(
        getattr(info, "rVariables", []) or []
    )
    lower_map = {var.lower(): var for var in variables}
    for option in options:
        if option.lower() in lower_map:
            return np.asarray(cdf.varget(lower_map[option.lower()]), dtype=float)
    for option in options:
        option_lower = option.lower()
        for lower, original in lower_map.items():
            if option_lower in lower:
                return np.asarray(cdf.varget(original), dtype=float)
    return None


def _read_cdf(path: Path, instrument: str) -> pd.DataFrame:
    try:
        from cdflib import CDF
    except ImportError as exc:
        raise RuntimeError("cdflib is required to parse CDF files") from exc

    spec = _validate_instrument(instrument)
    with CDF(str(path)) as cdf:
        columns: dict[str, np.ndarray] = {}
        for target_col in spec.columns:
            values = _first_matching_variable(
                cdf, VARIABLE_ALIASES.get(target_col, (target_col,))
            )
            if values is not None:
                columns[target_col] = np.ravel(values).astype(float)

        if not columns:
            raise ValueError(f"No recognized {instrument} variables in {path}")

        n_rows = min(len(values) for values in columns.values())
        index = _cdf_time_index(cdf, n_rows)
        frame = pd.DataFrame(
            {key: value[:n_rows] for key, value in columns.items()},
            index=index[:n_rows],
        )

    if instrument == "mag" and {"Bx_nT", "By_nT", "Bz_nT"}.issubset(frame.columns):
        frame["B_total_nT"] = np.sqrt(
            frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2
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
    """Fetch archived CDF data for an instrument.

    Useful for replaying historic events alongside the live feed.
    """

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
            filename = record.get("filename") or record.get("file_name") or record.get(
                "file_path"
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
    days: int = 1,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch the most recent live I-ALiRT samples for an instrument.

    Tries the live ``/space-weather`` endpoint first; if no live samples
    are available the archive endpoint is consulted; finally falls back to
    synthetic data if both are empty and ``fallback=True``.
    """

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=max(days, 0), minutes=0)
    frame = pd.DataFrame()

    try:
        frame = fetch_space_weather(
            instrument,
            time_utc_start=start,
            time_utc_end=end,
            api_url=api_url,
            fallback=False,
        )
    except RuntimeError:
        frame = pd.DataFrame()

    if not frame.empty:
        return frame

    try:
        frame = fetch_archive(
            instrument,
            since=start.date(),
            api_url=api_url,
            fallback=False,
        )
        if not frame.empty:
            return frame
    except RuntimeError:
        pass

    if not fallback:
        raise RuntimeError(f"No I-ALiRT data available for {instrument!r}")

    return _synthetic_data(instrument, days=days)


def fetch_range(
    instrument: str,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch I-ALiRT data over an explicit date range.

    Combines live ``/space-weather`` samples (when the range overlaps the
    last ~24h) and archive CDF products when available.
    """

    spec = _validate_instrument(instrument)

    try:
        live = fetch_space_weather(
            spec.instrument,
            time_utc_start=start_date,
            time_utc_end=end_date,
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
    days: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, physically plausible I-ALiRT-like data.

    Used when the upstream API is unreachable, so dashboards, tests, and the
    pub/sub service remain demonstrable offline.
    """

    spec = _validate_instrument(instrument)
    if n_points is None:
        cadence = max(spec.cadence_seconds, 300)
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
        bx = 4.5 * np.sin(phase) + 0.8 * rng.normal(size=n_points) + 0.7 * slow
        by = 3.0 * np.cos(phase / 2.0) + 0.7 * rng.normal(size=n_points) - 0.3 * slow
        bz = 2.0 * np.sin(phase / 3.0) + 0.9 * rng.normal(size=n_points) + 1.4 * slow
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 40)))
            bz[event] -= 7.0
            bx[event] += 3.0
        data = {
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
        density = 7 + 1.1 * np.sin(phase / 2.0) + rng.normal(0, 0.35, n_points)
        temp = 130_000 + 20_000 * np.cos(phase / 2.8) + rng.normal(0, 6_000, n_points)
        heat_flux = 0.8 + 0.25 * np.sin(phase) + rng.normal(0, 0.05, n_points)
        data = {
            "electron_density_cc": np.clip(density, 0.1, None),
            "electron_temp_K": np.clip(temp, 20_000, None),
            "heat_flux": np.clip(heat_flux, 0.01, None),
        }
    elif spec.instrument == "hit":
        h_flux = rng.lognormal(mean=2.0, sigma=0.25, size=n_points)
        he_flux = rng.lognormal(mean=0.9, sigma=0.25, size=n_points)
        heavy = rng.lognormal(mean=0.25, sigma=0.3, size=n_points)
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 30)))
            h_flux[event] *= 8
            he_flux[event] *= 5
            heavy[event] *= 4
        data = {"h_flux": h_flux, "he_flux": he_flux, "heavy_ion_flux": heavy}
    elif spec.instrument == "codice_lo":
        low_flux = rng.lognormal(mean=1.5, sigma=0.4, size=n_points)
        ion_temp = 1.0e5 + 1.5e4 * np.sin(phase / 2.4) + rng.normal(0, 6_000, n_points)
        data = {
            "ion_flux_low_energy": low_flux,
            "ion_temp_K": np.clip(ion_temp, 5_000, None),
        }
    else:  # codice_hi
        hi_flux = rng.lognormal(mean=1.0, sigma=0.5, size=n_points)
        energetic_temp = (
            5.0e5 + 5.0e4 * np.sin(phase / 3.0) + rng.normal(0, 20_000, n_points)
        )
        data = {
            "ion_flux_high_energy": hi_flux,
            "energetic_ion_temp_K": np.clip(energetic_temp, 5e4, None),
        }

    frame = pd.DataFrame(data, index=index)
    frame.attrs["source"] = "synthetic-fallback"
    frame.attrs["instrument"] = spec.instrument
    return frame.astype("float64")

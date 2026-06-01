"""Data access for public IMAP I-ALiRT products.

The module prefers the official ``imap-data-access`` package because that is
the supported interface for the IMAP Science Data Center. Direct REST access is
kept as a small fallback, and deterministic synthetic data keeps tests and demos
usable when a laptop or CI runner has no network.
"""

from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.imap-mission.com"


@dataclass(frozen=True)
class InstrumentSpec:
    """Expected I-ALiRT schema and sampling for one instrument."""

    instrument: str
    data_level: str
    descriptor: str | None
    cadence_seconds: int
    columns: tuple[str, ...]


IALIRT_INSTRUMENTS: dict[str, InstrumentSpec] = {
    "mag": InstrumentSpec(
        "mag",
        "l1c",
        None,
        1,
        ("Bx_nT", "By_nT", "Bz_nT", "B_total_nT"),
    ),
    "swe": InstrumentSpec(
        "swe",
        "l1b",
        None,
        12,
        ("electron_density_cc", "electron_temp_K", "heat_flux"),
    ),
    "swapi": InstrumentSpec(
        "swapi",
        "l1b",
        None,
        30,
        ("proton_speed_km_s", "proton_density_cc", "proton_temp_K"),
    ),
    "hit": InstrumentSpec(
        "hit",
        "l1b",
        None,
        60,
        ("h_flux", "he_flux", "heavy_ion_flux"),
    ),
}


def _validate_instrument(instrument: str) -> InstrumentSpec:
    key = instrument.lower()
    if key not in IALIRT_INSTRUMENTS:
        known = ", ".join(sorted(IALIRT_INSTRUMENTS))
        raise ValueError(f"Unknown instrument {instrument!r}. Expected one of: {known}")
    return IALIRT_INSTRUMENTS[key]


def _date_string(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("-", "")
    return value.strftime("%Y%m%d")


def _normalise_query_payload(payload: Any) -> list[dict[str, Any]]:
    """Return a list of file metadata from API/package response variants."""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        body = payload.get("body", payload)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return []
        if isinstance(body, list):
            return [item for item in body if isinstance(item, dict)]
    return []


def _query_with_package(**params: Any) -> list[dict[str, Any]]:
    try:
        import imap_data_access
    except ImportError:
        return []

    query = getattr(imap_data_access, "query", None)
    if query is None:
        return []

    try:
        return _normalise_query_payload(query(**params))
    except Exception as exc:  # pragma: no cover - depends on remote SDC state
        log.info("imap-data-access query failed: %s", exc)
        return []


def _query_with_rest(api_url: str, **params: Any) -> list[dict[str, Any]]:
    clean_params = {key: value for key, value in params.items() if value is not None}
    try:
        response = requests.get(f"{api_url.rstrip('/')}/query", params=clean_params, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.info("IMAP REST query failed: %s", exc)
        return []
    return _normalise_query_payload(response.json())


def list_available(
    instrument: str,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    data_level: str | None = None,
    descriptor: str | None = None,
    api_url: str = DEFAULT_API_URL,
    prefer_official_package: bool = True,
) -> list[dict[str, Any]]:
    """List public SDC files for an I-ALiRT instrument.

    Parameters match the public IMAP SDC query endpoint. Missing network access
    is treated as an empty result rather than an exception so callers can decide
    whether to fall back to local sample data.
    """

    spec = _validate_instrument(instrument)
    params = {
        "instrument": spec.instrument,
        "data_level": data_level or spec.data_level,
        "descriptor": descriptor if descriptor is not None else spec.descriptor,
        "start_date": _date_string(start_date),
        "end_date": _date_string(end_date),
        "extension": "cdf",
        "version": "latest",
    }

    if prefer_official_package:
        files = _query_with_package(**params)
        if files:
            return sorted(files, key=lambda item: item.get("start_date", ""))

    files = _query_with_rest(api_url, **params)
    return sorted(files, key=lambda item: item.get("start_date", ""))


def _download_with_package(file_path: str, target_dir: Path) -> Path | None:
    try:
        import imap_data_access
    except ImportError:
        return None

    download = getattr(imap_data_access, "download", None)
    if download is None:
        return None

    try:
        downloaded = download(file_path)
    except Exception as exc:  # pragma: no cover - depends on remote SDC state
        log.info("imap-data-access download failed for %s: %s", file_path, exc)
        return None

    if downloaded is None:
        candidate = target_dir / file_path
    else:
        candidate = Path(downloaded)
    return candidate if candidate.exists() else None


def _download_with_rest(file_path: str, target_dir: Path, api_url: str) -> Path | None:
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/download/{file_path}",
            headers={"Accept": "application/json"},
            timeout=30,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.info("IMAP REST download failed for %s: %s", file_path, exc)
        return None

    target = target_dir / Path(file_path).name
    target.write_bytes(response.content)
    return target


def _cdf_time_index(cdf: Any, n_rows: int) -> pd.DatetimeIndex:
    """Find a likely time variable in a CDF and convert it to a UTC index."""

    info = cdf.cdf_info()
    variables = list(getattr(info, "zVariables", []) or []) + list(
        getattr(info, "rVariables", []) or []
    )
    candidates = [var for var in variables if "epoch" in var.lower() or "time" in var.lower()]
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
    """Read a CDF file into the normalized project schema."""

    try:
        from cdflib import CDF
    except ImportError as exc:
        raise RuntimeError("cdflib is required to parse CDF files") from exc

    spec = _validate_instrument(instrument)
    with CDF(str(path)) as cdf:
        variable_options = {
            "Bx_nT": ("bx_gse", "bx", "b_gse_x", "mag_x", "x"),
            "By_nT": ("by_gse", "by", "b_gse_y", "mag_y", "y"),
            "Bz_nT": ("bz_gse", "bz", "b_gse_z", "mag_z", "z"),
            "B_total_nT": ("bt", "btotal", "b_total", "mag_total", "magnitude"),
            "proton_speed_km_s": ("proton_speed", "speed", "v_sw", "velocity"),
            "proton_density_cc": ("proton_density", "density", "n_p", "np"),
            "proton_temp_K": ("proton_temperature", "temperature", "t_p", "tp"),
            "electron_density_cc": ("electron_density", "density", "n_e", "ne"),
            "electron_temp_K": ("electron_temperature", "temperature", "t_e", "te"),
            "heat_flux": ("heat_flux", "q_e", "strahl"),
            "h_flux": ("h_flux", "proton_flux", "hydrogen_flux"),
            "he_flux": ("he_flux", "helium_flux", "alpha_flux"),
            "heavy_ion_flux": ("heavy_ion_flux", "ion_flux", "z_gt_2_flux"),
        }

        columns: dict[str, np.ndarray] = {}
        for target_col in spec.columns:
            values = _first_matching_variable(cdf, variable_options.get(target_col, (target_col,)))
            if values is not None:
                columns[target_col] = np.ravel(values).astype(float)

        if not columns:
            raise ValueError(f"No recognized {instrument} variables in {path}")

        n_rows = min(len(values) for values in columns.values())
        index = _cdf_time_index(cdf, n_rows)
        frame = pd.DataFrame(
            {key: value[:n_rows] for key, value in columns.items()}, index=index[:n_rows]
        )

    if instrument == "mag" and {"Bx_nT", "By_nT", "Bz_nT"}.issubset(frame.columns):
        frame["B_total_nT"] = np.sqrt(
            frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2
        )
    frame.index.name = "time"
    return frame.sort_index()


def _fetch_one_file(
    file_metadata: dict[str, Any], instrument: str, api_url: str
) -> pd.DataFrame | None:
    file_path = file_metadata.get("file_path")
    if not file_path:
        return None

    with tempfile.TemporaryDirectory(prefix="ialirt_") as tmp:
        tmp_path = Path(tmp)
        path = _download_with_package(file_path, tmp_path) or _download_with_rest(
            file_path, tmp_path, api_url
        )
        if path is None:
            return None
        try:
            return _read_cdf(path, instrument)
        except Exception as exc:
            log.info("Could not parse %s: %s", file_path, exc)
            return None


def _synthetic_data(
    instrument: str,
    *,
    n_points: int | None = None,
    days: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, physically plausible I-ALiRT-like data."""

    spec = _validate_instrument(instrument)
    if n_points is None:
        cadence = max(spec.cadence_seconds, 300)
        n_points = max(12, int(days * 24 * 3600 / cadence))

    rng = np.random.default_rng(seed + sum(ord(char) for char in spec.instrument))
    freq = f"{max(spec.cadence_seconds, 300)}s"
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
    else:
        h_flux = rng.lognormal(mean=2.0, sigma=0.25, size=n_points)
        he_flux = rng.lognormal(mean=0.9, sigma=0.25, size=n_points)
        heavy = rng.lognormal(mean=0.25, sigma=0.3, size=n_points)
        if n_points > 80:
            event = slice(n_points // 2, min(n_points, n_points // 2 + max(6, n_points // 30)))
            h_flux[event] *= 8
            he_flux[event] *= 5
            heavy[event] *= 4
        data = {"h_flux": h_flux, "he_flux": he_flux, "heavy_ion_flux": heavy}

    frame = pd.DataFrame(data, index=index)
    frame.attrs["source"] = "synthetic-fallback"
    frame.attrs["instrument"] = spec.instrument
    return frame.astype("float64")


def fetch_range(
    instrument: str,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    max_files: int = 4,
    parallel: bool = True,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch and normalize an I-ALiRT date range.

    If no public files are available or CDF parsing is not possible in the
    current environment, a deterministic fallback frame is returned by default.
    Set ``fallback=False`` to require live data.
    """

    spec = _validate_instrument(instrument)
    files = list_available(
        spec.instrument,
        start_date=start_date,
        end_date=end_date,
        api_url=api_url,
    )[-max_files:]

    frames: list[pd.DataFrame] = []
    if files:
        if parallel and len(files) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
                futures = [
                    executor.submit(_fetch_one_file, item, spec.instrument, api_url)
                    for item in files
                ]
                for future in as_completed(futures):
                    frame = future.result()
                    if frame is not None and not frame.empty:
                        frames.append(frame)
        else:
            for item in files:
                frame = _fetch_one_file(item, spec.instrument, api_url)
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.attrs["source"] = "imap-sdc"
        combined.attrs["instrument"] = spec.instrument
        return combined

    if not fallback:
        raise RuntimeError(f"No usable public IMAP files found for {spec.instrument}")

    log.info("Using synthetic fallback data for %s", spec.instrument)
    return _synthetic_data(spec.instrument, days=7)


def fetch_latest(
    instrument: str,
    *,
    days: int = 7,
    max_files: int = 4,
    api_url: str = DEFAULT_API_URL,
    fallback: bool = True,
) -> pd.DataFrame:
    """Fetch the most recent I-ALiRT data available for an instrument."""

    end = datetime.now(tz=UTC).date()
    start = end - timedelta(days=days)
    frame = fetch_range(
        instrument,
        start_date=start,
        end_date=end,
        max_files=max_files,
        api_url=api_url,
        fallback=fallback,
    )
    if frame.attrs.get("source") == "synthetic-fallback":
        return _synthetic_data(instrument, days=days)
    return frame

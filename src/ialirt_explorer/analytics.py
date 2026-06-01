"""Numerical analysis routines for I-ALiRT time series."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit
except ImportError:  # pragma: no cover - pure Python fallback is tested instead

    def njit(*args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(func: Any) -> Any:
            return func

        return decorator


MAG_VECTOR_COLUMNS = ("Bx_nT", "By_nT", "Bz_nT")


@njit(cache=True)
def _rolling_zscore_array(values: np.ndarray, window: int = 60) -> np.ndarray:
    """Trailing-window z-score using only samples before the current point."""

    n_values = values.shape[0]
    output = np.zeros(n_values, dtype=np.float64)
    window = max(2, window)

    for idx in range(n_values):
        start = max(0, idx - window)
        count = idx - start
        if count < 2 or math.isnan(values[idx]):
            output[idx] = 0.0
            continue

        total = 0.0
        valid = 0
        for jdx in range(start, idx):
            if not math.isnan(values[jdx]):
                total += values[jdx]
                valid += 1
        if valid < 2:
            output[idx] = 0.0
            continue

        mean = total / valid
        variance = 0.0
        for jdx in range(start, idx):
            if not math.isnan(values[jdx]):
                diff = values[jdx] - mean
                variance += diff * diff
        std = math.sqrt(variance / valid)

        if std < 1e-12:
            output[idx] = (
                0.0 if abs(values[idx] - mean) < 1e-12 else math.copysign(1e9, values[idx] - mean)
            )
        else:
            output[idx] = (values[idx] - mean) / std

    return output


@njit(cache=True)
def _rolling_below_threshold(values: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Flag windows whose mean has stayed below a physical threshold."""

    n_values = values.shape[0]
    output = np.zeros(n_values, dtype=np.bool_)
    window = max(2, window)

    for idx in range(n_values):
        start = max(0, idx - window + 1)
        total = 0.0
        valid = 0
        for jdx in range(start, idx + 1):
            if not math.isnan(values[jdx]):
                total += values[jdx]
                valid += 1
        output[idx] = valid == window and total / valid < threshold

    return output


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    """Compute compact quality-control and summary statistics."""

    if df.empty:
        return {}

    numeric = df.select_dtypes(include=[np.number])
    if len(df.index) > 1 and isinstance(df.index, pd.DatetimeIndex):
        duration_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        cadence_seconds = float(df.index.to_series().diff().dt.total_seconds().dropna().median())
    else:
        duration_hours = 0.0
        cadence_seconds = float("nan")

    column_stats: dict[str, dict[str, float]] = {}
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        column_stats[column] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p5": float(np.percentile(finite, 5)),
            "p95": float(np.percentile(finite, 95)),
        }

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "duration_hours": float(duration_hours),
        "cadence_seconds": cadence_seconds,
        "missing_fraction": float(df.isna().sum().sum() / max(1, df.size)),
        "column_stats": column_stats,
    }


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} requires columns {columns}; missing {missing}")


def calibrate_mag(df: pd.DataFrame, *, method: str = "offset", window: int = 121) -> pd.DataFrame:
    """Remove simple MAG baseline drift and recompute vector magnitude.

    This is intentionally transparent: it is a research screening tool, not a
    replacement for mission calibration. The output preserves the input index and
    normalizes the vector columns into a consistent schema for downstream plots.
    """

    _require_columns(df, MAG_VECTOR_COLUMNS, "MAG calibration")
    calibrated = df.copy()
    x = np.arange(len(calibrated), dtype=np.float64)

    for column in MAG_VECTOR_COLUMNS:
        series = calibrated[column].astype(float)
        if method == "offset":
            baseline = series.rolling(
                window=window, center=True, min_periods=max(5, window // 10)
            ).median()
            baseline = baseline.bfill().ffill()
            calibrated[column] = series - baseline + float(series.median())
        elif method == "detrend":
            mask = np.isfinite(series.to_numpy(dtype=np.float64))
            if mask.sum() < 2:
                calibrated[column] = series
            else:
                slope, intercept = np.polyfit(
                    x[mask], series.to_numpy(dtype=np.float64)[mask], deg=1
                )
                trend = slope * x + intercept
                calibrated[column] = series - trend + float(np.nanmedian(series))
        elif method == "zscore":
            std = float(series.std(ddof=0))
            calibrated[column] = 0.0 if std == 0 else (series - float(series.mean())) / std
        else:
            raise ValueError(
                f"Unknown method {method!r}. Expected 'offset', 'detrend', or 'zscore'."
            )

    calibrated["B_total_nT"] = np.sqrt(
        calibrated["Bx_nT"] ** 2 + calibrated["By_nT"] ** 2 + calibrated["Bz_nT"] ** 2
    )
    calibrated.attrs.update(df.attrs)
    calibrated.attrs["calibration_method"] = method
    return calibrated


def detect_anomalies(
    df: pd.DataFrame,
    instrument: str,
    *,
    sigma_threshold: float = 3.0,
    window: int = 48,
) -> pd.DataFrame:
    """Flag statistically unusual and physically meaningful events."""

    if df.empty:
        return pd.DataFrame(index=df.index)

    instrument = instrument.lower()
    flags = pd.DataFrame(index=df.index)
    numeric = df.select_dtypes(include=[np.number])

    for column in numeric.columns:
        zscore = _rolling_zscore_array(numeric[column].to_numpy(dtype=np.float64), window=window)
        flags[f"{column}_zscore"] = zscore
        flags[f"{column}_spike"] = np.abs(zscore) >= sigma_threshold

    if instrument == "mag" and "Bz_nT" in df:
        flags["storm_southward_Bz"] = _rolling_below_threshold(
            df["Bz_nT"].to_numpy(dtype=np.float64),
            window=max(6, min(window, 24)),
            threshold=-5.0,
        )
        if "B_total_nT" in df:
            flags["strong_field"] = df["B_total_nT"].to_numpy(dtype=np.float64) >= 15.0
    elif instrument == "swapi":
        if "proton_speed_km_s" in df:
            flags["high_speed_stream"] = df["proton_speed_km_s"].to_numpy(dtype=np.float64) >= 650.0
        if "proton_density_cc" in df:
            flags["density_compression"] = (
                df["proton_density_cc"].to_numpy(dtype=np.float64) >= 12.0
            )
    elif instrument == "hit":
        flux_cols = [column for column in df.columns if "flux" in column]
        if flux_cols:
            baseline = df[flux_cols].median()
            flags["energetic_particle_enhancement"] = (df[flux_cols] > 4 * baseline).any(axis=1)
    elif instrument == "swe" and "heat_flux" in df:
        flags["electron_heat_flux_enhancement"] = df["heat_flux"].to_numpy(dtype=np.float64) >= 1.5

    boolean_cols = [
        column
        for column in flags.columns
        if column.endswith("_spike") or flags[column].dtype == bool
    ]
    flags["any_anomaly"] = flags[boolean_cols].any(axis=1) if boolean_cols else False
    return flags


def compute_pressures(mag_df: pd.DataFrame, swapi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute solar-wind pressure terms from MAG and SWAPI measurements."""

    _require_columns(mag_df, ("B_total_nT",), "Pressure calculation")
    _require_columns(
        swapi_df,
        ("proton_speed_km_s", "proton_density_cc", "proton_temp_K"),
        "Pressure calculation",
    )

    merged = pd.merge_asof(
        mag_df[["B_total_nT"]].sort_index(),
        swapi_df[["proton_speed_km_s", "proton_density_cc", "proton_temp_K"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta("10min"),
    ).dropna()
    if merged.empty:
        return pd.DataFrame(index=mag_df.index)

    proton_mass_kg = 1.67262192369e-27
    boltzmann = 1.380649e-23
    mu0 = 4 * np.pi * 1e-7

    density_m3 = merged["proton_density_cc"].to_numpy(dtype=np.float64) * 1e6
    speed_ms = merged["proton_speed_km_s"].to_numpy(dtype=np.float64) * 1e3
    temp_k = merged["proton_temp_K"].to_numpy(dtype=np.float64)
    b_tesla = merged["B_total_nT"].to_numpy(dtype=np.float64) * 1e-9

    p_ram = density_m3 * proton_mass_kg * speed_ms**2 * 1e9
    p_mag = (b_tesla**2 / (2 * mu0)) * 1e9
    p_thermal = density_m3 * boltzmann * temp_k * 1e9
    p_total = p_ram + p_mag + p_thermal

    result = pd.DataFrame(
        {
            "P_ram_nPa": p_ram,
            "P_mag_nPa": p_mag,
            "P_thermal_nPa": p_thermal,
            "P_total_nPa": p_total,
            "plasma_beta": np.divide(
                p_thermal, p_mag, out=np.full_like(p_thermal, np.nan), where=p_mag > 0
            ),
        },
        index=merged.index,
    )
    result.index.name = "time"
    return result

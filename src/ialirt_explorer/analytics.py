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

# Anything whose absolute value exceeds this threshold is treated as a sentinel
# (CDF FILLVAL=-1e31, ISTP convention -9.9e30, ground-system -9999 as floats
# scaled to nT, instrument saturation rails written as extreme floats, etc.).
# No I-ALiRT physical quantity comes anywhere near 1e20 in any unit we plot:
#   * MAG is bounded by mission spec to a few thousand nT.
#   * Proton density / temperature / speed are <1e6 in their units.
#   * HIT/CoDICE rates are <1e9 counts/s.
# So this threshold is safely far above any real measurement and safely far
# below every documented FILLVAL convention we care about.
_FILL_VALUE_MAGNITUDE = 1.0e20


def _strip_fill_values(values: np.ndarray) -> np.ndarray:
    """Map absurd-magnitude sentinel values to NaN before they pollute stats.

    Real I-ALiRT samples that leak a CDF FILLVAL (-1e31), an ISTP-style
    -9.9e30, or an instrument saturation rail through the parser will
    otherwise silently destroy Welford running stats and inflate
    ``np.std``/``np.percentile`` for an entire trailing window — hiding
    real anomalies. The caller pays an O(N) copy in exchange for not
    having to trust upstream cleanliness.
    """

    cleaned = np.asarray(values, dtype=np.float64).copy()
    bad = ~np.isfinite(cleaned) | (np.abs(cleaned) > _FILL_VALUE_MAGNITUDE)
    cleaned[bad] = np.nan
    return cleaned


def _datetime_index_to_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a (possibly tz-aware, possibly non-ns-precision) DatetimeIndex
    to a flat float64 seconds-since-epoch array.

    Pandas 2.x can store datetimes at us/ms/ns resolution, so a raw
    ``index.view("int64") / 1e9`` silently produces 1000x-wrong seconds
    values for non-ns frames. We force ns precision explicitly. Tz-aware
    indices are first stripped to UTC-naive so ``astype('datetime64[ns]')``
    does not warn about the lossy tz drop.
    """

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return (
        index.to_numpy().astype("datetime64[ns]").view("int64").astype(np.float64)
        / 1.0e9
    )


@njit(cache=True)
def _rolling_zscore_array(
    values: np.ndarray,
    window: int = 60,
    time_seconds: np.ndarray | None = None,
    max_gap_seconds: float = 0.0,
) -> np.ndarray:
    """Trailing-window z-score using only samples strictly before the current point.

    Implementation: Welford's online algorithm (B. P. Welford, 1962,
    "Note on a method for calculating corrected sums of squares and products",
    Technometrics 4(3): 419-420) extended with the symmetric remove-sample
    update so it can run over a sliding window in a single pass.

    The classic Welford update for adding a new sample ``x`` to a running
    ``(n, mean, M2)`` state — where ``M2`` is the running sum of squared
    deviations from the *current* mean — is:

        n      += 1
        delta   = x - mean
        mean   += delta / n
        M2     += delta * (x - mean)        # second `(x - mean)` uses the new mean

    Variance is then ``M2 / n``. This is numerically stable because we never
    form the catastrophic cancellation ``sum(x_i^2) - (sum x_i)^2 / n`` that
    the textbook two-pass formula does.

    For a sliding window we also need the inverse update — remove an old
    sample ``y`` from the state:

        n_new     = n - 1
        mean_new  = (mean * n - y) / n_new
        M2       -= (y - mean_new) * (y - mean)

    Each iteration is O(1), so the whole pass is O(N) vs the naive O(N*window).

    Time-aware gap handling
    -----------------------
    Spacecraft telemetry has dropouts — LOS during umbra passes, downlink
    scheduling, etc. A purely index-based rolling window treats the sample
    before and after a 4-hour gap as if they were ``cadence`` apart, so a
    benign baseline shift across the gap fires a spurious spike flag. When
    ``time_seconds`` is provided and ``max_gap_seconds > 0``, the Welford
    state is reset whenever the inter-sample gap exceeds the threshold. The
    sample immediately after a gap is scored against itself (n<2 → 0) and
    new statistics rebuild from there.

    Numerical caveat: repeated add/remove can let floating-point error
    accumulate in ``M2`` over very long sequences. We clamp ``M2`` at zero
    on each deletion to keep std real-valued.
    """

    n_values = values.shape[0]
    output = np.zeros(n_values, dtype=np.float64)
    window = max(2, window)

    use_time_gaps = (
        time_seconds is not None
        and time_seconds.shape[0] == n_values
        and max_gap_seconds > 0.0
    )

    # Running Welford state over the trailing window [idx - window, idx).
    mean = 0.0
    m2 = 0.0
    n = 0
    last_time = 0.0
    have_last_time = False

    # Index of the most recent state reset. The reverse update at index
    # ``idx`` removes ``values[idx - window]``, but that's only valid if
    # that sample was ever *added* to the current Welford state. After a
    # gap reset, samples from before the reset were dropped wholesale —
    # trying to "remove" them again would underflow ``n`` and the state
    # never recovers. Tracking this index gates the reverse update so
    # post-reset Welford behaves correctly.
    reset_at_index = 0

    for idx in range(n_values):
        cur = values[idx]

        # Reset state if the inter-sample gap exceeds the configured budget.
        # The post-gap sample has no relevant pre-gap history to score against.
        if use_time_gaps and have_last_time:
            gap = time_seconds[idx] - last_time
            if gap > max_gap_seconds or gap < 0.0:
                n = 0
                mean = 0.0
                m2 = 0.0
                reset_at_index = idx

        # Score the current sample against the state of samples BEFORE it,
        # matching the original "strictly trailing" contract.
        if n < 2 or math.isnan(cur):
            output[idx] = 0.0
        else:
            std = math.sqrt(m2 / n)
            if std < 1e-12:
                output[idx] = (
                    0.0
                    if abs(cur - mean) < 1e-12
                    else math.copysign(1e9, cur - mean)
                )
            else:
                output[idx] = (cur - mean) / std

        # Welford forward update: incorporate the current sample into the
        # window so the *next* iteration sees it.
        if not math.isnan(cur):
            n += 1
            delta = cur - mean
            mean += delta / n
            m2 += delta * (cur - mean)

        # Welford reverse update: drop the sample that has just fallen off
        # the trailing edge of the window. Guarded against post-reset
        # corruption: we only remove a sample if it was added on or after
        # the last reset (otherwise it was never in the current state).
        leaving_idx = idx - window
        if leaving_idx >= reset_at_index and leaving_idx >= 0:
            leaving = values[leaving_idx]
            if not math.isnan(leaving):
                if n <= 1:
                    n = 0
                    mean = 0.0
                    m2 = 0.0
                else:
                    n_new = n - 1
                    mean_new = (mean * n - leaving) / n_new
                    m2 -= (leaving - mean_new) * (leaving - mean)
                    if m2 < 0.0:
                        m2 = 0.0
                    mean = mean_new
                    n = n_new

        if use_time_gaps:
            last_time = time_seconds[idx]
            have_last_time = True

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
    """Compute compact quality-control and summary statistics.

    Defensive against three classes of dirty real-world telemetry:

    * **Sentinel / FILLVAL leakage.** Per-column stats are computed only
      over values that pass :func:`_strip_fill_values`, so a single
      ``-1e31`` does not turn ``max`` into ``1e31``.
    * **Non-monotonic index.** Spacecraft packets can arrive out of order;
      ``duration_hours`` and ``cadence_seconds`` are derived from the
      timestamp extremes and the *absolute* inter-sample diff, so they
      never come out negative.
    * **Duplicate timestamps.** Zero-second diffs are filtered out before
      taking the cadence median, so a few retransmits don't claim a
      cadence of 0 s.
    """

    if df.empty:
        return {}

    numeric = df.select_dtypes(include=[np.number])
    if len(df.index) > 1 and isinstance(df.index, pd.DatetimeIndex):
        index_seconds = _datetime_index_to_seconds(df.index)
        duration_hours = (
            float(index_seconds.max() - index_seconds.min()) / 3600.0
        )
        diffs = np.abs(np.diff(index_seconds))
        positive = diffs[diffs > 0]
        cadence_seconds = float(np.median(positive)) if positive.size else float("nan")
    else:
        duration_hours = 0.0
        cadence_seconds = float("nan")

    column_stats: dict[str, dict[str, float]] = {}
    for column in numeric.columns:
        finite = _strip_fill_values(numeric[column].to_numpy(dtype=np.float64))
        finite = finite[np.isfinite(finite)]
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

    # Count any FILLVAL-magnitude rail as "missing" for the QC fraction — a
    # row whose vector components are all -1e31 isn't a real measurement.
    cleaned_numeric = numeric.where(numeric.abs() < _FILL_VALUE_MAGNITUDE)
    missing_fraction = float(
        cleaned_numeric.isna().sum().sum() / max(1, cleaned_numeric.size)
    )

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "duration_hours": float(duration_hours),
        "cadence_seconds": cadence_seconds,
        "missing_fraction": missing_fraction,
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

    # Refuse to silently emit an all-NaN frame for a vector component that
    # carries no real samples. The caller almost certainly wants to see this
    # case explicitly (instrument off, bad parser, all FILLVAL upstream)
    # rather than discover a flat-zero magnitude trace downstream.
    for column in MAG_VECTOR_COLUMNS:
        cleaned = _strip_fill_values(df[column].to_numpy(dtype=np.float64))
        if not np.any(np.isfinite(cleaned)):
            raise ValueError(
                f"MAG calibration requires real samples in {column!r}; "
                f"received an all-NaN/fill-value column."
            )

    for column in MAG_VECTOR_COLUMNS:
        # Substitute FILLVAL-magnitude rails with NaN so they cannot poison
        # the rolling median (offset), the polyfit baseline (detrend), or the
        # mean/std (zscore) of an otherwise clean component.
        cleaned_values = _strip_fill_values(df[column].to_numpy(dtype=np.float64))
        series = pd.Series(cleaned_values, index=df.index, name=column)
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

    # Build a seconds-since-epoch array so the Welford pass can detect
    # large telemetry gaps (LOS, downlink scheduling) and reset its running
    # mean/M2 instead of comparing post-gap samples against stale pre-gap
    # statistics. The default budget — 6x the median cadence, floored at
    # 5 minutes — is generous enough not to fire on routine packet jitter
    # and tight enough to catch real LOS events.
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        time_seconds = _datetime_index_to_seconds(df.index)
        positive_diffs = np.diff(time_seconds)
        positive_diffs = positive_diffs[positive_diffs > 0]
        if positive_diffs.size:
            cadence = float(np.median(positive_diffs))
            max_gap_seconds = max(6.0 * cadence, 300.0)
        else:
            max_gap_seconds = 0.0
    else:
        time_seconds = np.empty(0, dtype=np.float64)
        max_gap_seconds = 0.0

    for column in numeric.columns:
        cleaned = _strip_fill_values(numeric[column].to_numpy(dtype=np.float64))
        zscore = _rolling_zscore_array(
            cleaned,
            window=window,
            time_seconds=time_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        flags[f"{column}_zscore"] = zscore
        flags[f"{column}_spike"] = np.abs(zscore) >= sigma_threshold

    # Each threshold check below reads from ``clean_df`` rather than ``df``
    # so a FILLVAL of either sign (+1e31, -9.9e30) cannot trip a physical
    # threshold flag (e.g. ``high_speed_stream`` would otherwise fire on a
    # FILLVAL of +1e31 dressed as a proton speed).
    clean_df = df.where(numeric.abs() < _FILL_VALUE_MAGNITUDE)

    if instrument == "mag" and "Bz_nT" in df:
        flags["storm_southward_Bz"] = _rolling_below_threshold(
            _strip_fill_values(df["Bz_nT"].to_numpy(dtype=np.float64)),
            window=max(6, min(window, 24)),
            threshold=-5.0,
        )
        if "B_total_nT" in df:
            flags["strong_field"] = (
                clean_df["B_total_nT"].to_numpy(dtype=np.float64) >= 15.0
            )
    elif instrument == "swapi":
        if "proton_speed_km_s" in df:
            flags["high_speed_stream"] = (
                clean_df["proton_speed_km_s"].to_numpy(dtype=np.float64) >= 650.0
            )
        if "proton_density_cc" in df:
            flags["density_compression"] = (
                clean_df["proton_density_cc"].to_numpy(dtype=np.float64) >= 12.0
            )
    elif instrument == "hit":
        rate_cols = [column for column in df.columns if column.endswith("_en")]
        if rate_cols:
            baseline = clean_df[rate_cols].median().replace(0, 1.0)
            flags["energetic_particle_enhancement"] = (
                clean_df[rate_cols] > 4 * baseline
            ).any(axis=1)
    elif instrument == "swe":
        if "electron_counts_max" in df:
            flags["electron_burst"] = (
                clean_df["electron_counts_max"].to_numpy(dtype=np.float64) >= 200.0
            )
        if "counterstreaming_flag" in df:
            flags["counterstreaming_electrons"] = (
                clean_df["counterstreaming_flag"].fillna(0).to_numpy(dtype=np.float64)
                >= 1.0
            )
    elif instrument in {"codice_lo", "codice_hi"}:
        numeric_cols = [
            column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        if numeric_cols:
            valid = clean_df[numeric_cols].dropna(how="all")
            if not valid.empty:
                baseline = valid.median(skipna=True).replace(0, 1.0)
                flags["composition_excursion"] = (
                    (clean_df[numeric_cols] > 3 * baseline).any(axis=1).fillna(False)
                )

    boolean_cols = [
        column
        for column in flags.columns
        if column.endswith("_spike") or flags[column].dtype == bool
    ]
    flags["any_anomaly"] = flags[boolean_cols].any(axis=1) if boolean_cols else False
    return flags


CALIBRATION_METHODS: tuple[str, ...] = ("offset", "detrend", "zscore")


def calibration_quality(
    raw: pd.DataFrame, calibrated: pd.DataFrame, *, columns: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Quantify what a MAG calibration step actually did to the data.

    Researchers cannot trust calibration they cannot inspect. This returns
    per-component metrics describing the magnitude of the baseline that was
    removed, the residual drift after calibration, the noise floor, and how
    correlated the calibrated trace is with the raw input.

    Notes
    -----
    - ``baseline_amplitude_nT`` reports the peak-to-peak amplitude of the
      signal that calibration subtracted. A large value with a slow trend
      indicates an instrumental offset; a small value indicates the raw data
      was already stable.
    - ``residual_drift_per_hour_nT`` is the slope of the calibrated series
      against time. Close to zero means the baseline really has been removed.
    - ``raw_calibrated_correlation`` close to 1.0 means the high-frequency
      structure is preserved (calibration only removed low-frequency drift).
    """

    columns = columns or MAG_VECTOR_COLUMNS
    metrics: dict[str, dict[str, float]] = {}

    for column in columns:
        if column not in raw or column not in calibrated:
            continue
        raw_series = raw[column].astype(float)
        cal_series = calibrated[column].astype(float)
        baseline = (raw_series - cal_series).dropna()
        if baseline.empty:
            continue

        if isinstance(cal_series.index, pd.DatetimeIndex) and len(cal_series) > 1:
            elapsed_hours = np.asarray(
                (cal_series.index - cal_series.index[0]).total_seconds() / 3600,
                dtype=np.float64,
            )
            values = cal_series.to_numpy(dtype=np.float64)
            mask = np.isfinite(values)
            if mask.sum() >= 2:
                slope, _ = np.polyfit(elapsed_hours[mask], values[mask], deg=1)
            else:
                slope = float("nan")
        else:
            slope = float("nan")

        diff = cal_series.diff().abs().dropna()
        noise_floor = float(diff.quantile(0.5)) if not diff.empty else float("nan")

        corr = float(raw_series.corr(cal_series))

        metrics[column] = {
            "baseline_amplitude_nT": float(baseline.max() - baseline.min()),
            "baseline_mean_offset_nT": float(baseline.mean()),
            "residual_drift_per_hour_nT": float(slope),
            "noise_floor_nT": noise_floor,
            "raw_calibrated_correlation": corr,
            "std_before_nT": float(raw_series.std(ddof=0)),
            "std_after_nT": float(cal_series.std(ddof=0)),
        }

    if not metrics:
        return {}

    total_baseline = float(
        np.mean([m["baseline_amplitude_nT"] for m in metrics.values()])
    )
    total_drift = float(
        np.mean([abs(m["residual_drift_per_hour_nT"]) for m in metrics.values()])
    )
    return {
        "per_component": metrics,
        "baseline_amplitude_nT": total_baseline,
        "residual_drift_per_hour_nT": total_drift,
        "method": calibrated.attrs.get("calibration_method", "unknown"),
    }


def compare_calibration_methods(
    df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = CALIBRATION_METHODS,
    window: int = 121,
) -> dict[str, dict[str, Any]]:
    """Run several calibration methods on the same MAG frame and return metrics.

    The returned dict is keyed by method name. Each entry contains the
    quality metrics from :func:`calibration_quality` along with a compact
    per-component summary so a UI can render side-by-side comparisons
    without recomputing.
    """

    _require_columns(df, MAG_VECTOR_COLUMNS, "Calibration comparison")
    results: dict[str, dict[str, Any]] = {}
    for method in methods:
        calibrated = calibrate_mag(df, method=method, window=window)
        quality = calibration_quality(df, calibrated)
        results[method] = {
            "quality": quality,
            "score": _calibration_score(quality),
        }
    return results


def _calibration_score(quality: dict[str, Any]) -> float:
    """Heuristic combined score in [0, 1].

    Higher is better. Penalizes large residual drift and rewards preservation
    of high-frequency structure (raw <-> calibrated correlation).
    """

    if not quality:
        return 0.0
    drift = quality.get("residual_drift_per_hour_nT", float("nan"))
    drift_score = math.exp(-abs(drift) / 0.5) if math.isfinite(drift) else 0.0

    components = quality.get("per_component", {})
    if components:
        correlations = [m["raw_calibrated_correlation"] for m in components.values()]
        corr_score = float(np.nanmean(correlations))
    else:
        corr_score = 0.0

    return float(np.clip(0.5 * drift_score + 0.5 * corr_score, 0.0, 1.0))


def suggest_calibration_method(
    df: pd.DataFrame, *, window: int = 121
) -> dict[str, Any]:
    """Suggest the calibration method best suited to a MAG frame.

    The heuristic inspects each MAG component for:

    - a slow monotonic trend (-> ``detrend``)
    - a wandering DC offset relative to the noise floor (-> ``offset``)
    - already-stable data with no significant baseline (-> ``zscore`` to
      simply standardize for downstream comparison)
    """

    _require_columns(df, MAG_VECTOR_COLUMNS, "Calibration suggestion")
    diagnostics: dict[str, dict[str, float]] = {}
    method_votes: dict[str, int] = {method: 0 for method in CALIBRATION_METHODS}
    any_strong_trend = False
    any_strong_offset = False

    for column in MAG_VECTOR_COLUMNS:
        series = df[column].astype(float).dropna()
        if len(series) < 8:
            continue
        baseline = series.rolling(window=window, center=True, min_periods=5).median()
        baseline = baseline.bfill().ffill()
        baseline_amplitude = float(baseline.max() - baseline.min())
        noise_floor = float(series.diff().abs().median())

        if isinstance(series.index, pd.DatetimeIndex):
            elapsed_hours = (
                series.index - series.index[0]
            ).total_seconds().to_numpy() / 3600
        else:
            elapsed_hours = np.arange(len(series), dtype=np.float64)
        slope, _ = np.polyfit(elapsed_hours, series.to_numpy(dtype=np.float64), deg=1)
        trend_strength = abs(slope) * (elapsed_hours[-1] - elapsed_hours[0])

        diagnostics[column] = {
            "baseline_amplitude_nT": baseline_amplitude,
            "noise_floor_nT": noise_floor,
            "linear_trend_total_nT": float(trend_strength),
        }

        if trend_strength > 3 * max(noise_floor, 1e-6) and trend_strength > 2.0:
            method_votes["detrend"] += 1
            any_strong_trend = True
        elif baseline_amplitude > 5 * max(noise_floor, 1e-6) and baseline_amplitude > 1.5:
            method_votes["offset"] += 1
            any_strong_offset = True
        else:
            method_votes["zscore"] += 1

    # Priority: a sustained linear trend on *any* component requires a detrend
    # step before the other components can be compared. A wandering DC offset on
    # any component justifies an offset removal. Otherwise standardize.
    if any_strong_trend:
        chosen = "detrend"
    elif any_strong_offset:
        chosen = "offset"
    elif any(method_votes.values()):
        chosen = max(method_votes, key=method_votes.get)
    else:
        chosen = "offset"

    return {
        "recommendation": chosen,
        "votes": method_votes,
        "diagnostics": diagnostics,
        "rationale": _explain_recommendation(chosen, diagnostics),
    }


def _explain_recommendation(method: str, diagnostics: dict[str, dict[str, float]]) -> str:
    if method == "detrend":
        return (
            "At least one component shows a sustained linear trend several times "
            "larger than the noise floor; subtracting a linear fit will preserve "
            "the AC signal while removing the drift."
        )
    if method == "offset":
        return (
            "Components show a wandering DC baseline larger than the noise floor; "
            "a rolling-median offset is the conservative choice and leaves the "
            "high-frequency structure intact."
        )
    return (
        "Baseline drift is comparable to the noise floor. Z-score standardization "
        "is appropriate for cross-instrument comparison; no real calibration is "
        "required."
    )


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

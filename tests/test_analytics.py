from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ialirt_explorer.analytics import (
    CALIBRATION_METHODS,
    _rolling_below_threshold,
    _rolling_zscore_array,
    _strip_fill_values,
    analyze,
    calibrate_mag,
    calibration_quality,
    compare_calibration_methods,
    compute_pressures,
    detect_anomalies,
    suggest_calibration_method,
)
from ialirt_explorer.ingestion import _synthetic_data


@pytest.fixture
def mag_df() -> pd.DataFrame:
    return _synthetic_data("mag", n_points=400)


@pytest.fixture
def swapi_df() -> pd.DataFrame:
    return _synthetic_data("swapi", n_points=400)


def test_analyze_reports_core_quality_metrics(mag_df: pd.DataFrame) -> None:
    result = analyze(mag_df)

    assert result["n_rows"] == len(mag_df)
    assert result["duration_hours"] > 0
    assert result["cadence_seconds"] > 0
    assert 0 <= result["missing_fraction"] <= 1
    assert "Bz_nT" in result["column_stats"]


def test_analyze_empty_dataframe_returns_empty_dict() -> None:
    assert analyze(pd.DataFrame()) == {}


@pytest.mark.parametrize("method", ["offset", "detrend", "zscore"])
def test_calibrate_mag_preserves_shape_and_recomputes_magnitude(
    mag_df: pd.DataFrame, method: str
) -> None:
    calibrated = calibrate_mag(mag_df, method=method)
    expected = np.sqrt(
        calibrated["Bx_nT"] ** 2 + calibrated["By_nT"] ** 2 + calibrated["Bz_nT"] ** 2
    )

    assert calibrated.shape == mag_df.shape
    pd.testing.assert_index_equal(calibrated.index, mag_df.index)
    np.testing.assert_allclose(calibrated["B_total_nT"], expected)


def test_calibrate_mag_rejects_missing_vector_columns() -> None:
    with pytest.raises(ValueError, match="MAG calibration"):
        calibrate_mag(pd.DataFrame({"Bz_nT": [1.0, 2.0, 3.0]}))


def test_detrend_reduces_injected_linear_drift(mag_df: pd.DataFrame) -> None:
    trended = mag_df.copy()
    trended["Bz_nT"] = trended["Bz_nT"] + np.linspace(0, 25, len(trended))

    calibrated = calibrate_mag(trended, method="detrend")

    assert calibrated["Bz_nT"].std() < trended["Bz_nT"].std()


def test_detect_anomalies_finds_known_spike(mag_df: pd.DataFrame) -> None:
    spiky = mag_df.copy()
    spiky.iloc[220, spiky.columns.get_loc("Bz_nT")] = 100.0

    flagged = detect_anomalies(spiky, "mag", sigma_threshold=3.0)

    assert flagged["any_anomaly"].any()
    assert flagged.loc[spiky.index[220], "Bz_nT_spike"]


def test_detect_anomalies_threshold_is_monotonic(mag_df: pd.DataFrame) -> None:
    low = detect_anomalies(mag_df, "mag", sigma_threshold=2.0)["any_anomaly"].sum()
    high = detect_anomalies(mag_df, "mag", sigma_threshold=5.0)["any_anomaly"].sum()

    assert high <= low


@pytest.mark.parametrize("instrument", ["mag", "swapi", "hit", "swe"])
def test_detect_anomalies_supports_all_instruments(instrument: str) -> None:
    frame = _synthetic_data(instrument, n_points=200)
    flagged = detect_anomalies(frame, instrument)

    assert "any_anomaly" in flagged.columns
    assert flagged["any_anomaly"].dtype == bool
    pd.testing.assert_index_equal(flagged.index, frame.index)


def test_compute_pressures_returns_positive_components(
    mag_df: pd.DataFrame, swapi_df: pd.DataFrame
) -> None:
    pressure = compute_pressures(mag_df, swapi_df)

    assert {"P_ram_nPa", "P_mag_nPa", "P_thermal_nPa", "P_total_nPa", "plasma_beta"}.issubset(
        pressure.columns
    )
    assert (pressure["P_ram_nPa"] > 0).all()
    assert (pressure["P_mag_nPa"] > 0).all()
    np.testing.assert_allclose(
        pressure["P_total_nPa"],
        pressure["P_ram_nPa"] + pressure["P_mag_nPa"] + pressure["P_thermal_nPa"],
    )


def test_compute_pressures_rejects_missing_columns(swapi_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Pressure calculation"):
        compute_pressures(pd.DataFrame({"Bx_nT": [1.0]}), swapi_df)


def test_rolling_zscore_handles_constant_arrays() -> None:
    result = _rolling_zscore_array(np.ones(32, dtype=np.float64), window=8)

    np.testing.assert_array_equal(result, np.zeros(32))


def test_rolling_zscore_detects_large_outlier() -> None:
    values = np.zeros(80, dtype=np.float64)
    values[40] = 100.0

    result = _rolling_zscore_array(values, window=20)

    assert abs(result[40]) > 5


def test_rolling_zscore_welford_matches_naive_reference() -> None:
    """Lock in Welford's add/remove math against a brute-force reference.

    The brute-force implementation reproduces what the function did before we
    switched to Welford: per-index, two full passes over the trailing window.
    The Welford version must agree to within floating-point tolerance.
    """

    rng = np.random.default_rng(seed=42)
    values = rng.normal(loc=2.5, scale=1.7, size=500).astype(np.float64)
    values[100] = np.nan
    values[200:205] = np.nan

    window = 24

    def naive(values: np.ndarray, window: int) -> np.ndarray:
        out = np.zeros_like(values)
        for idx in range(len(values)):
            start = max(0, idx - window)
            chunk = values[start:idx]
            chunk = chunk[~np.isnan(chunk)]
            if chunk.size < 2 or np.isnan(values[idx]):
                continue
            mean = chunk.mean()
            std = chunk.std()
            if std < 1e-12:
                continue
            out[idx] = (values[idx] - mean) / std
        return out

    fast = _rolling_zscore_array(values, window=window)
    expected = naive(values, window=window)

    np.testing.assert_allclose(fast, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_quality_returns_expected_fields(
    mag_df: pd.DataFrame, method: str
) -> None:
    calibrated = calibrate_mag(mag_df, method=method)
    quality = calibration_quality(mag_df, calibrated)

    assert "per_component" in quality
    for column in ("Bx_nT", "By_nT", "Bz_nT"):
        component = quality["per_component"][column]
        for key in (
            "baseline_amplitude_nT",
            "residual_drift_per_hour_nT",
            "raw_calibrated_correlation",
            "std_before_nT",
            "std_after_nT",
        ):
            assert key in component
    assert quality["method"] == method


def test_compare_calibration_methods_runs_all_known_methods(
    mag_df: pd.DataFrame,
) -> None:
    comparison = compare_calibration_methods(mag_df)

    assert set(comparison) == set(CALIBRATION_METHODS)
    for entry in comparison.values():
        assert 0.0 <= entry["score"] <= 1.0
        assert entry["quality"]["per_component"]


def test_suggest_calibration_recommends_detrend_for_strong_linear_trend(
    mag_df: pd.DataFrame,
) -> None:
    trended = mag_df.copy()
    trended["Bz_nT"] = trended["Bz_nT"] + np.linspace(0, 200, len(trended))

    suggestion = suggest_calibration_method(trended)

    assert suggestion["recommendation"] == "detrend"
    assert "trend" in suggestion["rationale"].lower()


def test_suggest_calibration_recommends_offset_for_dc_baseline(
    mag_df: pd.DataFrame,
) -> None:
    offset = mag_df.copy()
    bump = np.concatenate(
        [
            np.zeros(len(offset) // 2),
            np.ones(len(offset) - len(offset) // 2) * 25,
        ]
    )
    offset["Bz_nT"] = offset["Bz_nT"] + bump

    suggestion = suggest_calibration_method(offset)

    assert suggestion["recommendation"] in {"offset", "detrend"}


def test_rolling_below_threshold_requires_sustained_interval() -> None:
    values = np.full(100, 5.0, dtype=np.float64)
    values[30:60] = -10.0

    sustained = _rolling_below_threshold(values, window=8, threshold=0.0)
    assert sustained[45:60].any()

    values = np.full(100, 5.0, dtype=np.float64)
    values[50] = -10.0
    isolated = _rolling_below_threshold(values, window=8, threshold=0.0)
    assert not isolated[50:60].any()


# ---------------------------------------------------------------------------
# Physical edge-case regression tests.
#
# Every test below corresponds to a real failure mode that the AI-suggested
# happy-path tests above missed and that probing against current behavior
# revealed as a live bug. They are intentionally written as user stories
# ("instrument did X; analytics must do Y"), not as 'doesn't crash' smoke.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill_value", [-1.0e31, -9.9e30, -9999.0e25, 1.0e31])
def test_strip_fill_values_replaces_cdf_sentinels_with_nan(
    fill_value: float,
) -> None:
    """CDF FILLVAL=-1e31, ISTP -9.9e30, and saturation rails of either sign
    must all become NaN; real measurements (within the physical envelope)
    must pass through unchanged.
    """

    values = np.array([1.5, fill_value, -3.2, 47.0, fill_value, 0.0])
    cleaned = _strip_fill_values(values)

    np.testing.assert_array_equal(np.isnan(cleaned), [False, True, False, False, True, False])
    np.testing.assert_array_equal(
        cleaned[~np.isnan(cleaned)], np.array([1.5, -3.2, 47.0, 0.0])
    )


def test_detect_anomalies_does_not_let_a_single_fillval_hide_a_real_spike() -> None:
    """The silent-killer case. A single -1e31 fill value sitting in the
    trailing window must NOT inflate the running std and mask a downstream
    legitimate 5-sigma spike. Before _strip_fill_values, this test failed:
    the spike scored ~0.2 sigma instead of >5.
    """

    rng = np.random.default_rng(seed=0)
    bx = rng.normal(0.0, 1.0, size=120)
    bx[10] = -1.0e31  # CDF FILLVAL leaks past the parser
    bx[60] = 6.0  # legitimate 6-sigma spike well after the fill

    times = pd.date_range("2026-06-05", periods=120, freq="4s")
    frame = pd.DataFrame(
        {"Bx_nT": bx, "By_nT": 0.0, "Bz_nT": 0.0, "B_total_nT": np.abs(bx)},
        index=times,
    )

    flags = detect_anomalies(frame, "mag", window=20, sigma_threshold=3.0)

    assert flags.iloc[60]["Bx_nT_spike"], (
        "real 6-sigma spike must still be flagged after a fill value passed through"
    )
    assert abs(flags.iloc[60]["Bx_nT_zscore"]) > 4.0


def test_detect_anomalies_does_not_fire_on_baseline_shift_across_clock_gap() -> None:
    """Spacecraft LOS / downlink scheduling can produce hour-long gaps. The
    sample immediately after a long gap MUST NOT be scored against the
    pre-gap mean/std; otherwise a ~1 nT seasonal drift across the gap
    masquerades as a high-sigma spike.

    Pre-fix: 2 false-positive Bx_nT_spike flags fired in the first three
    samples after a 4-hour gap. Post-fix: 0.
    """

    times_before = pd.date_range("2026-06-05 00:00", periods=30, freq="4s")
    times_after = pd.date_range("2026-06-05 04:00", periods=30, freq="4s")
    times = times_before.union(times_after)
    vals = np.concatenate([np.full(30, 10.0), np.full(30, 11.0)])

    frame = pd.DataFrame(
        {"Bx_nT": vals, "By_nT": 0.0, "Bz_nT": 0.0, "B_total_nT": np.abs(vals)},
        index=times,
    )
    flags = detect_anomalies(frame, "mag", window=20, sigma_threshold=3.0)

    spikes_after_gap = flags.iloc[30:35]["Bx_nT_spike"].sum()
    assert spikes_after_gap == 0, (
        f"baseline shift across a 4hr gap fired {spikes_after_gap} false-positive spikes"
    )


def test_detect_anomalies_still_fires_on_real_spike_after_clock_gap() -> None:
    """The dual of the previous test: the gap-reset must NOT swallow a real
    spike that occurs after enough post-gap samples have been seen.
    """

    times_before = pd.date_range("2026-06-05 00:00", periods=30, freq="4s")
    times_after = pd.date_range("2026-06-05 04:00", periods=60, freq="4s")
    times = times_before.union(times_after)

    rng = np.random.default_rng(seed=1)
    pre = rng.normal(0.0, 1.0, size=30)
    post = rng.normal(0.0, 1.0, size=60)
    post[40] = 20.0
    vals = np.concatenate([pre, post])

    frame = pd.DataFrame(
        {"Bx_nT": vals, "By_nT": 0.0, "Bz_nT": 0.0, "B_total_nT": np.abs(vals)},
        index=times,
    )
    flags = detect_anomalies(frame, "mag", window=20, sigma_threshold=3.0)

    assert flags.iloc[30 + 40]["Bx_nT_spike"]


def test_analyze_returns_non_negative_duration_for_out_of_order_packets() -> None:
    """Real telemetry can deliver packets out of arrival order. analyze()
    must report the *physical* duration covered (always non-negative)
    and a positive cadence, not negative numbers from a raw index.diff().
    """

    forward_times = pd.date_range("2026-06-05", periods=10, freq="4s")
    reversed_times = pd.DatetimeIndex(forward_times[::-1])
    frame = pd.DataFrame(
        {
            "Bx_nT": np.arange(10.0),
            "By_nT": 0.0,
            "Bz_nT": 0.0,
            "B_total_nT": np.arange(10.0),
        },
        index=reversed_times,
    )

    summary = analyze(frame)

    assert summary["duration_hours"] > 0
    assert summary["cadence_seconds"] == 4.0


def test_analyze_ignores_duplicate_timestamps_when_estimating_cadence() -> None:
    """A retransmitted packet (same timestamp as an earlier sample) must
    not drag the median cadence to 0 s.
    """

    base = pd.date_range("2026-06-05", periods=10, freq="4s")
    duplicates = pd.DatetimeIndex([base[5]] * 3)
    times = base.append(duplicates).sort_values()
    frame = pd.DataFrame(
        {
            "Bx_nT": np.arange(len(times), dtype=float),
            "By_nT": 0.0,
            "Bz_nT": 0.0,
            "B_total_nT": np.arange(len(times), dtype=float),
        },
        index=times,
    )

    summary = analyze(frame)

    assert summary["cadence_seconds"] == pytest.approx(4.0)


def test_analyze_treats_fillval_as_missing_not_as_extreme_measurement() -> None:
    """A FILLVAL sample must NOT show up as the column min/max in stats,
    and it must count toward missing_fraction. A scientist reading
    'min=-1e31 nT' would (rightly) panic.
    """

    times = pd.date_range("2026-06-05", periods=4, freq="4s")
    frame = pd.DataFrame(
        {"Bx_nT": [10.0, -1.0e31, 12.0, 11.0]}, index=times
    )

    summary = analyze(frame)
    stats = summary["column_stats"]["Bx_nT"]

    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(12.0)
    assert summary["missing_fraction"] == pytest.approx(0.25)


def test_calibrate_mag_refuses_all_nan_vector_component() -> None:
    """Silently emitting an all-NaN B_total_nT for a frame whose By_nT is
    entirely missing is the kind of bug that propagates undetected into
    plots and downstream products. Force an explicit failure instead.
    """

    times = pd.date_range("2026-06-05", periods=5, freq="4s")
    df = pd.DataFrame(
        {
            "Bx_nT": [1.0, 2.0, 3.0, 4.0, 5.0],
            "By_nT": [np.nan] * 5,
            "Bz_nT": [0.5, 0.6, 0.5, 0.6, 0.5],
            "B_total_nT": [1.0] * 5,
        },
        index=times,
    )

    with pytest.raises(ValueError, match=r"By_nT"):
        calibrate_mag(df, method="offset")


def test_calibrate_mag_treats_fillval_as_missing_in_baseline_computation() -> None:
    """If a single sample is a FILLVAL, the offset-method rolling median
    must not be dragged off scale; the calibrated frame must still produce
    a sensible (finite, near-the-real-baseline) magnitude on the clean
    samples.
    """

    times = pd.date_range("2026-06-05", periods=200, freq="4s")
    bx = np.full(200, 5.0)
    by = np.full(200, 5.0)
    bz = np.full(200, 0.0)
    bx[100] = -1.0e31  # one bad sample mid-frame
    df = pd.DataFrame(
        {
            "Bx_nT": bx,
            "By_nT": by,
            "Bz_nT": bz,
            "B_total_nT": np.sqrt(bx**2 + by**2 + bz**2),
        },
        index=times,
    )

    calibrated = calibrate_mag(df, method="offset")

    # The bad sample becomes NaN; everything else is finite and near the
    # true magnitude of sqrt(5^2 + 5^2) = 7.07 nT.
    finite_total = calibrated["B_total_nT"].to_numpy()
    finite_total = finite_total[np.isfinite(finite_total)]
    assert finite_total.size >= 195
    np.testing.assert_allclose(finite_total, np.sqrt(50.0), atol=1.0)


def test_detect_anomalies_threshold_checks_ignore_fillval_of_either_sign() -> None:
    """Threshold-based flags (e.g. SWAPI ``high_speed_stream`` at 650 km/s)
    must not fire on a +1e31 FILLVAL even though it is technically >= 650.
    """

    times = pd.date_range("2026-06-05", periods=30, freq="30s")
    speed = np.full(30, 400.0)
    speed[10] = 1.0e31
    speed[20] = -1.0e31
    df = pd.DataFrame(
        {
            "proton_speed_km_s": speed,
            "proton_density_cc": 5.0,
            "proton_temp_K": 1.0e5,
        },
        index=times,
    )

    flags = detect_anomalies(df, "swapi", window=8, sigma_threshold=3.0)

    assert not flags["high_speed_stream"].any()

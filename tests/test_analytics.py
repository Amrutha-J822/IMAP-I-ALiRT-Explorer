from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ialirt_explorer.analytics import (
    CALIBRATION_METHODS,
    _rolling_below_threshold,
    _rolling_zscore_array,
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

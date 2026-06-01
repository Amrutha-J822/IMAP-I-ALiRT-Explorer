from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ialirt_explorer.ingestion import (
    DEFAULT_API_URL,
    IALIRT_INSTRUMENTS,
    _records_to_frame,
    _synthetic_data,
    _validate_instrument,
    fetch_archive,
    fetch_latest,
    fetch_range,
    fetch_space_weather,
    list_available,
)


@pytest.mark.parametrize("instrument", sorted(IALIRT_INSTRUMENTS))
def test_synthetic_data_returns_typed_time_series(instrument: str) -> None:
    frame = _synthetic_data(instrument, n_points=128)

    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None
    assert frame.index.name == "time"
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 128
    assert frame.attrs["source"] == "synthetic-fallback"


def test_invalid_instrument_raises() -> None:
    with pytest.raises(ValueError, match="Unknown instrument"):
        _synthetic_data("not-real")


def test_mag_magnitude_is_recomputed_from_components() -> None:
    frame = _synthetic_data("mag", n_points=64)
    expected = np.sqrt(frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2)

    np.testing.assert_allclose(frame["B_total_nT"], expected)


def test_solar_wind_values_are_physically_plausible() -> None:
    frame = _synthetic_data("swapi", n_points=256)

    assert frame["proton_speed_km_s"].between(250, 950).all()
    assert (frame["proton_density_cc"] > 0).all()
    assert (frame["proton_temp_K"] > 0).all()


def test_default_api_url_targets_real_ialirt_endpoint() -> None:
    assert DEFAULT_API_URL.startswith("https://ialirt.imap-mission.com")


def test_records_to_frame_normalizes_known_aliases() -> None:
    spec = _validate_instrument("mag")
    records = [
        {"time_utc": "2026-01-01T00:00:00Z", "bx": 1.0, "by": 2.0, "bz": 3.0},
        {"time_utc": "2026-01-01T00:00:01Z", "bx": 1.1, "by": 2.1, "bz": 3.1},
    ]

    frame = _records_to_frame(records, spec)

    assert list(frame.columns) == ["Bx_nT", "By_nT", "Bz_nT", "B_total_nT"]
    assert len(frame) == 2
    np.testing.assert_allclose(
        frame["B_total_nT"],
        np.sqrt(frame["Bx_nT"] ** 2 + frame["By_nT"] ** 2 + frame["Bz_nT"] ** 2),
    )


def test_fetch_space_weather_parses_mocked_rest_payload() -> None:
    payload = [
        {"time_utc": "2026-05-21T00:00:00Z", "bx": 1.0, "by": 2.0, "bz": 3.0},
        {"time_utc": "2026-05-21T00:00:01Z", "bx": 1.5, "by": 2.5, "bz": 3.5},
    ]

    with (
        patch(
            "ialirt_explorer.ingestion._query_space_weather_package",
            return_value=[],
        ),
        patch("requests.get") as mock_get,
    ):
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        frame = fetch_space_weather(
            "mag",
            time_utc_start="2026-05-21T00:00:00",
            time_utc_end="2026-05-21T00:01:00",
        )

    assert not frame.empty
    assert frame.attrs["source"] == "ialirt-sdc"
    assert frame.attrs["instrument"] == "mag"
    assert len(frame) == 2


def test_fetch_space_weather_falls_back_when_no_records() -> None:
    with (
        patch(
            "ialirt_explorer.ingestion._query_space_weather_package",
            return_value=[],
        ),
        patch(
            "ialirt_explorer.ingestion._query_space_weather_rest",
            return_value=[],
        ),
    ):
        frame = fetch_space_weather("mag")

    assert frame.attrs["source"] == "synthetic-fallback"


def test_list_available_uses_archive_query_endpoint() -> None:
    payload = [
        {
            "filename": "imap_ialirt_l1_realtime_20260101_v001.cdf",
            "version": 1,
            "date": "20260101",
        }
    ]
    with patch("requests.get") as mock_get:
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        files = list_available(since=date(2026, 1, 1))

    assert len(files) == 1
    assert files[0]["filename"].endswith(".cdf")
    args, kwargs = mock_get.call_args
    assert "ialirt-archive-query" in args[0]
    assert kwargs["params"]["since"] == "20260101"


@pytest.mark.parametrize("instrument", sorted(IALIRT_INSTRUMENTS))
def test_fetch_latest_falls_back_to_non_empty_dataframe(instrument: str) -> None:
    with (
        patch(
            "ialirt_explorer.ingestion.fetch_space_weather",
            side_effect=RuntimeError,
        ),
        patch(
            "ialirt_explorer.ingestion.fetch_archive",
            side_effect=RuntimeError,
        ),
    ):
        frame = fetch_latest(instrument, days=1)

    assert not frame.empty
    assert frame.attrs["source"] == "synthetic-fallback"


def test_fetch_range_respects_no_fallback_mode() -> None:
    with (
        patch(
            "ialirt_explorer.ingestion.fetch_space_weather",
            side_effect=RuntimeError,
        ),
        patch(
            "ialirt_explorer.ingestion.fetch_archive",
            side_effect=RuntimeError,
        ),
    ):
        with pytest.raises(RuntimeError, match="No usable public IMAP files"):
            fetch_range("mag", fallback=False)


def test_fetch_range_accepts_date_arguments() -> None:
    start = date.today() - timedelta(days=2)
    end = date.today()

    with (
        patch(
            "ialirt_explorer.ingestion.fetch_space_weather",
            side_effect=RuntimeError,
        ),
        patch(
            "ialirt_explorer.ingestion.fetch_archive",
            side_effect=RuntimeError,
        ),
    ):
        frame = fetch_range("mag", start_date=start, end_date=end)

    assert not frame.empty
    assert frame.attrs["source"] == "synthetic-fallback"


def test_fetch_archive_no_files_falls_back_to_synthetic() -> None:
    with patch("ialirt_explorer.ingestion.list_available", return_value=[]):
        frame = fetch_archive("swapi", since=datetime.now(tz=UTC).date())

    assert not frame.empty
    assert frame.attrs["source"] == "synthetic-fallback"

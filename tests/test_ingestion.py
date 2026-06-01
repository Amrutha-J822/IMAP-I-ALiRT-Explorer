from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ialirt_explorer.ingestion import (
    IALIRT_INSTRUMENTS,
    _synthetic_data,
    fetch_latest,
    fetch_range,
    list_available,
)


@pytest.mark.parametrize("instrument", ["mag", "swapi", "hit", "swe"])
def test_synthetic_data_returns_typed_time_series(instrument: str) -> None:
    frame = _synthetic_data(instrument, n_points=128)

    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None
    assert frame.index.name == "time"
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 128
    assert frame.attrs["source"] == "synthetic-fallback"
    assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)


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


def test_list_available_parses_mocked_rest_response() -> None:
    mock_body = [
        {
            "file_path": "imap/mag/l1c/2026/001/imap_mag_l1c_ialirt_20260101_v001.cdf",
            "instrument": "mag",
            "data_level": "l1c",
            "start_date": "20260101",
        }
    ]

    with (
        patch("ialirt_explorer.ingestion._query_with_package", return_value=[]),
        patch("requests.get") as mock_get,
    ):
        response = MagicMock()
        response.json.return_value = mock_body
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        files = list_available("mag", start_date="20260101", end_date="20260102")

    assert len(files) == 1
    assert files[0]["instrument"] == "mag"
    assert files[0]["file_path"].endswith(".cdf")


def test_list_available_accepts_all_known_instruments_when_offline() -> None:
    with (
        patch("ialirt_explorer.ingestion._query_with_package", return_value=[]),
        patch("ialirt_explorer.ingestion._query_with_rest", return_value=[]),
    ):
        for instrument in IALIRT_INSTRUMENTS:
            assert list_available(instrument) == []


@pytest.mark.parametrize("instrument", ["mag", "swapi", "hit", "swe"])
def test_fetch_latest_falls_back_to_non_empty_dataframe(instrument: str) -> None:
    with patch("ialirt_explorer.ingestion.list_available", return_value=[]):
        frame = fetch_latest(instrument, days=1)

    assert not frame.empty
    assert frame.attrs["source"] == "synthetic-fallback"


def test_fetch_range_respects_no_fallback_mode() -> None:
    with patch("ialirt_explorer.ingestion.list_available", return_value=[]):
        with pytest.raises(RuntimeError, match="No usable public IMAP files"):
            fetch_range("mag", fallback=False)


def test_fetch_range_accepts_date_arguments() -> None:
    start = date.today() - timedelta(days=2)
    end = date.today()

    with patch("ialirt_explorer.ingestion.list_available", return_value=[]):
        frame = fetch_range("mag", start_date=start, end_date=end, parallel=False)

    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty

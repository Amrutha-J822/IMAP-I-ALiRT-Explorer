from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ialirt_explorer.ingestion import _synthetic_data
from ialirt_explorer.service.api import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_endpoint_returns_poller_status(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "poller" in payload
    assert "topics" in payload


def test_instruments_endpoint_lists_known_instruments(client: TestClient) -> None:
    response = client.get("/instruments")
    assert response.status_code == 200
    payload = response.json()
    names = {item["name"] for item in payload["instruments"]}
    assert {"mag", "swapi", "swe", "hit"}.issubset(names)


def test_snapshot_endpoint_returns_frame(client: TestClient) -> None:
    fake_frame = _synthetic_data("mag", n_points=64)

    with patch(
        "ialirt_explorer.service.api.fetch_latest", return_value=fake_frame
    ):
        response = client.get("/snapshot/mag?days=1&calibrate=true&method=offset")

    assert response.status_code == 200
    payload = response.json()
    assert payload["frame"]["instrument"] == "mag"
    assert payload["calibration"]["method"] == "offset"
    assert payload["stats"]["n_rows"] == len(fake_frame)


def test_snapshot_unknown_instrument_returns_404(client: TestClient) -> None:
    response = client.get("/snapshot/unknown")
    assert response.status_code == 404


def test_calibration_compare_returns_suggestion(client: TestClient) -> None:
    fake_frame = _synthetic_data("mag", n_points=200)

    with patch(
        "ialirt_explorer.service.api.fetch_latest", return_value=fake_frame
    ):
        response = client.get("/calibration/mag/compare?days=1")

    assert response.status_code == 200
    payload = response.json()
    assert "comparison" in payload
    assert "suggested" in payload
    assert payload["suggested"]["recommendation"] in {"offset", "detrend", "zscore"}


def test_calibration_rejected_for_non_mag(client: TestClient) -> None:
    response = client.get("/calibration/swapi/compare")
    assert response.status_code == 400

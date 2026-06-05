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


def test_snapshot_propagates_synthetic_kill_switch_to_fetch_latest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The /snapshot endpoint must ask fetch_latest for *real data only*
    when IALIRT_ALLOW_SYNTHETIC_FALLBACK is off (the deployed default).

    Without this guarantee, an instrument whose upstream has gone quiet
    would silently get filled in with fabricated rows — exactly what
    happened to codice_hi in production before this kill switch landed.
    """

    monkeypatch.delenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    fake_frame = _synthetic_data("mag", n_points=4)

    with patch(
        "ialirt_explorer.service.api.fetch_latest", return_value=fake_frame
    ) as mocked:
        client.get("/snapshot/mag?days=1")

    mocked.assert_called_once()
    assert mocked.call_args.kwargs["fallback"] is False


def test_snapshot_opts_into_synthetic_when_env_flag_is_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in path: setting the env flag flips fallback to True."""

    monkeypatch.setenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", "true")
    fake_frame = _synthetic_data("mag", n_points=4)

    with patch(
        "ialirt_explorer.service.api.fetch_latest", return_value=fake_frame
    ) as mocked:
        client.get("/snapshot/mag?days=1")

    assert mocked.call_args.kwargs["fallback"] is True


def test_snapshot_returns_empty_frame_when_upstream_dry_and_synthetic_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When fetch_latest raises (no upstream data + fallback off) the API
    must return an empty frame, not a 500. This is the 'codice_hi
    currently silent' user-visible state: the chart goes empty, the
    source field is blank, and no synthetic line gets drawn.
    """

    monkeypatch.delenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", raising=False)

    with patch(
        "ialirt_explorer.service.api.fetch_latest",
        side_effect=RuntimeError("no data"),
    ):
        response = client.get("/snapshot/codice_hi?days=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["frame"]["time"] == []
    assert payload["frame"]["columns"] == {}


def test_poller_config_default_disables_synthetic_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh PollerConfig in a clean environment must refuse to invent
    samples — this is what guarantees the live Render deploy never
    publishes a 'synthetic-fallback' message to the WebSocket.
    """

    from ialirt_explorer.service.poller import PollerConfig

    monkeypatch.delenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    assert PollerConfig().fallback_to_synthetic is False


def test_poller_config_honors_opt_in_env_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests, notebooks, and offline demos can still opt in explicitly."""

    from ialirt_explorer.service.poller import PollerConfig

    monkeypatch.setenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", "true")
    assert PollerConfig().fallback_to_synthetic is True

    monkeypatch.setenv("IALIRT_ALLOW_SYNTHETIC_FALLBACK", "off")
    assert PollerConfig().fallback_to_synthetic is False

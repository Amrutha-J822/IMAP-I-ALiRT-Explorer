"""FastAPI application exposing REST + WebSocket access to I-ALiRT data."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ialirt_explorer import (
    IALIRT_INSTRUMENTS,
    analyze,
    calibrate_mag,
    calibration_quality,
    compare_calibration_methods,
    detect_anomalies,
    fetch_latest,
    suggest_calibration_method,
)
from ialirt_explorer.service.poller import IALiRTPoller, PollerConfig
from ialirt_explorer.service.pubsub import Broker

log = logging.getLogger(__name__)

# Single switch that gates *every* synthetic-fallback path the service can
# take. Default: off, because the live deployment exists to serve real
# I-ALiRT downlinks — not to fabricate plausible-looking samples when the
# upstream returns empty. The library-level fetch_* helpers still default
# to fallback=True so the offline test suite and notebooks keep working;
# this knob narrows that down to "production service mode" only.
_ALLOW_SYNTHETIC_ENV = "IALIRT_ALLOW_SYNTHETIC_FALLBACK"


def _allow_synthetic_fallback() -> bool:
    return os.environ.get(_ALLOW_SYNTHETIC_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _fetch_latest_or_empty(instrument: str, *, days: int) -> pd.DataFrame:
    """Wrapper around :func:`fetch_latest` that honors the synthetic-fallback
    kill switch and surfaces 'upstream had nothing' as an empty DataFrame
    instead of fabricated rows.
    """

    allow_synth = _allow_synthetic_fallback()
    try:
        return await asyncio.to_thread(
            fetch_latest, instrument, days=days, fallback=allow_synth
        )
    except RuntimeError as exc:
        log.info("No upstream samples for %s: %s", instrument, exc)
        return pd.DataFrame()


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame to a JSON-safe dict suitable for the frontend."""

    if frame.empty:
        return {"time": [], "columns": {}, "source": frame.attrs.get("source", "")}
    numeric = frame.select_dtypes(include="number")
    return {
        "time": [
            stamp.isoformat() if isinstance(stamp, pd.Timestamp) else str(stamp)
            for stamp in frame.index
        ],
        "columns": {column: numeric[column].astype(float).tolist() for column in numeric.columns},
        "source": frame.attrs.get("source", ""),
        "instrument": frame.attrs.get("instrument", ""),
    }


def create_app(
    *, broker: Broker | None = None, config: PollerConfig | None = None
) -> FastAPI:
    """Create the FastAPI app, broker, and background poller."""

    broker = broker or Broker()
    poller = IALiRTPoller(broker=broker, config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        await poller.start()
        try:
            yield
        finally:
            await poller.stop()

    app = FastAPI(
        title="IMAP I-ALiRT Explorer",
        description=(
            "Live ingestion and analytics for the public IMAP I-ALiRT space-weather "
            "feed (https://ialirt.imap-mission.com). Exposes REST endpoints for "
            "one-shot queries, calibration helpers, and a WebSocket for pub/sub "
            "delivery of new samples per instrument."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.broker = broker
    app.state.poller = poller

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "poller": poller.status, "topics": broker.topics}

    @app.get("/instruments")
    async def instruments() -> dict[str, Any]:
        return {
            "instruments": [
                {
                    "name": name,
                    "cadence_seconds": spec.cadence_seconds,
                    "columns": list(spec.columns),
                }
                for name, spec in IALIRT_INSTRUMENTS.items()
            ]
        }

    @app.get("/snapshot/{instrument}")
    async def snapshot(
        instrument: str,
        days: int = Query(1, ge=0, le=14, description="Lookback in days."),
        calibrate: bool = Query(False, description="Apply MAG calibration if applicable."),
        method: str = Query("offset", description="Calibration method for MAG."),
        with_anomalies: bool = Query(True),
    ) -> dict[str, Any]:
        if instrument not in IALIRT_INSTRUMENTS:
            raise HTTPException(status_code=404, detail=f"Unknown instrument {instrument!r}")
        frame = await _fetch_latest_or_empty(instrument, days=days)
        applied_calibration: dict[str, Any] | None = None
        if calibrate and instrument == "mag":
            calibrated = calibrate_mag(frame, method=method)
            applied_calibration = calibration_quality(frame, calibrated)
            applied_calibration["method"] = method
            frame = calibrated
        stats = analyze(frame)
        anomalies = (
            detect_anomalies(frame, instrument) if with_anomalies else pd.DataFrame()
        )
        return {
            "frame": _frame_payload(frame),
            "stats": stats,
            "calibration": applied_calibration,
            "anomalies": (
                {
                    "time": [
                        stamp.isoformat()
                        for stamp in anomalies.index[anomalies.get("any_anomaly", False)]
                    ],
                    "flag_counts": {
                        column: int(anomalies[column].sum())
                        for column in anomalies.columns
                        if anomalies[column].dtype == bool
                    },
                }
                if not anomalies.empty
                else {"time": [], "flag_counts": {}}
            ),
        }

    @app.get("/calibration/{instrument}/suggest")
    async def calibration_suggest(instrument: str, days: int = 1) -> dict[str, Any]:
        if instrument != "mag":
            raise HTTPException(
                status_code=400,
                detail="Calibration tools are only defined for the MAG instrument.",
            )
        frame = await _fetch_latest_or_empty(instrument, days=days)
        return suggest_calibration_method(frame)

    @app.get("/calibration/{instrument}/compare")
    async def calibration_compare(instrument: str, days: int = 1) -> dict[str, Any]:
        if instrument != "mag":
            raise HTTPException(
                status_code=400,
                detail="Calibration tools are only defined for the MAG instrument.",
            )
        frame = await _fetch_latest_or_empty(instrument, days=days)
        return {
            "comparison": compare_calibration_methods(frame),
            "suggested": suggest_calibration_method(frame),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        instruments: str = Query(
            ",".join(IALIRT_INSTRUMENTS),
            description="Comma-separated instrument list to subscribe to.",
        ),
    ) -> None:
        await websocket.accept()

        topics = tuple(
            name.strip()
            for name in instruments.split(",")
            if name.strip() in IALIRT_INSTRUMENTS
        ) or tuple(IALIRT_INSTRUMENTS)

        async with broker.subscribe(topics) as subscription:
            for topic in topics:
                last = broker.latest(topic)
                if last is not None:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "topic": last.topic,
                                "sequence": last.sequence,
                                "payload": last.payload,
                            }
                        )
                    )
            try:
                while True:
                    message = await subscription.receive()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "topic": message.topic,
                                "sequence": message.sequence,
                                "payload": message.payload,
                            }
                        )
                    )
            except WebSocketDisconnect:
                log.info("ws client disconnected")
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("ws error: %s", exc)
                await websocket.close(code=1011)

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):  # type: ignore[no-untyped-def]
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()

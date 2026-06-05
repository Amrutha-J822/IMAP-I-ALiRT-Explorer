"""Background poller that publishes live I-ALiRT samples to the broker.

The poller treats the public ``/space-weather`` endpoint as the source of
truth, requests a moving lookback window per instrument, and only republishes
samples it has not already seen. Each new sample is published as one message
on the per-instrument topic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from ialirt_explorer.ingestion import (
    DEFAULT_API_URL,
    IALIRT_INSTRUMENTS,
    _synthetic_data,
    fetch_space_weather_async,
)
from ialirt_explorer.service.pubsub import Broker

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PollerConfig:
    """Runtime configuration for the live poller.

    ``fallback_to_synthetic`` defaults to ``False`` so the deployed service
    only publishes real I-ALiRT downlinks. Opt in by setting
    ``IALIRT_ALLOW_SYNTHETIC_FALLBACK=true`` if you want the poller to
    invent deterministic placeholder rows when the upstream returns empty
    (useful for offline demos, never appropriate for production).
    """

    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "IALIRT_DATA_ACCESS_URL", DEFAULT_API_URL
        ).rstrip("/")
    )
    poll_interval_seconds: float = field(
        default_factory=lambda: float(os.environ.get("IALIRT_POLL_INTERVAL_SECONDS", "30"))
    )
    instruments: tuple[str, ...] = field(
        default_factory=lambda: tuple(IALIRT_INSTRUMENTS)
    )
    fallback_to_synthetic: bool = field(
        default_factory=lambda: _env_flag("IALIRT_ALLOW_SYNTHETIC_FALLBACK", default=False)
    )


def _frame_to_records(frame: pd.DataFrame, instrument: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for stamp, row in frame.iterrows():
        record: dict[str, Any] = {
            "instrument": instrument,
            "time_utc": stamp.isoformat() if isinstance(stamp, pd.Timestamp) else str(stamp),
        }
        for column, value in row.items():
            try:
                record[column] = None if pd.isna(value) else float(value)
            except (TypeError, ValueError):
                record[column] = None
        out.append(record)
    return out


class IALiRTPoller:
    """Periodic publisher of live I-ALiRT samples to the in-memory broker."""

    def __init__(self, broker: Broker, config: PollerConfig | None = None) -> None:
        self.broker = broker
        self.config = config or PollerConfig()
        self._last_seen: dict[str, pd.Timestamp] = {}
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._used_synthetic: dict[str, bool] = {}

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._client = httpx.AsyncClient(
            base_url=self.config.api_url, timeout=httpx.Timeout(20.0)
        )
        self._task = asyncio.create_task(self._run(), name="ialirt-poller")
        log.info(
            "I-ALiRT poller started api_url=%s interval=%.1fs instruments=%s",
            self.config.api_url,
            self.config.poll_interval_seconds,
            ",".join(self.config.instruments),
        )

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # pragma: no cover - shutdown
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("I-ALiRT poller stopped")

    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                await self.poll_once()
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(),
                        timeout=self.config.poll_interval_seconds,
                    )
                except TimeoutError:
                    continue
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise

    async def poll_once(self) -> dict[str, int]:
        """Run one polling cycle and return per-instrument new-message counts."""

        assert self._client is not None, "Poller must be started before polling"

        results: dict[str, int] = {}
        tasks = [
            self._poll_instrument(instrument)
            for instrument in self.config.instruments
        ]
        for instrument, count in zip(
            self.config.instruments,
            await asyncio.gather(*tasks, return_exceptions=False),
            strict=True,
        ):
            results[instrument] = count
        return results

    async def _poll_instrument(self, instrument: str) -> int:
        """Pull the latest open-ended window from /space-weather.

        The endpoint returns its native recent-samples window for queries
        without explicit time bounds and rejects long explicit windows with
        a 400, so we let it pick the window and deduplicate locally.
        """

        assert self._client is not None

        try:
            frame = await fetch_space_weather_async(self._client, instrument)
        except Exception as exc:  # pragma: no cover - upstream surprises
            log.warning("Polling %s failed: %s", instrument, exc)
            frame = pd.DataFrame()

        used_synthetic = False
        if frame.empty and self.config.fallback_to_synthetic:
            frame = _synthetic_data(instrument, n_points=12)
            used_synthetic = True

        self._used_synthetic[instrument] = used_synthetic
        if frame.empty:
            return 0

        last = self._last_seen.get(instrument)
        if last is not None:
            frame = frame[frame.index > last]
        if frame.empty:
            return 0

        records = _frame_to_records(frame, instrument)
        for record in records:
            record["source"] = "synthetic-fallback" if used_synthetic else "ialirt-sdc"
            await self.broker.publish(instrument, record)

        self._last_seen[instrument] = frame.index.max()
        log.info(
            "poll instrument=%s new_samples=%d source=%s",
            instrument,
            len(records),
            "synthetic" if used_synthetic else "live",
        )
        return len(records)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "api_url": self.config.api_url,
            "interval_seconds": self.config.poll_interval_seconds,
            "instruments": list(self.config.instruments),
            "last_seen": {
                instrument: stamp.isoformat()
                for instrument, stamp in self._last_seen.items()
            },
            "using_synthetic_for": [
                instrument
                for instrument, value in self._used_synthetic.items()
                if value
            ],
        }

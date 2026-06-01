"""Parallel orchestration for multi-instrument I-ALiRT analysis."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ialirt_explorer.analytics import analyze, calibrate_mag, detect_anomalies
from ialirt_explorer.ingestion import IALIRT_INSTRUMENTS, fetch_latest


def _analyze_one(instrument: str, days: int) -> dict[str, Any]:
    frame = fetch_latest(instrument, days=days)
    analysis_frame = calibrate_mag(frame) if instrument == "mag" else frame
    return {
        "data": analysis_frame,
        "stats": analyze(analysis_frame),
        "flagged": detect_anomalies(analysis_frame, instrument),
    }


def parallel_analyze(
    instruments: Iterable[str] | None = None,
    *,
    days: int = 3,
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch, summarize, and flag multiple instruments concurrently."""

    selected = [item.lower() for item in (instruments or IALIRT_INSTRUMENTS.keys())]
    workers = max_workers or min(4, len(selected))
    results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_one, instrument, days): instrument for instrument in selected
        }
        for future in as_completed(futures):
            instrument = futures[future]
            results[instrument] = future.result()

    return results

#!/usr/bin/env python3
"""End-to-end IMAP I-ALiRT Explorer demo."""

from __future__ import annotations

import logging
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ialirt_mpl_cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")

import ialirt_explorer as ie

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def separator(title: str) -> None:
    """Print a readable section break in demo logs."""

    log.info("%s", "-" * 72)
    log.info("%s", title)
    log.info("%s", "-" * 72)


def main() -> None:
    """Run the full research workflow and save example figures."""

    separator("1. Fetch MAG data")
    start = time.perf_counter()
    mag = ie.fetch_latest("mag", days=1)
    log.info(
        "Loaded %d rows from %s in %.2fs",
        len(mag),
        mag.attrs.get("source"),
        time.perf_counter() - start,
    )
    log.info("Columns: %s", ", ".join(mag.columns))

    separator("2. Analyze and calibrate")
    stats = ie.analyze(mag)
    suggestion = ie.suggest_calibration_method(mag)
    log.info("Suggested calibration: %s (%s)", suggestion["recommendation"], suggestion["rationale"])
    calibrated = ie.calibrate_mag(mag, method=suggestion["recommendation"])
    quality = ie.calibration_quality(mag, calibrated)
    drift_delta = float(np.nanstd(mag["Bz_nT"]) - np.nanstd(calibrated["Bz_nT"]))
    log.info("Duration: %.1f hours", stats["duration_hours"])
    log.info("Cadence: %.0f seconds", stats["cadence_seconds"])
    log.info("Bz standard-deviation change after calibration: %.3f nT", drift_delta)
    log.info(
        "Baseline removed: %.2f nT, residual drift: %.3f nT/h",
        quality.get("baseline_amplitude_nT", float("nan")),
        quality.get("residual_drift_per_hour_nT", float("nan")),
    )

    separator("3. Detect candidate events")
    flagged = ie.detect_anomalies(calibrated, "mag", sigma_threshold=3.0)
    log.info("Anomalous points: %d", int(flagged["any_anomaly"].sum()))
    log.info("Southward Bz intervals: %d", int(flagged.get("storm_southward_Bz", []).sum()))

    separator("4. Compute pressure terms")
    swapi = ie.fetch_latest("swapi", days=1)
    pressure = ie.compute_pressures(calibrated, swapi)
    if pressure.empty:
        log.info("No overlapping MAG/SWAPI samples available.")
    else:
        log.info("Mean total pressure: %.3f nPa", float(pressure["P_total_nPa"].mean()))
        log.info("Mean plasma beta: %.3f", float(pressure["plasma_beta"].mean()))

    separator("5. Parallel multi-instrument analysis")
    start = time.perf_counter()
    results = ie.parallel_analyze(["mag", "swe", "swapi", "hit"], days=1)
    log.info("Analyzed %d instruments in %.2fs", len(results), time.perf_counter() - start)
    for instrument, result in sorted(results.items()):
        log.info(
            "%-5s rows=%4d anomalies=%3d",
            instrument.upper(),
            result["stats"].get("n_rows", 0),
            int(result["flagged"]["any_anomaly"].sum()),
        )

    separator("6. Save figures")
    ie.plot_dashboard(calibrated, stats, instrument="mag", save_path="output_mag_dashboard.png")
    ie.plot_timeseries(
        calibrated,
        instrument="mag",
        anomaly_df=flagged,
        title="IMAP MAG calibrated field with anomaly markers",
        save_path="output_mag_timeseries.png",
    )
    ie.plot_hodogram(calibrated, save_path="output_mag_hodogram.png")
    ie.plot_anomaly_summary(flagged, instrument="mag", save_path="output_anomaly_summary.png")
    ie.plot_dashboard(results, save_path="output_multi_instrument.png")
    log.info("Saved output_*.png")


if __name__ == "__main__":
    main()

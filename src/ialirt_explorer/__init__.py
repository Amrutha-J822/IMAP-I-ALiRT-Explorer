"""Live ingestion, calibration, analytics, and visualization for IMAP I-ALiRT."""

from ialirt_explorer.analytics import (
    CALIBRATION_METHODS,
    analyze,
    calibrate_mag,
    calibration_quality,
    compare_calibration_methods,
    compute_pressures,
    detect_anomalies,
    suggest_calibration_method,
)
from ialirt_explorer.ingestion import (
    DEFAULT_API_URL,
    IALIRT_INSTRUMENTS,
    fetch_archive,
    fetch_latest,
    fetch_range,
    fetch_space_weather,
    fetch_space_weather_async,
    list_available,
)
from ialirt_explorer.parallel import parallel_analyze
from ialirt_explorer.visualization import (
    plot_anomaly_summary,
    plot_dashboard,
    plot_hodogram,
    plot_timeseries,
)

__all__ = [
    "CALIBRATION_METHODS",
    "DEFAULT_API_URL",
    "IALIRT_INSTRUMENTS",
    "analyze",
    "calibrate_mag",
    "calibration_quality",
    "compare_calibration_methods",
    "compute_pressures",
    "detect_anomalies",
    "fetch_archive",
    "fetch_latest",
    "fetch_range",
    "fetch_space_weather",
    "fetch_space_weather_async",
    "list_available",
    "parallel_analyze",
    "plot_anomaly_summary",
    "plot_dashboard",
    "plot_hodogram",
    "plot_timeseries",
    "suggest_calibration_method",
]

__version__ = "0.2.0"

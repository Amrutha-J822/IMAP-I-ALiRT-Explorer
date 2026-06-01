"""Research-grade tools for exploring IMAP I-ALiRT space-weather data."""

from ialirt_explorer.analytics import (
    analyze,
    calibrate_mag,
    compute_pressures,
    detect_anomalies,
)
from ialirt_explorer.ingestion import (
    IALIRT_INSTRUMENTS,
    fetch_latest,
    fetch_range,
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
    "IALIRT_INSTRUMENTS",
    "analyze",
    "calibrate_mag",
    "compute_pressures",
    "detect_anomalies",
    "fetch_latest",
    "fetch_range",
    "list_available",
    "parallel_analyze",
    "plot_anomaly_summary",
    "plot_dashboard",
    "plot_hodogram",
    "plot_timeseries",
]

__version__ = "0.1.0"

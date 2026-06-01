"""Publication-ready Matplotlib/Seaborn visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_timeseries(
    df: pd.DataFrame,
    *,
    instrument: str = "mag",
    anomaly_df: pd.DataFrame | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot all numeric columns with optional anomaly markers."""

    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 5))
    for column in numeric.columns[:6]:
        ax.plot(numeric.index, numeric[column], linewidth=1.2, label=column)

    if anomaly_df is not None and "any_anomaly" in anomaly_df:
        anomaly_times = anomaly_df.index[anomaly_df["any_anomaly"]]
        for stamp in anomaly_times[:100]:
            ax.axvline(stamp, color="#d62728", alpha=0.08, linewidth=1)

    ax.set_title(title or f"IMAP {instrument.upper()} I-ALiRT time series")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Normalized instrument units")
    ax.legend(loc="best", fontsize="small")
    return _save_or_show(fig, save_path)


def plot_hodogram(df: pd.DataFrame, *, save_path: str | Path | None = None) -> plt.Figure:
    """Plot a MAG By/Bz hodogram for magnetic-field rotation inspection."""

    required = {"By_nT", "Bz_nT"}
    if not required.issubset(df.columns):
        raise ValueError("Hodogram requires By_nT and Bz_nT columns")

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.linspace(0, 1, len(df))
    scatter = ax.scatter(df["By_nT"], df["Bz_nT"], c=colors, cmap="viridis", s=12, alpha=0.75)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("By (nT)")
    ax.set_ylabel("Bz (nT)")
    ax.set_title("MAG hodogram")
    fig.colorbar(scatter, ax=ax, label="Time progression")
    return _save_or_show(fig, save_path)


def plot_anomaly_summary(
    flagged: pd.DataFrame,
    *,
    instrument: str = "mag",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot counts for boolean anomaly flags."""

    bool_cols = [
        column
        for column in flagged.columns
        if flagged[column].dtype == bool and column != "any_anomaly"
    ]
    counts = (
        flagged[bool_cols].sum().sort_values(ascending=True) if bool_cols else pd.Series(dtype=int)
    )

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * max(1, len(counts)))))
    if counts.empty:
        ax.text(0.5, 0.5, "No anomaly flags available", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4c78a8")
        ax.set_xlabel("Flagged samples")
        ax.set_ylabel("")
    ax.set_title(f"{instrument.upper()} anomaly summary")
    return _save_or_show(fig, save_path)


def _plot_multi_instrument(
    results: dict[str, dict[str, Any]], save_path: str | Path | None
) -> plt.Figure:
    instruments = list(results)
    fig, axes = plt.subplots(
        len(instruments), 1, figsize=(12, 2.8 * len(instruments)), sharex=False
    )
    if len(instruments) == 1:
        axes = [axes]

    for ax, instrument in zip(axes, instruments, strict=False):
        frame = results[instrument]["data"]
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        column = numeric.columns[0]
        ax.plot(numeric.index, numeric[column], linewidth=1)
        ax.set_title(f"{instrument.upper()} - {column}")
        ax.set_ylabel(column)

    return _save_or_show(fig, save_path)


def plot_dashboard(
    data: pd.DataFrame | dict[str, dict[str, Any]],
    stats: dict[str, Any] | None = None,
    *,
    instrument: str = "mag",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a compact research dashboard for one frame or many instruments."""

    if isinstance(data, dict):
        return _plot_multi_instrument(data, save_path)

    frame = data.select_dtypes(include=[np.number])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    primary_cols = list(frame.columns[:4])
    for column in primary_cols:
        axes[0].plot(frame.index, frame[column], linewidth=1, label=column)
    axes[0].set_title(f"{instrument.upper()} calibrated time series")
    axes[0].legend(fontsize="small")

    if "B_total_nT" in frame:
        sns.histplot(frame["B_total_nT"].dropna(), ax=axes[1], bins=40, color="#59a14f")
        axes[1].set_title("|B| distribution")
    else:
        sns.histplot(frame.iloc[:, 0].dropna(), ax=axes[1], bins=40, color="#59a14f")
        axes[1].set_title(f"{frame.columns[0]} distribution")

    corr = frame[primary_cols].corr() if primary_cols else pd.DataFrame()
    sns.heatmap(corr, ax=axes[2], cmap="vlag", center=0, annot=True, fmt=".2f", cbar=False)
    axes[2].set_title("Column correlation")

    summary = stats or {}
    summary_text = [
        f"Rows: {summary.get('n_rows', len(data))}",
        f"Duration: {summary.get('duration_hours', 0):.1f} h",
        f"Cadence: {summary.get('cadence_seconds', float('nan')):.0f} s",
        f"Missing: {100 * summary.get('missing_fraction', 0):.2f}%",
        f"Source: {data.attrs.get('source', 'unknown')}",
    ]
    axes[3].text(0.03, 0.95, "\n".join(summary_text), transform=axes[3].transAxes, va="top")
    axes[3].set_title("Data quality")
    axes[3].set_axis_off()

    return _save_or_show(fig, save_path)

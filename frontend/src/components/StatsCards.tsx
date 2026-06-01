import type { CalibrationQuality, SnapshotStats } from "../types";

interface Props {
  stats: SnapshotStats | null;
  calibration: CalibrationQuality | null;
  liveStatus: string;
  source: string;
}

function fmt(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

export function StatsCards({ stats, calibration, liveStatus, source }: Props) {
  return (
    <div className="stats-grid">
      <div className="stat-card">
        <div className="stat-label">Live stream</div>
        <div className={`stat-value status-${liveStatus}`}>{liveStatus}</div>
        <div className="stat-meta">Source: {source || "—"}</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Samples in snapshot</div>
        <div className="stat-value">{stats?.n_rows ?? 0}</div>
        <div className="stat-meta">
          {stats ? `${fmt(stats.duration_hours, 1)} h · ${fmt(stats.cadence_seconds, 0)} s cadence` : "—"}
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Missing fraction</div>
        <div className="stat-value">
          {stats ? `${fmt(100 * stats.missing_fraction, 2)}%` : "—"}
        </div>
        <div className="stat-meta">Across all numeric columns</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Calibration</div>
        <div className="stat-value">
          {calibration ? calibration.method : "raw"}
        </div>
        <div className="stat-meta">
          {calibration
            ? `baseline ${fmt(calibration.baseline_amplitude_nT)} nT · drift ${fmt(
                calibration.residual_drift_per_hour_nT,
                3,
              )} nT/h`
            : "Toggle in Calibration Lab to apply"}
        </div>
      </div>
    </div>
  );
}

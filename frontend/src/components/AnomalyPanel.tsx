import type { AnomalyEnvelope } from "../types";

interface Props {
  anomalies: AnomalyEnvelope | null;
}

export function AnomalyPanel({ anomalies }: Props) {
  if (!anomalies || Object.keys(anomalies.flag_counts).length === 0) {
    return (
      <div className="card">
        <h3>Event flags</h3>
        <p className="muted">No anomaly flags returned in the current snapshot.</p>
      </div>
    );
  }

  const sorted = Object.entries(anomalies.flag_counts)
    .filter(([name]) => name !== "any_anomaly")
    .sort((a, b) => b[1] - a[1]);
  const total = anomalies.flag_counts.any_anomaly ?? sorted.reduce((acc, [, v]) => acc + v, 0);

  return (
    <div className="card">
      <div className="card-header">
        <h3>Event flags</h3>
        <span className="card-meta">{anomalies.time.length} flagged samples</span>
      </div>
      <p className="muted small">
        Total any-anomaly samples: <strong>{total}</strong>
      </p>
      <ul className="flag-list">
        {sorted.map(([name, count]) => (
          <li key={name}>
            <span className="flag-name">{name}</span>
            <span className="flag-bar" style={{ width: `${Math.min(100, count)}%` }} />
            <span className="flag-count">{count}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

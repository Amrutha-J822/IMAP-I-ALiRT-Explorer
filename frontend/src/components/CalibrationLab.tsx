import { useEffect, useState } from "react";
import { compareCalibration } from "../api";
import type { CalibrationComparison } from "../types";

interface Props {
  instrument: string;
  method: string;
  onMethodChange: (method: string) => void;
  active: boolean;
  onToggleActive: (active: boolean) => void;
}

const METHOD_DESCRIPTIONS: Record<string, string> = {
  offset:
    "Rolling-median baseline subtraction. Removes slow wandering of the DC level while preserving fast structure.",
  detrend:
    "Linear detrend. Removes a single sustained trend; appropriate when drift is monotonic over the window.",
  zscore:
    "Standardization to zero-mean / unit-variance. Use for cross-instrument comparison rather than calibration.",
};

export function CalibrationLab({
  instrument,
  method,
  onMethodChange,
  active,
  onToggleActive,
}: Props) {
  const [comparison, setComparison] = useState<CalibrationComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (instrument !== "mag") {
      setComparison(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    compareCalibration(instrument)
      .then((data) => {
        if (!cancelled) setComparison(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [instrument]);

  if (instrument !== "mag") {
    return (
      <div className="card">
        <h3>Calibration Lab</h3>
        <p className="muted">
          Calibration helpers are only defined for the MAG instrument. Select{" "}
          <strong>MAG</strong> to inspect, compare, and apply calibration methods.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3>Calibration Lab</h3>
        <label className="toggle">
          <input
            type="checkbox"
            checked={active}
            onChange={(event) => onToggleActive(event.target.checked)}
          />
          <span>Apply to snapshot</span>
        </label>
      </div>

      {loading && <p className="muted">Analyzing recent MAG data...</p>}
      {error && <p className="error">Failed to load comparison: {error}</p>}

      {comparison && (
        <>
          <div className="suggestion">
            <strong>Suggested:</strong>{" "}
            <code>{comparison.suggested.recommendation}</code>
            <p className="muted">{comparison.suggested.rationale}</p>
          </div>

          <table className="comparison-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Score</th>
                <th>Baseline (nT)</th>
                <th>Residual drift (nT/h)</th>
                <th>Mean corr</th>
                <th>Choose</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.comparison).map(([name, entry]) => {
                const components = Object.values(entry.quality.per_component ?? {});
                const meanCorr =
                  components.length > 0
                    ? components.reduce(
                        (acc, c) => acc + c.raw_calibrated_correlation,
                        0,
                      ) / components.length
                    : Number.NaN;
                const recommended = comparison.suggested.recommendation === name;
                return (
                  <tr key={name} className={recommended ? "recommended" : ""}>
                    <td>
                      <code>{name}</code>
                      <div className="muted small">{METHOD_DESCRIPTIONS[name]}</div>
                    </td>
                    <td>{entry.score.toFixed(2)}</td>
                    <td>{entry.quality.baseline_amplitude_nT.toFixed(2)}</td>
                    <td>{entry.quality.residual_drift_per_hour_nT.toFixed(3)}</td>
                    <td>{Number.isFinite(meanCorr) ? meanCorr.toFixed(3) : "n/a"}</td>
                    <td>
                      <button
                        type="button"
                        className={`pill ${method === name ? "pill-active" : ""}`}
                        onClick={() => onMethodChange(name)}
                      >
                        {method === name ? "Selected" : "Select"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

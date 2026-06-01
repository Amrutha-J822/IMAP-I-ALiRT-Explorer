export interface InstrumentMeta {
  name: string;
  cadence_seconds: number;
  columns: string[];
}

export interface FramePayload {
  time: string[];
  columns: Record<string, number[]>;
  source: string;
  instrument: string;
}

export interface ColumnStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  p5: number;
  p95: number;
}

export interface SnapshotStats {
  n_rows: number;
  n_columns: number;
  duration_hours: number;
  cadence_seconds: number;
  missing_fraction: number;
  column_stats: Record<string, ColumnStats>;
}

export interface CalibrationQualityComponent {
  baseline_amplitude_nT: number;
  baseline_mean_offset_nT: number;
  residual_drift_per_hour_nT: number;
  noise_floor_nT: number;
  raw_calibrated_correlation: number;
  std_before_nT: number;
  std_after_nT: number;
}

export interface CalibrationQuality {
  per_component: Record<string, CalibrationQualityComponent>;
  baseline_amplitude_nT: number;
  residual_drift_per_hour_nT: number;
  method: string;
}

export interface CalibrationSuggestion {
  recommendation: string;
  votes: Record<string, number>;
  diagnostics: Record<string, Record<string, number>>;
  rationale: string;
}

export interface CalibrationComparison {
  comparison: Record<
    string,
    { quality: CalibrationQuality; score: number }
  >;
  suggested: CalibrationSuggestion;
}

export interface AnomalyEnvelope {
  time: string[];
  flag_counts: Record<string, number>;
}

export interface SnapshotResponse {
  frame: FramePayload;
  stats: SnapshotStats;
  calibration: CalibrationQuality | null;
  anomalies: AnomalyEnvelope;
}

export interface LiveSample {
  topic: string;
  sequence: number;
  payload: {
    instrument: string;
    time_utc: string;
    source: string;
    [column: string]: string | number | null;
  };
}

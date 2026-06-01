import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FramePayload, LiveSample } from "../types";

interface Props {
  instrument: string;
  snapshot: FramePayload | null;
  liveSamples: LiveSample[];
}

const COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"];

interface PlotPoint {
  time: number;
  iso: string;
  [series: string]: number | string;
}

export function LiveTimeSeries({ instrument, snapshot, liveSamples }: Props) {
  const merged = useMemo<PlotPoint[]>(() => {
    const points = new Map<number, PlotPoint>();

    if (snapshot && snapshot.instrument === instrument) {
      const seriesNames = Object.keys(snapshot.columns);
      snapshot.time.forEach((iso, idx) => {
        const stamp = Date.parse(iso);
        if (Number.isNaN(stamp)) return;
        const row: PlotPoint = { time: stamp, iso };
        seriesNames.forEach((name) => {
          const value = snapshot.columns[name][idx];
          row[name] = Number.isFinite(value) ? value : NaN;
        });
        points.set(stamp, row);
      });
    }

    for (const sample of liveSamples) {
      if (sample.payload.instrument !== instrument) continue;
      const stamp = Date.parse(sample.payload.time_utc);
      if (Number.isNaN(stamp)) continue;
      const existing = points.get(stamp) ?? { time: stamp, iso: sample.payload.time_utc };
      for (const [column, value] of Object.entries(sample.payload)) {
        if (column === "instrument" || column === "time_utc" || column === "source") continue;
        if (typeof value !== "number") continue;
        existing[column] = value;
      }
      points.set(stamp, existing);
    }

    return Array.from(points.values()).sort((a, b) => a.time - b.time);
  }, [snapshot, liveSamples, instrument]);

  const seriesNames = useMemo(() => {
    if (snapshot && snapshot.instrument === instrument) {
      return Object.keys(snapshot.columns);
    }
    const fromLive = new Set<string>();
    liveSamples.forEach((sample) => {
      if (sample.payload.instrument !== instrument) return;
      Object.entries(sample.payload).forEach(([k, v]) => {
        if (k !== "instrument" && k !== "time_utc" && k !== "source" && typeof v === "number") {
          fromLive.add(k);
        }
      });
    });
    return Array.from(fromLive);
  }, [snapshot, liveSamples, instrument]);

  if (merged.length === 0) {
    return (
      <div className="card empty-chart">
        <h3>{instrument.toUpperCase()} live feed</h3>
        <p>Waiting for the first sample to arrive on the WebSocket...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3>{instrument.toUpperCase()} live time series</h3>
        <span className="card-meta">{merged.length} samples</span>
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={merged} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(value) =>
                new Date(value).toISOString().substring(11, 19)
              }
              stroke="#9ca3af"
              minTickGap={48}
            />
            <YAxis stroke="#9ca3af" />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
              labelFormatter={(value) => new Date(Number(value)).toISOString()}
              formatter={(value: number) =>
                Number.isFinite(value) ? value.toFixed(3) : "n/a"
              }
            />
            <Legend wrapperStyle={{ paddingTop: 8 }} />
            {seriesNames.map((name, idx) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={COLORS[idx % COLORS.length]}
                dot={false}
                strokeWidth={1.6}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

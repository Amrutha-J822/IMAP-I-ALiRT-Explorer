import { useEffect, useRef, useState } from "react";
import type {
  CalibrationComparison,
  CalibrationSuggestion,
  InstrumentMeta,
  LiveSample,
  SnapshotResponse,
} from "./types";

const HTTP_BASE =
  (import.meta.env.VITE_BACKEND_HTTP as string | undefined) ?? "/api";
const WS_BASE =
  (import.meta.env.VITE_BACKEND_WS as string | undefined) ??
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
    : "ws://127.0.0.1:8000");

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${HTTP_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} on ${path}`);
  }
  return (await response.json()) as T;
}

export async function listInstruments(): Promise<InstrumentMeta[]> {
  const payload = await getJson<{ instruments: InstrumentMeta[] }>(
    "/instruments",
  );
  return payload.instruments;
}

export async function getSnapshot(
  instrument: string,
  options: {
    days?: number;
    calibrate?: boolean;
    method?: string;
    withAnomalies?: boolean;
  } = {},
): Promise<SnapshotResponse> {
  const params = new URLSearchParams();
  params.set("days", String(options.days ?? 1));
  if (options.calibrate) params.set("calibrate", "true");
  if (options.method) params.set("method", options.method);
  if (options.withAnomalies === false) params.set("with_anomalies", "false");
  return getJson<SnapshotResponse>(
    `/snapshot/${instrument}?${params.toString()}`,
  );
}

export async function compareCalibration(
  instrument: string,
  days = 1,
): Promise<CalibrationComparison> {
  return getJson<CalibrationComparison>(
    `/calibration/${instrument}/compare?days=${days}`,
  );
}

export async function suggestCalibration(
  instrument: string,
  days = 1,
): Promise<CalibrationSuggestion> {
  return getJson<CalibrationSuggestion>(
    `/calibration/${instrument}/suggest?days=${days}`,
  );
}

export type LiveStatus = "connecting" | "open" | "closed" | "error";

export interface UseLiveStream {
  status: LiveStatus;
  samples: LiveSample[];
  lastSample: LiveSample | null;
}

const MAX_BUFFER = 600;

export function useLiveStream(instruments: string[]): UseLiveStream {
  const [status, setStatus] = useState<LiveStatus>("connecting");
  const [samples, setSamples] = useState<LiveSample[]>([]);
  const lastSampleRef = useRef<LiveSample | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (instruments.length === 0) {
      setStatus("closed");
      return;
    }
    const url = `${WS_BASE}/ws?instruments=${instruments.join(",")}`;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      setStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!cancelled) setStatus("open");
      };
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as LiveSample;
          lastSampleRef.current = parsed;
          setSamples((prev) => {
            const next = [...prev, parsed];
            return next.length > MAX_BUFFER
              ? next.slice(next.length - MAX_BUFFER)
              : next;
          });
        } catch (error) {
          console.warn("malformed live sample", error);
        }
      };
      ws.onerror = () => {
        if (!cancelled) setStatus("error");
      };
      ws.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        retryTimer = setTimeout(connect, 2000);
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, [instruments.join(",")]);

  return { status, samples, lastSample: lastSampleRef.current };
}

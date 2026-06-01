import { useEffect, useMemo, useState } from "react";
import { getSnapshot, listInstruments, useLiveStream } from "./api";
import { AnomalyPanel } from "./components/AnomalyPanel";
import { CalibrationLab } from "./components/CalibrationLab";
import { InstrumentSelector } from "./components/InstrumentSelector";
import { LiveTimeSeries } from "./components/LiveTimeSeries";
import { StatsCards } from "./components/StatsCards";
import type { InstrumentMeta, SnapshotResponse } from "./types";

const SNAPSHOT_REFRESH_MS = 30_000;

export default function App() {
  const [instruments, setInstruments] = useState<InstrumentMeta[]>([]);
  const [selected, setSelected] = useState<string>("mag");
  const [calibrate, setCalibrate] = useState<boolean>(false);
  const [method, setMethod] = useState<string>("offset");
  const [snapshot, setSnapshot] = useState<SnapshotResponse | null>(null);
  const [loadingSnapshot, setLoadingSnapshot] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const subscription = useMemo(() => [selected], [selected]);
  const { status, samples } = useLiveStream(subscription);

  useEffect(() => {
    listInstruments()
      .then((list) => {
        setInstruments(list);
        if (list.length > 0 && !list.find((entry) => entry.name === selected)) {
          setSelected(list[0].name);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;

    const load = () => {
      setLoadingSnapshot(true);
      getSnapshot(selected, {
        days: 1,
        calibrate: calibrate && selected === "mag",
        method,
        withAnomalies: true,
      })
        .then((data) => {
          if (!cancelled) {
            setSnapshot(data);
            setError(null);
          }
        })
        .catch((err: Error) => {
          if (!cancelled) setError(err.message);
        })
        .finally(() => {
          if (!cancelled) setLoadingSnapshot(false);
        });
    };

    load();
    const interval = window.setInterval(load, SNAPSHOT_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [selected, calibrate, method]);

  const liveSamplesForInstrument = useMemo(
    () => samples.filter((sample) => sample.payload.instrument === selected),
    [samples, selected],
  );

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">IMAP I-ALiRT Explorer</h1>
        <p className="brand-sub">
          Live ingestion · calibration · anomaly screening
        </p>
        <InstrumentSelector
          instruments={instruments}
          selected={selected}
          onSelect={setSelected}
        />
        <div className="connection">
          <span className={`status-dot status-${status}`} />
          <span>WS {status}</span>
        </div>
        {error && <div className="error">Error: {error}</div>}
      </aside>

      <main className="main">
        <StatsCards
          stats={snapshot?.stats ?? null}
          calibration={snapshot?.calibration ?? null}
          liveStatus={status}
          source={snapshot?.frame.source ?? ""}
        />

        <LiveTimeSeries
          instrument={selected}
          snapshot={snapshot?.frame ?? null}
          liveSamples={liveSamplesForInstrument}
        />

        <div className="grid-two">
          <CalibrationLab
            instrument={selected}
            method={method}
            onMethodChange={setMethod}
            active={calibrate}
            onToggleActive={setCalibrate}
          />
          <AnomalyPanel anomalies={snapshot?.anomalies ?? null} />
        </div>

        {loadingSnapshot && <div className="muted small">Refreshing snapshot…</div>}
      </main>
    </div>
  );
}

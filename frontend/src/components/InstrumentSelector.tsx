import type { InstrumentMeta } from "../types";

interface Props {
  instruments: InstrumentMeta[];
  selected: string;
  onSelect: (name: string) => void;
}

export function InstrumentSelector({ instruments, selected, onSelect }: Props) {
  return (
    <div className="instrument-selector">
      <h3>Instruments</h3>
      <div className="selector-list">
        {instruments.map((instrument) => {
          const active = instrument.name === selected;
          return (
            <button
              key={instrument.name}
              type="button"
              className={`selector-button ${active ? "active" : ""}`}
              onClick={() => onSelect(instrument.name)}
              title={`Cadence: ${instrument.cadence_seconds}s`}
            >
              <span className="selector-name">{instrument.name.toUpperCase()}</span>
              <span className="selector-meta">
                {instrument.cadence_seconds}s · {instrument.columns.length} fields
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

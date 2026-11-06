import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  advanceChordIndex,
  capoDiagramSymbol,
  capoLabel,
  CHORD_DIAGRAM_LAYOUT,
  chordShape,
  displaySymbol,
  formatChordSymbol,
  lyricLineAt,
  measureAt,
  replaceChordSymbol,
  scrollLeftToCenter,
  sectionAt,
  type AnalysisData,
  type ChordLevel,
  type ChordShape,
  type TabData,
} from "../mixer/analysis";

type PracticeTab = "chords" | "sections" | "lyrics" | "tab";

const TABS: { id: PracticeTab; label: string }[] = [
  { id: "chords", label: "Chords" },
  { id: "sections", label: "Sections" },
  { id: "lyrics", label: "Lyrics" },
  { id: "tab", label: "Tab" },
];

const EMPTY_ALL = "No chords, sections, lyrics, or tab in this run.";
const EMPTY_ANALYSIS = "No analysis in this run.";
const EMPTY_CHORDS = "No chords in this run.";
const EMPTY_SECTIONS = "No sections in this run.";
const EMPTY_LYRICS = "No lyrics found (instrumental, or the lyrics model is unavailable).";
const EMPTY_TAB = "No tab for this run. Re-run with guitar separation.";
const CHORD_DEMO_NOTE =
  "Demo — chord chart is a work in progress; roots can drift on dense mixes — verify by ear.";

function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function Diagram({ symbol }: { symbol: string }) {
  const shape: ChordShape | null = chordShape(symbol);
  if (!shape) {
    return (
      <div className="chord-no-shape" data-diagram-symbol={symbol}>
        {formatChordSymbol(symbol)}
      </div>
    );
  }
  const { width, height, left, right, top, bottom, fretLabelX, dotRadius } =
    CHORD_DIAGRAM_LAYOUT;
  const stringGap = (right - left) / 5;
  const fretGap = (bottom - top) / 4;
  return (
    <svg
      className="chord-diagram"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${formatChordSymbol(symbol)} chord diagram`}
      data-diagram-symbol={symbol}
    >
      {shape.baseFret <= 1 ? (
        <rect x={left - 2} y={top - 3} width={right - left + 4} height={3} rx={1} className="chord-nut" />
      ) : (
        <text x={fretLabelX} y={top + fretGap * 0.7} textAnchor="end" className="chord-fret-label">
          {shape.baseFret}
        </text>
      )}
      {Array.from({ length: 6 }, (_, index) => {
        const x = left + index * stringGap;
        return <line key={`s${index}`} x1={x} y1={top} x2={x} y2={bottom} className="chord-string" />;
      })}
      {Array.from({ length: 5 }, (_, index) => {
        const y = top + index * fretGap;
        return <line key={`f${index}`} x1={left} y1={y} x2={right} y2={y} className="chord-fret" />;
      })}
      {shape.frets.map((fret, index) => {
        const x = left + index * stringGap;
        if (fret === null) {
          return (
            <text key={index} x={x} y={top - 6} textAnchor="middle" className="chord-mute">
              ×
            </text>
          );
        }
        if (fret === 0) {
          return <circle key={index} cx={x} cy={top - 6} r={3} className="chord-open" />;
        }
        const relative = fret - shape.baseFret;
        if (relative < 0 || relative >= 4) return null;
        return (
          <circle
            key={index}
            cx={x}
            cy={top + relative * fretGap + fretGap / 2}
            r={dotRadius}
            className="chord-dot"
          />
        );
      })}
    </svg>
  );
}

export function PracticeSheet({
  id,
  hidden = false,
  analysis = null,
  tab = null,
  currentTime,
  playing = false,
  onSeek,
}: {
  id?: string;
  hidden?: boolean;
  analysis?: AnalysisData | null;
  tab?: TabData | null;
  currentTime: number;
  playing?: boolean;
  onSeek: (time: number) => void;
}) {
  const [panel, setPanel] = useState<PracticeTab>("chords");
  const [capo, setCapo] = useState(0);
  const [level, setLevel] = useState<ChordLevel>("medium");
  const [edits, setEdits] = useState<Record<number, string>>({});
  const [draft, setDraft] = useState<{ index: number; value: string } | null>(null);
  const [frame, setFrame] = useState(0);
  const cancelEdit = useRef(false);
  const indexRef = useRef(0);
  const stripRef = useRef<HTMLDivElement | null>(null);
  const userScrollAt = useRef(0);
  const programmatic = useRef(false);
  const timeRef = useRef(currentTime);
  timeRef.current = currentTime;

  const chords = analysis?.chords ?? [];
  const sections = analysis?.sections ?? [];
  const lyrics = analysis?.lyrics ?? [];
  const measures = tab?.measures ?? [];
  const nothing = chords.length === 0 && sections.length === 0 && lyrics.length === 0 && measures.length === 0;

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      setFrame((count) => (count + 1) % 1_000_000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const playhead = playing ? timeRef.current : currentTime;
  void frame;
  const symbolAt = (index: number, stored: string): string =>
    edits[index] ?? formatChordSymbol(stored);
  indexRef.current = advanceChordIndex(chords, indexRef.current, playhead);
  const activeIndex = chords.length ? indexRef.current : -1;
  const current = activeIndex >= 0 ? chords[activeIndex] : undefined;
  const sounding = current ? symbolAt(activeIndex, displaySymbol(current, level)) : "—";
  const diagramSymbol = sounding !== "—" ? capoDiagramSymbol(sounding, capo) : "";

  useEffect(() => {
    const strip = stripRef.current;
    const chip = strip?.querySelector<HTMLElement>(`[data-chord-index="${activeIndex}"]`);
    if (!strip || !chip) return;
    if (performance.now() - userScrollAt.current < 3000) return;
    const left = scrollLeftToCenter(strip, chip);
    if (Math.abs(strip.scrollLeft - left) <= 1) return;
    programmatic.current = true;
    strip.scrollLeft = left;
    programmatic.current = false;
  }, [activeIndex, chords, panel]);
  const section = sectionAt(sections, playhead);
  const line = lyricLineAt(lyrics, playhead);
  const measure = measureAt(measures, playhead);
  const nextMeasure =
    measure && measures.length
      ? measures[Math.min(measures.indexOf(measure) + 1, measures.length - 1)]
      : null;

  const commit = (index: number, typed: string, stored: string) => {
    if (cancelEdit.current) {
      cancelEdit.current = false;
      setDraft(null);
      return;
    }
    setDraft(null);
    const next = replaceChordSymbol(edits[index] ?? stored, typed);
    if (!next) return;
    setEdits((prev) => ({ ...prev, [index]: next }));
  };

  let body: ReactNode;
  if (nothing) {
    body = <p className="practice-empty">{EMPTY_ALL}</p>;
  } else if (panel === "chords") {
    body = chords.length ? (
      <div className="chord-sheet">
        <div className="chord-now">
          <div className="chord-now-symbol">{sounding}</div>
          {diagramSymbol ? <Diagram symbol={diagramSymbol} /> : null}
          <div className="chord-now-meta">
            <span>{section?.label ?? "—"}</span>
            <label className="capo-label">
              <span className="visually-hidden">Capo fret</span>
              <select
                aria-label="Capo fret"
                value={String(capo)}
                onChange={(event) =>
                  setCapo(Math.max(0, Math.min(11, Math.round(Number(event.target.value) || 0))))
                }
              >
                {Array.from({ length: 12 }, (_, fret) => (
                  <option key={fret} value={fret}>
                    {capoLabel(fret)}
                  </option>
                ))}
              </select>
            </label>
            <label className="capo-label">
              <span className="visually-hidden">Chord detail</span>
              <select
                id="chord-level"
                aria-label="Chord detail"
                value={level}
                onChange={(event) => {
                  const next = event.target.value;
                  setLevel(next === "simple" || next === "medium" ? next : "chart");
                }}
              >
                <option value="simple">Simple</option>
                <option value="medium">Medium</option>
                <option value="chart">Chart</option>
              </select>
            </label>
          </div>
        </div>
        <div
          className="chord-strip"
          ref={stripRef}
          onScroll={() => {
            if (programmatic.current) return;
            userScrollAt.current = performance.now();
          }}
        >
          {chords.map((chord, index) => {
            const shown =
              draft?.index === index
                ? draft.value
                : symbolAt(index, displaySymbol(chord, level));
            return (
              <div
                className={index === activeIndex ? "chord-chip active" : "chord-chip"}
                data-chord-index={index}
                key={`${chord.start}-${index}`}
              >
                <input
                  className="chord-chip-symbol"
                  aria-label="Chord symbol"
                  value={shown}
                  onChange={(event) => setDraft({ index, value: event.target.value })}
                  onBlur={(event) => {
                    const baseline = symbolAt(index, displaySymbol(chord, level));
                    if (event.currentTarget.value.trim() === baseline) return;
                    commit(index, event.currentTarget.value, chord.symbol);
                  }}
                  onKeyDown={(event) => {
                    if (event.key !== "Escape") return;
                    event.preventDefault();
                    cancelEdit.current = true;
                    setDraft(null);
                    event.currentTarget.blur();
                  }}
                />
                <button type="button" onClick={() => onSeek(chord.start)}>
                  {clock(chord.start)}
                </button>
              </div>
            );
          })}
        </div>
        <p className="practice-note">{CHORD_DEMO_NOTE}</p>
      </div>
    ) : (
      <p className="practice-empty">{EMPTY_CHORDS}</p>
    );
  } else if (panel === "sections") {
    body = sections.length ? (
      <div className="section-list">
        {sections.map((item) => (
          <button type="button" key={`${item.start}-${item.label}`} onClick={() => onSeek(item.start)}>
            <span>{clock(item.start)}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    ) : (
      <p className="practice-empty">{EMPTY_SECTIONS}</p>
    );
  } else if (panel === "lyrics") {
    body = lyrics.length ? (
      <div className="lyrics-list">
        {lyrics.map((item) => (
          <button type="button" key={`${item.start}-${item.text}`} onClick={() => onSeek(item.start)}>
            {item.text}
          </button>
        ))}
        {line ? <p className="visually-hidden">{line.text}</p> : null}
      </div>
    ) : (
      <p className="practice-empty">{EMPTY_LYRICS}</p>
    );
  } else {
    body = measures.length ? (
      <div className="tab-measure-wrap">
        <div>Measure</div>
        <pre className="tab-measure">{measure?.text ?? ""}</pre>
        <div>Next</div>
        <pre className="tab-measure">{nextMeasure && nextMeasure !== measure ? nextMeasure.text : ""}</pre>
      </div>
    ) : (
      <p className="practice-empty">{EMPTY_TAB}</p>
    );
  }

  const summary = analysis
    ? [analysis.bpm ? `${Math.round(Number(analysis.bpm))} BPM` : "", analysis.key ?? ""]
        .filter(Boolean)
        .join(" · ")
    : EMPTY_ANALYSIS;

  return (
    <section id={id} className="practice-sheet" hidden={hidden} aria-label="Practice">
      <div className="practice-sheet-head">
        <h3>Practice</h3>
        <p>{summary}</p>
      </div>
      <div className="practice-tabs" role="tablist" aria-label="Practice view">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={panel === item.id}
            onClick={() => setPanel(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="practice-sheet-body">{body}</div>
    </section>
  );
}

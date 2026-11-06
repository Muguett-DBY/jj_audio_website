/** Pure practice-panel helpers. No DOM, no audio — safe to unit test later. */

export type ChordLevel = "simple" | "medium" | "chart";

export type ChordEvent = {
  start: number;
  end: number;
  symbol: string;
  confidence?: number;
  simple?: string;
  medium?: string;
};

export type Section = {
  start: number;
  end: number;
  label: string;
  confidence?: number;
};

export type LyricWord = { start: number; end: number; text: string };
export type LyricLine = {
  start: number;
  end: number;
  text: string;
  words?: LyricWord[];
};

export type Measure = { start: number; end: number; text: string };

export type AnalysisData = {
  bpm?: number | null;
  bpm_confidence?: string;
  key?: string | null;
  key_confidence?: string;
  beats?: number[];
  downbeats?: number[];
  beats_per_measure?: number;
  chords?: ChordEvent[];
  chart?: unknown;
  sections?: Section[];
  lyrics?: LyricLine[];
  lyrics_language?: string | null;
  warnings?: string[];
};

export type TabData = {
  document?: unknown;
  measures?: Measure[];
};

function finite(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Event covering ``t``; tolerant of gaps (returns null) and unsorted input. */
function eventAt<T extends { start: number; end: number }>(
  events: T[] | undefined,
  t: number
): T | null {
  if (!Array.isArray(events) || !Number.isFinite(t)) return null;
  for (const event of events) {
    const start = finite(event.start, Number.NaN);
    const end = finite(event.end, Number.NaN);
    if (Number.isFinite(start) && Number.isFinite(end) && t >= start && t < end) {
      return event;
    }
  }
  return null;
}

export function chordAt(
  chords: ChordEvent[] | undefined,
  t: number
): ChordEvent | null {
  return eventAt(chords, t);
}

export function sectionAt(
  sections: Section[] | undefined,
  t: number
): Section | null {
  return eventAt(sections, t);
}

export function lyricLineAt(
  lines: LyricLine[] | undefined,
  t: number
): LyricLine | null {
  return eventAt(lines, t);
}

export function measureAt(
  measures: Measure[] | undefined,
  t: number
): Measure | null {
  if (!Array.isArray(measures) || measures.length === 0 || !Number.isFinite(t)) {
    return null;
  }
  for (const measure of measures) {
    if (t >= finite(measure.start) && t < finite(measure.end)) return measure;
  }
  return t >= finite(measures[measures.length - 1]!.end)
    ? measures[measures.length - 1]!
    : null;
}

/** Pretty display for canonical symbols; leaves already-readable symbols alone. */
export function formatChordSymbol(raw: unknown): string {
  const text = String(raw ?? "").trim();
  if (!text || text.toUpperCase() === "N") return "—";
  return text
    .replace(/:maj7\b/g, "maj7")
    .replace(/:maj\b/g, "")
    .replace(/:min7\b/g, "m7")
    .replace(/:min\b/g, "m")
    .replace(/:sus4\b/g, "sus4")
    .replace(/:dim\b/g, "dim")
    .replace(/:aug\b/g, "aug")
    .replace(/:7\b/g, "7");
}

/** Clamp an A/B region to the song; null when unusable (min 0.5 s). */
export function clampLoop(
  start: number,
  end: number,
  duration: number
): { start: number; end: number } | null {
  const dur = finite(duration, 0);
  if (!(dur > 0)) return null;
  const s = Math.max(0, Math.min(dur, finite(start, Number.NaN)));
  const e = Math.max(0, Math.min(dur, finite(end, Number.NaN)));
  if (!Number.isFinite(s) || !Number.isFinite(e) || e - s < 0.5) return null;
  return { start: s, end: e };
}

const ROOT_PC: Record<string, number> = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
};

type Quality =
  | "maj"
  | "min"
  | "7"
  | "maj7"
  | "m7"
  | "sus4"
  | "dim"
  | "aug";

const QUALITY_FROM_SUFFIX: Record<string, Quality> = {
  "": "maj",
  m: "min",
  "7": "7",
  maj7: "maj7",
  m7: "m7",
  sus4: "sus4",
  dim: "dim",
  aug: "aug",
};

const E_SHAPES: Partial<Record<Quality, Array<number | null>>> = {
  maj: [0, 2, 2, 1, 0, 0],
  min: [0, 2, 2, 0, 0, 0],
  "7": [0, 2, 0, 1, 0, 0],
  maj7: [0, 2, 1, 1, 0, 0],
  m7: [0, 2, 0, 0, 0, 0],
  sus4: [0, 2, 2, 2, 0, 0],
};

const A_SHAPES: Partial<Record<Quality, Array<number | null>>> = {
  maj: [null, 0, 2, 2, 2, 0],
  min: [null, 0, 2, 2, 1, 0],
  "7": [null, 0, 2, 0, 2, 0],
  maj7: [null, 0, 2, 1, 2, 0],
  m7: [null, 0, 2, 0, 1, 0],
  sus4: [null, 0, 2, 2, 3, 0],
};

/** Common open voicings, preferred over a needless barre. */
const OPEN_SHAPES: Record<string, Array<number | null>> = {
  C: [null, 3, 2, 0, 1, 0],
  Cm: [null, 3, 5, 5, 4, 3],
  C7: [null, 3, 2, 3, 1, 0],
  D: [null, null, 0, 2, 3, 2],
  Dm: [null, null, 0, 2, 3, 1],
  D7: [null, null, 0, 2, 1, 2],
  E: [0, 2, 2, 1, 0, 0],
  Em: [0, 2, 2, 0, 0, 0],
  E7: [0, 2, 0, 1, 0, 0],
  G: [3, 2, 0, 0, 0, 3],
  Gm: [3, 5, 5, 3, 3, 3],
  G7: [3, 2, 0, 0, 0, 1],
  A: [null, 0, 2, 2, 2, 0],
  Am: [null, 0, 2, 2, 1, 0],
  A7: [null, 0, 2, 0, 2, 0],
  F: [1, 3, 3, 2, 1, 1],
  Fm: [1, 3, 3, 1, 1, 1],
  F7: [1, 3, 1, 2, 1, 1],
  B: [null, 2, 4, 4, 4, 2],
  Bm: [null, 2, 4, 4, 3, 2],
  B7: [null, 2, 1, 2, 0, 2],
};

export type ChordShape = {
  /** Low-E to high-E; null = muted, 0 = open. */
  frets: Array<number | null>;
  /** Lowest fret shown when the shape is a barre (1 for open shapes). */
  baseFret: number;
};

/** Shared SVG metrics for practice chord diagrams (desktop mixer + website). */
export const CHORD_DIAGRAM_LAYOUT = {
  width: 108,
  height: 112,
  /** Left edge of the fretboard; leaves a gutter for the base-fret label. */
  left: 26,
  right: 98,
  top: 20,
  bottom: 100,
  /** text-anchor=end x for the base-fret number beside the first fret space. */
  fretLabelX: 18,
  dotRadius: 4.2,
} as const;

function parseChord(
  symbol: string
): { rootPc: number; quality: Quality; head: string } | null {
  const head = symbol.split("/")[0]!.trim();
  const match = head.match(/^([A-G][#b]?)(.*)$/);
  if (!match) return null;
  const pc = ROOT_PC[match[1]!];
  if (pc === undefined) return null;
  const quality = QUALITY_FROM_SUFFIX[match[2]!.toLowerCase()];
  if (!quality) return null;
  return { rootPc: pc, quality, head };
}

/**
 * Guitar shape for a canonical chord symbol. Tries open voicings first, then
 * E- and A-shape barres. Returns null for qualities we do not draw (dim/aug).
 */
export function chordShape(symbol: string): ChordShape | null {
  const parsed = parseChord(formatChordSymbol(symbol));
  if (!parsed) return null;
  const open = OPEN_SHAPES[parsed.head];
  if (open) {
    const fretted = open.filter((value): value is number => value !== null && value > 0);
    const baseFret =
      open.includes(0) || fretted.length === 0 ? 1 : Math.max(1, Math.min(...fretted));
    return { frets: open.slice(), baseFret };
  }

  const eShape = E_SHAPES[parsed.quality];
  const aShape = A_SHAPES[parsed.quality];
  if (!eShape && !aShape) return null;

  const eFret = (parsed.rootPc - 4 + 12) % 12;
  const aFret = (parsed.rootPc - 9 + 12) % 12;
  const useE = eShape && (!aShape || eFret <= aFret);
  const shape = useE ? eShape : aShape;
  const fret = useE ? eFret : aFret;
  if (!shape) return null;
  const frets = shape.map((value) =>
    value === null ? null : value === 0 ? fret : value + fret
  );
  return { frets, baseFret: fret > 0 ? fret : 1 };
}

const PC_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const SUFFIX_FROM_QUALITY: Record<Quality, string> = {
  maj: "",
  min: "m",
  "7": "7",
  maj7: "maj7",
  m7: "m7",
  sus4: "sus4",
  dim: "dim",
  aug: "aug",
};

function capoFret(capo: number): number {
  return Math.max(0, Math.min(11, Math.round(Number(capo) || 0)));
}

function transposePitch(pc: number, semitones: number): string {
  const steps = ((semitones % 12) + 12) % 12;
  return PC_NAME[(pc + steps) % 12]!;
}

/** Sounding symbol transposed by semitones. Unknown text is returned unchanged. */
export function transposeChordSymbol(symbol: string, semitones: number): string {
  const formatted = formatChordSymbol(symbol);
  if (formatted === "—") return formatted;
  const [head, bass] = formatted.split("/");
  const parsed = parseChord(head ?? "");
  if (!parsed) return formatted;
  const nextHead =
    transposePitch(parsed.rootPc, semitones) + SUFFIX_FROM_QUALITY[parsed.quality];
  if (!bass) return nextHead;
  const bassPc = ROOT_PC[bass.trim()];
  if (bassPc === undefined) return nextHead;
  return `${nextHead}/${transposePitch(bassPc, semitones)}`;
}

/** Shape to draw above the capo. The stored chord name does not change. */
export function capoDiagramSymbol(symbol: string, capo: number): string {
  const fret = capoFret(capo);
  const formatted = formatChordSymbol(symbol);
  if (fret === 0) return formatted === "—" ? String(symbol ?? "") : formatted;
  return transposeChordSymbol(formatted, -fret);
}

export function capoLabel(capo: number): string {
  const fret = capoFret(capo);
  return fret === 0 ? "No capo" : `Capo ${fret}`;
}

/** Accept a typed chord, or null when it is not a symbol this sheet can draw. */
export function replaceChordSymbol(current: string, typed: string): string | null {
  void current;
  const formatted = formatChordSymbol(typed);
  if (!formatted || formatted === "—") return null;
  const [headRaw, bassRaw] = formatted.split("/");
  const headMatch = (headRaw ?? "").trim().match(/^([A-Ga-g])([#b]?)(.*)$/);
  if (!headMatch) return null;
  const head = headMatch[1]!.toUpperCase() + (headMatch[2] ?? "") + (headMatch[3] ?? "");
  if (!parseChord(head)) return null;
  if (!bassRaw) return head;
  const bassMatch = bassRaw.trim().match(/^([A-Ga-g])([#b]?)$/);
  if (!bassMatch) return null;
  return `${head}/${bassMatch[1]!.toUpperCase()}${bassMatch[2] ?? ""}`;
}

const PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

export type Spelling = "sharp" | "flat";

export function spellPitchClass(pc: number, spelling: Spelling): string {
  const names = spelling === "flat" ? PC_FLAT : PC_NAME;
  const index = ((Math.round(pc) % 12) + 12) % 12;
  return names[index]!;
}

const PC_FROM_NAME: Record<string, number> = {
  C: 0,
  "C#": 1,
  Db: 1,
  D: 2,
  "D#": 3,
  Eb: 3,
  E: 4,
  F: 5,
  "F#": 6,
  Gb: 6,
  G: 7,
  "G#": 8,
  Ab: 8,
  A: 9,
  "A#": 10,
  Bb: 10,
  B: 11,
};

export function spellChordSymbol(symbol: string, spelling: Spelling): string {
  const formatted = formatChordSymbol(symbol);
  if (formatted === "—") return formatted;
  const rewrite = (token: string): string => {
    const match = token.match(/^([A-Ga-g][#b]?)(.*)$/);
    if (!match) return token;
    const raw = match[1] ?? "";
    const name = raw.charAt(0).toUpperCase() + raw.slice(1);
    const pc = PC_FROM_NAME[name];
    if (pc === undefined) return token;
    return spellPitchClass(pc, spelling) + (match[2] ?? "");
  };
  const [head, bass] = formatted.split("/");
  const next = rewrite(head ?? "");
  return bass ? `${next}/${rewrite(bass)}` : next;
}

/** Easy is a triad or sus. Medium is a 6th or 7th. Everything else is Hard. */
export function chordDifficulty(symbol: string): "" | "Easy" | "Medium" | "Hard" {
  const formatted = formatChordSymbol(symbol);
  if (!formatted || formatted === "—") return "";
  const quality = (formatted.split("/")[0] ?? "").replace(/^[A-G][#b]?/, "");
  if (quality === "" || quality === "m" || quality === "5" || quality === "sus2" || quality === "sus4") {
    return "Easy";
  }
  if (["6", "m6", "7", "maj7", "m7", "add9"].includes(quality)) return "Medium";
  return "Hard";
}

export const CAPO_PRESETS = [
  { id: "open", label: "Open", fret: 0 },
  { id: "capo2", label: "Capo 2", fret: 2 },
  { id: "capo5", label: "Capo 5", fret: 5 },
  { id: "capo7", label: "Capo 7", fret: 7 },
] as const;

export function replaceSectionLabel(current: string, typed: string): string {
  const text = typed.replace(/\s+/g, " ").trim();
  return text || current;
}

export function replaceLyricLine(current: string, typed: string): string {
  const text = typed.replace(/\s+/g, " ").trim();
  return text || current;
}

export function replaceMeasureText(current: string, typed: string): string | null {
  void current;
  const text = typed.trim();
  if (!/^[\d\-xX|.\s]+$/.test(text)) return null;
  return text;
}

export function formatBassTab(pitches: number[]): string {
  const strings = [
    { name: "G", open: 43 },
    { name: "D", open: 38 },
    { name: "A", open: 33 },
    { name: "E", open: 28 },
  ];
  return strings
    .map((string) => {
      const frets = pitches.map((pitch) => {
        const fret = Math.round(pitch) - string.open;
        return fret >= 0 && fret <= 24 ? String(fret) : "-";
      });
      return `${string.name}|${frets.join("-")}`;
    })
    .join("\n");
}

export function listenCents(hz: number): number {
  if (!Number.isFinite(hz) || hz <= 0) return 0;
  return Math.round(1200 * Math.log2(hz / 440) * 100) / 100;
}

/** Output latency plus a user offset, clamped so a bad device cannot shove the chart. */
export function outputLatencyMs(
  ctx: { outputLatency?: number; baseLatency?: number } | null | undefined,
  userOffsetMs = 0
): number {
  const seconds = ctx?.outputLatency ?? ctx?.baseLatency ?? 0;
  const raw = (Number.isFinite(seconds) ? seconds : 0) * 1000 + (Number.isFinite(userOffsetMs) ? userOffsetMs : 0);
  return Math.min(250, Math.max(0, raw));
}

/** Acoustic time: the transport clock plus the time the speakers are behind it. */
export function chartTimeSeconds(audioTime: number, latencyMs: number): number {
  const time = Number.isFinite(audioTime) ? audioTime : 0;
  const latency = Number.isFinite(latencyMs) ? latencyMs : 0;
  return time + latency / 1000;
}

const CHORD_HYSTERESIS_SEC = 0.02;

/** Move forward only after the next chord has been current for 20 ms. Rewind on a seek. */
export function advanceChordIndex(
  chords: { start: number }[] | undefined,
  index: number,
  chartTime: number
): number {
  if (!Array.isArray(chords) || chords.length === 0 || !Number.isFinite(chartTime)) return 0;
  let cursor = Math.max(0, Math.min(Math.floor(index) || 0, chords.length - 1));
  const startAt = (i: number): number => finite(chords[i]?.start, Number.POSITIVE_INFINITY);
  if (chartTime < startAt(cursor)) cursor = 0;
  while (cursor + 1 < chords.length && chartTime >= startAt(cursor + 1) + CHORD_HYSTERESIS_SEC) {
    cursor += 1;
  }
  return cursor;
}

/** Scroll offset that puts the chip's centre on the strip's centre, clamped to the scroll range. */
export function scrollLeftToCenter(
  container: { clientWidth: number; scrollWidth: number },
  chip: { offsetLeft: number; offsetWidth: number }
): number {
  const view = Math.max(0, container.clientWidth);
  const target = chip.offsetLeft + chip.offsetWidth / 2 - view / 2;
  const max = Math.max(0, container.scrollWidth - view);
  return Math.max(0, Math.min(target, max));
}

/** Symbol for the selected display level. An empty reduced form falls back to the chart symbol. */
export function displaySymbol(
  chord: { symbol: string; simple?: string; medium?: string },
  level: ChordLevel
): string {
  if (level === "simple" && chord.simple) return chord.simple;
  if (level === "medium" && chord.medium) return chord.medium;
  return chord.symbol;
}

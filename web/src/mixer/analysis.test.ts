import { describe, expect, it } from "vitest";
import * as desktop from "../../../ui/stem_mixer_component/frontend/src/analysis";
import {
  advanceChordIndex,
  capoDiagramSymbol,
  capoLabel,
  chartTimeSeconds,
  CHORD_DIAGRAM_LAYOUT,
  chordDifficulty,
  chordShape,
  displaySymbol,
  formatBassTab,
  formatChordSymbol,
  listenCents,
  outputLatencyMs,
  replaceChordSymbol,
  replaceMeasureText,
  scrollLeftToCenter,
  spellChordSymbol,
} from "./analysis";

const samples = ["Am", "G", "F#m", "Bb", "Cmaj7", "Am/G", "N", "nope", "   "];

describe("website chord helpers match the desktop sheet", () => {
  it("draws G# as an E-shape barre at fret 4 with room for the fret label", () => {
    const shape = chordShape("G#");
    expect(shape).toEqual({ frets: [4, 6, 6, 5, 4, 4], baseFret: 4 });
    expect(desktop.chordShape("G#")).toEqual(shape);

    const layout = CHORD_DIAGRAM_LAYOUT;
    expect(desktop.CHORD_DIAGRAM_LAYOUT).toEqual(layout);
    // text-anchor=end: label's right edge is fretLabelX; leftmost barre dot
    // sits on string 0 at x=left with radius dotRadius.
    const clearance = layout.left - layout.dotRadius - layout.fretLabelX;
    expect(clearance).toBeGreaterThanOrEqual(3);
  });

  it("labels capo 0 as No capo and does not transpose the diagram", () => {
    expect(capoLabel(0)).toBe("No capo");
    expect(capoLabel(0)).toBe(desktop.capoLabel(0));
    expect(capoDiagramSymbol("Am", 0)).toBe(formatChordSymbol("Am"));
    expect(capoDiagramSymbol("Am", 0)).toBe(desktop.capoDiagramSymbol("Am", 0));
    expect(capoDiagramSymbol("G", 2)).toBe(desktop.capoDiagramSymbol("G", 2));
    expect(capoDiagramSymbol("G", 2)).not.toBe(formatChordSymbol("G"));
    for (const capo of [0, 1, 5, 11, 12, -1]) {
      expect(capoLabel(capo)).toBe(desktop.capoLabel(capo));
      for (const symbol of samples) {
        expect(capoDiagramSymbol(symbol, capo)).toBe(desktop.capoDiagramSymbol(symbol, capo));
      }
    }
  });

  it("accepts a typed symbol and rejects a blank or unusable one", () => {
    expect(replaceChordSymbol("Am", "C")).toBe("C");
    expect(replaceChordSymbol("Am", "C")).toBe(desktop.replaceChordSymbol("Am", "C"));
    expect(replaceChordSymbol("Am", "   ")).toBeNull();
    expect(replaceChordSymbol("Am", "")).toBeNull();
    expect(replaceChordSymbol("Am", "   ")).toBe(desktop.replaceChordSymbol("Am", "   "));
    for (const typed of ["F#m", "Bb", "Am/G", "nope", " ", "N"]) {
      expect(replaceChordSymbol("Am", typed)).toBe(desktop.replaceChordSymbol("Am", typed));
    }
  });

  it("spells flats, rates difficulty, formats bass, and offsets 432 Hz", () => {
    expect(spellChordSymbol("F#", "flat")).toBe("Gb");
    expect(spellChordSymbol("Bb", "sharp")).toBe("A#");
    expect(chordDifficulty("Am")).toBe("Easy");
    expect(chordDifficulty("Cmaj7")).toBe("Medium");
    expect(chordDifficulty("C9")).toBe("Hard");
    expect(formatBassTab([28])).toBe("G|-\nD|-\nA|-\nE|0");
    expect(listenCents(440)).toBe(0);
    expect(listenCents(432)).toBeLessThan(0);
    expect(replaceMeasureText("0-2", "1-3")).toBe("1-3");
    expect(replaceMeasureText("0-2", "hello")).toBeNull();
    expect(spellChordSymbol("F#", "flat")).toBe(desktop.spellChordSymbol("F#", "flat"));
    expect(formatBassTab([28, 43])).toBe(desktop.formatBassTab([28, 43]));
  });

  it("follows a click grid at 60 Hz with the active cell centred", () => {
    const chords = Array.from({ length: 40 }, (_, index) => ({
      start: index * 0.5,
      end: (index + 1) * 0.5,
      symbol: index % 2 === 0 ? "Cmaj7" : "G7",
      simple: index % 2 === 0 ? "C" : "G",
      medium: index % 2 === 0 ? "Cmaj7" : "G7",
    }));
    const latency = outputLatencyMs({ outputLatency: 0.04 }, 0);
    expect(latency).toBe(40);
    expect(latency).toBe(desktop.outputLatencyMs({ outputLatency: 0.04 }, 0));
    expect(outputLatencyMs({ outputLatency: 9 }, 0)).toBe(250);
    expect(displaySymbol(chords[0]!, "simple")).toBe("C");
    expect(displaySymbol(chords[0]!, "medium")).toBe(desktop.displaySymbol(chords[0]!, "medium"));
    expect(displaySymbol({ symbol: "Cmaj7" }, "simple")).toBe("Cmaj7");

    let index = 0;
    let desktopIndex = 0;
    const errors: number[] = [];
    const frame = 1 / 60;
    const seconds = 8;
    let frames = 0;
    for (let audio = 0; audio < seconds; audio += frame) {
      frames += 1;
      const chart = chartTimeSeconds(audio, latency);
      expect(chart).toBeCloseTo(desktop.chartTimeSeconds(audio, latency));
      index = advanceChordIndex(chords, index, chart);
      desktopIndex = desktop.advanceChordIndex(chords, desktopIndex, chart);
      expect(index).toBe(desktopIndex);
      const active = chords[index]!;
      const overhang =
        chart < active.start
          ? (active.start - chart) * 1000
          : chart >= active.end
            ? (chart - active.end) * 1000
            : 0;
      errors.push(overhang);
    }
    errors.sort((left, right) => left - right);
    const p95 = errors[Math.min(errors.length - 1, Math.floor(errors.length * 0.95))] ?? 0;
    expect(p95).toBeLessThanOrEqual(50);
    expect(frames / seconds).toBeGreaterThanOrEqual(30);
    expect(advanceChordIndex(chords, 12, 0.1)).toBe(0);

    const left = scrollLeftToCenter(
      { clientWidth: 1000, scrollWidth: 4000 },
      { offsetLeft: 2000, offsetWidth: 100 }
    );
    expect(left).toBe(desktop.scrollLeftToCenter(
      { clientWidth: 1000, scrollWidth: 4000 },
      { offsetLeft: 2000, offsetWidth: 100 }
    ));
    const centre = 2000 + 50 - left;
    expect(centre / 1000).toBeGreaterThanOrEqual(0.2);
    expect(centre / 1000).toBeLessThanOrEqual(0.8);
  });
});

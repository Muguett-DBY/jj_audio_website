import { act, type ReactNode } from "react";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { PracticeSheet } from "./PracticeSheet";
import type { AnalysisData, TabData } from "../mixer/analysis";

const analysis: AnalysisData = {
  bpm: 100,
  key: "Am",
  chords: [{ start: 0, end: 2, symbol: "Am" }],
  sections: [{ start: 0, end: 2, label: "Verse" }],
  lyrics: [{ start: 0, end: 2, text: "hello world" }],
};

const tab: TabData = {
  measures: [
    { start: 0, end: 2, text: "e|--0--|" },
    { start: 2, end: 4, text: "e|--3--|" },
  ],
};

let root: Root | null = null;
let host: HTMLDivElement | null = null;

async function render(node: ReactNode) {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(node);
  });
  return host;
}

afterEach(() => {
  if (root) {
    act(() => {
      root?.unmount();
    });
  }
  host?.remove();
  root = null;
  host = null;
});

function tabButton(scope: ParentNode, name: string): HTMLButtonElement {
  const button = [...scope.querySelectorAll('[role="tab"]')].find(
    (item) => item.textContent === name
  );
  if (!(button instanceof HTMLButtonElement)) throw new Error(`missing tab ${name}`);
  return button;
}

describe("practice sheet", () => {
  it("shows a chord, a section, a lyric line, and a tab measure", async () => {
    const scope = await render(
      <PracticeSheet analysis={analysis} tab={tab} currentTime={0.4} onSeek={() => {}} />
    );
    expect(scope.textContent).toContain("Am");
    expect(scope.textContent).toContain("Verse");
    await act(async () => {
      tabButton(scope, "Sections").click();
    });
    expect(scope.textContent).toContain("Verse");
    await act(async () => {
      tabButton(scope, "Lyrics").click();
    });
    expect(scope.textContent).toContain("hello world");
    await act(async () => {
      tabButton(scope, "Tab").click();
    });
    expect(scope.textContent).toContain("e|--0--|");
  });

  it("uses the empty copy when analysis and measures are absent", async () => {
    const scope = await render(<PracticeSheet currentTime={0} onSeek={() => {}} />);
    expect(scope.textContent).toContain("No analysis in this run.");
    expect(scope.textContent).toContain("No chords, sections, lyrics, or tab in this run.");
  });

  it("names the missing piece on that tab", async () => {
    const scope = await render(
      <PracticeSheet
        analysis={{ sections: [{ start: 0, end: 1, label: "Intro" }] }}
        currentTime={0}
        onSeek={() => {}}
      />
    );
    await act(async () => {
      tabButton(scope, "Chords").click();
    });
    expect(scope.textContent).toContain("No chords in this run.");
    await act(async () => {
      tabButton(scope, "Lyrics").click();
    });
    expect(scope.textContent).toContain(
      "No lyrics found (instrumental, or the lyrics model is unavailable)."
    );
    await act(async () => {
      tabButton(scope, "Tab").click();
    });
    expect(scope.textContent).toContain("No tab for this run. Re-run with guitar separation.");
    await act(async () => {
      tabButton(scope, "Sections").click();
    });
    expect(scope.textContent).toContain("Intro");
    expect(scope.textContent).not.toContain("No sections in this run.");
  });

  it("says there are no sections when that list is empty", async () => {
    const scope = await render(
      <PracticeSheet
        analysis={{ chords: [{ start: 0, end: 1, symbol: "C" }] }}
        currentTime={0}
        onSeek={() => {}}
      />
    );
    await act(async () => {
      tabButton(scope, "Sections").click();
    });
    expect(scope.textContent).toContain("No sections in this run.");
  });

  it("keeps the previous symbol when the typed chord is blank or cancelled", async () => {
    const scope = await render(
      <PracticeSheet analysis={analysis} currentTime={0.4} onSeek={() => {}} />
    );
    const input = scope.querySelector('input[aria-label="Chord symbol"]');
    if (!(input instanceof HTMLInputElement)) throw new Error("missing chord input");
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    if (!setValue) throw new Error("missing value setter");

    await act(async () => {
      input.focus();
      setValue.call(input, "   ");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.blur();
    });
    expect(input.value).toBe("Am");

    await act(async () => {
      input.focus();
      setValue.call(input, "C");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(input.value).toBe("Am");

    await act(async () => {
      input.focus();
      setValue.call(input, "C");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.blur();
    });
    expect(input.value).toBe("C");
    expect(scope.textContent).toContain("C");
  });

  it("labels the chord chart as a demo work in progress", async () => {
    const scope = await render(
      <PracticeSheet analysis={analysis} currentTime={0.4} onSeek={() => {}} />
    );
    expect(scope.textContent).toContain(
      "Demo — chord chart is a work in progress; roots can drift on dense mixes — verify by ear."
    );
  });

  it("draws the stored chord at capo 0", async () => {
    const scope = await render(
      <PracticeSheet analysis={analysis} currentTime={0.4} onSeek={() => {}} />
    );
    const diagram = scope.querySelector("[data-diagram-symbol]");
    expect(diagram?.getAttribute("data-diagram-symbol")).toBe("Am");
    const select = scope.querySelector('select[aria-label="Capo fret"]');
    if (!(select instanceof HTMLSelectElement)) throw new Error("missing capo");
    expect(select.value).toBe("0");
    expect(select.options[0]?.textContent).toBe("No capo");
  });
});

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { ThemeSelector } from "../components/ThemeSelector";
import { ThemeProvider } from "./ThemeContext";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

describe("ThemeProvider", () => {
  let root: Root | null = null;
  let host: HTMLDivElement | null = null;

  afterEach(() => {
    act(() => root?.unmount());
    host?.remove();
    root = null;
    host = null;
    try {
      localStorage.clear();
    } catch {
      /* Node can expose a disabled localStorage stub. */
    }
  });

  it("renders ThemeSelector on the first paint", () => {
    window.matchMedia = (query: string) =>
      ({
        matches: false,
        media: query,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
        onchange: null,
      }) as MediaQueryList;
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    expect(() => {
      act(() => {
        root?.render(
          <ThemeProvider>
            <ThemeSelector />
          </ThemeProvider>
        );
      });
    }).not.toThrow();
    expect(host.textContent).toContain("Theme Mode");
  });
});

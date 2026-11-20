import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf-8");

describe("field layout", () => {
  it("keeps text fields stacked and puts checkbox and radio labels in a row", () => {
    expect(css).toMatch(/\.field\s*\{[^}]*flex-direction:\s*column/);
    expect(css).toContain('label.field:has(> input[type="checkbox"])');
    expect(css).toContain('label.field:has(> input[type="radio"])');
    const row = css.slice(css.indexOf('label.field:has(> input[type="checkbox"])'));
    expect(row.slice(0, 400)).toMatch(/flex-direction:\s*row/);
    expect(row.slice(0, 400)).toMatch(/align-items:\s*flex-start/);
  });

  it("uses the theme secondary color for hints and ledes", () => {
    expect(css).toMatch(/\.hint\s*\{[^}]*color:\s*var\(--color-text-secondary\)/);
    expect(css).toMatch(/\.lede\s*\{[^}]*color:\s*var\(--color-text-secondary\)/);
    expect(css).not.toMatch(/\.hint\s*\{[^}]*rgba\(26,\s*31,\s*28/);
    expect(css).not.toMatch(/\.lede\s*\{[^}]*rgba\(26,\s*31,\s*28/);
  });
});

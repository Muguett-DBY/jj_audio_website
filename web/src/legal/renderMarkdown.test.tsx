import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./renderMarkdown";

describe("renderMarkdown headings", () => {
  it("renders a third-level heading instead of the hash marks", () => {
    const nodes = renderMarkdown("### Website\n\n### Desktop\n");
    expect(nodes.map((node) => (node as { type: string }).type)).toEqual(["h3", "h3"]);
    expect(nodes.map((node) => (node as { props: { children: string[] } }).props.children)).toEqual([
      ["Website"],
      ["Desktop"],
    ]);
  });

  it("keeps first and second level headings", () => {
    const nodes = renderMarkdown("# Privacy policy\n\n## What Audio Tools is\n");
    expect(nodes.map((node) => (node as { type: string }).type)).toEqual(["h1", "h2"]);
  });
});

import type { ReactNode } from "react";

/** Minimal markdown renderer for legal drafts (headings, lists, paragraphs, links, code). */
export function renderMarkdown(md: string): ReactNode[] {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const nodes: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    if (line.startsWith("### ")) {
      nodes.push(<h3 key={key++}>{inline(line.slice(4))}</h3>);
      i += 1;
      continue;
    }
    if (line.startsWith("## ")) {
      nodes.push(<h2 key={key++}>{inline(line.slice(3))}</h2>);
      i += 1;
      continue;
    }
    if (line.startsWith("# ")) {
      nodes.push(<h1 key={key++}>{inline(line.slice(2))}</h1>);
      i += 1;
      continue;
    }
    if (line.startsWith("|")) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        rows.push(lines[i]);
        i += 1;
      }
      nodes.push(renderTable(rows, key++));
      continue;
    }
    if (/^[-*] /.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        items.push(lines[i].replace(/^[-*] /, ""));
        i += 1;
      }
      nodes.push(
        <ul key={key++}>
          {items.map((item, idx) => (
            <li key={idx}>{inline(item)}</li>
          ))}
        </ul>
      );
      continue;
    }
    const para: string[] = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].startsWith("#") &&
      !lines[i].startsWith("|") &&
      !/^[-*] /.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    nodes.push(<p key={key++}>{inline(para.join(" "))}</p>);
  }
  return nodes;
}

function renderTable(rows: string[], key: number): ReactNode {
  const parse = (row: string) =>
    row
      .split("|")
      .slice(1, -1)
      .map((c) => c.trim());
  const bodyStart = rows.length > 1 && /^\|?\s*-+/.test(rows[1]) ? 2 : 1;
  const head = parse(rows[0]);
  const body = rows.slice(bodyStart).map(parse);
  return (
    <div className="legal-table-wrap" key={key}>
      <table className="legal-table">
        <thead>
          <tr>
            {head.map((c, i) => (
              <th key={i}>{inline(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri}>
              {row.map((c, ci) => (
                <td key={ci}>{inline(c)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function inline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const token = m[0];
    if (token.startsWith("**")) {
      parts.push(<strong key={k++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      parts.push(<code key={k++}>{token.slice(1, -1)}</code>);
    } else {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      if (link) {
        const href = link[2];
        const internalMap: Record<string, string> = {
          "./privacy.md": "/legal/privacy",
          "./terms.md": "/legal/terms",
          "./refund.md": "/legal/refund",
          "./cookies.md": "/legal/cookies",
          "./NOTICES.md": "/legal/notices",
          "./THIRD_PARTY_SDKS.md": "/legal/third-party",
        };
        const mapped = internalMap[href];
        parts.push(
          <a
            key={k++}
            href={mapped ?? href}
            {...(mapped
              ? {}
              : { target: "_blank", rel: "noopener noreferrer" })}
          >
            {link[1]}
          </a>
        );
      }
    }
    last = m.index + token.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

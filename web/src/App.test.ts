import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const webRoot = process.cwd();

function readSource(rel: string): string {
  return readFileSync(resolve(webRoot, rel), "utf-8");
}

describe("App routing (I-601)", () => {
  it("routes /login to the LoginPage component", () => {
    const app = readSource("src/App.tsx");
    expect(app).toContain('path="/login"');
    expect(app).toContain("<LoginPage");
  });

  it("Shell shows Sign in / Sign out and the session email", () => {
    const app = readSource("src/App.tsx");
    expect(app).toMatch(/Sign (in|out)/);
    expect(app).toContain("user-info");
    expect(app).toContain("handleAuthClick");
  });

  it("leaves the Sign in control on screen when anonymous sessions are off", () => {
    const app = readSource("src/App.tsx");
    expect(app).toContain("sessionError !== ANON_SESSION_UNAVAILABLE");
    expect(app).toContain("handleAuthClick");
  });

  it("LoginPage navigates to /isolate after sign-in, not /tab", () => {
    const login = readSource("src/pages/LoginPage.tsx");
    expect(login).toContain('navigate("/isolate")');
    expect(login).not.toContain('navigate("/tab")');
  });

  it("LoginPage redirects already-authenticated users to /isolate", () => {
    const login = readSource("src/pages/LoginPage.tsx");
    expect(login).toContain('<Navigate to="/isolate"');
  });

  it("routes /legal/:docId and mounts cookie banner + footer", () => {
    const app = readSource("src/App.tsx");
    expect(app).toContain('path="/legal/:docId"');
    expect(app).toContain("<LegalPage");
    expect(app).toContain("<CookieBanner");
    expect(app).toContain("<SiteFooter");
    expect(app).toContain("skip-link");
  });

  it("opens the shell without a Loading gate", () => {
    const app = readSource("src/App.tsx");
    expect(app).not.toContain("Loading…");
    expect(app).toContain("app-shell");
    const html = readSource("index.html");
    expect(html).toContain('translate="no"');
    expect(html).toContain('name="google" content="notranslate"');
  });

  it("LoginPage requires unchecked Terms/Privacy consent", () => {
    const login = readSource("src/pages/LoginPage.tsx");
    expect(login).toContain("acceptTerms");
    expect(login).not.toContain("ageOk");
    expect(login).toContain("useState(false)");
    expect(login).toContain("/legal/privacy");
  });
});

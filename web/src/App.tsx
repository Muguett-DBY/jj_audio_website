import { NavLink, Navigate, Outlet, Route, Routes, useNavigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth";
import { CookieBanner } from "./components/CookieBanner";
import { SiteFooter } from "./components/SiteFooter";
import { IsolatePage } from "./pages/IsolatePage";
import { LegalPage } from "./pages/LegalPage";
import { LoginPage } from "./pages/LoginPage";
import { TabPage } from "./pages/TabPage";
import { SettingsPage } from "./pages/SettingsPage";

function Shell() {
  const { loading, sessionError, retrySession, email, logout } = useAuth();
  const navigate = useNavigate();

  const handleAuthClick = () => {
    if (email) {
      void logout();
    } else {
      void navigate("/login");
    }
  };

  if (!loading && sessionError) {
    return (
      <main>
        <p className="error">{sessionError}</p>
        <p className="hint">Could not start a session. Check that the API is running, then retry.</p>
        <button type="button" onClick={() => void retrySession()}>
          Retry
        </button>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="topbar">
        <NavLink to="/isolate" className="brand">
          Audio Tools
        </NavLink>
        <nav className="nav" aria-label="Primary">
          <NavLink to="/tab" className={({ isActive }) => (isActive ? "active" : "")}>
            Tab PDF
          </NavLink>
          <NavLink
            to="/isolate"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            Isolate
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            Settings
          </NavLink>
          <NavLink
            to="/legal/privacy"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            Legal
          </NavLink>
          <span className="user-info">
            {email ? `Signed in as ${email}` : "Anonymous"}
          </span>
          <button
            type="button"
            className="secondary"
            onClick={handleAuthClick}
          >
            {email ? "Sign out" : "Sign in"}
          </button>
          <a
            href="https://github.com/notjj1234/jj_audio_website/issues"
            target="_blank"
            rel="noopener noreferrer"
          >
            Send feedback
          </a>
        </nav>
      </header>
      <main id="main-content">
        <Outlet />
      </main>
      <SiteFooter />
      <CookieBanner />
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/tab" element={<TabPage />} />
          <Route path="/isolate" element={<IsolatePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/legal" element={<Navigate to="/legal/privacy" replace />} />
          <Route path="/legal/:docId" element={<LegalPage />} />
          <Route path="*" element={<Navigate to="/isolate" replace />} />
        </Route>
      </Routes>
    </AuthProvider>
  );
}

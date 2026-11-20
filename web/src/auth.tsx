import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import * as api from "./api";

type AuthState = {
  email: string | null;
  loading: boolean;
  sessionError: string | null;
  retrySession: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

export const ANON_SESSION_UNAVAILABLE =
  "Sign in to continue. This server is not accepting anonymous sessions.";

export const API_DOWN_HINT =
  "Could not start a session. Check that the API is running, then retry.";

export function sessionFailureMessage(err: unknown): string {
  if (err instanceof api.ApiError && err.status === 404) return ANON_SESSION_UNAVAILABLE;
  if (err instanceof Error && err.message) return err.message;
  return "Could not start a session.";
}

export function sessionFailureHint(message: string | null): string {
  if (message === ANON_SESSION_UNAVAILABLE) return ANON_SESSION_UNAVAILABLE;
  return API_DOWN_HINT;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [email, setEmail] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionError, setSessionError] = useState<string | null>(null);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    setSessionError(null);
    const token = api.getAccessToken();
    if (token) {
      try {
        const u = await api.me();
        setEmail(u.email);
        setLoading(false);
        return;
      } catch {
        api.setAccessToken(null);
      }
    }
    try {
      const res = await api.startAnonymousSession();
      setEmail(res.email);
    } catch (err) {
      setEmail(null);
      setSessionError(sessionFailureMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  const login = useCallback(async (e: string, password: string) => {
    const res = await api.login(e, password);
    setEmail(res.email);
    setSessionError(null);
  }, []);

  const logout = useCallback(async () => {
    await api.logout();
    setEmail(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        email,
        loading,
        sessionError,
        retrySession: bootstrap,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}

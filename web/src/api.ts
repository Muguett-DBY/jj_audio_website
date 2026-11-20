export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export type JobKind = "tab" | "isolate";

export type JobResponse = {
  id: string;
  status: JobStatus;
  kind: JobKind;
  stage: string;
  message: string;
  error: string | null;
  artifacts: Record<string, string>;
  title?: string | null;
  /** 0–1 or 0–100 when the job payload includes it. Absent today. */
  percent?: number | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const TOKEN_KEY = "att_access_token";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getAccessToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string | null): void {
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  else sessionStorage.removeItem(TOKEN_KEY);
}

async function api<T>(
  path: string,
  init: RequestInit = {},
  auth = true
): Promise<T> {
  const headers = new Headers(init.headers);
  if (auth) {
    const token = getAccessToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers, credentials: "include" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* */
    }
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<{ email: string }> {
  const data = await api<{ access_token: string; email: string }>(
    "/v1/auth/login",
    { method: "POST", body: JSON.stringify({ email, password }) },
    false
  );
  setAccessToken(data.access_token);
  return { email: data.email };
}

export async function startAnonymousSession(): Promise<{ email: string }> {
  const data = await api<{ access_token: string; email: string }>(
    "/v1/auth/session",
    { method: "POST" },
    false
  );
  setAccessToken(data.access_token);
  return { email: data.email };
}

export async function logout(): Promise<void> {
  try {
    await api("/v1/auth/logout", { method: "POST" });
  } finally {
    setAccessToken(null);
  }
}

export async function me(): Promise<{ id: string; email: string }> {
  return api("/v1/auth/me");
}

export async function deleteAccount(): Promise<{
  status: string;
  deleted_jobs: number;
  deleted_artifacts: number;
  deleted_uploads: number;
}> {
  return api("/v1/auth/me", { method: "DELETE" });
}

export async function uploadAudio(file: File): Promise<{ upload_id: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  return api("/v1/uploads/audio", { method: "POST", body: form });
}

export async function createTabJob(body: Record<string, unknown>): Promise<JobResponse> {
  return api("/v1/jobs", { method: "POST", body: JSON.stringify(body) });
}

export async function createIsolateJob(body: Record<string, unknown>): Promise<JobResponse> {
  return api("/v1/isolate/jobs", { method: "POST", body: JSON.stringify(body) });
}

export async function getJob(jobId: string): Promise<JobResponse> {
  return api(`/v1/jobs/${jobId}`);
}

export async function listJobs(opts?: {
  kind?: JobKind;
  limit?: number;
}): Promise<JobResponse[]> {
  const params = new URLSearchParams();
  if (opts?.kind) params.set("kind", opts.kind);
  if (opts?.limit != null) params.set("limit", String(opts.limit));
  const qs = params.toString();
  return api(`/v1/jobs${qs ? `?${qs}` : ""}`);
}

export type ProcessingModeInfo = {
  id: string;
  label: string;
  enabled: boolean;
  reason: string | null;
  device: string;
  max_duration_sec: number;
};

export type SystemCapabilities = {
  device_options: string[];
  recommended_mode: string;
  detected_device: string;
  ram_gb: number | null;
  notes: string;
  low_ram: boolean;
  allow_youtube: boolean;
  modes: ProcessingModeInfo[];
};

export async function getSystemCapabilities(): Promise<SystemCapabilities> {
  return api("/v1/system/capabilities", {}, false);
}

export async function cancelJob(jobId: string): Promise<JobResponse> {
  return api(`/v1/jobs/${jobId}/cancel`, { method: "POST" });
}

export type ProgressHandler = (job: JobResponse) => void;

/** WebSocket progress with 2s polling fallback. */
export function watchJob(jobId: string, onUpdate: ProgressHandler): () => void {
  let stopped = false;
  let ws: WebSocket | null = null;
  let pollTimer: number | null = null;

  const poll = async () => {
    if (stopped) return;
    try {
      const job = await getJob(jobId);
      onUpdate(job);
      if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        stop();
        return;
      }
    } catch {
      /* keep polling */
    }
    pollTimer = window.setTimeout(poll, 2000);
  };

  const token = getAccessToken();
  const qs = token ? `?access_token=${encodeURIComponent(token)}` : "";
  const wsUrl = API_BASE
    ? `${API_BASE.replace(/^http/, "ws")}/v1/jobs/${jobId}/ws${qs}`
    : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/jobs/${jobId}/ws${qs}`;
  try {
    ws = new WebSocket(wsUrl);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "heartbeat") return;
        onUpdate(data as JobResponse);
        if (["succeeded", "failed", "cancelled"].includes(data.status)) stop();
      } catch {
        /* */
      }
    };
    ws.onerror = () => {
      /* fall through to poll */
    };
    ws.onclose = () => {
      if (!stopped) void poll();
    };
  } catch {
    void poll();
  }

  // Always start polling as mobile fallback
  pollTimer = window.setTimeout(poll, 2000);

  function stop() {
    stopped = true;
    if (ws) {
      try {
        ws.close();
      } catch {
        /* */
      }
      ws = null;
    }
    if (pollTimer !== null) window.clearTimeout(pollTimer);
  }

  return stop;
}

"""Desktop launcher: native window over a local Streamlit server (dev or PyInstaller freeze)."""

from __future__ import annotations

import inspect
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from audio_to_tab.edition import (  # noqa: E402
    desktop_edition,
    edition_app_name,
    edition_window_title,
)

DEFAULT_PORT = 8501
PORT_FALLBACKS = (8501, 8502, 8503, 8504, 8505)
LOCK_FILENAME = "launcher.lock"
# Frozen first-start imports Torch/Streamlit; 90s is tight on Windows HDD / Intel Mac.
HEALTH_TIMEOUT_SEC = 180.0
HEALTH_POLL_SEC = 0.3
# Diagnostic only: how long to wait for the `shown` event before logging window
# handles. Never used to swap the native window for a browser.
SHOWN_TIMEOUT_SEC = 20.0
SERVER_STOP_TIMEOUT_SEC = 10.0
WINDOW_SIZE = (1280, 860)
WINDOW_MIN_SIZE = (960, 640)

# Streamlit installs signal handlers, which only work on a main thread, so the
# server runs in a child copy of this process while the parent owns the window.
SERVER_ROLE_ENV = "AUDIO_TOOLS_SERVER_ROLE"
SERVER_PORT_ENV = "AUDIO_TOOLS_SERVER_PORT"
# Windows and macOS always ship the native window. Only an explicit opt-in (or a
# platform with no supported webview, e.g. a headless Linux box) may use a browser.
BROWSER_FALLBACK_ENV = "AUDIO_TOOLS_ALLOW_BROWSER"
# Unfrozen only: watch ui/ and rerun in the native window. Frozen installers stay off.
DEV_RELOAD_ENV = "AUDIO_TOOLS_DEV"

# Native open-panel filter when HTML accept MIME types fail to map (common on macOS).
# The list lives in ui/opened_audio.py so Open With and this dialog stay the same.

# Native window wash. Light matches the page wash in ui/theme_presets.py.
_WINDOW_BG_DARK = "#192542"
_WINDOW_BG_LIGHT = "#FFF6DC"
_THEME_CONFIG_NAME = "theme_config.json"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _dev_reload_enabled() -> bool:
    """Live Streamlit reruns in the native window. Never on in a frozen .app / Setup."""
    if _is_frozen():
        return False
    return os.environ.get(DEV_RELOAD_ENV, "").strip() == "1"


def app_name() -> str:
    """%LOCALAPPDATA% folder. CUDA is a separate Windows identity."""
    return edition_app_name(desktop_edition(bundle=_bundle_root()), platform=sys.platform)


def window_title() -> str:
    """Native window title — CPU and NVIDIA editions must not share an HWND title.

    Unfrozen (repo) launches append Demo so they are obvious vs the installed .app.
    """
    title = edition_window_title(desktop_edition(bundle=_bundle_root()))
    if not _is_frozen():
        return f"{title} Demo"
    return title


def _windows_local_appdata() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local)


def _bundle_root() -> Path:
    if _is_frozen():
        # PyInstaller onedir: _MEIPASS is the extracted/bundled resources dir
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _user_data_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AudioTools"
    if sys.platform.startswith("win"):
        return _windows_local_appdata() / app_name()
    return Path.home() / ".local" / "share" / "audio-tools"


def _user_cache_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "AudioTools"
    if sys.platform.startswith("win"):
        # Shared Demucs weights for CPU + NVIDIA editions.
        return _windows_local_appdata() / "AudioTools" / "models"
    return Path.home() / ".cache" / "audio-tools"


def _user_log_dir() -> Path:
    log_dir = _user_data_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _log(message: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with (_user_log_dir() / "launcher.log").open("a", encoding="utf-8") as log:
            log.write(f"{ts} {message.rstrip()}\n")
    except OSError:
        pass


def _enable_crash_log() -> None:
    """Keep a trace when the process dies natively (console=False hides stderr)."""
    try:
        import faulthandler

        # Held for the process lifetime on purpose; faulthandler writes to it on fault.
        handle = (_user_log_dir() / "crash.log").open("a", encoding="utf-8")
        faulthandler.enable(file=handle, all_threads=True)
    except Exception:
        _log("Could not enable the native crash log")

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        name = getattr(args.thread, "name", "?")
        _log(
            f"Thread {name} crashed: {args.exc_type.__name__}: {args.exc_value}\n"
            f"{''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_tb))}"
        )

    threading.excepthook = _thread_hook


def _attach_pywebview_log() -> None:
    """pywebview logs to stderr; console=False freezes swallow that, so tee into launcher.log."""
    try:
        path = _user_log_dir() / "launcher.log"
        logger = logging.getLogger("pywebview")
        target = str(path)
        if any(getattr(handler, "baseFilename", None) == target for handler in logger.handlers):
            return
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s pywebview %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
    except Exception:
        _log("Could not attach the pywebview log")


def _browser_fallback_allowed() -> bool:
    """A browser is never the shipping UX on Windows/macOS; both have a native window."""
    if os.environ.get(BROWSER_FALLBACK_ENV) == "1":
        return True
    return not (sys.platform.startswith("win") or sys.platform == "darwin")


def _lock_path() -> Path:
    return _user_data_root() / LOCK_FILENAME


def _clear_lock() -> None:
    """Remove launcher.lock regardless of which pid wrote it (orphan recovery)."""
    try:
        _lock_path().unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        import ctypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            # Tests on macOS/Linux can fake platform=win32; fall back to POSIX.
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            return True
        kernel32 = windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock() -> dict | None:
    path = _lock_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid = int(data.get("pid", 0))
        port = int(data.get("port", 0))
    except (TypeError, ValueError):
        return None
    if pid <= 0 or port <= 0:
        return None
    return {"pid": pid, "port": port}


def _write_lock(port: int) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "port": int(port)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clear_lock_if_ours() -> None:
    data = _read_lock()
    if data is None or data["pid"] != os.getpid():
        return
    _clear_lock()


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _health_ok(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return 200 <= int(getattr(resp, "status", 0) or 0) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _show_dialog(message: str, *, error: bool = True) -> None:
    _log(("ERROR " if error else "INFO ") + message)
    print(message, file=sys.stderr if error else sys.stdout)
    if sys.platform == "darwin":
        escaped = message.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'display dialog "{escaped}" with title "{window_title()}" '
            'buttons {"OK"} default button 1'
        )
        try:
            subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        except OSError:
            pass
        return
    if sys.platform.startswith("win"):
        try:
            import ctypes

            style = 0x10 if error else 0x40
            ctypes.windll.user32.MessageBoxW(0, message, window_title(), style)  # type: ignore[attr-defined]
        except Exception:
            pass


def _show_error(message: str) -> None:
    _show_dialog(message, error=True)


def _running_from_dmg() -> bool:
    if sys.platform != "darwin":
        return False
    exe = str(Path(sys.executable).resolve())
    return exe.startswith("/Volumes/")


def _pick_port() -> int | None:
    """First loopback port nothing else is listening on.

    Callers must rule out an existing instance first; a port serving a live copy
    of this app is never reusable for a *new* server because the bind would fail.
    """
    for port in PORT_FALLBACKS:
        if _port_free(port):
            return port
    return None


def _open_browser(url: str) -> None:
    try:
        if sys.platform == "darwin":
            # `open` brings the default browser forward; webbrowser.open often does not.
            completed = subprocess.run(["open", url], check=False, capture_output=True)
            if completed.returncode == 0:
                return
        webbrowser.open(url)
    except Exception:
        _log(f"Failed to open browser for {url}")


def _macos_activate_pid(pid: int) -> bool:
    """Raise another process's windows via NSRunningApplication (no extra deps)."""
    if sys.platform != "darwin":
        return False
    try:
        import ctypes

        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        libobjc.objc_getClass.restype = ctypes.c_void_p
        libobjc.objc_getClass.argtypes = [ctypes.c_char_p]
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.cast(libobjc.objc_msgSend, ctypes.c_void_p).value
        cls = libobjc.objc_getClass(b"NSRunningApplication")
        if not cls or not send:
            return False
        app_for_pid = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        )(send)
        app = app_for_pid(
            cls,
            libobjc.sel_registerName(b"runningApplicationWithProcessIdentifier:"),
            int(pid),
        )
        if not app:
            return False
        activate = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong
        )(send)
        # NSApplicationActivateAllWindows | NSApplicationActivateIgnoringOtherApps
        return bool(activate(app, libobjc.sel_registerName(b"activateWithOptions:"), 0x3))
    except Exception:
        _log("Could not activate the running Audio Tools process")
        return False


def _windows_window_handles(pid: int) -> list[int]:
    """Visible top-level HWNDs owned by pid. Title-independent, unlike FindWindowW."""
    if not sys.platform.startswith("win") or pid <= 0:
        return []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        found: list[int] = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @WNDENUMPROC
        def _enum(hwnd, _lparam):
            proc_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
            if int(proc_id.value) == int(pid) and user32.IsWindowVisible(hwnd):
                found.append(int(hwnd))
            return True

        user32.EnumWindows(_enum, 0)
        return found
    except Exception:
        _log("Could not enumerate Audio Tools windows")
        return []


def _windows_activate_pid(pid: int) -> bool:
    """Restore/foreground every visible top-level window owned by pid (title-independent)."""
    handles = _windows_window_handles(pid)
    if not handles:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        SWP_FLAGS = 0x0001 | 0x0002 | 0x0040  # NOSIZE | NOMOVE | SHOWWINDOW
        for hwnd in handles:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            # Windows refuses SetForegroundWindow to a background process, but a
            # topmost bounce still lifts the window above the current z-order.
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, SWP_FLAGS)  # HWND_TOPMOST
            user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, SWP_FLAGS)  # HWND_NOTOPMOST
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        _log(f"Could not foreground windows for pid={pid}")
        return False
    _log(f"Foregrounded {len(handles)} window(s) for pid={pid} hwnds={handles}")
    return True


def _windows_activate_window(title: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            if user32.SetForegroundWindow(hwnd):
                return True
    except Exception:
        _log("Could not activate the running Audio Tools window by title")
    return _windows_activate_pid(os.getpid())


def _focus_existing_instance(pid: int) -> bool:
    """Bring the already-running instance's window forward. Never starts a server."""
    if sys.platform == "darwin":
        return _macos_activate_pid(pid)
    if sys.platform.startswith("win"):
        return _windows_activate_pid(pid) or _windows_activate_window(window_title())
    return False


def _wait_until_healthy(port: int, timeout: float, is_alive=None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _health_ok(port):
            return True
        if is_alive is not None and not is_alive():
            _log("Local server exited before it became healthy")
            return False
        time.sleep(HEALTH_POLL_SEC)
    return False


def _existing_instance() -> dict | None:
    """The pid/port of a live sibling instance, or None if we should start one."""
    data = _read_lock()
    if data is None:
        return None
    pid = data["pid"]
    port = data["port"]
    if not _pid_alive(pid):
        _log(f"Stale lock for dead pid={pid}; clearing")
        _clear_lock()
        return None
    if _health_ok(port):
        _log(f"Existing healthy instance pid={pid} port={port}")
        return data
    # Another copy is still starting — wait briefly for it instead of launching a second server.
    _log(f"Lock held by pid={pid} port={port}; waiting for health")
    deadline = time.monotonic() + min(HEALTH_TIMEOUT_SEC, 30.0)
    while time.monotonic() < deadline:
        if _health_ok(port):
            _log(f"Existing instance became healthy pid={pid} port={port}")
            return data
        if not _pid_alive(pid):
            _log(f"Lock holder pid={pid} exited while waiting; clearing")
            _clear_lock()
            return None
        time.sleep(HEALTH_POLL_SEC)
    # Zombie parent / hung start: clear so a new launch can proceed.
    _log(f"Lock holder pid={pid} never became healthy; clearing stale lock")
    _clear_lock()
    return None


def _prepend_path(directory: Path) -> None:
    if not directory.is_dir():
        return
    current = os.environ.get("PATH", "")
    prefix = str(directory)
    sep = ";" if sys.platform.startswith("win") else ":"
    parts = current.split(sep) if current else []
    if prefix not in parts:
        os.environ["PATH"] = prefix + (sep + current if current else "")


def _ffmpeg_dir_candidates(bundle: Path) -> list[Path]:
    """Locations that may contain ffmpeg / ffprobe (and Windows ffmpeg DLLs)."""
    exe_dir = Path(sys.executable).resolve().parent
    dirs = [
        bundle / "ffmpeg",
        bundle / "ffmpeg" / "bin",
        bundle / "ffmpeg" / "lib",
        exe_dir / "ffmpeg",
        exe_dir / "ffmpeg" / "bin",
        exe_dir / "ffmpeg" / "lib",
    ]
    # PyInstaller .app: executable is Contents/MacOS, datas live in Contents/Frameworks.
    if sys.platform == "darwin" and exe_dir.name == "MacOS":
        frameworks = exe_dir.parent / "Frameworks"
        dirs.extend(
            (
                frameworks / "ffmpeg",
                frameworks / "ffmpeg" / "bin",
                frameworks / "ffmpeg" / "lib",
            )
        )
    return dirs


WEBVIEW2_INSTALL_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
WEBVIEW2_HINT = (
    "Audio Tools needs the Microsoft Edge WebView2 Runtime for its window "
    "(included with Windows 11 and recent Windows 10). Install it from "
    f"{WEBVIEW2_INSTALL_URL} then start Audio Tools again."
)
# Evergreen WebView2 Runtime client id (Microsoft docs).
_WEBVIEW2_CLIENT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def _webview2_available() -> bool:
    """True when Edge WebView2 Runtime looks installed (Windows only; else True)."""
    if not sys.platform.startswith("win"):
        return True
    try:
        import winreg
    except ImportError:
        return False
    roots = (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT_GUID}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT_GUID}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT_GUID}"),
    )
    for hive, path in roots:
        try:
            with winreg.OpenKey(hive, path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
            ver = str(version or "").strip()
            # EdgeUpdate uses an all-zero pv when the runtime is not really present.
            if ver and ver.replace("0", "").replace(".", ""):
                _log(f"WebView2 Runtime detected ({ver}) via {path}")
                return True
        except OSError:
            continue
    _log("WebView2 Runtime not found in registry")
    return False


def configure_environment(bundle: Path) -> Path:
    """
    Set user cache/data dirs and PATH for bundled ffmpeg.
    Returns the working directory that contains ui/app.py.
    """
    data_root = _user_data_root()
    cache_root = _user_cache_root()
    runs_dir = data_root / "runs"
    # TORCH_HOME: platform user cache for Demucs weights
    if sys.platform.startswith("win"):
        torch_home = _windows_local_appdata() / "AudioTools" / "models"
    else:
        torch_home = cache_root / "torch"

    for path in (data_root, runs_dir, torch_home, _user_log_dir()):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AUDIO_TOOLS_DATA_DIR", str(runs_dir))
    os.environ.setdefault("AUDIO_TOOLS_PENDING_OPEN", str(data_root / "pending_open"))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("HF_HOME", str(torch_home / "huggingface"))
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    # PyInstaller bundles lack site-packages in Streamlit's __file__ path, which
    # auto-enables global.developmentMode and rejects explicit --server.port.
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    if _dev_reload_enabled():
        os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "auto"
        os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "true"
    else:
        os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
        os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

    for cand in _ffmpeg_dir_candidates(bundle):
        _prepend_path(cand)

    # Prefer repo root (dev) or _MEIPASS (frozen) as cwd so ui/app.py resolves
    if (bundle / "ui" / "app.py").is_file():
        return bundle
    # Dev: launcher lives in packaging/
    repo = Path(__file__).resolve().parents[1]
    if (repo / "ui" / "app.py").is_file():
        return repo
    return bundle


def _server_command() -> list[str]:
    """Re-run this launcher; frozen builds have no separate Python interpreter."""
    if _is_frozen():
        return [sys.executable]
    return [sys.executable, str(Path(__file__).resolve())]


_KILL_ON_CLOSE_JOB: int | None = None


def _kill_child_with_parent(pid: int) -> bool:
    """Windows: tie the server child's lifetime to this process via a job object.

    Without this, anything that kills the window process (including a native crash
    that Python cannot catch) leaves the Streamlit child running with no window:
    the idle "AudioTools still in Task Manager" process users report.
    """
    global _KILL_ON_CLOSE_JOB
    if not sys.platform.startswith("win") or pid <= 0:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        job = _KILL_ON_CLOSE_JOB
        if not job:
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return False
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):  # JobObjectExtendedLimitInformation
                kernel32.CloseHandle(job)
                return False
            _KILL_ON_CLOSE_JOB = job
        handle = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not handle:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(job, handle))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        _log(f"Could not attach server pid={pid} to a job object")
        return False


def _windows_process_image(pid: int) -> str | None:
    if not sys.platform.startswith("win") or pid <= 0:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buffer))
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return None
            return buffer.value
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def _reclaim_orphan(pid: int) -> bool:
    """End a windowless copy of this app so a fresh launch can own the port."""
    if pid <= 0 or pid == os.getpid():
        return False
    image = _windows_process_image(pid)
    expected = Path(sys.executable).name.lower()
    if image and Path(image).name.lower() != expected:
        _log(f"Refusing to end pid={pid}: image {image} is not {expected}")
        return False
    try:
        if sys.platform.startswith("win"):
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
            if not handle:
                return False
            try:
                ended = bool(kernel32.TerminateProcess(handle, 1))
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 15)
            ended = True
    except OSError:
        _log(f"Could not end orphaned instance pid={pid}")
        return False
    _log(f"Ended orphaned instance pid={pid} (image={image})")
    return ended


def _spawn_server(workdir: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env[SERVER_ROLE_ENV] = "1"
    env[SERVER_PORT_ENV] = str(port)
    kwargs: dict = {"cwd": str(workdir), "env": env, "stdin": subprocess.DEVNULL}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_file = None
    try:
        log_file = (_user_log_dir() / "server.log").open("a", encoding="utf-8")
    except OSError:
        log_file = None
    try:
        server = subprocess.Popen(
            _server_command(),
            stdout=log_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
    finally:
        if log_file is not None:
            log_file.close()
    _kill_child_with_parent(server.pid)
    return server


def _stop_server(server: subprocess.Popen | None) -> None:
    if server is None or server.poll() is not None:
        return
    _log(f"Stopping local server pid={server.pid}")
    try:
        server.terminate()
    except OSError:
        return
    try:
        server.wait(timeout=SERVER_STOP_TIMEOUT_SEC)
        return
    except subprocess.TimeoutExpired:
        _log("Server ignored terminate; killing it")
    try:
        server.kill()
        server.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _streamlit_server_argv(workdir: Path, port: int) -> list[str]:
    watch = "auto" if _dev_reload_enabled() else "none"
    run_on_save = "true" if _dev_reload_enabled() else "false"
    return [
        "streamlit",
        "run",
        str(workdir / "ui" / "app.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--browser.serverAddress=127.0.0.1",
        f"--server.fileWatcherType={watch}",
        f"--server.runOnSave={run_on_save}",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--client.toolbarMode=minimal",
    ]


def run_streamlit_server(workdir: Path, port: int) -> int:
    """Server role: hand the main thread to Streamlit (it installs signal handlers)."""
    _log(f"Server process {os.getpid()} running Streamlit on 127.0.0.1:{port}")
    from streamlit.web import cli as stcli

    sys.argv = _streamlit_server_argv(workdir, port)
    return int(stcli.main() or 0)


def _is_windows_icon(path: Path) -> bool:
    """True for a real ICO. System.Drawing.Icon throws on anything else (e.g. PNG)."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x00\x00\x01\x00"
    except OSError:
        return False


def _webview_storage_dir() -> Path:
    """WebView2/WKWebView profile under the user data root (never Program Files).

    A leftover Edge process holding this folder can stall EnsureCoreWebView2Async
    inside webview.start — the idle ~50 MB process with no HWND. If the default
    profile looks locked, use a per-pid folder so a new window can still appear.
    """
    preferred = _user_data_root() / "webview"
    preferred.mkdir(parents=True, exist_ok=True)
    if _webview_profile_locked(preferred):
        alt = _user_data_root() / f"webview-{os.getpid()}"
        alt.mkdir(parents=True, exist_ok=True)
        _log(f"WebView2 user-data {preferred} looks locked; using {alt}")
        return alt
    _log(f"WebView2 user-data={preferred} existing_profile={(preferred / 'EBWebView').is_dir()}")
    return preferred


def _webview_profile_locked(folder: Path) -> bool:
    """True when Chromium/WebView2 already owns this user-data folder."""
    for candidate in (
        folder / "lockfile",
        folder / "EBWebView" / "lockfile",
        folder / "EBWebView" / "SingletonLock",
    ):
        if not candidate.is_file():
            continue
        try:
            with candidate.open("a+b") as handle:
                handle.write(b"")
        except OSError:
            return True
    return False


def _webview_icon() -> str | None:
    """Window icon in the format the platform toolkit accepts.

    WinForms builds the icon with System.Drawing.Icon, which only reads ICO and
    raises ArgumentException on a PNG. That exception lands on the STA thread that
    creates the form, so the window never shows. Cocoa (macOS) reads PNG happily.
    """
    windows = sys.platform.startswith("win")
    name = "icon.ico" if windows else "icon.png"
    for candidate in (
        _bundle_root() / "packaging" / name,
        _bundle_root() / "ui" / name,
        Path(__file__).resolve().parent / name,
    ):
        if not candidate.is_file():
            continue
        if windows and not _is_windows_icon(candidate):
            _log(f"Ignoring window icon {candidate}: not a Windows ICO")
            continue
        return str(candidate)
    _log(f"No usable window icon found ({name}); using the executable icon")
    return None


def _patch_webview_audio_open_dialogs() -> None:
    """Force native open panels to the audio and video types Isolate accepts.

    Streamlit's ``accept`` often sends MIME types that pywebview cannot map to
    UTTypes on macOS, leaving every file selectable. OPEN dialogs use the same
    extensions as the Home uploader, including video. SAVE (downloads) is
    unchanged.
    """
    try:
        from webview import FileDialog
        from webview.platforms import cocoa
    except ImportError:
        return

    browser_view = cocoa.BrowserView
    if getattr(browser_view.create_file_dialog, "_audio_tools_patched", False):
        return

    original = browser_view.create_file_dialog
    audio_filter = [["Audio or video", list(_audio_open_extensions())]]

    def create_file_dialog(
        self,
        dialog_type,
        directory,
        allow_multiple,
        save_filename,
        file_filter,
        main_thread=False,
    ):
        use_filter = file_filter
        if dialog_type == FileDialog.OPEN:
            use_filter = audio_filter
        return original(
            self,
            dialog_type,
            directory,
            allow_multiple,
            save_filename,
            use_filter,
            main_thread=main_thread,
        )

    create_file_dialog._audio_tools_patched = True  # type: ignore[attr-defined]
    browser_view.create_file_dialog = create_file_dialog  # type: ignore[method-assign]


# Frozen x64 installers drop unused WebView2Loader ABIs. pywebview 6.x still
# looks up every folder in this list at import and raises FileNotFoundError.
_OPTIONAL_WEBVIEW2_RUNTIMES = frozenset({"win-arm64", "win-x86"})


def _patch_pywebview_optional_runtimes(webview_module=None) -> None:
    """Allow pywebview to import when win-arm64 / win-x86 are not in the freeze.

    ``webview.platforms.edgechromium`` does ``interop_dll_path(platform)`` for
    ``win-arm64``, ``win-x64``, and ``win-x86`` at module import. Missing
    optional ABIs used to surface as "Cannot find win-arm64" and kill the
    native window. Patch *before* ``webview.start()`` imports that module.
    """
    util = getattr(webview_module, "util", None) if webview_module is not None else None
    if util is None:
        if webview_module is not None:
            return
        try:
            from webview import util as util
        except ImportError:
            return
    original = getattr(util, "interop_dll_path", None)
    if original is None or getattr(original, "_audio_tools_patched", False):
        return

    def interop_dll_path(dll_name: str) -> str:
        try:
            return original(dll_name)
        except FileNotFoundError:
            if str(dll_name) in _OPTIONAL_WEBVIEW2_RUNTIMES:
                _log(f"Skipping missing WebView2 runtime {dll_name}")
                return ""
            raise

    interop_dll_path._audio_tools_patched = True  # type: ignore[attr-defined]
    util.interop_dll_path = interop_dll_path  # type: ignore[method-assign]


def _start_webview(webview, boot=None) -> None:
    """webview.start() must own the main thread; pass only kwargs this version knows."""
    _patch_pywebview_optional_runtimes(webview)
    storage = _webview_storage_dir()
    candidates: dict = {
        # Persist localStorage so streamlit-local-storage keeps user settings.
        "private_mode": False,
        "storage_path": str(storage),
    }
    # WinForms BrowserForm.__init__ does System.Drawing.Icon(path) on the STA
    # thread *before* Form.Show. A PNG (or a bad ICO) raises there; pythonnet
    # can leave that thread alive-but-stuck, which is the idle hang inside
    # webview.start with no HWND. Skip icon= on Windows and let WinForms pull
    # the ICO already embedded in AudioTools.exe via ExtractIconW.
    if not sys.platform.startswith("win"):
        icon = _webview_icon()
        if icon:
            candidates["icon"] = icon
    if boot is not None:
        candidates["func"] = boot
    if sys.platform.startswith("win"):
        candidates["gui"] = "edgechromium"
    allowed = set(inspect.signature(webview.start).parameters)
    kwargs = {k: v for k, v in candidates.items() if k in allowed and v is not None}
    _log(
        f"Entering webview.start storage={storage} icon_kwarg={'icon' in kwargs} "
        f"kwargs={sorted(kwargs)}"
    )
    try:
        webview.start(**kwargs)
    except Exception:
        _log(f"webview.start raised:\n{traceback.format_exc()}")
        raise
    _log("webview.start returned")


def _run_in_browser(url: str, port: int, server: subprocess.Popen) -> int:
    """Last resort when no native window can be created (headless dev box)."""
    _log(f"Browser fallback for {url}")
    if not _wait_until_healthy(port, HEALTH_TIMEOUT_SEC, lambda: server.poll() is None):
        _show_error(_not_ready_message(url))
        return 1
    _open_browser(url)
    server.wait()
    return int(server.returncode or 0)


def _window_was_shown(window) -> bool:
    """Whether the GUI toolkit ever raised `shown` for this window."""
    shown = getattr(getattr(window, "events", None), "shown", None)
    if shown is None or not hasattr(shown, "is_set"):
        return True
    try:
        return bool(shown.is_set())
    except Exception:
        return True


def window_background_color(mode: str | None) -> str:
    """Light uses the page wash. Dark, system, and a missing file stay dark."""
    if str(mode or "").strip().lower() == "light":
        return _WINDOW_BG_LIGHT
    return _WINDOW_BG_DARK


def launch_shell_html(background: str) -> str:
    """Quiet Home-shaped first paint. No logo, no splash branding."""
    bg = str(background or _WINDOW_BG_DARK).strip() or _WINDOW_BG_DARK
    light = bg.upper() == _WINDOW_BG_LIGHT
    rail = "#142033" if not light else "#F0E6C8"
    wash = "#2a4467" if not light else "#E8DCC0"
    text = "#d5deea" if not light else "#3a3a3a"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title></title>
<style>
  html, body {{
    margin: 0;
    height: 100%;
    background: {bg};
    color: {text};
    font-family: system-ui, sans-serif;
  }}
  .shell {{
    display: flex;
    height: 100%;
    min-height: 100%;
  }}
  .rail {{
    width: 240px;
    flex: 0 0 240px;
    background: {rail};
  }}
  .main {{
    flex: 1;
    min-width: 0;
    padding: 24px 20px 0 33px;
  }}
  .heading {{
    margin: 32px 0 12px;
    font-size: 18px;
    line-height: 26px;
    font-weight: 500;
    color: {text};
  }}
  .skel {{
    height: 56px;
    margin: 0;
    background: {wash};
    border-bottom: 1px solid transparent;
  }}
  .cue {{
    margin: 8px 0 16px;
    font-size: 13px;
    line-height: 18px;
    font-weight: 500;
    letter-spacing: 0.01em;
    opacity: 0.85;
  }}
  @media (prefers-reduced-motion: reduce) {{
    .cue {{ opacity: 1; }}
  }}
</style>
</head>
<body>
  <div class="shell">
    <aside class="rail" aria-hidden="true"></aside>
    <main class="main">
      <div class="cue">Starting…</div>
      <div class="heading">Files</div>
      <div class="skel" data-skel="0"></div>
      <div class="skel" data-skel="1"></div>
      <div class="skel" data-skel="2"></div>
    </main>
  </div>
</body>
</html>
"""


def _saved_window_theme_mode() -> str | None:
    """Read Appearance from theme_config.json. None when the file is missing."""
    raw = os.environ.get("AUDIO_TOOLS_DATA_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw) / _THEME_CONFIG_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    mode = data.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        return None
    return mode.strip()


def _no_window_failure(detail: str) -> int:
    """Report a native-window failure instead of quietly degrading to a browser."""
    _log(f"Native window failed: {detail}")
    _show_error(
        f"Audio Tools could not open its window ({detail}). "
        f"Logs: {_user_log_dir() / 'launcher.log'}"
    )
    return 1


def _run_native_window(url: str, port: int, server: subprocess.Popen) -> int:
    if sys.platform.startswith("win") and not _webview2_available():
        _show_dialog(WEBVIEW2_HINT, error=True)
        if not _browser_fallback_allowed():
            return 1
        return _run_in_browser(url, port, server)

    try:
        if sys.platform.startswith("win"):
            # pythonnet/WinForms require STA; set before importing clr via pywebview.
            sys.coinit_flags = 2  # type: ignore[attr-defined]
            _log("STA requested (sys.coinit_flags=2) before importing pywebview")
        _attach_pywebview_log()
        _log("Importing pywebview")
        import webview

        _patch_pywebview_optional_runtimes(webview)
    except ImportError:
        if not _browser_fallback_allowed():
            return _no_window_failure("pywebview is unavailable")
        _log("pywebview is unavailable; falling back to the system browser")
        return _run_in_browser(url, port, server)

    # st.download_button and st.file_uploader need real download / open panels.
    webview.settings["ALLOW_DOWNLOADS"] = True
    _patch_webview_audio_open_dialogs()

    window = None
    stopping = threading.Event()
    failed = threading.Event()
    theme_mode = _saved_window_theme_mode()
    wash = window_background_color(theme_mode)

    def close_window() -> None:
        if window is None:
            return
        try:
            window.destroy()
        except Exception:
            pass

    def watch_server() -> None:
        while not stopping.wait(0.5):
            if server.poll() is not None:
                _log(f"Server exited (rc={server.returncode}); closing the window")
                close_window()
                return

    def watch_window() -> None:
        """Foreground the window once it exists, and log HWNDs either way.

        Do not call window.load_url from this thread: pythonnet can deadlock the GIL.
        Navigation after health runs only from the webview.start boot callback.
        """
        shown = getattr(getattr(window, "events", None), "shown", None)
        visible = False
        try:
            if shown is not None and hasattr(shown, "wait"):
                visible = bool(shown.wait(SHOWN_TIMEOUT_SEC))
            else:
                visible = stopping.wait(SHOWN_TIMEOUT_SEC)
        except Exception as exc:
            _log(f"shown wait failed: {exc}")
        if stopping.is_set():
            return
        if visible:
            handles = _windows_window_handles(os.getpid())
            _log(f"Native window shown (hwnds={handles or 'n/a'})")
            if not _windows_activate_pid(os.getpid()) and not _windows_activate_window(
                window_title()
            ):
                _log("Window shown but no HWND could be foregrounded")
            return
        _log(
            f"Native window not shown within {SHOWN_TIMEOUT_SEC}s; "
            f"hwnds={_windows_window_handles(os.getpid()) or 'none'}"
        )

    def gui_heartbeat() -> None:
        """Prove whether start() is deadlocked (no ticks) vs waiting on WebView2."""
        started = time.monotonic()
        while not stopping.wait(2.0):
            elapsed = time.monotonic() - started
            handles = _windows_window_handles(os.getpid())
            _log(
                f"GUI heartbeat +{elapsed:.1f}s hwnds={handles or 'none'} "
                f"shown={_window_was_shown(window)}"
            )

    def boot() -> None:
        """After the shell is up, wait for Streamlit, then navigate once."""
        _log(f"Shell open; waiting for Streamlit health before {url}")
        if not _wait_until_healthy(port, HEALTH_TIMEOUT_SEC, lambda: server.poll() is None):
            failed.set()
            _show_error(_not_ready_message(url))
            close_window()
            return
        if stopping.is_set():
            return
        _log(f"Health check OK; navigating once to {url}")
        try:
            window.load_url(url)
        except Exception as exc:
            failed.set()
            _log(f"load_url failed: {exc}")
            _show_error(_not_ready_message(url))
            close_window()

    try:
        create_kwargs: dict = {
            "html": launch_shell_html(wash),
            "width": WINDOW_SIZE[0],
            "height": WINDOW_SIZE[1],
            "min_size": WINDOW_MIN_SIZE,
            "text_select": True,
            # Light mode matches the page wash. Dark, system, and a missing file stay dark.
            # Keep the window visible with the shell; hiding until loaded can flash twice.
            "background_color": wash,
        }
        if sys.platform.startswith("win"):
            create_kwargs["shadow"] = False
        _log("Creating native window with Home-shaped shell (navigate after health)")
        window = webview.create_window(window_title(), **create_kwargs)
        shown = getattr(getattr(window, "events", None), "shown", None)
        if shown is not None and hasattr(shown, "__iadd__"):
            def _on_form_shown(*_args, **_kwargs) -> None:
                handles = _windows_window_handles(os.getpid())
                _log(f"Form.Shown (hwnds={handles or 'n/a'})")

            shown += _on_form_shown
        threading.Thread(target=watch_server, name="audiotools-server-watch", daemon=True).start()
        threading.Thread(target=watch_window, name="audiotools-window-watch", daemon=True).start()
        threading.Thread(target=gui_heartbeat, name="audiotools-gui-heartbeat", daemon=True).start()
        _start_webview(webview, boot=boot)
    except Exception as exc:
        stopping.set()
        detail = str(exc).lower()
        if sys.platform.startswith("win") and any(
            token in detail for token in ("webview", "clr", "chromium", "edge")
        ):
            _show_dialog(WEBVIEW2_HINT, error=True)
        if not _browser_fallback_allowed():
            return _no_window_failure(str(exc))
        _log(f"Native window unavailable ({exc}); falling back to the system browser")
        return _run_in_browser(url, port, server)
    finally:
        stopping.set()
    if not _window_was_shown(window):
        # webview.start() returned without the form ever appearing: the GUI thread
        # died during window creation (a bad icon file used to do exactly this).
        return _no_window_failure("the window closed before it appeared")
    _log("Native window closed")
    return 1 if failed.is_set() else 0


def _not_ready_message(url: str) -> str:
    return (
        f"Audio Tools could not start its local UI within {int(HEALTH_TIMEOUT_SEC)}s. "
        f"Logs: {_user_log_dir() / 'launcher.log'} and "
        f"{_user_log_dir() / 'server.log'} (server URL was {url})."
    )


def _audio_open_extensions() -> tuple[str, ...]:
    from ui.opened_audio import OPENED_AUDIO_EXTENSIONS

    return OPENED_AUDIO_EXTENSIONS


def _pending_open_dir() -> Path:
    raw = os.environ.get("AUDIO_TOOLS_PENDING_OPEN", "").strip()
    if raw:
        return Path(raw)
    return _user_data_root() / "pending_open"


def _handoff_opened_files(argv: list[str] | None = None) -> None:
    """Queue audio the OS passed on the command line for the running Isolate page.

    A second launch (the app is already open) writes the queue and then focuses
    the existing window. The first launch writes it before Streamlit starts.
    """
    from ui.opened_audio import enqueue_opened_paths, paths_from_argv

    accepted, declined = paths_from_argv(list(sys.argv if argv is None else argv))
    items = [{"path": str(path)} for path in accepted]
    items.extend({"path": str(path), "declined": True} for path in declined)
    if not items:
        return
    enqueue_opened_paths(_pending_open_dir(), items)
    _log(
        f"Queued {len(accepted)} opened file(s)"
        + (f" and declined {len(declined)}" if declined else "")
    )


def run_desktop_app(workdir: Path) -> int:
    _enable_crash_log()
    if _is_frozen() and _running_from_dmg():
        _show_error(
            "Move AudioTools to the Applications folder, then launch it from there. "
            "Do not run it from the disk image."
        )
        return 1

    _handoff_opened_files()

    existing = _existing_instance()
    if existing is not None:
        pid = existing["pid"]
        port = existing["port"]
        url = f"http://127.0.0.1:{port}"
        _log(f"Existing instance pid={pid} port={port}; focusing it")
        if _focus_existing_instance(pid):
            return 0
        _log(f"Focus failed for pid={pid}; checking health for recovery")
        if _browser_fallback_allowed() and _health_ok(port):
            _log(f"Server healthy but no window; opening browser at {url}")
            _open_browser(url)
            _show_dialog(
                "Audio Tools is running but its window could not be shown. "
                f"Opened {url} in your browser instead. "
                "If that fails, end AudioTools in Task Manager and try again.",
                error=False,
            )
            return 0
        # A copy with no window is an orphan (crashed parent, leftover server). End
        # it and start clean rather than sending the user to a browser tab.
        _log(f"No window for pid={pid}; reclaiming it and starting a fresh instance")
        _reclaim_orphan(pid)
        _clear_lock()
        # Fall through to a fresh start below.

    port = _pick_port()
    if port is None:
        _show_error("Ports 8501–8505 are in use by other programs. Quit them and try again.")
        return 1

    url = f"http://127.0.0.1:{port}"
    _write_lock(port)
    try:
        server = _spawn_server(workdir, port)
    except OSError as exc:
        _clear_lock_if_ours()
        _show_error(f"Audio Tools could not start its local server: {exc}")
        return 1

    _log(f"Server pid={server.pid} starting at {url} cwd={workdir}")
    try:
        return _run_native_window(url, port, server)
    finally:
        _stop_server(server)
        _clear_lock_if_ours()


def main() -> int:
    bundle = _bundle_root()
    workdir = configure_environment(bundle)
    os.chdir(workdir)

    # Ensure src/ is importable when not frozen
    src = workdir / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    if os.environ.get(SERVER_ROLE_ENV) == "1":
        try:
            port = int(os.environ.get(SERVER_PORT_ENV, "") or 0)
        except ValueError:
            port = 0
        if port <= 0:
            _log(f"Server role started without a valid {SERVER_PORT_ENV}")
            return 1
        return run_streamlit_server(workdir, port)

    return run_desktop_app(workdir)


if __name__ == "__main__":
    raise SystemExit(main())

"""Desktop boot screen: launch the real jobs and reopen local separations."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path

import streamlit as st

from ui.common import DATA_DIR, edition_product_name, list_recent_runs
from ui.library_song import (
    TRASH_DIR_NAME,
    delete_library_song,
    list_trashed_songs,
    locale_text,
    recover_library_song,
    trashed_row,
)
from ui.setlists import (
    add_song,
    create_setlist,
    delete_setlist,
    load_setlists,
    remove_song,
    rename_setlist,
    save_setlists,
)
from ui.design_tokens import ui_v2_enabled
from ui.isolate_jobs import list_jobs, merge_library_runs
from ui.isolate_state import (
    ISOLATE_SKIP_REHYDRATE_KEY,
    ISOLATE_USER_ID_FILENAME,
    LISTEN_PICKER_NEXT_KEY,
    SHELL_TAB_KEY,
    SOURCE_KIND_YOUTUBE,
    add_open_mix_tab,
    format_source_title,
    infer_source_kind,
    load_persist_isolate_user_id,
    open_mix_shell,
    open_new_draft_tab,
    recent_runs_with_owner_fallback,
    youtube_video_id,
)

_HOME_CSS = """
<style>
  .st-key-att_home [data-testid="stButton"] button {
    min-height: 4.6rem;
    border-radius: 14px;
    justify-content: flex-start;
    text-align: left;
    white-space: normal;
    line-height: 1.25;
    transition:
      transform 160ms cubic-bezier(0.22, 1, 0.36, 1),
      border-color 160ms cubic-bezier(0.22, 1, 0.36, 1);
  }
  .st-key-att_home [data-testid="stButton"] button:hover:not(:disabled) {
    border-color: var(--color-accent-primary, #f6ad49);
  }
  @media (prefers-reduced-motion: reduce) {
    .st-key-att_home [data-testid="stButton"] button {
      transition: none;
    }
  }
</style>
"""


def relative_age(created_at: object, *, now: float | None = None) -> str:
    """Short age label for a run timestamp. Unknown times stay blank."""
    try:
        stamp = float(created_at)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if stamp <= 0:
        return ""
    seconds = max(0.0, (time.time() if now is None else float(now)) - stamp)
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"
    hours = int(minutes // 60)
    if hours < 24:
        return "1 hour ago" if hours == 1 else f"{hours} hours ago"
    days = int(hours // 24)
    if days < 30:
        return "1 day ago" if days == 1 else f"{days} days ago"
    months = int(days // 30)
    if months < 12:
        return "1 month ago" if months == 1 else f"{months} months ago"
    years = int(days // 365)
    return "1 year ago" if years <= 1 else f"{years} years ago"


def library_row_is_present(row: dict) -> bool:
    """True when the mix folder and a wav are still in the library.

    delete_library_song moves the folder into library_trash and leaves the
    isolate_jobs status in place. That status still names .wav paths, so a
    suffix check put the mix straight back on Home.
    """
    raw = str(row.get("run_dir") or "")
    if not raw:
        return False
    run_dir = Path(raw)
    if not run_dir.is_dir():
        return False
    trash = (DATA_DIR / TRASH_DIR_NAME).resolve()
    try:
        run_dir.resolve().relative_to(trash)
    except (OSError, ValueError):
        pass
    else:
        return False
    arts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
    for path in arts.values():
        wav = Path(str(path))
        if wav.suffix.lower() == ".wav" and wav.is_file():
            return True
    try:
        return any(
            path.is_file() and path.suffix.lower() == ".wav" for path in run_dir.iterdir()
        )
    except OSError:
        return False


def library_rows(owner: str | None, *, limit: int = 40) -> list[dict]:
    """Isolate runs that still have a wav, newest first. Same owner rule as Isolate."""
    owned = list_recent_runs("isolate", limit=limit, owner=owner)
    everyone = list_recent_runs("isolate", limit=limit)
    recent = recent_runs_with_owner_fallback(owned, everyone)
    merged = merge_library_runs(recent, list_jobs(limit=limit))
    return [row for row in merged if library_row_is_present(row)][:limit]


def _share_token(run_dir: str) -> str:
    if not run_dir:
        return ""
    path = Path(run_dir) / "share.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, UnicodeError):
        return ""
    token = str(data.get("token") or "") if isinstance(data, dict) else ""
    return "" if "/" in token or "\\" in token else token


def run_label(row: dict) -> str:
    return format_source_title(
        row.get("title") or "tracks",
        infer_source_kind(
            source_kind=row.get("source_kind"),
            source_fingerprint=row.get("source_fingerprint"),
        ),
    )


def recent_art_url(row: dict) -> str:
    """Path-free Recents thumb: public YouTube CDN URL, else empty (generic motif)."""
    kind = infer_source_kind(
        source_kind=row.get("source_kind"),
        source_fingerprint=row.get("source_fingerprint"),
    )
    if kind != SOURCE_KIND_YOUTUBE:
        return ""
    fingerprint = str(row.get("source_fingerprint") or "")
    prefix = "youtube:"
    url = fingerprint[len(prefix) :] if fingerprint.startswith(prefix) else ""
    vid = youtube_video_id(url) or ""
    if not vid or not all(c.isalnum() or c in "-_" for c in vid):
        return ""
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def _go(page_name: str) -> None:
    st.session_state["_nav_loading"] = True
    st.switch_page(str(Path(__file__).with_name(page_name)))


def open_new_separation() -> None:
    """Land on Isolate's upload form, with the New draft selected."""
    open_new_draft_tab(st.session_state, fresh=True)
    _go("isolate.py")


def analysis_summary(row: dict) -> dict | None:
    """Compact analysis summary from meta.json, or from analysis.json on old runs."""
    summary = row.get("analysis")
    if isinstance(summary, dict) and summary:
        return summary
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    raw_path = artifacts.get("analysis")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_file():
        return None
    try:
        from audio_to_tab.analysis import load_analysis

        result = load_analysis(path)
    except Exception:
        return None
    if result is None:
        return None
    summary = {
        "bpm": result.bpm,
        "key": result.key,
        "chord_count": len(result.chords),
        "section_count": len(result.sections),
    }
    try:
        duration = float(result.duration_sec)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        summary["duration_sec"] = duration
    return summary


def analysis_line(summary: dict | None) -> str:
    """Small Home row line: BPM · Key · Sections."""
    if not summary:
        return "Not analysed yet"
    bpm_text = "BPM —"
    try:
        bpm = float(summary.get("bpm"))
    except (TypeError, ValueError):
        bpm = 0.0
    if bpm > 0:
        bpm_text = f"BPM {bpm:.0f}" if abs(bpm - round(bpm)) < 0.05 else f"BPM {bpm:.1f}"
    key_text = str(summary.get("key") or "Key —")
    raw_sections = summary.get("section_count")
    try:
        sections = int(raw_sections)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sections = 0
    section_text = "1 section" if sections == 1 else f"{sections} sections"
    if sections <= 0:
        section_text = "No sections"
    return f"{bpm_text} · {key_text} · {section_text}"


def row_analysis_state(summary: dict | None, run_dir: str, status: dict | None) -> str:
    """``ready``, ``analysing``, ``waiting`` (locked), or ``failed`` / ``idle`` (open)."""
    if summary:
        return "ready"
    data = status if isinstance(status, dict) else {}
    state = str(data.get("state") or "")
    skipped = {str(item) for item in (data.get("skipped") or []) if item}
    if run_dir and run_dir in skipped:
        return "failed"
    if state == "analysing" and run_dir and run_dir == str(data.get("run_dir") or ""):
        return "analysing"
    if state in {"analysing", "waiting"}:
        return "waiting"
    return "idle"


def row_analysis_subtitle(summary: dict | None, run_dir: str, status: dict | None) -> str:
    """Home row line, including the background key-and-chord pass."""
    state = row_analysis_state(summary, run_dir, status)
    if state == "ready":
        return analysis_line(summary)
    if state == "analysing":
        return "Getting key and chords…"
    if state == "waiting":
        return "Waiting for key and chords"
    if state == "failed":
        return "Could not get key and chords. Use the row menu to retry."
    return analysis_line(None)


def analysis_status_notice(status: dict | None) -> str:
    """One sentence for the on-screen progress line, or empty when nothing is queued."""
    if not isinstance(status, dict):
        return ""
    state = str(status.get("state") or "")
    if state not in {"analysing", "waiting"}:
        return ""
    try:
        remaining = int(status.get("remaining") or 0)
    except (TypeError, ValueError):
        remaining = 0
    if remaining <= 0:
        return ""
    title = " ".join(str(status.get("title") or "").split()) or "a saved mix"
    if state == "waiting":
        tail = f" {remaining} still waiting." if remaining else ""
        return f"Key and chords will continue after the current separation.{tail}"
    extra = remaining - 1
    tail = f" {extra} more still waiting." if extra > 0 else ""
    return f"Getting key and chords for {title}.{tail}"


def analysis_status_markup(notice: str) -> str:
    """Visible progress bar. The sentence is escaped plain text."""
    import html as html_lib

    safe = html_lib.escape(notice)
    return f"""
<style>
  .att-analysis-status {{
    box-sizing: border-box;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    width: 100%;
    margin: 0 0 12px 0;
    padding: 10px 12px;
    border: 1px solid #F6AD49;
    border-radius: 10px;
    background: #192542;
    color: #F4F7FB;
    font-size: 0.95rem;
    line-height: 1.4;
  }}
  .att-analysis-dot {{
    width: 10px;
    height: 10px;
    margin-top: 0.35rem;
    border-radius: 99px;
    background: #F6AD49;
    flex: none;
    animation: att-analysis-pulse 1.2s ease-in-out infinite;
  }}
  @keyframes att-analysis-pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.35; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .att-analysis-dot {{ animation: none; }}
  }}
</style>
<div class="att-analysis-status" role="status" aria-live="polite">
  <span class="att-analysis-dot" aria-hidden="true"></span>
  <span>{safe}</span>
</div>
"""


def analysis_status_token(status: dict | None) -> str:
    if not isinstance(status, dict):
        return "idle"
    return "|".join(
        str(status.get(key) or "")
        for key in ("state", "run_dir", "remaining", "finished_run")
    )


def analysis_status_should_refresh_page(page: str, token: str, previous: str | None) -> bool:
    """Home reruns when progress changes. The mixer page must stay put."""
    if page not in {"", "home"}:
        return False
    idle = token in {"", "idle", "|||"}
    if previous is None:
        return not idle
    return token != previous


def next_auto_analysis(
    rows: list[dict],
    *,
    busy: bool,
    skipped: set[str],
    active_run: str | None = None,
) -> str | None:
    """One finished run that still has no analysis summary, or None.

    ``active_run`` is the song open in the mixer. Other unfinished songs go
    first. It is chosen only when it is the last unfinished song.
    """
    if busy:
        return None
    pending: list[str] = []
    for row in rows:
        run_dir = str(row.get("run_dir") or "")
        if not run_dir or run_dir in skipped:
            continue
        if analysis_summary(row):
            continue
        pending.append(run_dir)
    if not pending:
        return None
    current = str(active_run or "")
    if current and current in pending and len(pending) > 1:
        pending = [item for item in pending if item != current]
    return pending[0]


def auto_analysis_should_skip(ok: bool, message: str) -> bool:
    """A no-stem or failed analysis stays skipped. A busy worker does not."""
    if ok:
        return False
    return "separation" not in str(message).casefold()


def open_saved_run(run_dir: str) -> None:
    """Open one local separation in the Isolate mixer."""
    st.session_state.pop(ISOLATE_SKIP_REHYDRATE_KEY, None)
    st.session_state["isolate_run_dir"] = run_dir
    st.session_state["isolate_viewing_run_dir"] = run_dir
    st.session_state[LISTEN_PICKER_NEXT_KEY] = run_dir
    st.session_state.pop("isolate_listen_applied_dir", None)
    add_open_mix_tab(st.session_state, run_dir)
    open_mix_shell(st.session_state)
    st.session_state[SHELL_TAB_KEY] = run_dir
    _go("isolate.py")


def open_practice_run(run_dir: str) -> None:
    """Open a run in the mixer; the practice panel is part of that view."""
    open_saved_run(run_dir)


def _flash(message: str, *, error: bool = False) -> None:
    st.session_state["home_flash"] = (str(message), bool(error))


_TOKEN_MAP: dict[str, str] = {}


def clock_duration(raw: object) -> str:
    """Clock label for a stored length. Missing or empty values stay blank."""
    try:
        seconds = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if seconds != seconds or seconds <= 0:
        return ""
    total = int(round(seconds))
    if total <= 0:
        return ""
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def existing_duration(row: dict, summary: dict | None) -> str:
    """Song length already stored on the job or the analysis. Never probed."""
    raw = row.get("audio_duration_sec")
    if raw is None and isinstance(summary, dict):
        raw = summary.get("duration_sec")
    return clock_duration(raw)


def existing_tempo(summary: dict | None) -> str:
    """Detected BPM already on the analysis summary. Blank when it was not stored."""
    if not isinstance(summary, dict):
        return ""
    try:
        bpm = float(summary.get("bpm"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if bpm != bpm or bpm <= 0:
        return ""
    if abs(bpm - round(bpm)) < 0.05:
        return str(int(round(bpm)))
    return f"{bpm:.1f}"


def build_home_props(
    rows: list[dict],
    *,
    product: str = "Audio Tools",
    nonce: str | None = None,
    flash: tuple[str, bool] | None = None,
    loading: bool = False,
    setlists: list[dict] | None = None,
    locale: str = "en",
    trashed: list[dict] | None = None,
    analysis_status: dict | None = None,
) -> dict:
    """Browser payload. Paths stay in ``_TOKEN_MAP`` and never in the dict."""
    _TOKEN_MAP.clear()
    issued = nonce or secrets.token_hex(8)
    public_rows: list[dict[str, str]] = []
    tokens_by_path: dict[str, str] = {}
    for row in rows:
        run_dir = str(row.get("run_dir") or "")
        summary = analysis_summary(row)
        token = ""
        if run_dir:
            token = hashlib.sha256(f"{issued}\n{run_dir}".encode()).hexdigest()[:16]
            _TOKEN_MAP[token] = run_dir
            tokens_by_path[run_dir] = token
        art = recent_art_url(row)
        public_rows.append(
            {
                "token": token,
                "share": _share_token(run_dir),
                "title": run_label(row),
                "subtitle": row_analysis_subtitle(summary, run_dir, analysis_status),
                "state": row_analysis_state(summary, run_dir, analysis_status),
                "age": relative_age(row.get("created_at")),
                "duration": existing_duration(row, summary),
                "tempo": existing_tempo(summary),
                "action": "practice" if summary else "analyse",
                "art": art,
            }
        )
    public_setlists: list[dict] = []
    for item in setlists or []:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "").strip()
        name = " ".join(str(item.get("name") or "").split())
        songs = item.get("songs")
        if not ident or not name or not isinstance(songs, list):
            continue
        tokens = [tokens_by_path[str(song)] for song in songs if str(song) in tokens_by_path]
        public_setlists.append({"id": ident, "name": name, "tokens": tokens})
    public_trashed: list[dict[str, str]] = []
    for row in trashed or []:
        run_dir = str(row.get("run_dir") or "")
        token = ""
        if run_dir:
            token = hashlib.sha256(f"{issued}\n{run_dir}".encode()).hexdigest()[:16]
            _TOKEN_MAP[token] = run_dir
        public_trashed.append(
            {
                "token": token,
                "title": str(row.get("title") or "Song"),
                "subtitle": "",
                "age": "",
                "action": "recover",
            }
        )
    chosen = locale if locale in {"en", "ko"} else "en"
    strings = {
        key: locale_text(chosen, key)
        for key in ("search", "title", "all", "practice", "analyse", "deleted", "delete", "recover")
    }
    flash_payload = None
    if isinstance(flash, tuple) and len(flash) == 2:
        flash_payload = {"message": str(flash[0]), "error": bool(flash[1])}
    return {
        "nonce": issued,
        "product": product,
        "cards": [],
        "recents": public_rows[:6],
        "rows": public_rows,
        "trashed": public_trashed,
        "setlists": public_setlists,
        "loading": bool(loading),
        "flash": flash_payload,
        "locale": chosen,
        "strings": strings,
    }


def _setlists_path() -> Path:
    return DATA_DIR / "setlists.json"


def _remember_setlist(setlist_id: str) -> None:
    st.session_state["home_v2_setlist"] = setlist_id


def active_setlist_id(setlists: list[dict]) -> str:
    """The setlist pill that should stay selected after a rerun."""
    active = str(st.session_state.get("home_v2_setlist") or "")
    if any(str(item.get("id") or "") == active for item in setlists):
        return active
    return ""


def song_deletion_blocked(run_dir: str, status: dict | None) -> bool:
    """The mix currently being analysed cannot be moved while that child is running."""
    data = status if isinstance(status, dict) else {}
    return str(data.get("state") or "") == "analysing" and str(data.get("run_dir") or "") == str(
        run_dir
    )


def apply_library_deletes(
    token_map: dict[str, str],
    ident: str,
    *,
    data_dir: Path,
    status: dict | None,
) -> tuple[int, int]:
    """Trash each resolved token. Unknown tokens are ignored. Returns (moved, failed)."""
    moved = 0
    failed = 0
    seen: set[str] = set()
    for raw in str(ident or "").split(","):
        token = raw.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        run_dir = token_map.get(token)
        if not run_dir:
            continue
        if song_deletion_blocked(run_dir, status):
            failed += 1
            continue
        if delete_library_song(Path(run_dir), data_dir) is None:
            failed += 1
        else:
            moved += 1
    return moved, failed


def library_delete_message(moved: int, failed: int) -> tuple[str, bool] | None:
    """Flash text, and whether it is an error. None when nothing was attempted."""
    if moved and failed:
        return (f"Deleted {moved}. {failed} could not be deleted.", True)
    if moved == 1:
        return ("Deleted 1 mix. You can recover it from Deleted.", False)
    if moved:
        return (f"Deleted {moved} mixes. You can recover them from Deleted.", False)
    if failed == 1:
        return ("Could not delete 1 mix.", True)
    if failed:
        return (f"Could not delete {failed} mixes.", True)
    return None


def _handle_home_event(event: dict, tokens: dict[str, str]) -> None:
    kind = str(event.get("type") or "")
    ident = str(event.get("id") or "")
    if kind == "new_setlist":
        lists = load_setlists(_setlists_path())
        created = create_setlist(lists, ident)
        save_setlists(_setlists_path(), lists)
        _remember_setlist(str(created["id"]))
        st.rerun()
        return
    if kind == "add_to_setlist":
        setlist_id, sep, token = ident.partition("|")
        run_dir = tokens.get(token) if sep else ""
        if not setlist_id or not run_dir:
            _flash("Create a setlist first.", error=True)
            st.rerun()
            return
        lists = load_setlists(_setlists_path())
        if add_song(lists, setlist_id, run_dir):
            save_setlists(_setlists_path(), lists)
        _remember_setlist(setlist_id)
        st.rerun()
        return
    if kind == "remove_from_setlist":
        setlist_id, sep, token = ident.partition("|")
        run_dir = tokens.get(token) if sep else ""
        if setlist_id and run_dir:
            lists = load_setlists(_setlists_path())
            if remove_song(lists, setlist_id, run_dir):
                save_setlists(_setlists_path(), lists)
            _remember_setlist(setlist_id)
        st.rerun()
        return
    if kind == "rename_setlist":
        setlist_id, sep, name = ident.partition("|")
        if sep and setlist_id:
            lists = load_setlists(_setlists_path())
            if rename_setlist(lists, setlist_id, name):
                save_setlists(_setlists_path(), lists)
            _remember_setlist(setlist_id)
        st.rerun()
        return
    if kind == "set_locale":
        st.session_state["home_v2_locale"] = "ko" if ident == "ko" else "en"
        st.rerun()
        return
    if kind in {"delete_song", "delete_songs"}:
        from ui.isolate_jobs import read_background_analysis_status

        moved, failed = apply_library_deletes(
            tokens,
            ident,
            data_dir=DATA_DIR,
            status=read_background_analysis_status(),
        )
        message = library_delete_message(moved, failed)
        if message is not None:
            _flash(message[0], error=message[1])
        st.rerun()
        return
    if kind == "recover_song":
        run_dir = tokens.get(ident)
        if run_dir:
            recover_library_song(Path(run_dir), DATA_DIR)
        st.rerun()
        return
    if kind == "delete_setlist":
        lists = load_setlists(_setlists_path())
        if delete_setlist(lists, ident):
            save_setlists(_setlists_path(), lists)
        if str(st.session_state.get("home_v2_setlist") or "") == ident:
            _remember_setlist("")
        st.rerun()
        return
    if kind == "select_setlist":
        _remember_setlist(ident)
        st.rerun()
        return
    if kind == "open_card":
        # "Pick a local file" still opens Isolation. Page cards were removed;
        # YouTube / Tab PDF live in the sidebar only.
        if ident == "learn":
            open_new_separation()
        return
    run_dir = tokens.get(ident)
    if not run_dir:
        return
    if kind in {"open_recent", "open_row"}:
        open_saved_run(run_dir)
    elif kind == "practice":
        open_practice_run(run_dir)
    elif kind == "analyse":
        st.session_state["home_v2_pending"] = run_dir
        st.rerun()


def _main_v2() -> None:
    st.html('<div class="att-home-v2" hidden></div>')
    stored_nonce = st.session_state.get("home_v2_nonce")
    stored_tokens = dict(st.session_state.get("home_v2_tokens") or {})
    owner = load_persist_isolate_user_id(st.session_state, DATA_DIR / ISOLATE_USER_ID_FILENAME)
    rows = library_rows(owner if isinstance(owner, str) else None)
    # Automatic analysis lives in the background worker. This pending path is
    # only the explicit Analyse click, which still shows Home's loading state.
    pending = st.session_state.get("home_v2_pending")
    if not st.session_state.get("home_v2_first_paint_done"):
        st.session_state["home_v2_first_paint_done"] = True
    show_loading = bool(pending) and not st.session_state.get("home_v2_pending_shown")
    flash = None
    if not show_loading:
        raw_flash = st.session_state.pop("home_flash", None)
        if isinstance(raw_flash, tuple) and len(raw_flash) == 2:
            flash = (str(raw_flash[0]), bool(raw_flash[1]))
    from ui.isolate_jobs import read_background_analysis_status

    props = build_home_props(
        rows,
        product=edition_product_name(),
        flash=flash,
        loading=show_loading,
        setlists=load_setlists(_setlists_path()),
        locale=str(st.session_state.get("home_v2_locale") or "en"),
        trashed=[trashed_row(path) for path in list_trashed_songs(DATA_DIR)],
        analysis_status=read_background_analysis_status(),
    )
    props["activeSetlist"] = active_setlist_id(props["setlists"])
    st.session_state["home_v2_nonce"] = props["nonce"]
    st.session_state["home_v2_tokens"] = dict(_TOKEN_MAP)
    try:
        from ui.home_component import home_build_is_complete, home_library
    except Exception:
        _main_legacy()
        return
    if not home_build_is_complete():
        _main_legacy()
        return
    try:
        event = home_library(props, key="home_v2")
    except Exception:
        _main_legacy()
        return
    if show_loading and pending:
        st.session_state["home_v2_pending_shown"] = True
        st.rerun()
    if isinstance(event, dict) and event.get("nonce") == stored_nonce:
        _handle_home_event(event, stored_tokens)
    if pending and st.session_state.get("home_v2_pending_shown"):
        run_dir = str(pending)
        st.session_state.pop("home_v2_pending", None)
        st.session_state.pop("home_v2_pending_shown", None)
        from ui.isolate_jobs import analyse_existing_run

        ok, message = analyse_existing_run(run_dir)
        _flash(
            message,
            error=not ok or str(message).startswith("Analysis ready with warnings"),
        )
        st.rerun()


def _matches(row: dict, query: str) -> bool:
    needle = query.strip().casefold()
    if not needle:
        return True
    hay = f"{run_label(row)} {row.get('title') or ''}".casefold()
    return needle in hay


def _main_legacy() -> None:
    st.html(_HOME_CSS)
    product = edition_product_name()
    st.title(product, anchor=False)
    st.caption("Separate a song, save audio, or sketch a tab. Everything stays on this computer.")
    flash = st.session_state.pop("home_flash", None)
    if isinstance(flash, tuple) and len(flash) == 2:
        message, is_error = flash
        if is_error:
            st.warning(str(message))
        else:
            st.success(str(message))

    owner = load_persist_isolate_user_id(st.session_state, DATA_DIR / ISOLATE_USER_ID_FILENAME)
    rows = library_rows(owner if isinstance(owner, str) else None)

    st.subheader("Recents")
    if not rows:
        st.info("No separations yet. Separate a song to see it here.")
        if st.button("Separate a song", key="home_empty_separate"):
            open_new_separation()
    else:
        recent = rows[:6]
        cols = st.columns(len(recent))
        for index, row in enumerate(recent):
            run_dir = str(row.get("run_dir") or "")
            age = relative_age(row.get("created_at"))
            with cols[index]:
                if st.button(
                    run_label(row),
                    key=f"home_recent_{index}",
                    width="stretch",
                    help=age or "Open this mix",
                ):
                    open_saved_run(run_dir)
                if age:
                    st.caption(age)

    st.subheader("Library")
    query = st.text_input(
        "Search library",
        key="home_library_query",
        placeholder="Search by name",
        label_visibility="collapsed",
    )
    shown = [row for row in rows if _matches(row, str(query or ""))]
    if rows and not shown:
        st.caption("Nothing in the library matches that search.")
    elif not rows:
        st.caption("Finished separations show up here.")
    else:
        for index, row in enumerate(shown):
            run_dir = str(row.get("run_dir") or "")
            age = relative_age(row.get("created_at"))
            label = run_label(row)
            if age:
                label = f"{label}  ·  {age}"
            summary = analysis_summary(row)
            open_col, action_col = st.columns([6, 1], gap="small")
            with open_col:
                if st.button(label, key=f"home_file_{index}", width="stretch"):
                    open_saved_run(run_dir)
                st.caption(analysis_line(summary))
            with action_col:
                if summary:
                    if st.button(
                        "Practice",
                        key=f"home_practice_{index}",
                        width="stretch",
                        help="Open the mixer with chords, sections, lyrics, and tab.",
                    ):
                        open_practice_run(run_dir)
                else:
                    if st.button(
                        "Analyse",
                        key=f"home_analyse_{index}",
                        width="stretch",
                        help="Build chords, sections, lyrics, key, and BPM for this run.",
                    ):
                        from ui.isolate_jobs import analyse_existing_run

                        with st.spinner("Analysing song…"):
                            ok, message = analyse_existing_run(run_dir)
                        _flash(
                            message,
                            error=not ok
                            or message.startswith("Analysis ready with warnings"),
                        )
                        st.rerun()


def main() -> None:
    if ui_v2_enabled():
        _main_v2()
        return
    _main_legacy()


if __name__ == "__main__":
    main()

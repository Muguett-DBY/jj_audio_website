"""Streamlit page: Audio Isolation — separate songs into instrument tracks."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

from ui.common import (
    AUDIO_UPLOAD_TYPES,
    DATA_DIR,
    ensure_src_path,
    list_recent_runs,
    rename_run_title,
    run_output_dir,
    save_upload,
)
from ui.desktop_export import (
    EXPORT_FORMAT_LABELS,
    EXPORT_FORMAT_PLAIN_LABELS,
    EXPORT_FORMATS,
    choose_export_dir,
    default_export_dir,
    export_mix_to_folder,
    export_song_dir,
    export_tracks_to_folder,
    write_wav_48_24,
    open_path_in_os,
)
from ui.desktop_notify import notify as desktop_notify
from ui.guitar_fixup import (
    GUITAR_BACKUP_NAME,
    GUITAR_HPF_TUNING_CAPTION,
    GUITAR_PREREFINE_NAME,
    GUITAR_REFINED_NAME,
    apply_mixer_guitar_fixup,
    has_guitar_backup,
    has_guitar_prerefine,
    invalidate_mixer_playback,
    load_fold_other_diagnostics,
    load_guitar_stem_quality,
    merge_artifact_updates,
    reset_mixer_guitar_fixup,
    switch_mixer_guitar_variant,
)
from ui.isolate_jobs import (
    IsolateJobSpec,
    active_job_id,
    apply_finished_job_poll_outcome,
    apply_succeeded_job_to_session,
    delete_all_finished_jobs,
    delete_finished_job,
    delete_library_run,
    enqueue_job,
    enqueue_local_batch,
    ensure_worker_started,
    format_job_error,
    jobs_visible_in_queue,
    list_jobs,
    merge_library_runs,
    pause_job,
    queued_job_ids,
    read_status,
    remove_job,
    requeue_job,
    resume_job,
    separation_in_progress,
    video_offset_sec_for_run,
    worker_busy,
)
from ui.isolate_state import (
    CUSTOM_OUTCOME_CARD,
    CUSTOM_STEM_CHOICES,
    CUSTOM_STEM_TILE_ORDER,
    DEFAULT_OUTCOME_CARD,
    DEFAULT_SPEED_PRESET,
    DEFAULT_TRACK_OPTIONS,
    OUTCOME_CARD_KEY,
    OUTCOME_CARD_ORDER,
    OUTCOME_CARDS,
    get_effective_theme_mode,
    outcome_icon_markdown,
    stem_icon_markdown,
    DEMUCS_STEM_CHECKBOX_IDS,
    DISMISSED_JOB_IDS_KEY,
    GUITAR_TRACK_OPTION_IDS,
    ISOLATE_BATCH_FP_KEY,
    ISOLATE_BATCH_PATHS_KEY,
    ISOLATE_EXPORT_DIR_KEY,
    ISOLATE_OUTPUT_NAME_PENDING_KEY,
    ISOLATE_UI_STATE_FILENAME,
    ISOLATE_YOUTUBE_URL_PENDING_KEY,
    ISOLATE_USER_ID_FILENAME,
    OPEN_MIX_TABS_KEY,
    ISOLATE_SKIP_REHYDRATE_KEY,
    LISTEN_PICKER_KEY,
    LISTEN_PICKER_NEXT_KEY,
    MIX_TAB_NOTICE_KEY,
    add_open_mix_tab,
    apply_shell_view,
    clamp_pan,
    close_open_mix_tab,
    current_setup_step,
    draft_tab_job_overlays,
    mix_tab_event_index,
    mixer_focus_run_dir,
    move_open_mix_tab,
    pending_open_job_ids,
    focus_new_draft_tab,
    open_opened_audio_tab,
    is_loaded_mix_tab,
    is_new_draft_tab,
    mark_draft_tab_processing,
    open_home_shell,
    open_mix_shell,
    open_mix_tabs_for_session,
    open_new_draft_tab,
    ensure_upload_tab_focused,
    draft_tab_title,
    promote_job_origin_tab,
    save_active_draft_source,
    SHELL_TAB_KEY,
    SHELL_VIEW_KEY,
    SHELL_VIEW_NEXT_KEY,
    WORKSPACE_NEXT_KEY,
    ROFORMER_BACKEND_UI_HINT,
    SPEED_PRESETS,
    TRACK_OPTIONS,
    VOCALS_INSTRUMENTAL_OPTION_ID,
    apply_listen_picker_pending,
    apply_pending_output_name,
    apply_pending_youtube_url,
    apply_stored_isolate_ui_state,
    apply_workspace_tab,
    apply_youtube_output_name_sync,
    LITE_MAX_DURATION_SEC,
    DEFAULT_LITE_GUITAR_ENGINE,
    LITE_GUITAR_ENGINE_KEY,
    LITE_GUITAR_ENGINE_LABELS,
    LITE_GUITAR_ENGINE_ORDER,
    apply_lite_guitar_engine,
    clamp_region_bounds,
    custom_stems_from_options,
    default_guitar_track_option,
    dismiss_failed_job,
    effective_clip_seconds,
    format_source_caption,
    format_source_title,
    guitar_track_radio_ids,
    infer_source_kind,
    is_pro_mode,
    is_stopping_previous_job,
    isolate_ui_state_payload,
    job_requires_roformer_backend,
    jobs_needing_os_notify,
    lite_guitar_engine_allowed,
    lite_guitar_engine_disabled_reason,
    lite_roformer_enqueue_block_reason,
    resolve_lite_guitar_engine,
    DEMUCS_OVERLAP_OPTIONS,
    DEMUCS_SHIFT_OPTIONS,
    FOLD_OTHER_MODE_ORDER,
    PRO_LOW_END_RESTORE_MAX_DB,
    demucs_compute_is_overridden,
    quality_demucs_overlap,
    quality_demucs_shifts,
    resolve_pro_engine_job_fields,
    sync_demucs_compute_defaults,
    listen_picker_default,
    load_persist_isolate_user_id,
    mixer_local_video_path,
    mixer_youtube_video_id,
    prefer_local_mixer_picture,
    migrate_track_options,
    mixer_component_key,
    ISOLATE_METRO_ACCENT_KEY,
    ISOLATE_METRO_APPLIED_KEY,
    ISOLATE_METRO_FOLLOW_KEY,
    ISOLATE_METRO_RATE_KEY,
    ISOLATE_METRO_SOUND_KEY,
    metronome_applied_stamp,
    metronome_render_from_session,
    seed_metronome_widgets_for_run,
    normalize_guitar_track_selection,
    os_notify_message,
    outcome_card_for_options,
    partition_queue_jobs,
    paused_job_caption,
    pending_audio_needs_resave,
    pending_upload_fp_for_stale,
    consume_mix_tab_event,
    isolate_poll_requires_full_rerun,
    isolate_scroll_top_token,
    MIX_TAB_EVENT_SEQ_KEY,
    plan_isolate_job_poll,
    promote_default_guitar_option,
    request_isolate_scroll_top,
    queue_reopen_output_name,
    queue_youtube_url,
    queued_wait_caption,
    read_isolate_ui_state,
    recent_runs_with_owner_fallback,
    reset_new_tab_source,
    adopt_audio_into_run,
    default_isolate_device,
    discard_youtube_staging,
    resolve_outcome_card,
    resolve_speed_preset,
    resolve_tab_pdf_carry_from_isolate,
    resolve_track_selection,
    resolve_youtube_job_name,
    roformer_speed_note,
    running_progress_view,
    refresh_open_mix_action,
    select_rehydrate_row,
    SETUP_STEP_KEY,
    SETUP_STEP_SEPARATE,
    SETUP_STEP_SOURCE,
    session_mixer_artifacts_ok,
    setup_continue_enabled,
    should_hide_stale_results,
    should_show_failed_job,
    staged_audio_for_new_tab,
    status_strip_waiting_caption,
    stem_group_fields,
    stem_label_for_id,
    sync_output_name_on_upload,
    toggle_custom_stem_options,
    tracks_picker_help,
    upload_fingerprint,
    write_isolate_ui_state,
    get_iframe_theme_vars,
)
from ui.opened_audio import (
    PENDING_OPEN_ENV,
    claim_opened_paths,
    is_opened_audio,
    is_opened_video,
)
from ui.media import (
    ensure_mixer_audio_paths,
    ensure_region_preview_wav,
    media_url_for_file,
    register_mixer_media,
    stem_media_urls,
)
from ui.project import (
    PROJECT_FILENAME,
    load_analysis_for_run,
    load_project,
    load_tab_for_run,
    save_project,
    seed_mixer,
    update_from_mixer,
)
from ui.stem_mixer_component import component_build_available, stem_mixer
from ui.youtube_search import (
    has_next_page,
    hits_as_dicts,
    meta_line,
    normalize_page,
    page_offset,
    result_range_caption,
)

ensure_src_path()

from audio_to_tab.hardware import (  # noqa: E402
    CUDA_UNAVAILABLE_MESSAGE,
    desktop_device_options,
    desktop_recommend_caption,
    desktop_system_summary,
    ensure_cuda_available,
    get_desktop_probe,
    lite_auto_choice,
    lite_auto_speed_id,
    lite_detected_caption,
    lite_device_choice_ids,
    lite_device_plain_label,
    lite_using_caption,
    roformer_max_audio_sec,
)
from audio_to_tab.ingest import (  # noqa: E402
    YouTubeDownloadError,
    YouTubeSearchError,
    download_direct_media,
    download_youtube_audio,
    download_youtube_preview_video,
    format_youtube_duration,
    is_direct_media_url,
    is_youtube_url,
    normalize_audio,
    search_youtube_videos,
)
from audio_to_tab.isolate import (  # noqa: E402
    DEMUCS_INSTALL_HINT,
    MIN_REGION_SEC,
    RegionError,
    format_region_label,
    format_region_label_filename,
    format_time_sec,
    probe_duration_sec,
    resolve_region,
)
from audio_to_tab.metronome import (  # noqa: E402
    METRONOME_DIAGNOSTICS_NAME,
    analysis_wav_for_downbeat,
    click_grid_for_playback,
    coerce_metronome_render_options,
    load_metronome_diagnostics,
    rebake_metronome_artifact,
    steady_click_times,
)
from audio_to_tab.mixer import (  # noqa: E402
    DB_DEFAULT,
    PLAYBACK_RATE_DEFAULT,
    PLAYBACK_SEMITONES_DEFAULT,
    apply_playback_transform,
    clamp_playback_rate,
    clamp_playback_semitones,
    default_muted_for,
    effective_linear_gains,
    mix_stems_to_wav,
    sort_stem_names,
    stem_display_name,
    waveform_peaks,
)
from audio_to_tab.pipeline import YOUTUBE_DISCLAIMER  # noqa: E402
from audio_to_tab.roformer import (  # noqa: E402
    ROFORMER_INSTALL_HINT,
    is_roformer_backend_available,
)
from audio_to_tab.separate import is_demucs_available  # noqa: E402

logger = logging.getLogger(__name__)

STEM_HINTS = {
    "piano": "Less accurate than other tracks. Demucs piano bleeds heavily.",
    "guitar": "Isolation is harder than vocals/drums/bass; other instruments still bleed in",
    "guitar1": "Legacy spatial label",
    "guitar2": "Legacy spatial label",
    "metronome": "Click track",
}

PAGE_TITLE_HELP = (
    "Split a song into separate tracks on this computer. "
    "Use Home to separate, open a mix tab to listen, and open Queue for jobs. "
    "For download-only YouTube saves, use the YouTube to MP3 page."
)
SEPARATE_TRACKS_HELP = (
    "Pick what you want out. Lite auto-picks speed and device for this machine; "
    "guitar can still pick up other instruments."
)
SECTION_OPTIONAL_HELP = (
    "Choose how much of the file to separate. Default is the whole track. "
    "On Lite, long songs use a lot of RAM and time. Try Safer: first 90 s if needed."
)
DOWNLOADS_HELP = "Export stems or a mix from the current Mixer run."

ISOLATE_YOUTUBE_SEARCH_OPEN_KEY = "isolate_youtube_search_open"
ISOLATE_YOUTUBE_SEARCH_HITS_KEY = "isolate_youtube_search_hits"
ISOLATE_YOUTUBE_SEARCH_ERROR_KEY = "isolate_youtube_search_error"
ISOLATE_YOUTUBE_SEARCH_PAGE_KEY = "isolate_youtube_search_page"
ISOLATE_YOUTUBE_SEARCH_QUERY_KEY = "isolate_youtube_search_query"
ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE = 9
ISOLATE_YOUTUBE_AUTO_DOWNLOAD_KEY = "isolate_youtube_auto_download"
ISOLATE_YOUTUBE_DOWNLOADING_KEY = "_isolate_youtube_downloading"
LITE_GUITAR_FELL_BACK_KEY = "_isolate_lite_guitar_fell_back"
ISOLATE_QUEUE_PANEL_OPEN_KEY = "isolate_queue_panel_open"
BUSY_TAB_CLOSE_KEY = "_isolate_busy_tab_close"


def _close_youtube_search_dialog() -> None:
    st.session_state[ISOLATE_YOUTUBE_SEARCH_OPEN_KEY] = False


def _clear_youtube_search_results() -> None:
    st.session_state.pop(ISOLATE_YOUTUBE_SEARCH_HITS_KEY, None)
    st.session_state.pop(ISOLATE_YOUTUBE_SEARCH_ERROR_KEY, None)
    st.session_state[ISOLATE_YOUTUBE_SEARCH_PAGE_KEY] = 1


def _run_youtube_search(query: str, *, page: int) -> None:
    """Fetch one page of results into session state."""
    page = normalize_page(page)
    st.session_state[ISOLATE_YOUTUBE_SEARCH_PAGE_KEY] = page
    try:
        with st.spinner("Searching YouTube…"):
            hits = search_youtube_videos(
                query,
                max_results=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE,
                offset=page_offset(page, page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE),
            )
        st.session_state[ISOLATE_YOUTUBE_SEARCH_HITS_KEY] = hits_as_dicts(hits)
        st.session_state.pop(ISOLATE_YOUTUBE_SEARCH_ERROR_KEY, None)
        if not hits:
            st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = (
                "No public videos matched that search."
                if page <= 1
                else "No more public videos for this search."
            )
    except ValueError as exc:
        st.session_state[ISOLATE_YOUTUBE_SEARCH_HITS_KEY] = []
        st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = str(exc)
    except YouTubeSearchError as exc:
        st.session_state[ISOLATE_YOUTUBE_SEARCH_HITS_KEY] = []
        st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = str(exc)
    except Exception as exc:
        st.session_state[ISOLATE_YOUTUBE_SEARCH_HITS_KEY] = []
        st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = (
            f"YouTube search failed: {exc}"
        )


def _stage_youtube_audio(
    url: str,
    *,
    show_spinner: bool = True,
    spinner_label: str | None = None,
) -> tuple[Path | None, str | None]:
    """Download YouTube audio into pending staging. Returns (path, error)."""
    prev_path = st.session_state.get("isolate_pending_audio_path")
    prev_fp = st.session_state.get("isolate_pending_fp") or st.session_state.get(
        "isolate_upload_fp"
    )
    label = spinner_label or "Downloading YouTube audio…"
    try:
        if show_spinner:
            with st.status(label, expanded=True) as status:
                st.write("Fetching audio from YouTube. This can take a minute.")
                path = download_youtube_audio(url, run_output_dir())
                status.update(label=f"Ready: {path.stem}", state="complete")
        else:
            path = download_youtube_audio(url, run_output_dir())
        if prev_path and str(prev_path) != str(path):
            discard_youtube_staging(
                str(prev_path),
                fingerprint=str(prev_fp) if prev_fp else None,
                retain_paths=(path,),
            )
        st.session_state["isolate_pending_audio_path"] = str(path)
        st.session_state["isolate_pending_fp"] = f"youtube:{url}"
        st.session_state["isolate_upload_fp"] = f"youtube:{url}"
        st.session_state.pop("isolate_duration_sec", None)
        st.session_state.pop("isolate_duration_fp", None)
        apply_youtube_output_name_sync(
            st.session_state,
            downloaded_stem=path.stem,
            apply_now=True,
        )
        return path, None
    except YouTubeDownloadError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"YouTube download failed: {exc}"


def _download_youtube_with_status(url: str, *, title: str) -> str | None:
    """Show a visible download status, then stage audio. Returns error or None."""
    label = f"Downloading “{title}”…"
    st.session_state[ISOLATE_YOUTUBE_DOWNLOADING_KEY] = title
    try:
        with st.status(label, expanded=True) as status:
            st.write("Fetching audio from YouTube. This can take a minute.")
            path, err = _stage_youtube_audio(url, show_spinner=False)
            if err:
                status.update(label="Download failed", state="error")
                return err
            ready_name = Path(path).stem if path else title
            status.update(label=f"Ready: {ready_name}", state="complete")
            return None
    finally:
        st.session_state.pop(ISOLATE_YOUTUBE_DOWNLOADING_KEY, None)


@st.dialog("Search YouTube", width="large", on_dismiss=_close_youtube_search_dialog)
def _youtube_search_dialog() -> None:
    """Centered modal: search public videos, pick one to fill the URL and auto-download."""
    st.caption(
        "Find a public video. **Use** downloads it so you can preview a section. "
        "**Separate** queues the outcome already chosen on Home."
    )
    with st.form("isolate_youtube_search_form", clear_on_submit=False, border=False):
        search_cols = st.columns([4, 1], vertical_alignment="bottom")
        with search_cols[0]:
            search_q = st.text_input(
                "Song or artist",
                key=ISOLATE_YOUTUBE_SEARCH_QUERY_KEY,
                placeholder="e.g. artist, song title",
            )
        with search_cols[1]:
            do_search = st.form_submit_button(
                "Search",
                type="primary",
                width="stretch",
            )
    secondary = st.columns([1, 1, 4], vertical_alignment="center")
    with secondary[0]:
        if st.button(
            "Clear",
            key="isolate_youtube_search_clear",
            type="secondary",
            width="stretch",
        ):
            _clear_youtube_search_results()
            st.rerun()
    with secondary[1]:
        if st.button(
            "Close",
            key="isolate_youtube_search_close",
            type="secondary",
            width="stretch",
        ):
            _close_youtube_search_dialog()
            st.rerun()
    if do_search:
        _run_youtube_search(search_q, page=1)
    search_error = st.session_state.get(ISOLATE_YOUTUBE_SEARCH_ERROR_KEY)
    if search_error:
        st.warning(str(search_error))
    downloading_title = st.session_state.get(ISOLATE_YOUTUBE_DOWNLOADING_KEY)
    if downloading_title:
        st.info(f"Downloading **{downloading_title}**…")
    page = normalize_page(st.session_state.get(ISOLATE_YOUTUBE_SEARCH_PAGE_KEY, 1))
    hits_state = st.session_state.get(ISOLATE_YOUTUBE_SEARCH_HITS_KEY) or []
    if not isinstance(hits_state, list):
        hits_state = []
    busy = bool(downloading_title)
    if hits_state:
        st.caption(
            result_range_caption(
                page,
                len(hits_state),
                page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE,
            )
        )
        hits_for_grid = [hit for hit in hits_state if isinstance(hit, dict)][
            :ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE
        ]
        for row_index in range(3):
            grid_row = st.columns(3)
            for col_index in range(3):
                with grid_row[col_index]:
                    slot = row_index * 3 + col_index
                    if slot >= len(hits_for_grid):
                        continue
                    hit = hits_for_grid[slot]
                    vid = str(hit.get("video_id") or "")
                    title = str(hit.get("title") or "Untitled")
                    dur = format_youtube_duration(hit.get("duration_sec"))
                    url = str(hit.get("url") or "").strip()
                    thumb = str(hit.get("thumbnail_url") or "").strip()
                    meta = meta_line(hit, duration_label=dur)
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    st.markdown(f"**{title}**")
                    if meta:
                        st.caption(meta)
                    if st.button(
                        "Use",
                        key=f"isolate_youtube_pick_p{page}_{vid}",
                        type="primary",
                        disabled=not url or busy,
                        width="stretch",
                        help="Download this track’s audio into New.",
                    ):
                        queue_youtube_url(st.session_state, url, title=title)
                        st.session_state.pop(ISOLATE_YOUTUBE_SEARCH_ERROR_KEY, None)
                        # Download in the dialog so Use shows progress instead of
                        # closing first and leaving a blank gap.
                        err = _download_youtube_with_status(url, title=title)
                        if err:
                            st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = err
                            st.rerun()
                        st.session_state[SETUP_STEP_KEY] = SETUP_STEP_SEPARATE
                        _close_youtube_search_dialog()
                        st.session_state.pop(ISOLATE_YOUTUBE_AUTO_DOWNLOAD_KEY, None)
                        st.session_state["isolate_flash"] = (
                            f"Downloaded **{title}**. Pick a section or separate tracks."
                        )
                        st.rerun()
                    if st.button(
                        "Separate",
                        key=f"isolate_youtube_separate_p{page}_{vid}",
                        type="secondary",
                        disabled=not url or busy,
                        width="stretch",
                        help="Download this track and queue the outcome already chosen on Home.",
                    ):
                        queue_youtube_url(st.session_state, url, title=title)
                        st.session_state.pop(ISOLATE_YOUTUBE_SEARCH_ERROR_KEY, None)
                        err = _download_youtube_with_status(url, title=title)
                        if err:
                            st.session_state[ISOLATE_YOUTUBE_SEARCH_ERROR_KEY] = err
                            st.rerun()
                        _close_youtube_search_dialog()
                        st.session_state.pop(ISOLATE_YOUTUBE_AUTO_DOWNLOAD_KEY, None)
                        st.session_state["isolate_search_enqueue"] = True
                        st.rerun()
                    if url:
                        st.markdown(f"[Open on YouTube]({url})")
    if hits_state or page > 1:
        nav = st.columns([1, 2, 1], vertical_alignment="center")
        with nav[0]:
            if st.button(
                "Previous",
                key="isolate_youtube_search_prev",
                type="secondary",
                disabled=page <= 1 or busy,
                width="stretch",
            ):
                _run_youtube_search(
                    str(st.session_state.get(ISOLATE_YOUTUBE_SEARCH_QUERY_KEY) or ""),
                    page=page - 1,
                )
                st.rerun()
        with nav[1]:
            st.caption(
                result_range_caption(
                    page,
                    len(hits_state),
                    page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE,
                )
            )
        with nav[2]:
            if st.button(
                "Next",
                key="isolate_youtube_search_next",
                type="secondary",
                disabled=not has_next_page(len(hits_state), page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE)
                or busy,
                width="stretch",
            ):
                _run_youtube_search(
                    str(st.session_state.get(ISOLATE_YOUTUBE_SEARCH_QUERY_KEY) or ""),
                    page=page + 1,
                )
                st.rerun()

def _stateful_expander(label: str, *, key: str, default: bool = False):
    """
    Expander that remembers whether the user left it open.

    The live mixer reruns the page on every interaction. Without a key the
    expander would snap back to ``default`` each time; with one, Streamlit
    tracks the open state in ``st.session_state[key]``. ``default`` and
    ``label`` must stay constant across reruns or the widget identity changes
    and the state is lost.
    """
    return st.expander(label, expanded=default, key=key)


def _get_browser_user_id() -> str:
    """Anonymous owner id for this session, persisted to a local file."""
    return load_persist_isolate_user_id(st.session_state, DATA_DIR / ISOLATE_USER_ID_FILENAME)


def _isolate_ui_state_path() -> Path:
    return DATA_DIR / ISOLATE_UI_STATE_FILENAME


def _persist_isolate_ui_state() -> None:
    write_isolate_ui_state(_isolate_ui_state_path(), isolate_ui_state_payload(st.session_state))


def _restore_isolate_ui_state() -> None:
    apply_stored_isolate_ui_state(
        st.session_state,
        read_isolate_ui_state(_isolate_ui_state_path()),
    )


def _session_has_mixer_wavs() -> bool:
    return session_mixer_artifacts_ok(
        st.session_state,
        wav_exists=lambda path: Path(path).is_file(),
    )


def _wav_exists(path: str) -> bool:
    return Path(path).is_file()


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_waveform_peaks(path_str: str, mtime_ns: int, num_points: int) -> list[float]:
    try:
        return waveform_peaks(Path(path_str), num_points=num_points).tolist()
    except Exception as exc:
        logger.warning("Could not compute waveform peaks for %s: %s", path_str, exc)
        return []


def _stem_waveform_peaks(path: Path, *, num_points: int = 512) -> list[float]:
    """Mixer waveform peaks, cached per file version.

    Uncached, every stem was decoded again on each rerun of the mixer fragment —
    six full-song stems per pass. Keyed on mtime so a guitar fix-up that rewrites
    a stem still invalidates.
    """
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return []
    return _cached_waveform_peaks(str(path), mtime_ns, num_points)


def _artifact_fingerprint(stem_paths: dict[str, Path]) -> str:
    payload = "|".join(f"{k}:{v}" for k, v in sorted((n, str(p)) for n, p in stem_paths.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stem_paths_from_artifacts(artifacts_map: dict) -> dict[str, Path]:
    return {
        name: Path(path)
        for name, path in artifacts_map.items()
        if Path(path).exists()
        and name
        not in (
            "guitar_split_diagnostics",
            "stem_presence_diagnostics",
            "bass_bleed_diagnostics",
        )
        and Path(path).suffix.lower() == ".wav"
    }


def _clear_pending_source() -> None:
    st.session_state.pop("isolate_pending_audio_path", None)
    st.session_state.pop("isolate_pending_fp", None)
    st.session_state.pop("isolate_upload_fp", None)
    st.session_state.pop("isolate_duration_sec", None)
    st.session_state.pop("isolate_duration_fp", None)


def _duration_cache_key(audio_path: Path) -> str:
    upload_fp = st.session_state.get("isolate_upload_fp")
    if upload_fp:
        return str(upload_fp)
    try:
        stat = audio_path.stat()
        return f"path:{audio_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"path:{audio_path.resolve()}"


def _cached_probe_duration_sec(audio_path: Path) -> float | None:
    """Probe file length once per upload; avoid ffprobe on every Streamlit rerun."""
    cache_key = _duration_cache_key(audio_path)
    cached_fp = st.session_state.get("isolate_duration_fp")
    if cached_fp == cache_key and "isolate_duration_sec" in st.session_state:
        return st.session_state.get("isolate_duration_sec")
    duration = probe_duration_sec(audio_path)
    st.session_state["isolate_duration_fp"] = cache_key
    st.session_state["isolate_duration_sec"] = duration
    return duration


def _sync_upload_output_name(uploaded: object) -> None:
    last_fp = st.session_state.get("isolate_upload_fp")
    current_name = st.session_state.get("isolate_output_name", "")
    new_fp, synced_name, changed = sync_output_name_on_upload(
        uploaded,
        last_fp=last_fp,
        output_name=current_name,
    )
    if new_fp is not None:
        st.session_state["isolate_upload_fp"] = new_fp
        if last_fp != new_fp:
            st.session_state.pop("isolate_duration_sec", None)
            st.session_state.pop("isolate_duration_fp", None)
    if changed:
        st.session_state["isolate_output_name"] = synced_name


def _selection_fingerprint(stem_paths: dict[str, Path]) -> str:
    return _artifact_fingerprint(stem_paths)


def _mixer_export_fingerprint(
    stem_names: list[str],
    volumes_db: dict,
    muted: dict,
    soloed: dict,
    master_volume_db: float = DB_DEFAULT,
    *,
    playback_rate: float = 1.0,
    playback_semitones: int = 0,
    pans: dict | None = None,
) -> str:
    parts = [
        f"master:{float(master_volume_db):.2f}",
        f"rate:{float(playback_rate):.2f}",
        f"key:{int(playback_semitones)}",
    ]
    any_solo = any(bool(soloed.get(n, False)) for n in stem_names)
    for name in stem_names:
        part = (
            f"{name}:{float(volumes_db.get(name, DB_DEFAULT)):.2f}"
            f":{int(bool(muted.get(name, False)))}:{int(bool(soloed.get(name, False)))}"
        )
        # Only audible pan changes the file; silence has no stereo image.
        audible = bool(soloed.get(name, False)) if any_solo else not bool(
            muted.get(name, False)
        )
        if pans is not None and audible:
            part += f":pan{clamp_pan(pans.get(name))}"
        parts.append(part)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _load_bass_bleed_diagnostics(artifacts: dict) -> dict:
    path = artifacts.get("bass_bleed_diagnostics")
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_low_end_recovery_diagnostics(artifacts: dict) -> dict:
    path = artifacts.get("low_end_recovery_diagnostics")
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_guitar_fixup_panel(
    stem_paths: dict[str, Path],
    run_dir: Path,
    artifacts_map: dict,
    bass_bleed: dict,
) -> None:
    """Post-separation guitar tweak — no Demucs re-run required."""
    if "guitar" not in stem_paths:
        return

    guitar_path = stem_paths["guitar"]
    run_fp = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:12]
    has_bass = bool(stem_paths.get("bass") and stem_paths["bass"].is_file())
    backup_exists = has_guitar_backup(guitar_path)
    prerefine_exists = has_guitar_prerefine(guitar_path)
    refined_exists = (run_dir / GUITAR_REFINED_NAME).is_file()
    flagged = bool(bass_bleed.get("flagged"))
    recovery = _load_low_end_recovery_diagnostics(artifacts_map)
    quality = load_guitar_stem_quality(artifacts_map or {})
    fold = load_fold_other_diagnostics(artifacts_map or {})

    debleed_key = f"mixer_fixup_debleed_{run_fp}"
    restore_key = f"mixer_fixup_restore_{run_fp}"
    variant_key = f"mixer_guitar_variant_{run_fp}"
    if debleed_key not in st.session_state:
        st.session_state[debleed_key] = flagged and has_bass
    if restore_key not in st.session_state:
        st.session_state[restore_key] = 0.0

    with st.expander(
        "Guitar fix-up (after separation, no re-run)",
        expanded=flagged or bool(recovery.get("attempted")) or prerefine_exists,
    ):
        st.caption(
            "Fix boomy bass bleed or thin low notes on the **guitar** stem only. "
            f"Each apply resets from `{GUITAR_BACKUP_NAME}` (saved on first apply)."
        )

        if quality.get("high_end_energy_share") is not None:
            high_share = quality["high_end_energy_share"]
            competitor_overlap = quality.get("competitor_overlap") or {}
            overlap = max(competitor_overlap.values()) if competitor_overlap else None
            if overlap is None and isinstance(
                quality.get("low_band_energy_share"), (int, float)
            ):
                overlap = quality["low_band_energy_share"]
            st.caption(
                f"Brightness: {'high' if high_share > 0.3 else 'moderate' if high_share > 0.15 else 'low'}"
                f" | Cross-bleed: {'significant' if overlap > 0.3 else 'moderate' if overlap > 0.15 else 'minimal'}"
            )
            if fold.get("folded"):
                st.caption(f"Other folded: {fold.get('reason', 'yes')}")

        if prerefine_exists and refined_exists:
            options = {
                "refined": f"MelBand refined (`{GUITAR_REFINED_NAME}`)",
                "prerefine": f"Pre-refine BS-RoFormer (`{GUITAR_PREREFINE_NAME}`)",
            }
            if variant_key not in st.session_state:
                st.session_state[variant_key] = "refined"
            choice = st.radio(
                "Guitar version",
                options=list(options.keys()),
                format_func=lambda k: options[k],
                key=variant_key,
                help="MelBand refine can soften highs. Compare without re-separating.",
            )
            use_prerefine = choice == "prerefine"
            if st.button(
                "Load selected guitar version",
                key=f"mixer_variant_apply_{run_fp}",
            ):
                result = switch_mixer_guitar_variant(
                    guitar_path=guitar_path,
                    stem_paths=stem_paths,
                    run_dir=run_dir,
                    use_prerefine=use_prerefine,
                )
                if result.ok:
                    st.session_state["isolate_artifacts"] = merge_artifact_updates(
                        st.session_state.get("isolate_artifacts") or artifacts_map,
                        result.artifact_updates,
                    )
                    invalidate_mixer_playback(st.session_state)
                    st.success(result.message)
                    st.rerun()
                else:
                    st.error(result.message)

        if flagged:
            share = bass_bleed.get("low_band_energy_share")
            hint = (
                f"Bass-heavy guitar detected (low-band share {share:.2f}). "
                "Try **Quick de-bleed** below."
                if isinstance(share, (int, float))
                else "Bass-heavy guitar detected. Try **Quick de-bleed** below."
            )
            st.info(hint)

        col_debleed, col_restore = st.columns(2)
        with col_debleed:
            st.checkbox(
                "Subtractive bass de-bleed",
                help=(
                    "Subtract leaked bass/drum energy below ~150 Hz using the separated "
                    "bass stem from this run. "
                    + GUITAR_HPF_TUNING_CAPTION
                ),
                disabled=not has_bass,
                key=debleed_key,
            )
            if not has_bass:
                st.caption("No bass stem in this run. De-bleed unavailable.")
            st.caption(GUITAR_HPF_TUNING_CAPTION)
        with col_restore:
            st.slider(
                "Low-end restore (dB)",
                min_value=0.0,
                max_value=6.0,
                step=1.0,
                help="Optional 60–200 Hz boost after de-bleed if the guitar sounds thin.",
                key=restore_key,
            )

        if recovery.get("attempted"):
            st.caption(
                f"Last fix: {recovery.get('reason', 'applied')} "
                f"(de-bleed={recovery.get('sub_bass_debleed_applied')}, "
                f"restore={recovery.get('harmonic_restore_applied')})"
            )

        btn_apply, btn_quick, btn_reset = st.columns(3)
        with btn_apply:
            apply_clicked = st.button(
                "Apply to guitar",
                type="primary",
                key=f"mixer_fixup_apply_{run_fp}",
            )
        with btn_quick:
            quick_clicked = st.button(
                "Quick de-bleed",
                key=f"mixer_fixup_quick_{run_fp}",
                disabled=not has_bass,
                help="Subtractive bass de-bleed only (no restore). Add restore manually if needed.",
            )
        with btn_reset:
            reset_clicked = st.button(
                "Reset fix-up",
                key=f"mixer_fixup_reset_{run_fp}",
                disabled=not backup_exists,
                help=f"Restore guitar from `{GUITAR_BACKUP_NAME}` before fix-up.",
            )

        result = None
        if apply_clicked:
            with st.spinner("Applying guitar fix-up..."):
                result = apply_mixer_guitar_fixup(
                    guitar_path=guitar_path,
                    stem_paths=stem_paths,
                    run_dir=run_dir,
                    sub_bass_debleed=bool(st.session_state.get(debleed_key)),
                    low_end_restore_db=float(st.session_state.get(restore_key) or 0.0),
                )
        elif quick_clicked:
            with st.spinner("Applying quick de-bleed..."):
                result = apply_mixer_guitar_fixup(
                    guitar_path=guitar_path,
                    stem_paths=stem_paths,
                    run_dir=run_dir,
                    sub_bass_debleed=True,
                    low_end_restore_db=0.0,
                )
            st.session_state[debleed_key] = True
            st.session_state[restore_key] = 0.0
        elif reset_clicked:
            with st.spinner("Resetting guitar fix-up..."):
                result = reset_mixer_guitar_fixup(
                    guitar_path=guitar_path,
                    stem_paths=stem_paths,
                    run_dir=run_dir,
                )

        if result is None:
            return

        if result.ok:
            st.session_state["isolate_artifacts"] = merge_artifact_updates(
                st.session_state.get("isolate_artifacts") or artifacts_map,
                result.artifact_updates,
            )
            invalidate_mixer_playback(st.session_state)
            st.success(result.message)
            st.rerun()
        else:
            st.error(result.message)


def _load_guitar_split_diagnostics(artifacts: dict) -> dict:
    path = artifacts.get("guitar_split_diagnostics")
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolved_export_dir() -> Path:
    raw = st.session_state.get(ISOLATE_EXPORT_DIR_KEY)
    if raw:
        path = Path(str(raw)).expanduser()
        if path.exists() or path.parent.exists():
            return path
    path = default_export_dir()
    st.session_state[ISOLATE_EXPORT_DIR_KEY] = str(path)
    return path


def _build_current_mix(
    selected_stem_paths: dict[str, Path],
    run_dir: Path,
    stem_names: list[str],
    volumes_db: dict,
    muted: dict,
    soloed: dict,
    master_volume_db: float = DB_DEFAULT,
    *,
    playback_rate: float = PLAYBACK_RATE_DEFAULT,
    playback_semitones: int = PLAYBACK_SEMITONES_DEFAULT,
    pans: dict | None = None,
) -> Path:
    mix_path = run_dir / "current_mix.wav"
    rate = clamp_playback_rate(playback_rate)
    semis = clamp_playback_semitones(playback_semitones)
    gains = effective_linear_gains(
        stem_names,
        muted={n: bool(muted.get(n, False)) for n in stem_names},
        soloed={n: bool(soloed.get(n, False)) for n in stem_names},
        volume_db={n: float(volumes_db.get(n, DB_DEFAULT)) for n in stem_names},
        master_volume_db=master_volume_db,
    )
    mix_stems_to_wav(
        selected_stem_paths,
        output_path=mix_path,
        gains=gains,
        pans={n: clamp_pan((pans or {}).get(n)) for n in stem_names},
    )
    apply_playback_transform(mix_path, rate=rate, semitones=semis)
    fp = _mixer_export_fingerprint(
        stem_names,
        volumes_db,
        muted,
        soloed,
        master_volume_db,
        playback_rate=rate,
        playback_semitones=semis,
        pans=pans,
    )
    st.session_state["isolate_mix_ready"] = str(mix_path)
    st.session_state["isolate_mix_fp"] = fp
    return mix_path


def _metronome_payload_for_mixer(run_dir: Path | None) -> dict | None:
    """Grid + options for live mixer clicks; None falls back to baked WAV."""
    if run_dir is None:
        return None
    diag = load_metronome_diagnostics(run_dir / METRONOME_DIAGNOSTICS_NAME)
    seed_metronome_widgets_for_run(st.session_state, run_dir, diag)
    if not isinstance(diag, dict):
        return None
    times = diag.get("click_times_1x")
    if not isinstance(times, list) or not times:
        return None
    options = metronome_render_from_session(st.session_state)
    try:
        times_f = [float(t) for t in times]
    except (TypeError, ValueError):
        return None
    duration = float(diag.get("duration_sec") or 0.0)
    measure = diag.get("beats_per_measure", 4)
    times_a, ref, bpm = click_grid_for_playback(
        times_f,
        diag,
        analysis_wav_for_downbeat(run_dir),
    )
    times_f = [float(t) for t in times_a]
    steady = steady_click_times(
        duration_sec=duration,
        bpm=bpm,
        body_start_sec=float(ref or 0.0),
    )
    return {
        "times1x": times_f,
        "steadyTimes1x": [float(t) for t in steady],
        "refSec": float(ref or 0.0),
        "beatsPerMeasure": max(1, int(measure or 4)),
        "durationSec": duration,
        "options": {
            "accent": bool(options.accent),
            "rate": float(options.rate),
            "sound": str(options.sound),
            "follow": str(options.follow),
        },
    }


def _sync_metronome_options_from_mixer(
    mixer_state: dict,
    *,
    run_dir: Path,
    stem_paths: dict[str, Path],
) -> None:
    """Persist mixer click options and rebake export WAV without remounting stems."""
    raw = mixer_state.get("metronomeOptions")
    if not isinstance(raw, dict):
        return
    if stem_paths.get("metronome") is None:
        return
    options = coerce_metronome_render_options(
        accent=raw.get("accent", True),
        rate=raw.get("rate", 1.0),
        sound=raw.get("sound", "classic"),
        follow=raw.get("follow", "smart"),
    )
    st.session_state[ISOLATE_METRO_ACCENT_KEY] = options.accent
    st.session_state[ISOLATE_METRO_RATE_KEY] = options.rate
    st.session_state[ISOLATE_METRO_SOUND_KEY] = options.sound
    st.session_state[ISOLATE_METRO_FOLLOW_KEY] = options.follow
    stamp = metronome_applied_stamp(run_dir, options)
    if st.session_state.get(ISOLATE_METRO_APPLIED_KEY) == stamp:
        return
    result = rebake_metronome_artifact(run_dir, options)
    st.session_state[ISOLATE_METRO_APPLIED_KEY] = stamp
    if result is None:
        return
    # Keep playback mounted; only refresh download URLs on the next fragment paint.
    _persist_isolate_ui_state()


def _persist_project_from_mixer(
    run_dir: Path,
    mixer_state: dict | None,
    *,
    force: bool = False,
) -> bool:
    """Save player state at most once per second (or immediately with ``force``)."""
    if not isinstance(mixer_state, dict):
        return False
    stamp_key = f"isolate_project_save_at::{run_dir}"
    now = time.time()
    if not force and now - float(st.session_state.get(stamp_key) or 0.0) < 1.0:
        return False
    project, _warning = load_project(run_dir)
    update_from_mixer(project, mixer_state)
    wrote = save_project(run_dir, project)
    st.session_state[stamp_key] = now
    return wrote


def _render_live_mixer(
    stem_paths: dict[str, Path],
    *,
    track_title: str = "",
    base_name: str = "stems",
    run_dir: Path | None = None,
    media_urls: dict[str, str] | None = None,
    download_urls: dict[str, str] | None = None,
    tab_pdf_url: str | None = None,
) -> dict | None:
    if not component_build_available():
        st.error(
            "Live mixer is not built. "
            "Run: npm install && npm run build in ui/stem_mixer_component/frontend "
            "(or .\\scripts\\dev.ps1 mixer-build)."
        )
        return None

    mixer_paths = ensure_mixer_audio_paths(stem_paths)

    try:
        urls = media_urls if media_urls is not None else stem_media_urls(mixer_paths)
        if download_urls is None:
            download_urls = stem_media_urls(stem_paths, coord_prefix="isolate.download")
    except Exception as exc:
        st.error("Could not prepare track audio for the mixer.")
        with st.expander("Details"):
            st.exception(exc)
        return None

    stem_names = sort_stem_names(stem_paths.keys())
    project = None
    project_warning = None
    seed: dict = {}
    if run_dir is not None:
        project, project_warning = load_project(run_dir)
        if project_warning:
            st.caption(project_warning)
        seed = seed_mixer(project)
        if not (run_dir / PROJECT_FILENAME).is_file():
            # First open of an older run: create the mutable player manifest now
            # so the first mixer report has a file to update.
            save_project(run_dir, project, force=True)
    seed_mixer_state = project.get("mixer") if project else {}
    seed_mixer_state = seed_mixer_state if isinstance(seed_mixer_state, dict) else {}
    volumes = dict(seed_mixer_state.get("volumesDb") or {})
    st.session_state["isolate_volumes_db"] = volumes
    for name in stem_names:
        volumes.setdefault(name, DB_DEFAULT)
    master_db = float(
        seed_mixer_state.get(
            "masterVolumeDb",
            st.session_state.setdefault("isolate_master_volume_db", DB_DEFAULT),
        )
    )
    saved_muted = seed_mixer_state.get("muted") or {}
    saved_soloed = seed_mixer_state.get("soloed") or {}
    saved_pan = (
        seed_mixer_state.get("pan")
        if isinstance(seed_mixer_state.get("pan"), dict)
        else {}
    )
    practice_seed = {
        key: seed.get(key)
        for key in (
            "playback_rate",
            "playback_semitones",
            "initial_loop",
            "count_in_bars",
        )
    }

    analysis_payload = None
    tab_payload = None
    if run_dir is not None and project is not None:
        analysis_payload = load_analysis_for_run(run_dir, project)
        tab_payload = load_tab_for_run(run_dir, project, pdf_url=tab_pdf_url)

    safe_base = (base_name or "stems").strip() or "stems"
    stems_arg = [
        {
            "id": name,
            "label": stem_display_name(name),
            "url": urls[name],
            "downloadUrl": download_urls.get(name, urls[name]),
            "downloadFilename": f"{safe_base}_{name}.wav",
            "peaks": _stem_waveform_peaks(stem_paths[name]),
            "hint": STEM_HINTS.get(name, ""),
            **stem_group_fields(name),
        }
        for name in stem_names
        if name in urls
    ]
    key_src = str(run_dir.resolve()) if run_dir is not None else _artifact_fingerprint(stem_paths)
    mixer_key = mixer_component_key(key_src)
    metro = _metronome_payload_for_mixer(run_dir)
    youtube_id, local_path = prefer_local_mixer_picture(
        mixer_youtube_video_id(st.session_state),
        mixer_local_video_path(run_dir),
    )
    local_url = _local_mixer_video_url(run_dir) if local_path is not None else None
    if local_path is not None and not local_url:
        youtube_id = mixer_youtube_video_id(st.session_state)
    theme_vars = get_iframe_theme_vars(st.session_state)
    return stem_mixer(
        stems_arg,
        initial_volumes_db={n: float(volumes.get(n, DB_DEFAULT)) for n in stem_names},
        initial_muted={
            n: bool(saved_muted.get(n, default_muted_for(n))) for n in stem_names
        },
        initial_soloed={n: bool(saved_soloed.get(n, False)) for n in stem_names},
        initial_pan={n: clamp_pan(saved_pan.get(n)) for n in stem_names},
        initial_master_volume_db=master_db,
        track_title=track_title,
        metronome=metro,
        youtube_video_id=youtube_id,
        hide_youtube_video=bool(st.session_state.get("isolate_hide_youtube_video")),
        local_video_url=local_url,
        video_offset_sec=video_offset_sec_for_run(run_dir),
        theme_vars=theme_vars,
        analysis=analysis_payload,
        tab=tab_payload,
        initial_loop=practice_seed.get("initial_loop"),
        count_in_bars=int(practice_seed.get("count_in_bars") or 0),
        initial_rate=float(practice_seed.get("playback_rate") or 1.0),
        initial_semitones=int(practice_seed.get("playback_semitones") or 0),
        stem_trims=project.get("stem_trims") if isinstance(project, dict) else {},
        seek_sec=st.session_state.pop("isolate_seek_sec", None),
        key=mixer_key,
    )


def _init_track_picker_session(*, prefer_roformer: bool) -> None:
    """Seed track picker widget keys from legacy preset or defaults (once)."""
    if st.session_state.get("isolate_track_picker_initialized"):
        return
    migrated = migrate_track_options(st.session_state)
    migrated = promote_default_guitar_option(
        migrated,
        roformer_available=is_roformer_backend_available(),
        prefer_roformer=prefer_roformer,
    )
    for oid in DEMUCS_STEM_CHECKBOX_IDS:
        st.session_state[f"isolate_track_{oid}"] = oid in migrated
    guitar_pick = next((g for g in migrated if g in GUITAR_TRACK_OPTION_IDS), None)
    st.session_state["isolate_guitar_track"] = guitar_pick or "none"
    st.session_state["isolate_vocals_instrumental_only"] = (
        VOCALS_INSTRUMENTAL_OPTION_ID in migrated
    )
    st.session_state["isolate_track_options"] = list(migrated)
    st.session_state["isolate_track_picker_initialized"] = True
    st.session_state.pop("isolate_separation_preset", None)


def _sync_track_widgets_from_options(option_ids: list[str]) -> None:
    """Keep checkbox/radio session keys aligned with a resolved option list."""
    ids = list(option_ids)
    karaoke = ids == [VOCALS_INSTRUMENTAL_OPTION_ID]
    st.session_state["isolate_vocals_instrumental_only"] = karaoke
    if karaoke:
        for oid in DEMUCS_STEM_CHECKBOX_IDS:
            st.session_state[f"isolate_track_{oid}"] = False
        st.session_state["isolate_guitar_track"] = "none"
    else:
        for oid in DEMUCS_STEM_CHECKBOX_IDS:
            st.session_state[f"isolate_track_{oid}"] = oid in ids
        guitar_pick = next((g for g in ids if g in GUITAR_TRACK_OPTION_IDS), None)
        st.session_state["isolate_guitar_track"] = normalize_guitar_track_selection(
            guitar_pick or "none",
            roformer_available=is_roformer_backend_available(),
        )
    st.session_state["isolate_track_options"] = list(ids)


def _collect_track_options_from_session() -> list[str]:
    if st.session_state.get("isolate_vocals_instrumental_only"):
        return [VOCALS_INSTRUMENTAL_OPTION_ID]
    options: list[str] = []
    for oid in DEMUCS_STEM_CHECKBOX_IDS:
        if st.session_state.get(f"isolate_track_{oid}"):
            options.append(oid)
    guitar = str(st.session_state.get("isolate_guitar_track") or "none")
    if guitar in GUITAR_TRACK_OPTION_IDS:
        options.append(guitar)
    return options


def _render_track_picker(*, persist: dict) -> None:
    """Pro: guitar engine after the shared stem grid. Individual stems are tiles."""
    roformer_ok = is_roformer_backend_available()
    st.caption(tracks_picker_help(roformer_available=roformer_ok))
    if not roformer_ok:
        st.info(ROFORMER_BACKEND_UI_HINT)
    if st.session_state.get("isolate_vocals_instrumental_only"):
        return
    normalized_guitar = normalize_guitar_track_selection(
        str(st.session_state.get("isolate_guitar_track") or "none"),
        roformer_available=roformer_ok,
    )
    if st.session_state.get("isolate_guitar_track") != normalized_guitar:
        st.session_state["isolate_guitar_track"] = normalized_guitar
    guitar_radio_ids = guitar_track_radio_ids(roformer_available=roformer_ok)
    guitar_labels = {
        "none": "No guitar",
        **{oid: TRACK_OPTIONS[oid]["label"] for oid in guitar_radio_ids if oid != "none"},
    }
    st.radio(
        "Guitar",
        options=list(guitar_labels.keys()),
        format_func=lambda oid: guitar_labels[oid],
        key="isolate_guitar_track",
        **persist,
    )


def _tiles_use_light_icons() -> bool:
    return get_effective_theme_mode(st.session_state) == "light"


def _outcome_tile_label(card_id: str, *, selected: bool) -> str:
    """Icon above copy (same stack as Custom); stems + N tracks on one meta line."""
    spec = OUTCOME_CARDS[card_id]
    n_tracks = int(spec["n_tracks"])
    tracks = "1\u00a0track" if n_tracks == 1 else f"{n_tracks}\u00a0tracks"
    return (
        f"{outcome_icon_markdown(card_id, selected=selected, light=_tiles_use_light_icons())}\n\n"
        f"{spec['label']}\n"
        f"{spec['stems_line']} · {tracks}"
    )


def _render_outcome_tile(
    card_id: str, *, selected: bool, prefer_roformer: bool, guitar_option: str | None = None
) -> None:
    """Preset outcome button. Clicking writes options + OUTCOME_CARD_KEY and reruns."""
    clicked = st.button(
        _outcome_tile_label(card_id, selected=selected),
        type="primary" if selected else "secondary",
        key=f"isolate_outcome_pick_{card_id}",
        help=str(OUTCOME_CARDS[card_id]["help"]),
        width="stretch",
    )
    if clicked and not selected:
        option_ids = resolve_outcome_card(
            card_id,
            roformer_available=is_roformer_backend_available(),
            prefer_roformer=prefer_roformer,
            guitar_option=guitar_option,
        )
        st.session_state[OUTCOME_CARD_KEY] = card_id
        _sync_track_widgets_from_options(option_ids)
        _rerun_after_tile_pick()


def _render_stem_tile(
    stem_id: str, *, selected: bool, prefer_roformer: bool, guitar_option: str | None = None
) -> None:
    """One custom-grid stem. Clicking toggles the existing Pro session keys."""
    clicked = st.button(
        f"{stem_icon_markdown(stem_id, selected=selected, light=_tiles_use_light_icons())}\n"
        f"{CUSTOM_STEM_CHOICES[stem_id]}",
        type="primary" if selected else "secondary",
        key=f"isolate_stem_pick_{stem_id}",
        width="stretch",
    )
    if clicked:
        roformer_ok = is_roformer_backend_available()
        new_options = toggle_custom_stem_options(
            _collect_track_options_from_session(),
            stem_id,
            roformer_available=roformer_ok,
            prefer_roformer=prefer_roformer,
            guitar_option=guitar_option,
        )
        _sync_track_widgets_from_options(new_options)
        matched = outcome_card_for_options(
            new_options,
            roformer_available=roformer_ok,
            prefer_roformer=prefer_roformer,
            guitar_option=guitar_option,
        )
        st.session_state[OUTCOME_CARD_KEY] = matched or CUSTOM_OUTCOME_CARD
        _rerun_after_tile_pick()


def _rerun_preserve_scroll() -> None:
    """In-place control: remount only the current fragment so main-pane scroll stays."""
    st.rerun(scope="fragment")


PENDING_QUEUE_DELETE_KEY = "isolate_pending_queue_delete"
_DELETING_TRACKS_CAPTION = "Deleting separated tracks…"
_DELETED_TRACKS_FLASH = "Deleted separated tracks."
_DELETE_FAILED_FLASH = "Could not delete that separation."


def _rerun_after_tile_pick() -> None:
    """Remount only this fragment so New-tab scroll stays put (Lite and Pro)."""
    _rerun_preserve_scroll()


def _rerun_scroll_top() -> None:
    """Navigation / workspace switch: full remount, then scroll main to top."""
    request_isolate_scroll_top(st.session_state)
    st.rerun()


def _render_custom_stem_grid(
    *, selected_stems: set[str], prefer_roformer: bool, guitar_option: str | None = None
) -> None:
    st.markdown("**Custom**")
    st.caption("Pick individual tracks")
    cols = st.columns(3)
    for idx, stem_id in enumerate(CUSTOM_STEM_TILE_ORDER):
        with cols[idx % 3]:
            _render_stem_tile(
                stem_id,
                selected=stem_id in selected_stems,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )


def _render_lite_guitar_engine_chooser(
    *,
    selected: str,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
    roformer_available: bool,
) -> None:
    """Always list the three Lite guitar engines; disable (don't hide) over-cap ones."""
    st.markdown("**Guitar quality**")
    st.caption("How guitar is separated. Other tracks still use Demucs.")
    cols = st.columns(3)
    reasons: list[str] = []
    for idx, engine_id in enumerate(LITE_GUITAR_ENGINE_ORDER):
        allowed = lite_guitar_engine_allowed(
            engine_id,
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_available,
        )
        reason = lite_guitar_engine_disabled_reason(
            engine_id,
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_available,
        )
        if reason and reason not in reasons:
            reasons.append(reason)
        with cols[idx]:
            clicked = st.button(
                LITE_GUITAR_ENGINE_LABELS[engine_id],
                type="primary" if selected == engine_id else "secondary",
                key=f"isolate_lite_guitar_{engine_id}",
                disabled=not allowed,
                help=reason,
                width="stretch",
            )
        if clicked and allowed and engine_id != selected:
            st.session_state[LITE_GUITAR_ENGINE_KEY] = engine_id
            st.session_state.pop(LITE_GUITAR_FELL_BACK_KEY, None)
            _rerun_after_tile_pick()
    for note in reasons:
        st.caption(note)
    if st.session_state.get(LITE_GUITAR_FELL_BACK_KEY):
        st.caption("Reset to Default (Demucs) because this clip cannot use BS-RoFormer.")


def _render_outcome_picker(
    *,
    persist: dict,
    prefer_roformer: bool,
    effective_sec: float | None = None,
    cap_sec: float = 180.0,
    duration_known: bool = False,
) -> tuple[list[str], dict, str | None]:
    """Shared New picker: preset cards plus the six-stem custom grid.

    Called from ``_render_section_and_outcomes`` (one fragment) so tile and
    region clicks remount the same block and keep New-tab scroll.
    """
    # Allow one machine-panel draw per fragment/full paint (see _render_machine_panel).
    st.session_state.pop("_isolate_machine_panel_drawn", None)
    roformer_ok = is_roformer_backend_available()
    pro = is_pro_mode(st.session_state)
    guitar_option: str | None = None
    if not pro:
        guitar_option, fell_back = resolve_lite_guitar_engine(
            st.session_state.get(LITE_GUITAR_ENGINE_KEY),
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_ok,
        )
        st.session_state[LITE_GUITAR_ENGINE_KEY] = guitar_option
        if fell_back:
            st.session_state[LITE_GUITAR_FELL_BACK_KEY] = True
        elif lite_guitar_engine_allowed(
            "guitar_roformer",
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_ok,
        ):
            st.session_state.pop(LITE_GUITAR_FELL_BACK_KEY, None)

    _init_track_picker_session(prefer_roformer=prefer_roformer)
    option_ids = _collect_track_options_from_session()
    matched = outcome_card_for_options(
        option_ids,
        roformer_available=roformer_ok,
        prefer_roformer=prefer_roformer,
        guitar_option=guitar_option,
    )
    if matched:
        st.session_state[OUTCOME_CARD_KEY] = matched
    else:
        st.session_state[OUTCOME_CARD_KEY] = CUSTOM_OUTCOME_CARD
    selected = str(st.session_state.get(OUTCOME_CARD_KEY) or DEFAULT_OUTCOME_CARD)

    # Lite: rematerialize with the chosen guitar engine so outcomes honor the
    # chooser (and a prior Pro RoFormer pick does not stick when illegal).
    if not pro:
        if selected in OUTCOME_CARDS:
            option_ids = resolve_outcome_card(
                selected,
                roformer_available=roformer_ok,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )
            _sync_track_widgets_from_options(option_ids)
            st.session_state[OUTCOME_CARD_KEY] = selected
        else:
            option_ids = apply_lite_guitar_engine(option_ids, guitar_option or DEFAULT_LITE_GUITAR_ENGINE)
            _sync_track_widgets_from_options(option_ids)
            matched = outcome_card_for_options(
                option_ids,
                roformer_available=roformer_ok,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )
            st.session_state[OUTCOME_CARD_KEY] = matched or CUSTOM_OUTCOME_CARD
            selected = str(st.session_state.get(OUTCOME_CARD_KEY) or CUSTOM_OUTCOME_CARD)

    selected_stems = set(custom_stems_from_options(option_ids))

    st.subheader(
        "Separate tracks",
        anchor=False,
        help=SEPARATE_TRACKS_HELP,
    )
    st.caption("Select what you want out")
    st.markdown("**Full mix**")
    mix_cols = st.columns(2)
    with mix_cols[0]:
        _render_outcome_tile(
            "band",
            selected=selected == "band",
            prefer_roformer=prefer_roformer,
            guitar_option=guitar_option,
        )
    with mix_cols[1]:
        _render_outcome_tile(
            "karaoke",
            selected=selected == "karaoke",
            prefer_roformer=prefer_roformer,
            guitar_option=guitar_option,
        )
    _render_custom_stem_grid(
        selected_stems=selected_stems,
        prefer_roformer=prefer_roformer,
        guitar_option=guitar_option,
    )
    if not pro:
        _render_lite_guitar_engine_chooser(
            selected=guitar_option or DEFAULT_LITE_GUITAR_ENGINE,
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_ok,
        )

    if pro:
        _render_track_picker(persist=persist)
        option_ids = _collect_track_options_from_session()
        matched = outcome_card_for_options(
            option_ids,
            roformer_available=roformer_ok,
            prefer_roformer=prefer_roformer,
        )
        if matched:
            st.session_state[OUTCOME_CARD_KEY] = matched
        else:
            st.session_state[OUTCOME_CARD_KEY] = CUSTOM_OUTCOME_CARD

    st.session_state["isolate_track_options"] = list(option_ids)
    preset_error: str | None = None
    try:
        resolved = resolve_track_selection(option_ids)
    except ValueError as exc:
        preset_error = str(exc)
        st.warning(preset_error)
        resolved = resolve_track_selection(list(DEFAULT_TRACK_OPTIONS))

    pro = is_pro_mode(st.session_state)
    if pro and resolved.get("caveat"):
        st.caption(resolved["caveat"])

    probe = get_desktop_probe()
    if pro:
        speed_id = st.radio(
            "Speed",
            options=list(SPEED_PRESETS.keys()),
            format_func=_speed_preset_radio_label,
            key="isolate_speed_preset",
            horizontal=True,
            **persist,
        )
    else:
        # GPU-first auto from installed hardware (CUDA / eligible MPS).
        lite_auto = lite_auto_choice(probe)
        speed_id = str(lite_auto["speed"])
    speed = resolve_speed_preset(speed_id, probe, model=resolved["model"])
    if pro:
        if speed["help"]:
            st.caption(speed["help"])
        if job_requires_roformer_backend(resolved["model"]):
            st.caption(roformer_speed_note())
    applied = f"{speed['id']}:{resolved['model']}"
    if st.session_state.get("isolate_speed_applied") != applied:
        st.session_state["isolate_speed_applied"] = applied
        st.session_state["isolate_quality"] = speed["quality"]
        st.session_state["isolate_device"] = speed["device"]
    if pro:
        _render_engine_panel(
            resolved=resolved,
            speed=speed,
            speed_id=str(speed_id),
            persist=persist,
        )
        # Keep Device inside this fragment with Engine. A sibling expander after
        # the fragment call was duplicating on fragment/full-script handoffs
        # (Streamlit additive writes / stale keyed expanders).
        _render_machine_panel(
            probe=probe,
            device_options=desktop_device_options(probe),
        )
    else:
        guitar_engine = next(
            (oid for oid in option_ids if oid in GUITAR_TRACK_OPTION_IDS),
            guitar_option
            or default_guitar_track_option(
                roformer_available=roformer_ok,
                prefer_roformer=prefer_roformer,
            ),
        )
        st.caption(lite_detected_caption(probe))
        run_choices = lite_device_choice_ids(probe)
        using_reason = lite_auto.get("reason")
        if run_choices:
            if not st.session_state.get("isolate_lite_run_on_manual"):
                auto_dev = str(lite_auto["device"])
                st.session_state["isolate_lite_run_on"] = (
                    auto_dev if auto_dev in run_choices else run_choices[0]
                )
            elif st.session_state.get("isolate_lite_run_on") not in run_choices:
                default_run = (
                    speed["device"] if speed["device"] in run_choices else run_choices[0]
                )
                st.session_state["isolate_lite_run_on"] = default_run

            def _mark_lite_run_on_manual() -> None:
                st.session_state["isolate_lite_run_on_manual"] = True

            run_on = st.radio(
                "Run on",
                options=run_choices,
                format_func=lite_device_plain_label,
                key="isolate_lite_run_on",
                horizontal=True,
                on_change=_mark_lite_run_on_manual,
                help=(
                    "GPU is the default when this computer has one — usually faster. "
                    "Switch to CPU only if you need to."
                ),
                **persist,
            )
            st.session_state["isolate_device"] = run_on
            if run_on == "cpu":
                # Balanced on CPU is far slower than Faster; Lite CPU means Faster.
                st.session_state["isolate_quality"] = "fast"
                using_speed = "faster"
            else:
                st.session_state["isolate_quality"] = speed["quality"]
                using_speed = str(speed["id"])
            # Caption reason is unused under GPU-first auto; keep the wire for
            # any residual reason string from older session state.
            caption_reason = (
                using_reason
                if (
                    not st.session_state.get("isolate_lite_run_on_manual")
                    and using_reason
                )
                else None
            )
            st.caption(
                lite_using_caption(
                    probe,
                    guitar_engine=str(guitar_engine),
                    device=str(run_on),
                    speed=using_speed,
                    reason=caption_reason,
                )
            )
        else:
            st.caption(
                lite_using_caption(
                    probe,
                    guitar_engine=str(guitar_engine),
                    device=str(lite_auto["device"]),
                    speed=str(lite_auto["speed"]),
                    reason=using_reason,
                )
            )
    return list(option_ids), resolved, preset_error


def _render_engine_panel(
    *,
    resolved: dict,
    speed: dict,
    speed_id: str,
    persist: dict,
) -> None:
    """Pro: how the separation is done. Model-specific toggles live here."""
    with _stateful_expander(
        "Engine", key="isolate_options_expanded", default=False
    ):
        quality_now = st.session_state.get("isolate_quality", speed["quality"])
        # Speed and Quality are the same axis. Speed writes Quality whenever it
        # changes, which used to silently discard a hand-picked Quality with no
        # hint that it had happened.
        st.caption(
            f"Speed **{SPEED_PRESETS[speed_id]['label']}** sets Quality to "
            f"**{speed['quality']}**. Change Quality below to override. Picking "
            "a different Speed resets it again."
        )
        _closed_selectbox(
            "Quality",
            ["fast", "balanced", "high", "extreme"],
            key="isolate_quality",
            help="Higher quality is slower, especially on CPU.",
        )
        if quality_now != speed["quality"]:
            st.caption(f"Overriding Speed: running **{quality_now}**, not {speed['quality']}.")

        sync_demucs_compute_defaults(st.session_state, quality=str(quality_now))
        demucs_compute = resolved["model"] in ("htdemucs_6s", "htdemucs", "htdemucs_ft")
        compute_help = (
            ""
            if demucs_compute
            else " Only applies to Demucs (RoFormer ignores shifts, overlap, segment, and jobs)."
        )
        st.caption(
            "Quality sets Demucs shifts and overlap. Overrides below win." + compute_help
        )
        compute_cols = st.columns(2)
        with compute_cols[0]:
            _closed_selectbox(
                "Shifts",
                list(DEMUCS_SHIFT_OPTIONS),
                key="isolate_demucs_shifts",
                help="Demucs --shifts. Higher is slower and usually cleaner." + compute_help,
            )
        with compute_cols[1]:
            _closed_selectbox(
                "Overlap",
                list(DEMUCS_OVERLAP_OPTIONS),
                key="isolate_demucs_overlap",
                help="Demucs --overlap. Higher reduces chunk-boundary artifacts." + compute_help,
            )
        if demucs_compute_is_overridden(
            quality=str(quality_now),
            shifts=st.session_state.get("isolate_demucs_shifts"),
            overlap=st.session_state.get("isolate_demucs_overlap"),
        ):
            st.caption(
                f"Overriding Quality: shifts **{st.session_state.get('isolate_demucs_shifts')}**, "
                f"overlap **{st.session_state.get('isolate_demucs_overlap')}** "
                f"(Quality {quality_now} is {quality_demucs_shifts(str(quality_now))}/"
                f"{quality_demucs_overlap(str(quality_now))})."
            )
        seg_cols = st.columns(2)
        with seg_cols[0]:
            st.number_input(
                "Demucs segment (s)",
                min_value=0,
                max_value=20,
                step=1,
                format="%d",
                key="isolate_demucs_segment",
                disabled=not demucs_compute,
                help=(
                    "Chunk length in seconds. Default 8 (clamped per model). "
                    "0 = model default (full-track tensors; more RAM)." + compute_help
                ),
                **persist,
            )
        with seg_cols[1]:
            st.number_input(
                "Demucs jobs",
                min_value=1,
                max_value=8,
                step=1,
                format="%d",
                key="isolate_demucs_jobs",
                disabled=not demucs_compute,
                help=(
                    "Parallel Demucs chunks. Keep 1 on 8–16 GB machines; "
                    "raising this can OOM." + compute_help
                ),
                **persist,
            )

        # Opt-in until eval/lead_rhythm Stage-1 3-clip listen beats stock 6s.
        st.checkbox(
            "Use guitar-focused Demucs weights (experimental)",
            help=(
                "Optional guitar-focused htdemucs_6s weights (~330 MB first download, "
                "cached under TORCH_HOME/checkpoints). Falls back to stock if download "
                "or load fails. Piano/synth and bass can still bleed into guitar. "
                "Meta Demucs weights are provided for scientific/research use."
            ),
            key="isolate_guitar_ft",
            **persist,
        )
        four_stem = resolved["model"] in ("htdemucs", "htdemucs_ft")
        four_stem_help = (
            ""
            if four_stem
            else " Only applies when the track mix is 4-stem Demucs (no guitar/piano)."
        )
        st.checkbox(
            "Hi-Fi (fine-tuned Demucs)",
            disabled=not four_stem,
            help=(
                "Meta htdemucs_ft: usually cleaner vocals/drums/bass than stock 4-stem. "
                "Slower. Ignored for 6-stem guitar/piano jobs." + four_stem_help
            ),
            key="isolate_htdemucs_ft",
            **persist,
        )
        st.checkbox(
            "Split lead and rhythm guitar",
            disabled=resolved["model"] != "htdemucs_6s",
            help="Uses the guitar stem already in this job. No extra model.",
            key="isolate_lead_rhythm",
            **persist,
        )
        # Rendered unconditionally but disabled off-model, with the reason stated.
        # Appearing and disappearing as the track picker changed made these look
        # like a glitch, and a hidden-but-checked box still altered the run.
        demucs_guitar = resolved["model"] == "htdemucs_6s"
        off_model_help = (
            ""
            if demucs_guitar
            else " Only applies to the Demucs 6-stem guitar option."
        )
        st.checkbox(
            "Two-pass guitar isolation (experimental)",
            disabled=not demucs_guitar,
            help=(
                "Runs a 4-stem split first, then isolates guitar from the leftover mix. "
                "About twice as slow. Can reduce competing vocals/drums/bass in the "
                "guitar stem. Stay opt-in until the stage-1 listen pass." + off_model_help
            ),
            key="isolate_two_pass",
            **persist,
        )
        st.checkbox(
            "Refine guitar with MelBand specialist (experimental)",
            disabled=not demucs_guitar,
            help=(
                "Second-pass guitar extraction (becruily MelBand-Roformer, ~45 MB first "
                "download). De-bleeds the Demucs guitar stem. Skipped when the refine "
                "runtime is unavailable." + off_model_help
            ),
            key="isolate_guitar_refine",
            **persist,
        )
        if resolved["model"] == "bs_roformer_sw" and resolved.get("guitar_refine"):
            st.caption(
                "BS-RoFormer-SW runs first; MelBand refine follows when guitar is selected."
            )

        emit = set(resolved.get("emit_stems") or ())
        has_guitar = "guitar" in emit
        fold_on = bool(resolved.get("fold_other_into_guitar", True))
        guitar_post_help = (
            ""
            if has_guitar
            else " Only applies when the mix includes guitar."
        )
        fold_help = (
            ""
            if fold_on
            else " Fold is off because Other is in the mix."
        )
        ensemble_ok = (
            is_roformer_backend_available()
            and not resolved.get("two_stems")
            and not (
                st.session_state.get("isolate_two_pass")
                and resolved["model"] == "htdemucs_6s"
            )
            and resolved["model"] in ("htdemucs_6s", "htdemucs", "htdemucs_ft")
        )
        st.markdown("**Guitar post**")
        st.caption(
            "Applied during separation. Mixer can still tweak de-bleed/restore after."
            + guitar_post_help
        )
        _closed_selectbox(
            "Fold Other → Guitar",
            list(FOLD_OTHER_MODE_ORDER),
            key="isolate_fold_other_mode",
            help=(
                "best_effort skips piano-like Other; band_limited mixes the guitar band; "
                "full mixes all leftover Other." + fold_help + guitar_post_help
            ),
        )
        st.checkbox(
            "Adaptive fold gain",
            disabled=not (has_guitar and fold_on),
            help=(
                "Search the Other→Guitar mix gain instead of the fixed 0.5."
                + fold_help
                + guitar_post_help
            ),
            key="isolate_adaptive_fold_gain",
            **persist,
        )
        st.checkbox(
            "Spectral bleed gate",
            disabled=not has_guitar,
            help=(
                "Scrub bass/cymbal flutter from the guitar stem when competitor stems dominate."
                + guitar_post_help
            ),
            key="isolate_bleed_gate",
            **persist,
        )
        st.checkbox(
            "Bass-bleed high-pass",
            disabled=not has_guitar,
            help=(
                "Lossy HPF on guitar when bass-bleed diagnostics flag the stem. "
                "Can thin Drop C / 7-string fundamentals." + guitar_post_help
            ),
            key="isolate_bass_bleed_mitigation",
            **persist,
        )
        st.checkbox(
            "Subtractive bass de-bleed",
            disabled=not has_guitar,
            help=(
                "Subtract scaled bass/drum energy below ~150 Hz from the guitar stem."
                + guitar_post_help
            ),
            key="isolate_sub_bass_debleed",
            **persist,
        )
        st.slider(
            "Low-end restore (dB)",
            min_value=0.0,
            max_value=float(PRO_LOW_END_RESTORE_MAX_DB),
            step=1.0,
            key="isolate_low_end_restore_db",
            help=(
                "Boost 60–200 Hz on the guitar stem. 0 = off." + guitar_post_help
            ),
            **persist,
        )
        st.checkbox(
            "Cross-model guitar ensemble",
            disabled=not (has_guitar and ensemble_ok),
            help=(
                "Also run BS-RoFormer-SW and per-band blend the two guitar stems. "
                "Much slower; requires the RoFormer extra. Off for karaoke, two-pass, "
                "and RoFormer-primary jobs."
                + (
                    ""
                    if ensemble_ok
                    else " Needs Demucs primary + BS-RoFormer runtime."
                )
                + guitar_post_help
            ),
            key="isolate_guitar_ensemble",
            **persist,
        )

        st.caption(
            "Guitar low-end fix-up (bass de-bleed, restore) is also on the **Mixer** tab after "
            "separation. No need to re-run Demucs. MelBand refine improves isolation but can "
            "soften highs. Use **Guitar (BS-RoFormer)** without refine if guitar sounds dull."
        )


def _render_machine_panel(*, probe: object, device_options: list[str]) -> None:
    """Pro: what this computer will run it on, plus the hardware readout."""
    # One draw per full script run. Fragment remounts + a second call site used
    # to paint two identical "This computer" expanders after separation.
    if st.session_state.get("_isolate_machine_panel_drawn"):
        return
    st.session_state["_isolate_machine_panel_drawn"] = True
    options = list(device_options) or ["cpu"]
    if st.session_state.get("isolate_device") not in options:
        st.session_state["isolate_device"] = options[0]
    with _stateful_expander(
        "This computer", key="isolate_machine_expanded", default=False
    ):
        _closed_selectbox(
            "Device",
            options,
            key="isolate_device",
            help=(
                "GPU: NVIDIA CUDA on Windows; Apple GPU (MPS) on Apple Silicon "
                "with ≥12 GB RAM. CPU is always available."
            ),
        )
        st.caption(desktop_system_summary(probe))
        st.caption(desktop_recommend_caption(probe))


def _speed_preset_radio_label(preset_id: str) -> str:
    return SPEED_PRESETS[preset_id]["label"]


_VIDEO_UPLOAD_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}
_HOME_UPLOAD_TYPES = [
    *AUDIO_UPLOAD_TYPES,
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
]
_VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


def _local_mixer_video_url(run_dir: Path | None) -> str | None:
    """Browser URL for a video copied into this run. Audio-only runs return None."""
    video = mixer_local_video_path(run_dir)
    if video is None:
        return None
    try:
        from streamlit.runtime import get_instance

        runtime = get_instance()
        if runtime is None:
            return None
        return runtime.media_file_mgr.add(
            str(video.resolve()),
            _VIDEO_MEDIA_TYPES.get(video.suffix.lower(), "video/mp4"),
            f"isolate.video.{video.name}",
            file_name=video.name,
        )
    except Exception:
        logger.warning("Could not register mixer video %s", video, exc_info=True)
        return None


def _stage_opened_file(src: Path) -> tuple[Path | None, str | None, str | None]:
    """Copy an OS-opened file into app storage. Video becomes wav.

    Returns ``(audio_path, video_path, error)``. The user's file is not modified.
    """
    name = Path(src.name).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        return None, None, f"Could not open {src.name}."
    dest = run_output_dir() / name
    try:
        shutil.copy2(src, dest)
    except OSError:
        return None, None, f"Could not open {src.name}."
    if not is_opened_video(src):
        return dest, None, None
    wav = dest.with_name(f"{dest.stem}.extracted.wav")
    try:
        audio = normalize_audio(dest, wav)
    except Exception as exc:
        logger.warning("Could not read audio from opened video %s: %s", src, exc)
        return None, None, f"Could not read audio from {src.name}."
    return audio, str(dest), None


def _consume_opened_audio(session=None) -> bool:
    """Stage files the OS handed the desktop app onto new Home tabs.

    Returns True when a tab was opened, so a fragment tick can remount the page.
    A direct call from ``main`` paints the new tab in the same run.
    """
    state = st.session_state if session is None else session
    folder = os.environ.get(PENDING_OPEN_ENV, "").strip()
    if not folder:
        return False
    items = claim_opened_paths(Path(folder))
    if not items:
        return False
    opened: list[str] = []
    notes: list[str] = []
    for item in items:
        raw = str(item.get("path") or "")
        path = Path(raw) if raw else None
        name = path.name if path is not None and path.name else "That file"
        declined = bool(item.get("declined")) or path is None or not is_opened_audio(path)
        if declined or path is None or not path.is_file():
            notes.append(f"Could not open {name}. Use an audio or video file.")
            continue
        staged, video, err = _stage_opened_file(path)
        if err or staged is None:
            notes.append(err or f"Could not open {name}.")
            continue
        open_opened_audio_tab(
            state,
            staged,
            title=path.stem,
            video_path=video,
        )
        opened.append(name)
    if len(opened) == 1:
        notes.insert(0, f"Opened {opened[0]} on a new tab.")
    elif opened:
        notes.insert(0, f"Opened {len(opened)} files on new tabs.")
    if notes:
        state[MIX_TAB_NOTICE_KEY] = " ".join(notes)
    return bool(opened or notes)


def _extract_uploaded_audio(uploaded: object) -> tuple[Path | None, str | None]:
    """Save an upload. Video files become wav through the existing ffmpeg normalizer."""
    path = save_upload(uploaded)
    if path.suffix.lower() not in _VIDEO_UPLOAD_SUFFIXES:
        st.session_state.pop("isolate_pending_video_path", None)
        return path, None
    st.session_state["isolate_pending_video_path"] = str(path)
    wav = path.with_name(f"{path.stem}.extracted.wav")
    try:
        audio = normalize_audio(path, wav)
    except Exception as exc:
        wav.unlink(missing_ok=True)
        st.session_state.pop("isolate_pending_video_path", None)
        logger.warning("Could not read audio from uploaded video: %s", exc)
        return None, "Could not read audio from this video."
    return audio, None


def _ensure_pending_audio(uploaded: object) -> Path | None:
    """Save upload to disk when fingerprint changes so duration/preview work."""
    fp = upload_fingerprint(uploaded)
    if fp is None:
        return None
    pending = st.session_state.get("isolate_pending_audio_path")
    pending_fp = st.session_state.get("isolate_pending_fp")
    pending_exists = bool(pending and Path(pending).exists())
    if pending_audio_needs_resave(
        uploaded_fp=fp,
        pending_fp=pending_fp if isinstance(pending_fp, str) else None,
        pending_path_exists=pending_exists,
    ):
        if uploaded is not None:
            path, err = _extract_uploaded_audio(uploaded)
            if err or path is None:
                st.session_state.pop("isolate_pending_audio_path", None)
                st.session_state.pop("isolate_pending_fp", None)
                st.error(err or "Could not read this file.")
                return None
            st.session_state["isolate_pending_audio_path"] = str(path)
            st.session_state["isolate_pending_fp"] = fp
            return path
        return None
    return Path(pending) if pending_exists else None


def _active_source_path(uploaded: object) -> Path | None:
    pending = st.session_state.get("isolate_pending_audio_path")
    pending_str = str(pending) if pending else None
    carry = st.session_state.get("carry_over_audio_path")
    carry_str = str(carry) if carry else None
    youtube_url = (st.session_state.get("isolate_youtube_url") or "").strip()
    chosen = staged_audio_for_new_tab(
        uploaded=uploaded is not None,
        youtube_url=youtube_url,
        pending_path=pending_str,
        pending_fp=(
            str(st.session_state.get("isolate_pending_fp"))
            if st.session_state.get("isolate_pending_fp")
            else None
        ),
        pending_exists=bool(pending_str and Path(pending_str).exists()),
        carry_path=carry_str,
        carry_exists=bool(carry_str and Path(carry_str).exists()),
    )
    return Path(chosen) if chosen else None


def _st_version() -> tuple[int, int, int]:
    """Return parsed Streamlit version as (major, minor, patch)."""
    try:
        from packaging import version as _pv

        v = _pv.Version(st.__version__)
        return (v.major, v.minor, v.micro)
    except Exception:
        return (0, 0, 0)


def _st_at_least(major: int, minor: int, patch: int = 0) -> bool:
    return _st_version() >= (major, minor, patch)


def _closed_selectbox(label: str, options: list[str], *, key: str, help: str | None = None):
    """Dropdown that does not accept typed free-text (Streamlit ≥1.55 combobox)."""
    kwargs: dict = {
        "label": label,
        "options": options,
        "key": key,
        "accept_new_options": False,
    }
    if help:
        kwargs["help"] = help
    if _st_at_least(1, 55, 0):
        return st.selectbox(**kwargs, filter_mode=None)
    return st.selectbox(**kwargs)


def _render_region_controls(audio_path: Path | None) -> tuple[float, float | None, str | None]:
    """
    Region picker. Returns (start_sec, max_duration_sec, region_label).

    Full-file processing uses ``max_duration_sec is None``. Lite and Pro both
    default to the whole file; Lite shows a RAM/time warning and an optional
    Safer: first 90 s control (never a silent clamp).
    """
    if audio_path is None or not audio_path.exists():
        return 0.0, None, None

    duration = _cached_probe_duration_sec(audio_path)
    if duration is None:
        st.caption("Length unknown. The full file will be processed.")
        return 0.0, None, None

    pro = is_pro_mode(st.session_state)
    if pro:
        st.caption(f"Length: **{format_time_sec(duration)}** ({duration:.1f} s)")

    region_key = f"isolate_region_{st.session_state.get('isolate_pending_fp') or st.session_state.get('isolate_upload_fp', 'none')}"
    if region_key not in st.session_state:
        st.session_state[region_key] = (0.0, float(duration))

    if (
        not pro
        and duration > LITE_MAX_DURATION_SEC
        and st.button(
            f"Safer: first {LITE_MAX_DURATION_SEC:.0f} s",
            key=f"{region_key}_safer_90",
            help=(
                "Long separations on Lite can use a lot of RAM and time. "
                f"Sets the section to 0–{LITE_MAX_DURATION_SEC:.0f} s."
            ),
        )
    ):
        st.session_state[region_key] = (0.0, float(LITE_MAX_DURATION_SEC))
        st.session_state[f"{region_key}_start"] = 0.0
        st.session_state[f"{region_key}_end"] = float(LITE_MAX_DURATION_SEC)
        st.session_state[f"{region_key}_wave_nonce"] = (
            int(st.session_state.get(f"{region_key}_wave_nonce") or 0) + 1
        )
        _rerun_preserve_scroll()

    start_default, end_default = st.session_state[region_key]
    start_default, end_default = clamp_region_bounds(
        start_default, end_default, duration, min_length=MIN_REGION_SEC
    )

    start_sec, end_sec = _render_region_bounds_widgets(
        audio_path=audio_path,
        region_key=region_key,
        duration=duration,
        start_default=start_default,
        end_default=end_default,
        pro=pro,
    )
    start_sec, end_sec = clamp_region_bounds(
        start_sec, end_sec, duration, min_length=MIN_REGION_SEC
    )
    st.session_state[region_key] = (start_sec, end_sec)

    length = end_sec - start_sec
    region_label = format_region_label(start_sec, length)
    if pro:
        st.caption(f"**{region_label}** ({length:.0f} s)")

    if length < MIN_REGION_SEC:
        st.error(f"Section must be at least {MIN_REGION_SEC:.0f} seconds.")
        return start_sec, None, region_label

    # Leave start/end at the file bounds → process the whole file (no trim).
    if start_sec <= 0.5 and end_sec >= duration - 0.5:
        return 0.0, None, None

    return start_sec, length, region_label


def _render_region_bounds_widgets(
    *,
    audio_path: Path,
    region_key: str,
    duration: float,
    start_default: float,
    end_default: float,
    pro: bool,
) -> tuple[float, float]:
    """Waveform region picker when built; otherwise dual sliders."""
    try:
        from ui.region_picker_component import (
            component_build_available,
            region_picker,
        )
    except Exception:
        component_build_available = None  # type: ignore[assignment]
        region_picker = None  # type: ignore[assignment]

    if component_build_available is not None and component_build_available() and region_picker:
        try:
            theme_vars = get_iframe_theme_vars(st.session_state)
            url = media_url_for_file(
                audio_path, coordinates=f"isolate.region.{region_key}"
            )
            with st.container(key="isolate_region_picker"):
                result = region_picker(
                    audio_url=url,
                    start_sec=float(start_default),
                    end_sec=float(end_default),
                    min_length_sec=float(MIN_REGION_SEC),
                    duration_sec=float(duration),
                    max_hint_sec=None if pro else float(LITE_MAX_DURATION_SEC),
                    theme_vars=theme_vars,
                    key=f"{region_key}_wave_{st.session_state.get(f'{region_key}_wave_nonce', 0)}",
                )
            if isinstance(result, dict):
                start_sec = float(result.get("startSec", start_default))
                end_sec = float(result.get("endSec", end_default))
                return start_sec, end_sec
            return float(start_default), float(end_default)
        except Exception as exc:
            logger.warning("Region picker failed; falling back to sliders: %s", exc)

    start_col, end_col = st.columns(2)
    with start_col:
        start_sec = st.slider(
            f"Start ({format_time_sec(start_default)})",
            min_value=0.0,
            max_value=float(duration),
            value=float(start_default),
            step=1.0,
            format="%.0f",
            key=f"{region_key}_start",
        )
    with end_col:
        end_sec = st.slider(
            f"End ({format_time_sec(end_default)})",
            min_value=float(MIN_REGION_SEC),
            max_value=float(duration),
            value=float(end_default),
            step=1.0,
            format="%.0f",
            key=f"{region_key}_end",
        )
    return float(start_sec), float(end_sec)


def _render_section_preview(
    audio_path: Path | None,
    start_sec: float,
    max_duration_sec: float | None,
    region_label: str | None,
) -> None:
    """Play the section that will be isolated (or the full file)."""
    if audio_path is None or not audio_path.exists():
        return
    fp = str(
        st.session_state.get("isolate_pending_fp")
        or st.session_state.get("isolate_upload_fp")
        or audio_path.name
    )
    if max_duration_sec is None:
        st.caption("Preview: full file (what will be processed)")
        try:
            st.audio(str(audio_path))
        except Exception:
            pass
        return
    label = region_label or format_region_label(start_sec, float(max_duration_sec))
    st.caption(f"Preview: **{label}** ({float(max_duration_sec):.0f} s)")
    try:
        with st.spinner("Preparing section preview…"):
            preview = ensure_region_preview_wav(
                audio_path,
                start_sec=start_sec,
                length_sec=float(max_duration_sec),
                fingerprint=fp,
                cache_dir=DATA_DIR / "region_previews",
            )
        st.audio(str(preview))
    except Exception:
        try:
            st.audio(str(audio_path))
        except Exception:
            pass


@st.fragment
def _render_section_and_outcomes(
    *,
    audio_path: Path | None,
    persist: dict,
    prefer_roformer: bool,
    cap_sec: float,
) -> tuple[float, float | None, str | None, list, dict, str | None]:
    """Section + outcomes in one fragment so region/tile clicks keep New-tab scroll."""
    start_sec, max_duration_sec, region_label = 0.0, None, None
    if audio_path and audio_path.exists():
        st.subheader(
            "Section (optional)",
            anchor=False,
            help=SECTION_OPTIONAL_HELP,
        )
        start_sec, max_duration_sec, region_label = _render_region_controls(audio_path)
    file_dur = (
        _cached_probe_duration_sec(audio_path)
        if audio_path and audio_path.exists()
        else None
    )
    effective_sec = effective_clip_seconds(
        file_duration_sec=file_dur,
        start_sec=start_sec,
        max_duration_sec=max_duration_sec,
    )
    track_options, resolved, preset_error = _render_outcome_picker(
        persist=persist,
        prefer_roformer=prefer_roformer,
        effective_sec=effective_sec,
        cap_sec=cap_sec,
        duration_known=effective_sec is not None,
    )
    return start_sec, max_duration_sec, region_label, track_options, resolved, preset_error


def _persist_kwargs() -> dict:
    """Keep isolate widgets across page switches when Streamlit supports it."""
    try:
        import inspect

        if "persist_state" in inspect.signature(st.radio).parameters:
            return {"persist_state": "session"}
    except Exception:
        pass
    return {}


def _park_unmounted_widget(drew: bool, widget_key: str, pending_key: str) -> None:
    """Keep a text widget across a screen that does not instantiate it.

    Skip when this run already queued a pending value (search pick or reset).
    """
    if drew or pending_key in st.session_state:
        return
    if widget_key not in st.session_state:
        return
    st.session_state[pending_key] = st.session_state.get(widget_key)


def _batch_path_strings() -> list[str]:
    raw = st.session_state.get(ISOLATE_BATCH_PATHS_KEY) or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item)]


def _stored_batch_paths() -> list[Path]:
    return [Path(item) for item in _batch_path_strings() if Path(item).is_file()]


def _render_batch_caption(count: int) -> None:
    if count <= 1:
        return
    extra = count - 20
    if extra == 1:
        st.caption(
            f"{count} files selected. The first 20 will be queued. 1 file was left out."
        )
    elif extra > 1:
        st.caption(
            f"{count} files selected. The first 20 will be queued. {extra} files were left out."
        )
    else:
        st.caption(
            f"{count} files selected. Each file is queued with the same tracks and speed."
        )


def _remember_batch_uploads(uploads: list) -> None:
    """Copy a multi-file selection before the uploader unmounts.

    An empty uploader is not a clear: Back remounts it with no files.
    One new file replaces the batch. ``save_upload`` makes a new directory
    on every call, so the same fingerprint is not staged again.
    """
    if len(uploads) == 1:
        st.session_state.pop(ISOLATE_BATCH_PATHS_KEY, None)
        st.session_state.pop(ISOLATE_BATCH_FP_KEY, None)
        return
    if len(uploads) < 2:
        return
    fp = "|".join(upload_fingerprint(item) or "" for item in uploads)
    stored = _batch_path_strings()
    if fp == st.session_state.get(ISOLATE_BATCH_FP_KEY) and len(stored) > 1:
        return
    paths: list[str] = []
    notes: list[str] = []
    for item in uploads:
        path, err = _stage_batch_upload(item)
        if err or path is None:
            notes.append(err or f"Could not read {getattr(item, 'name', 'file')}.")
            continue
        paths.append(str(path))
    st.session_state[ISOLATE_BATCH_PATHS_KEY] = paths
    st.session_state[ISOLATE_BATCH_FP_KEY] = fp
    if notes:
        st.warning(" ".join(notes))


def _region_bounds_from_session(audio_path: Path | None) -> tuple[float, float | None, str | None]:
    """Region already stored for this file, without drawing the picker.

    Missing tuple means the whole file. Does not create a region key.
    """
    if audio_path is None or not audio_path.exists():
        return 0.0, None, None
    duration = _cached_probe_duration_sec(audio_path)
    if duration is None:
        return 0.0, None, None
    fp = st.session_state.get("isolate_pending_fp") or st.session_state.get(
        "isolate_upload_fp", "none"
    )
    stored = st.session_state.get(f"isolate_region_{fp}")
    if not isinstance(stored, tuple) or len(stored) != 2:
        return 0.0, None, None
    start_sec, end_sec = clamp_region_bounds(
        stored[0], stored[1], duration, min_length=MIN_REGION_SEC
    )
    length = end_sec - start_sec
    region_label = format_region_label(start_sec, length)
    if length < MIN_REGION_SEC:
        return start_sec, None, region_label
    if start_sec <= 0.5 and end_sec >= duration - 0.5:
        return 0.0, None, None
    return start_sec, length, region_label


def _separation_choice_from_session(
    *,
    prefer_roformer: bool,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
) -> tuple[list[str], dict, str | None]:
    """Track choice for screen 1, including Search → Separate.

    Same session writes as the outcome picker before it paints. Does not
    draw widgets and does not assign the Run-on radio key.
    """
    roformer_ok = is_roformer_backend_available()
    pro = is_pro_mode(st.session_state)
    guitar_option: str | None = None
    if not pro:
        guitar_option, fell_back = resolve_lite_guitar_engine(
            st.session_state.get(LITE_GUITAR_ENGINE_KEY),
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_ok,
        )
        st.session_state[LITE_GUITAR_ENGINE_KEY] = guitar_option
        if fell_back:
            st.session_state[LITE_GUITAR_FELL_BACK_KEY] = True
        elif lite_guitar_engine_allowed(
            "guitar_roformer",
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=duration_known,
            roformer_available=roformer_ok,
        ):
            st.session_state.pop(LITE_GUITAR_FELL_BACK_KEY, None)
    _init_track_picker_session(prefer_roformer=prefer_roformer)
    option_ids = _collect_track_options_from_session()
    matched = outcome_card_for_options(
        option_ids,
        roformer_available=roformer_ok,
        prefer_roformer=prefer_roformer,
        guitar_option=guitar_option,
    )
    if matched:
        st.session_state[OUTCOME_CARD_KEY] = matched
    else:
        st.session_state[OUTCOME_CARD_KEY] = CUSTOM_OUTCOME_CARD
    selected = str(st.session_state.get(OUTCOME_CARD_KEY) or DEFAULT_OUTCOME_CARD)
    if not pro:
        if selected in OUTCOME_CARDS:
            option_ids = resolve_outcome_card(
                selected,
                roformer_available=roformer_ok,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )
            _sync_track_widgets_from_options(option_ids)
            st.session_state[OUTCOME_CARD_KEY] = selected
        else:
            option_ids = apply_lite_guitar_engine(
                option_ids, guitar_option or DEFAULT_LITE_GUITAR_ENGINE
            )
            _sync_track_widgets_from_options(option_ids)
            matched = outcome_card_for_options(
                option_ids,
                roformer_available=roformer_ok,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )
            st.session_state[OUTCOME_CARD_KEY] = matched or CUSTOM_OUTCOME_CARD
    preset_error: str | None = None
    try:
        resolved = resolve_track_selection(option_ids)
    except ValueError as exc:
        preset_error = str(exc)
        resolved = resolve_track_selection(list(DEFAULT_TRACK_OPTIONS))
    return list(option_ids), resolved, preset_error


def _render_separation_controls() -> dict:
    """New tab: source screen, then separate screen."""
    apply_pending_output_name(st.session_state)
    apply_pending_youtube_url(st.session_state)
    apply_youtube_output_name_sync(st.session_state)
    if st.session_state.get("isolate_speed_preset") not in SPEED_PRESETS:
        st.session_state["isolate_speed_preset"] = DEFAULT_SPEED_PRESET
    persist = _persist_kwargs()
    step = current_setup_step(st.session_state)
    drew_youtube = False
    drew_output_name = False
    uploads: list = []
    single = None
    youtube_error: str | None = None
    youtube_url = str(st.session_state.get("isolate_youtube_url") or "")
    default_name = ""

    if step == SETUP_STEP_SOURCE:
        with st.container(key="isolate_source_entry"):
            uploaded_raw = st.file_uploader(
                "Upload audio or video",
                type=_HOME_UPLOAD_TYPES,
                accept_multiple_files=True,
                help="MP3, WAV, FLAC, M4A, MP4, MOV, WEBM, or MKV. Video audio is extracted on this computer. Several files are queued one at a time.",
                key=f"isolate_upload_{st.session_state.get('isolate_upload_key', 0)}",
            )
            uploads = list(uploaded_raw) if isinstance(uploaded_raw, list) else []
            single = uploads[0] if len(uploads) == 1 else None
            _remember_batch_uploads(uploads)
            batch_count = len(uploads) if len(uploads) > 1 else (
                len(_batch_path_strings()) if not uploads else 0
            )
            _render_batch_caption(batch_count)
            if single is not None:
                _sync_upload_output_name(single)
                _ensure_pending_audio(single)

            if st.button(
                "Search YouTube",
                key="isolate_youtube_search_open_btn",
                type="secondary",
                width="stretch",
                help="Open a search panel to find a public video by song or artist.",
            ):
                st.session_state[ISOLATE_YOUTUBE_SEARCH_OPEN_KEY] = True
                st.rerun()
            if st.session_state.get(ISOLATE_YOUTUBE_SEARCH_OPEN_KEY):
                _youtube_search_dialog()
            st.caption(YOUTUBE_DISCLAIMER)

            carry_over_path = st.session_state.get("carry_over_audio_path")
            carry_over_name = st.session_state.get("carry_over_audio_name")
            if not single and len(uploads) <= 1 and carry_over_path and Path(carry_over_path).exists():
                st.caption(
                    f"Using carried-over audio: **{carry_over_name}**. "
                    "Upload a file above to use something else instead."
                )
            pending_path = st.session_state.get("isolate_pending_audio_path")
            pending_ready = bool(pending_path and Path(str(pending_path)).exists()) or bool(
                carry_over_path and Path(str(carry_over_path)).exists()
            )
            continue_ok = setup_continue_enabled(
                download_in_progress=bool(st.session_state.get(ISOLATE_YOUTUBE_DOWNLOADING_KEY)),
                batch_count=len(_stored_batch_paths()),
                live_upload_count=len(uploads),
                youtube_url=youtube_url,
                pending_fp=(
                    str(st.session_state.get("isolate_pending_fp"))
                    if st.session_state.get("isolate_pending_fp")
                    else None
                ),
                pending_ready=pending_ready,
            )
            if st.button(
                "Continue",
                type="primary",
                key="isolate_setup_continue",
                disabled=not continue_ok,
            ):
                direct_url = youtube_url.strip()
                if len(uploads) > 1 and len(_stored_batch_paths()) < 2:
                    st.error("Could not queue these files.")
                elif direct_url and is_direct_media_url(direct_url):
                    saved, direct_err = _resolve_audio_for_job(
                        {
                            "youtube_url": direct_url,
                            "uploaded": None,
                            "output_name": st.session_state.get("isolate_output_name") or "",
                            "audio_path": None,
                        }
                    )
                    if direct_err or saved is None:
                        st.error(direct_err or "Could not save that link.")
                    else:
                        st.session_state[SETUP_STEP_KEY] = SETUP_STEP_SEPARATE
                        _rerun_scroll_top()
                else:
                    st.session_state[SETUP_STEP_KEY] = SETUP_STEP_SEPARATE
                    _rerun_scroll_top()

    batch_paths = _stored_batch_paths()
    is_batch = len(batch_paths) > 1 or len(uploads) > 1
    uploaded = uploads if len(uploads) > 1 else single
    audio_path = None if is_batch else _active_source_path(single)
    carry_name = st.session_state.get("carry_over_audio_name")
    if single is not None:
        default_name = Path(single.name).stem
    elif carry_name:
        default_name = Path(str(carry_name)).stem
    if default_name and not st.session_state.get("isolate_output_name"):
        st.session_state["isolate_output_name"] = default_name
    if step == SETUP_STEP_SEPARATE and not is_batch:
        st.text_input(
            "Output name",
            help="Used for downloaded file names. Defaults to the uploaded file's name.",
            key="isolate_output_name",
            max_chars=100,
            **persist,
        )
        drew_output_name = True
        staged_name = st.session_state.get("isolate_output_name")
        if staged_name:
            sanitized_name = " ".join(str(staged_name).split())
            if sanitized_name != staged_name:
                st.caption(f"Will be saved as: **{sanitized_name}**")

    pro = is_pro_mode(st.session_state)
    # Full torch probe for Lite auto (MPS/CUDA) — without_torch would hide MPS.
    probe = get_desktop_probe()
    cap_sec = roformer_max_audio_sec(probe)
    prefer_roformer = True if pro else False
    if step == SETUP_STEP_SEPARATE:
        start_sec, max_duration_sec, region_label, track_options, resolved, preset_error = (
            _render_section_and_outcomes(
                audio_path=None if is_batch else audio_path,
                persist=persist,
                prefer_roformer=prefer_roformer,
                cap_sec=cap_sec,
            )
        )
    else:
        if is_batch:
            start_sec, max_duration_sec, region_label = 0.0, None, None
            effective_sec = None
        else:
            start_sec, max_duration_sec, region_label = _region_bounds_from_session(audio_path)
            file_dur = (
                _cached_probe_duration_sec(audio_path)
                if audio_path is not None and audio_path.exists()
                else None
            )
            effective_sec = effective_clip_seconds(
                file_duration_sec=file_dur,
                start_sec=start_sec,
                max_duration_sec=max_duration_sec,
            )
        track_options, resolved, preset_error = _separation_choice_from_session(
            prefer_roformer=prefer_roformer,
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=effective_sec is not None,
        )
    custom_stems = list(resolved.get("stems") or [])

    speed_id = st.session_state.get("isolate_speed_preset")
    lite_auto = lite_auto_choice(probe) if not pro else None
    if not pro or speed_id not in SPEED_PRESETS:
        speed_id = (
            str(lite_auto["speed"])
            if lite_auto is not None
            else lite_auto_speed_id(probe)
        )
    speed = resolve_speed_preset(speed_id, probe, model=resolved["model"])
    device_options = desktop_device_options(probe)
    allowed_devices = list(device_options)
    if not pro:
        for device_id in lite_device_choice_ids(probe):
            if device_id not in allowed_devices:
                allowed_devices.append(device_id)
    st.session_state["isolate_device"] = default_isolate_device(
        probe, allowed_devices, st.session_state.get("isolate_device")
    )

    quality = st.session_state.get("isolate_quality", speed["quality"])
    device = st.session_state.get("isolate_device", speed["device"])
    if lite_auto is not None and not st.session_state.get("isolate_lite_run_on_manual"):
        # Re-probe at form build / Separate so the job matches current headroom.
        # Do not write isolate_lite_run_on here — the Run-on radio already owns
        # that key (set before the widget in the outcome picker).
        device = str(lite_auto["device"])
        st.session_state["isolate_device"] = device
        if device == "cpu":
            quality = "fast"
            st.session_state["isolate_quality"] = "fast"
        else:
            quality = speed["quality"]
            st.session_state["isolate_quality"] = quality
    output_name = st.session_state.get("isolate_output_name", default_name or "")
    guitar_checkpoint = None
    if st.session_state.get("isolate_guitar_ft") and resolved["model"] == "htdemucs_6s":
        guitar_checkpoint = "htdemucs_6s_guitar_ft"
    two_pass = bool(
        st.session_state.get("isolate_two_pass") and resolved["model"] == "htdemucs_6s"
    )
    # Only honour the checkbox for the model it is offered for. It used to apply to
    # bs_roformer_sw / melband too, so a box ticked earlier under Demucs kept
    # forcing a refine pass on RoFormer runs while being invisible.
    guitar_refine = bool(resolved.get("guitar_refine")) or bool(
        st.session_state.get("isolate_guitar_refine") and resolved["model"] == "htdemucs_6s"
    )
    engine_fields = resolve_pro_engine_job_fields(
        st.session_state,
        is_pro=pro,
        model=resolved["model"],
        quality=str(quality),
        two_stems=resolved.get("two_stems"),
        two_pass=two_pass,
        fold_other_into_guitar=bool(resolved.get("fold_other_into_guitar", True)),
        fold_other_mode=str(resolved.get("fold_other_mode") or "best_effort"),
        roformer_available=is_roformer_backend_available(),
    )
    _park_unmounted_widget(drew_youtube, "isolate_youtube_url", ISOLATE_YOUTUBE_URL_PENDING_KEY)
    _park_unmounted_widget(
        drew_output_name, "isolate_output_name", ISOLATE_OUTPUT_NAME_PENDING_KEY
    )
    if step == SETUP_STEP_SEPARATE and st.button(
        "Back",
        key="isolate_setup_back",
        type="secondary",
    ):
        st.session_state[SETUP_STEP_KEY] = SETUP_STEP_SOURCE
        _rerun_scroll_top()
    return {
        "model": engine_fields["model"],
        "two_stems": resolved["two_stems"],
        "custom_stems": custom_stems,
        "track_options": list(track_options),
        "error": preset_error or youtube_error,
        "quality": quality,
        "device": device,
        "start_sec": start_sec,
        "max_duration_sec": max_duration_sec,
        "uploaded": uploaded,
        "output_name": output_name,
        "region_label": region_label,
        "guitar_checkpoint": guitar_checkpoint,
        "two_pass": two_pass,
        "guitar_refine": guitar_refine,
        "low_end_restore_db": engine_fields["low_end_restore_db"],
        "sub_bass_debleed": engine_fields["sub_bass_debleed"],
        "demucs_segment": engine_fields["demucs_segment"],
        "demucs_jobs": engine_fields["demucs_jobs"],
        "demucs_shifts": engine_fields["demucs_shifts"],
        "demucs_overlap": engine_fields["demucs_overlap"],
        "adaptive_fold_gain": engine_fields["adaptive_fold_gain"],
        "bleed_gate": engine_fields["bleed_gate"],
        "bass_bleed_mitigation": engine_fields["bass_bleed_mitigation"],
        "guitar_ensemble": engine_fields["guitar_ensemble"],
        "lead_rhythm": engine_fields["lead_rhythm"],
        "lead_rhythm_mode": engine_fields["lead_rhythm_mode"],
        "emit_stems": list(resolved["emit_stems"]) if resolved.get("emit_stems") else None,
        "fold_other_into_guitar": bool(resolved.get("fold_other_into_guitar", True)),
        "fold_other_mode": engine_fields["fold_other_mode"],
        "youtube_url": youtube_url.strip(),
        "tracks_label": resolved.get("tracks") or "",
        "preset_label": resolved.get("label") or "",
        "audio_path": audio_path,
    }


def _ffmpeg_install_hint() -> str:
    if sys.platform == "darwin":
        return "Install ffmpeg: brew install ffmpeg"
    if sys.platform.startswith("win"):
        return "Install ffmpeg: winget install Gyan.FFmpeg (full/shared build, not essentials-only)"
    return "Install ffmpeg: sudo apt install ffmpeg (or your distro equivalent)"


def _rehydrate_artifacts_from_disk(browser_id: str | None) -> bool:
    """Restore mixer results from last-viewed run, else the latest library row."""
    rows = _library_rows_available(browser_id)
    row = select_rehydrate_row(st.session_state, rows, wav_exists=_wav_exists)
    if row is None:
        return False
    return _apply_library_row(row, viewing_mode="latest", reopen_name=False)


def _sync_listen_name_if_current(run_dir: str, title: str) -> None:
    """Keep the Listening-to field aligned when Refresh renames the open mix."""
    if str(st.session_state.get("isolate_listen_name_for") or "") == run_dir:
        st.session_state["isolate_listen_name"] = title


def _refresh_isolate_from_disk(browser_id: str | None) -> None:
    """Reload the open mix from disk. A dismissed Home stays on Home."""
    ensure_worker_started()
    rows = _library_rows_available(browser_id)
    loaded = str(st.session_state.get("isolate_run_dir") or "")
    row = (
        next((item for item in rows if str(item.get("run_dir") or "") == loaded), None)
        if loaded
        else None
    )
    changed = False
    if row is not None:
        action = refresh_open_mix_action(st.session_state, row)
        if action == "stems":
            changed = _apply_library_row(
                row,
                viewing_mode=str(row.get("id") or loaded),
                reopen_name=False,
            )
            if changed:
                _sync_listen_name_if_current(
                    loaded, str(st.session_state.get("isolate_base_name") or "tracks")
                )
        elif action == "title":
            title = str(row.get("title") or "tracks")
            st.session_state["isolate_base_name"] = title
            _sync_listen_name_if_current(loaded, title)
            changed = True
    elif not st.session_state.get(ISOLATE_SKIP_REHYDRATE_KEY):
        changed = _rehydrate_artifacts_from_disk(browser_id)
    st.session_state["isolate_flash"] = (
        "Reloaded from disk." if changed else "Already up to date."
    )
    _persist_isolate_ui_state()
    st.rerun()


def _open_mixer_workspace() -> None:
    """Request the mix shell on the next full run (Moises tab strip)."""
    st.session_state.pop(ISOLATE_SKIP_REHYDRATE_KEY, None)
    open_mix_shell(st.session_state)


def _ensure_workspace_tab(*, has_artifacts: bool) -> None:
    apply_workspace_tab(st.session_state, has_artifacts=has_artifacts)
    apply_shell_view(st.session_state)


def _request_loading_overlay() -> None:
    """Ask the global overlay to show while a cross-page switch is in flight."""
    st.session_state["_nav_loading"] = True


def _scroll_main_to_top(token: int = 0) -> None:
    """Always-on 0-height slot. Scrolls stMain / section.main when token advances."""
    import streamlit.components.v1 as components

    token_js = int(token or 0)
    components.html(
        f"""
<script>
(function () {{
  var token = {token_js};
  var doc = window.parent && window.parent.document ? window.parent.document : document;
  var win = window.parent || window;
  if (!token || win.__isolateScrollToken === token) return;
  win.__isolateScrollToken = token;
  function go() {{
    var nodes = [
      doc.querySelector('[data-testid="stMain"]'),
      doc.querySelector('section.main'),
      doc.querySelector('[data-testid="stAppViewContainer"]'),
      doc.scrollingElement,
      doc.documentElement,
      doc.body
    ];
    for (var i = 0; i < nodes.length; i++) {{
      var el = nodes[i];
      if (!el) continue;
      try {{
        if (typeof el.scrollTo === "function") el.scrollTo(0, 0);
        el.scrollTop = 0;
      }} catch (e) {{}}
    }}
    try {{ win.scrollTo(0, 0); }} catch (e) {{}}
  }}
  go();
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(go);
  setTimeout(go, 50);
  setTimeout(go, 200);
}})();
</script>
        """,
        height=0,
        width=0,
    )


def _staged_source_name() -> str:
    pending = st.session_state.get("isolate_pending_audio_path")
    if pending and Path(str(pending)).exists():
        return Path(str(pending)).stem
    return "New file"


def _render_running_progress(status: dict) -> None:
    view = running_progress_view(status, time.time())
    st.progress(min(1.0, max(0.0, float(view["percent"]))))
    bits = [str(view.get("label") or ""), str(view.get("eta_line") or "")]
    st.markdown(" · ".join(b for b in bits if b) or "\u00a0")
    # Always paint the hint slot so a stage change does not grow/shrink the strip.
    st.caption(str(view.get("hint") or "\u00a0"))
    # Stage-by-stage breakdown is Pro detail. In Lite it competes with the single
    # percentage line that answers the only question being asked: how much longer.
    if is_pro_mode(st.session_state):
        st.markdown(str(view.get("checklist_md") or "\u00a0"))


def _render_compact_running_hint(status: dict) -> None:
    """One-line Home progress when Queue is closed. Pause/Stop stay in Queue."""
    view = running_progress_view(status, time.time())
    st.progress(min(1.0, max(0.0, float(view["percent"]))))
    title = _job_source_title(status)
    bits = [
        str(title),
        str(view.get("label") or ""),
        str(view.get("eta_line") or ""),
        "Open Queue for Pause/Stop",
    ]
    st.caption(" · ".join(b for b in bits if b))


def _retry_failed_job(job_id: str) -> None:
    """Re-queue a failed job from strip or Queue. Same spec, fresh output dir."""
    new_id = requeue_job(job_id)
    if new_id:
        dismiss_failed_job(st.session_state, job_id)
        ensure_worker_started()
        _rerun_preserve_scroll()
        return
    st.session_state["isolate_flash"] = (
        "Cannot retry. The original audio file is no longer on disk. "
        "Add the file again on Home."
    )
    _rerun_preserve_scroll()


def _job_source_title(job: dict) -> str:
    title = job.get("title") or str(job.get("id") or "")[:8] or "track"
    kind = infer_source_kind(
        source_kind=job.get("source_kind"),
        source_fingerprint=job.get("source_fingerprint"),
    )
    return format_source_title(str(title), kind)


def _toggle_queue_panel() -> None:
    st.session_state[ISOLATE_QUEUE_PANEL_OPEN_KEY] = not bool(
        st.session_state.get(ISOLATE_QUEUE_PANEL_OPEN_KEY)
    )


def _queue_visible_count() -> int:
    """Cheap badge for the header Queue control (in-flight + finished with wavs)."""
    try:
        return len(jobs_visible_in_queue(list_jobs(limit=30)))
    except Exception:
        return 0


@st.fragment
def _queue_header_fragment() -> None:
    """Queue toggle + float panel. Fragment-scoped so mixer playback stays mounted."""
    queue_open = bool(st.session_state.get(ISOLATE_QUEUE_PANEL_OPEN_KEY))
    queue_n = _queue_visible_count()
    queue_label = f"Queue ({queue_n})" if queue_n else "Queue"
    with st.container(key="isolate_queue_toggle_slot"):
        st.button(
            queue_label,
            key="isolate_queue_toggle",
            type="primary" if queue_open else "secondary",
            help="Jobs: open in mixer, delete, pause, or stop.",
            width="stretch",
            on_click=_toggle_queue_panel,
        )
    if st.session_state.get(ISOLATE_QUEUE_PANEL_OPEN_KEY):
        _render_queue_float_panel()


def _render_queue_float_panel() -> None:
    """Non-modal floating Queue card (Home or mix). Fixed overlay via CSS."""
    with st.container(key="isolate_queue_float", border=True):
        st.markdown("**Queue**")
        _queue_tab_fragment()


def _arm_confirm(flag: str) -> None:
    st.session_state[flag] = True


def _disarm_confirm(flag: str) -> None:
    st.session_state.pop(flag, None)


def accept_confirm_click(session: dict, flag: str, clicked: bool) -> bool:
    """True only when confirm is clicked while the flag is still set.

    The flag is cleared after the click is accepted, never before it is read.
    """
    if not session.get(flag) or not clicked:
        return False
    session.pop(flag, None)
    return True


def _confirm_button(
    label: str,
    *,
    key: str,
    confirm_label: str,
    confirm_help: str,
    help: str | None = None,
    width: str = "content",
) -> bool:
    """Two-step button for deletes with no undo. True only on the confirm click."""
    flag = f"{key}__confirm"
    if st.session_state.get(flag):
        confirmed = st.button(
            confirm_label,
            key=f"{key}__yes",
            type="primary",
            help=confirm_help,
            width=width,
        )
        if st.button(
            "Cancel",
            key=f"{key}__no",
            width=width,
        ):
            _disarm_confirm(flag)
            st.rerun(scope="app")
        if accept_confirm_click(st.session_state, flag, confirmed):
            return True
        return False
    idle = label if str(label).endswith("…") else f"{label}…"
    st.button(
        idle,
        key=key,
        help=help or "Click to arm confirm. Nothing is deleted yet.",
        width=width,
        on_click=_arm_confirm,
        args=(flag,),
    )
    return False


def _queue_confirm_pair(
    label: str,
    *,
    key: str,
    confirm_label: str,
    confirm_help: str,
    help: str | None = None,
    width: str = "content",
) -> bool:
    """Queue-only two-step delete. Confirm and Cancel share one equal row.

    The idle label stays a single button. Once armed, the pair replaces that
    control so the red confirm cannot sit beside a different action.
    """
    flag = f"{key}__confirm"
    if st.session_state.get(flag):
        with st.container(key=f"queue_confirm_row_{key}"):
            yes_col, no_col = st.columns(2)
            with yes_col:
                confirmed = st.button(
                    confirm_label,
                    key=f"{key}__yes",
                    type="primary",
                    help=confirm_help,
                    width="stretch",
                )
            with no_col:
                cancelled = st.button(
                    "Cancel",
                    key=f"{key}__no",
                    width="stretch",
                )
        if cancelled:
            _disarm_confirm(flag)
            st.rerun(scope="app")
        if accept_confirm_click(st.session_state, flag, confirmed):
            return True
        return False
    idle = label if str(label).endswith("…") else f"{label}…"
    st.button(
        idle,
        key=key,
        help=help or "Click to arm confirm. Nothing is deleted yet.",
        width=width,
        on_click=_arm_confirm,
        args=(flag,),
    )
    return False


def _render_job_queue_panel() -> None:
    ensure_worker_started()
    parts = partition_queue_jobs(jobs_visible_in_queue(list_jobs(limit=30)))
    if parts["empty"]:
        st.caption("No jobs yet.")
        return
    if parts["succeeded"]:
        n_done = len(parts["succeeded"])
        if _queue_confirm_pair(
            "Delete all finished",
            key="isolate_delete_all_finished",
            confirm_label=f"Delete {n_done} from disk",
            confirm_help="Removes the separated tracks for every finished job. No undo.",
        ):
            _request_queue_delete(parts["succeeded"])
    waiting_ids = queued_job_ids()
    active_id = active_job_id()
    stopping_previous = False
    if active_id and worker_busy():
        active_status = read_status(active_id) or {}
        stopping_previous = is_stopping_previous_job(
            active_id,
            str(active_status.get("status") or ""),
        )
    for job in parts["in_flight"]:
        with st.container(border=True):
            _render_queue_job_row(
                job,
                waiting_ids=waiting_ids,
                stopping_previous=stopping_previous,
            )
    for job in parts["succeeded"]:
        with st.container(border=True):
            _render_queue_job_row(
                job,
                waiting_ids=waiting_ids,
                stopping_previous=stopping_previous,
            )


def _request_queue_delete(jobs: list[dict], *, tab_ids: list[str] | None = None) -> None:
    """Remember finished jobs and rerun the whole app so the mixer can close."""
    slim = []
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        slim.append(
            {
                "id": str(job.get("id") or ""),
                "status": "succeeded",
                "run_dir": str(job.get("run_dir") or "") or None,
            }
        )
    extra_tabs = [str(tab) for tab in (tab_ids or []) if str(tab or "").strip()]
    if not slim and not extra_tabs:
        st.session_state["isolate_flash"] = _DELETE_FAILED_FLASH
        st.rerun(scope="app")
    pending: dict = {"phase": "show", "jobs": slim}
    if extra_tabs:
        pending["tab_ids"] = extra_tabs
    st.session_state[PENDING_QUEUE_DELETE_KEY] = pending
    st.rerun(scope="app")


def detach_deleted_mix_runs(session: dict, run_dirs: list[str]) -> None:
    """Close mix tabs for deleted runs and drop the loaded mixer.

    Path match uses the mix-tab normalizer, so a trailing slash or ``..``
    still counts as the open mix. Sets skip-rehydrate when that mix was loaded.
    """
    loaded = False
    for run_dir in run_dirs:
        if not run_dir:
            continue
        if is_loaded_mix_tab(session, run_dir):
            loaded = True
        close_open_mix_tab(session, run_dir)
    if not loaded:
        return
    _clear_loaded_mixer_keys(session, dismiss=True)
    open_home_shell(session)


def _clear_loaded_mixer_keys(session: dict, *, dismiss: bool) -> None:
    for key in (
        "isolate_artifacts",
        "isolate_base_name",
        "isolate_run_dir",
        "isolate_volumes_db",
        "isolate_master_volume_db",
        "isolate_mixer_state",
        "isolate_mix_ready",
        "isolate_mix_fp",
        "isolate_selected_stems",
        "isolate_results_fp",
        "isolate_source_audio_path",
        "isolate_source_kind",
        "isolate_region_label",
        "isolate_clip_length",
        "isolate_viewing_run_dir",
        "isolate_listen_applied_dir",
        "isolate_listen_missing",
        "isolate_results_source_fp",
        "isolate_last_export_path",
        LISTEN_PICKER_KEY,
        LISTEN_PICKER_NEXT_KEY,
        "isolate_flash",
    ):
        session.pop(key, None)
    if dismiss:
        session[ISOLATE_SKIP_REHYDRATE_KEY] = True
    else:
        session.pop(ISOLATE_SKIP_REHYDRATE_KEY, None)


def _execute_pending_queue_delete(
    jobs: list[dict], *, tab_ids: list[str] | None = None
) -> None:
    failed = False
    deleted_dirs: list[str] = []
    for job in jobs:
        try:
            result = delete_finished_job(job)
        except OSError:
            result = {"ok": False, "run_dir": str(job.get("run_dir") or "") or None}
        run_dir = str(result.get("run_dir") or job.get("run_dir") or "")
        if run_dir and not is_new_draft_tab(run_dir):
            try:
                delete_library_run(run_dir)
            except OSError:
                result = {"ok": False, "run_dir": run_dir}
        gone = bool(run_dir) and not is_new_draft_tab(run_dir) and not Path(run_dir).exists()
        if result.get("ok") or gone:
            if run_dir:
                deleted_dirs.append(run_dir)
            continue
        failed = True
    for tab_id in tab_ids or []:
        tab = str(tab_id or "").strip()
        if not tab or tab in deleted_dirs:
            continue
        if is_new_draft_tab(tab):
            if not failed:
                deleted_dirs.append(tab)
            continue
        try:
            removed = delete_library_run(tab)
        except OSError:
            removed = False
        if removed or not Path(tab).exists():
            deleted_dirs.append(tab)
        else:
            failed = True
    if deleted_dirs:
        detach_deleted_mix_runs(st.session_state, deleted_dirs)
    _persist_isolate_ui_state()
    st.session_state["isolate_flash"] = (
        _DELETE_FAILED_FLASH if failed else _DELETED_TRACKS_FLASH
    )


def _drain_pending_queue_delete() -> None:
    """Paint Deleting…, then remove the runs on the following full run."""
    pending = st.session_state.get(PENDING_QUEUE_DELETE_KEY)
    if not isinstance(pending, dict):
        return
    if pending.get("phase") != "run":
        st.caption(_DELETING_TRACKS_CAPTION)
        pending["phase"] = "run"
        st.session_state[PENDING_QUEUE_DELETE_KEY] = pending
        st.rerun(scope="app")
    jobs = list(pending.get("jobs") or [])
    tab_ids = [str(tab) for tab in (pending.get("tab_ids") or []) if str(tab or "").strip()]
    st.session_state.pop(PENDING_QUEUE_DELETE_KEY, None)
    _execute_pending_queue_delete(jobs, tab_ids=tab_ids)


def _render_job_failure(
    title: str,
    raw_error: str | None,
    *,
    detail_key: str,
    inline_caption: bool = False,
) -> None:
    """Short summary plus full error text in an expander."""
    summary = format_job_error(raw_error)
    detail = (raw_error or "").strip() or summary
    if inline_caption:
        st.caption(summary)
    else:
        st.error(f"**{title}** failed: {summary}")
    # Collapsed, and Pro only: an auto-expanded stack trace reads as a crash and
    # is unactionable for anyone who did not write the pipeline.
    if detail and is_pro_mode(st.session_state):
        with st.expander("Technical detail", expanded=False, key=detail_key):
            st.code(detail)


def _render_queue_job_row(
    job: dict,
    *,
    waiting_ids: list[str],
    stopping_previous: bool,
) -> None:
    status = job.get("status", "unknown")
    title = _job_source_title(job)
    job_id = str(job.get("id") or "")
    label = "done" if status == "succeeded" else status
    st.write(f"**{title}** · {label}")
    if status == "failed":
        _render_job_failure(
            title,
            job.get("error") or job.get("message"),
            detail_key=f"job_fail_details_{job_id}",
            inline_caption=True,
        )
    elif status == "queued":
        st.caption(
            queued_wait_caption(
                job_id,
                waiting_ids,
                stopping_previous=stopping_previous,
            )
        )
    elif status == "cancelled":
        st.caption("Removed from queue")
    elif status == "paused":
        st.caption(paused_job_caption(job.get("completed_stages")))
    elif status == "pausing":
        st.caption("Pausing…")
    # running: progress paints full-width below Pause/Stop (see below)
    if status == "succeeded" and job_id:
        if st.session_state.get(f"delete_finished_{job_id}__confirm"):
            if _queue_confirm_pair(
                "Delete",
                key=f"delete_finished_{job_id}",
                confirm_label="Delete from disk",
                confirm_help="Removes this job's separated tracks. No undo.",
                width="stretch",
            ):
                _request_queue_delete([read_status(job_id) or job])
            return
        open_col, delete_col = st.columns(2)
        with open_col:
            if st.button("Open in Mixer", key=f"open_mixer_{job_id}", width="stretch"):
                fresh = read_status(job_id) or job
                if apply_succeeded_job_to_session(
                    st.session_state,
                    fresh,
                    viewing_mode=job_id,
                ):
                    run_dir = promote_job_origin_tab(st.session_state, fresh)
                    if run_dir:
                        st.session_state["isolate_listen_applied_dir"] = str(run_dir)
                        st.session_state[LISTEN_PICKER_KEY] = str(run_dir)
                    _open_mixer_workspace()
                    _persist_isolate_ui_state()
                    _rerun_scroll_top()
                else:
                    st.caption("Those files are no longer available.")
        with delete_col:
            if _queue_confirm_pair(
                "Delete",
                key=f"delete_finished_{job_id}",
                confirm_label="Delete from disk",
                confirm_help="Removes this job's separated tracks. No undo.",
                width="stretch",
            ):
                _request_queue_delete([read_status(job_id) or job])
        return
    if status == "running" and job_id:
        pause_col, stop_col = st.columns(2)
        with pause_col:
            if st.button(
                "Pause",
                key=f"pause_job_{job_id}",
                width="stretch",
                help=(
                    "Pauses after the current step finishes (or stops heavy processing). "
                    "Resume continues from the last saved step."
                ),
            ):
                pause_job(job_id)
                ensure_worker_started()
                _rerun_preserve_scroll()
        with stop_col:
            # Two-step: this throws away minutes of finished compute and there is
            # no undo, so a single stray click must not be enough.
            confirm_key = f"isolate_confirm_stop_{job_id}"
            if st.session_state.get(confirm_key):
                if st.button(
                    "Discard",
                    key=f"stop_job_{job_id}",
                    type="primary",
                    help="Discard this job and its progress.",
                    width="stretch",
                ):
                    st.session_state.pop(confirm_key, None)
                    remove_job(job_id)
                    ensure_worker_started()
                    _rerun_preserve_scroll()
                if st.button("Keep going", key=f"isolate_keep_{job_id}", width="stretch"):
                    st.session_state.pop(confirm_key, None)
                    _rerun_preserve_scroll()
            elif st.button(
                "Stop",
                key=f"isolate_stop_ask_{job_id}",
                width="stretch",
                help="Cancel this job. Progress so far is lost.",
            ):
                st.session_state[confirm_key] = True
                _rerun_preserve_scroll()
        fresh = read_status(job_id) or job
        _render_running_progress(fresh)
        return
    if status == "paused" and job_id:
        resume_col, remove_col = st.columns(2)
        with resume_col:
            if st.button("Resume", key=f"resume_job_{job_id}", width="stretch"):
                resume_job(job_id)
                ensure_worker_started()
                _rerun_preserve_scroll()
        with remove_col:
            if st.button("Remove", key=f"remove_paused_{job_id}", width="stretch"):
                remove_job(job_id)
                ensure_worker_started()
                _rerun_preserve_scroll()
        return
    if status == "failed" and job_id:
        retry_col, remove_col = st.columns(2)
        with retry_col:
            if st.button(
                "Try again",
                key=f"isolate_queue_retry_{job_id}",
                type="primary",
                width="stretch",
                help="Queue the same track with the same settings.",
            ):
                _retry_failed_job(job_id)
        with remove_col:
            if st.button(
                "Remove",
                key=f"remove_job_{job_id}",
                width="stretch",
                help="Drop this job from the list.",
            ):
                remove_job(job_id)
                ensure_worker_started()
                _rerun_preserve_scroll()
        return
    if job_id and st.button(
        "Remove",
        key=f"remove_job_{job_id}",
        help="Drop this job from the list.",
    ):
        remove_job(job_id)
        ensure_worker_started()
        _rerun_preserve_scroll()


def _render_failed_strip(failed: dict) -> None:
    """Failure with the two things people actually want: retry, or make it go away."""
    job_id = str(failed.get("id") or "")
    title = _job_source_title(failed)
    st.error(f"**{title}** failed: {format_job_error(failed.get('error'))}")
    retry_col, dismiss_col, _ = st.columns([1, 1, 4])
    with retry_col:
        if st.button(
            "Try again",
            key=f"isolate_retry_{job_id}",
            type="primary",
            width="stretch",
            help="Queue the same track with the same settings.",
        ):
            _retry_failed_job(job_id)
    with dismiss_col:
        if st.button(
            "Dismiss",
            key=f"isolate_dismiss_{job_id}",
            width="stretch",
            help="Hide this here. The job stays in Queue.",
        ):
            dismiss_failed_job(st.session_state, job_id)
            _rerun_preserve_scroll()
    if is_pro_mode(st.session_state):
        detail = (failed.get("error") or "").strip()
        if detail:
            with st.expander("Technical detail", expanded=False, key="isolate_status_fail_details"):
                st.code(detail)


def _render_status_strip(jobs: list) -> None:
    """Non-running job state on Home, plus a compact running hint if Queue is closed.

    Detailed % / ETA / Pause / Stop live in the Queue float. Closing Queue must
    not hide that a separation is still running.
    """
    running = None
    failed = None
    dismissed = st.session_state.get(DISMISSED_JOB_IDS_KEY) or []
    now = time.time()
    for job in jobs:
        jid = job.get("id")
        if not jid:
            continue
        fresh = read_status(jid) or job
        status = fresh.get("status")
        if status == "running" and running is None:
            running = fresh
        elif status == "failed" and failed is None:
            if should_show_failed_job(fresh, now=now, dismissed_ids=dismissed):
                failed = fresh
    waiting_ids = queued_job_ids()
    active_id = active_job_id()
    stopping_previous = False
    if active_id and worker_busy():
        active_status = read_status(active_id) or {}
        stopping_previous = is_stopping_previous_job(
            active_id,
            str(active_status.get("status") or ""),
        )
    queue_open = bool(st.session_state.get(ISOLATE_QUEUE_PANEL_OPEN_KEY))
    show_running_hint = running is not None and not queue_open
    show_queued = running is None and failed is None and bool(waiting_ids)
    # Idle: paint nothing. An empty keyed container reserved a random vertical
    # gap between the Home|mix tabs and Upload on Home.
    if not (failed or show_running_hint or stopping_previous or show_queued):
        return
    with st.container(key="isolate_status_strip"):
        if failed:
            with st.container(border=True):
                _render_failed_strip(failed)
        elif show_running_hint:
            with st.container(border=True):
                _render_compact_running_hint(running)
        elif stopping_previous:
            with st.container(border=True):
                st.caption(status_strip_waiting_caption(waiting_ids, stopping_previous=True))
        else:
            with st.container(border=True):
                st.caption(
                    status_strip_waiting_caption(waiting_ids, stopping_previous=False)
                )


@st.fragment(run_every=1.0)
def _poll_running_jobs() -> None:
    """Keep progress visible across page switches; apply finished jobs when unpinned."""
    ensure_worker_started()
    jobs = list_jobs(limit=15)
    if "isolate_consumed_job_ids" not in st.session_state:
        consumed_ids = None
    else:
        consumed_ids = list(st.session_state.get("isolate_consumed_job_ids") or [])
    viewing = st.session_state.get("isolate_viewing_job_id")
    shell = st.session_state.get(SHELL_VIEW_KEY)
    viewing_run = st.session_state.get("isolate_viewing_run_dir")
    listen = st.session_state.get(LISTEN_PICKER_KEY)
    shell_tab = st.session_state.get(SHELL_TAB_KEY)
    plan = plan_isolate_job_poll(
        jobs,
        consumed_ids=consumed_ids,
        viewing_id=viewing if isinstance(viewing, str) else None,
        form_drawn=bool(st.session_state.get("_isolate_form_drawn")),
        shell_view=shell if isinstance(shell, str) else None,
        viewing_run_dir=viewing_run if isinstance(viewing_run, str) else None,
        listen_picker=listen if isinstance(listen, str) else None,
        shell_tab=shell_tab if isinstance(shell_tab, str) else None,
    )
    st.session_state["isolate_consumed_job_ids"] = plan["consumed_ids"]

    job = plan["apply_job"]
    applied = False
    if job:
        jid = str(job.get("id") or "")
        fresh = read_status(jid) or job
        st.session_state["isolate_consumed_job_id"] = jid
        # Promote draft → run tab always; load stems/picker only when auto-applying.
        open_mixer = apply_finished_job_poll_outcome(
            st.session_state,
            fresh,
            notify_only=bool(plan["notify_only"]),
        )
        applied = True
        if open_mixer:
            _open_mixer_workspace()

    if "isolate_notified_job_ids" not in st.session_state:
        notified_ids = None
    else:
        notified_ids = list(st.session_state.get("isolate_notified_job_ids") or [])
    notified_ids, pending_notify = jobs_needing_os_notify(jobs, notified_ids)
    st.session_state["isolate_notified_job_ids"] = notified_ids
    for nj in pending_notify:
        jid = str(nj.get("id") or "")
        fresh = read_status(jid) if jid else None
        row = fresh or nj
        message = os_notify_message(row)
        if message is None:
            continue
        desktop_notify(message[0], message[1])
        if row.get("status") == "failed" and not applied:
            fail_title = row.get("title") or "track"
            st.session_state["isolate_flash"] = f"**{fail_title}** failed. Open Queue."

    if isolate_poll_requires_full_rerun(applied=applied, plan_rerun=bool(plan["rerun"])):
        _persist_isolate_ui_state()
        _rerun_scroll_top()

    _render_status_strip(jobs)


@st.fragment(run_every=1.0)
def _queue_tab_fragment() -> None:
    _render_job_queue_panel()


@st.fragment
def _mixer_and_downloads_fragment(
    stem_paths: dict[str, Path],
    *,
    base_name: str,
    run_dir: Path,
    media_urls: dict[str, str],
    download_urls: dict[str, str],
    tab_pdf_url: str | None = None,
) -> None:
    """Mixer and downloads — fragment-scoped so the mixer does not remount the page.

    Mute/solo/volume reports only update session here. Export WAV is built on
    **Save current mix**, not on every mixer callback. Click-track options live on
    the metronome row and rebake the export WAV without remounting playback.
    """
    selected_stem_paths = stem_paths
    # Re-register download URLs each fragment pass so rebaked metronome.wav is fresh.
    try:
        download_urls = stem_media_urls(selected_stem_paths, coord_prefix="isolate.download")
    except Exception:
        pass
    mixer_state = _render_live_mixer(
        selected_stem_paths,
        track_title=base_name,
        base_name=base_name,
        run_dir=run_dir,
        media_urls=media_urls,
        download_urls=download_urls,
        tab_pdf_url=tab_pdf_url,
    )
    if mixer_state:
        st.session_state["isolate_mixer_state"] = mixer_state
        if "volumesDb" in mixer_state:
            st.session_state["isolate_volumes_db"] = {
                k: float(v) for k, v in mixer_state["volumesDb"].items()
            }
        if "masterVolumeDb" in mixer_state:
            st.session_state["isolate_master_volume_db"] = float(
                mixer_state["masterVolumeDb"]
            )
        if "playbackRate" in mixer_state:
            st.session_state["isolate_playback_rate"] = clamp_playback_rate(
                float(mixer_state["playbackRate"])
            )
        if "playbackSemitones" in mixer_state:
            st.session_state["isolate_playback_semitones"] = clamp_playback_semitones(
                mixer_state["playbackSemitones"]
            )
        _persist_project_from_mixer(run_dir, mixer_state)
        if "hideYoutubeVideo" in mixer_state:
            st.session_state["isolate_hide_youtube_video"] = bool(
                mixer_state["hideYoutubeVideo"]
            )
        _sync_metronome_options_from_mixer(
            mixer_state, run_dir=run_dir, stem_paths=selected_stem_paths
        )

    st.subheader("Downloads", anchor=False, help=DOWNLOADS_HELP)
    if not selected_stem_paths:
        st.info("No tracks available to download.")
        return

    stem_names = sort_stem_names(selected_stem_paths.keys())
    state = st.session_state.get("isolate_mixer_state") or {}
    volumes_db = state.get("volumesDb") or st.session_state.get("isolate_volumes_db") or {}
    muted = {
        n: bool((state.get("muted") or {}).get(n, default_muted_for(n)))
        for n in stem_names
    }
    soloed = state.get("soloed") or {n: False for n in stem_names}
    master_volume_db = float(
        state.get("masterVolumeDb", st.session_state.get("isolate_master_volume_db", DB_DEFAULT))
    )
    playback_rate = clamp_playback_rate(
        float(
            state.get(
                "playbackRate",
                st.session_state.get("isolate_playback_rate", PLAYBACK_RATE_DEFAULT),
            )
        )
    )
    playback_semitones = clamp_playback_semitones(
        state.get(
            "playbackSemitones",
            st.session_state.get("isolate_playback_semitones", PLAYBACK_SEMITONES_DEFAULT),
        )
    )
    pan = state.get("pan")
    if not isinstance(pan, dict) or not pan:
        pan = load_project(run_dir)[0]["mixer"].get("pan") or {}

    _render_downloads_panel(
        selected_stem_paths,
        base_name=base_name,
        run_dir=run_dir,
        stem_names=stem_names,
        volumes_db=volumes_db,
        muted=muted,
        soloed=soloed,
        master_volume_db=master_volume_db,
        playback_rate=playback_rate,
        playback_semitones=playback_semitones,
        pans=pan,
    )


def _save_all_tracks(selected_stem_paths: dict[str, Path], export_root: Path, base_name: str, fmt: str) -> None:
    dest = export_song_dir(export_root, str(base_name))
    export_tracks_to_folder(selected_stem_paths, dest, str(base_name), fmt)
    st.session_state["isolate_last_export_path"] = str(dest)
    st.success(f"Download finished. Saved to {dest}")


def _save_current_mix(ready: str, export_root: Path, base_name: str, fmt: str) -> None:
    if fmt == "wav" and st.session_state.get("isolate_wav_48_24"):
        dest_dir = export_song_dir(export_root, str(base_name))
        dest_file = write_wav_48_24(Path(ready), dest_dir / f"{base_name}_current_mix.wav")
        st.session_state["isolate_last_export_path"] = str(dest_file.parent)
        st.success(f"Download finished. Saved to {dest_file}")
        return
    dest_file = export_mix_to_folder(
        Path(ready),
        export_song_dir(export_root, str(base_name)),
        f"{base_name}_current_mix.wav",
        fmt,
    )
    st.session_state["isolate_last_export_path"] = str(dest_file.parent)
    st.success(f"Download finished. Saved to {dest_file}")


def _download_format_widget() -> str:
    """Format selector used by the Downloads panel. Returns an EXPORT_FORMATS key."""
    pro = is_pro_mode(st.session_state)
    labels = EXPORT_FORMAT_LABELS if pro else EXPORT_FORMAT_PLAIN_LABELS
    help_text = (
        "WAV is lossless; other formats are handled automatically."
        if pro
        else None
    )
    if _st_at_least(1, 41, 0):
        fmt = st.segmented_control(
            "Export format",
            options=list(EXPORT_FORMATS),
            format_func=lambda f: labels[f],
            default="wav",
            key="isolate_download_format",
            help=help_text,
        )
        return fmt or "wav"
    fmt = st.selectbox(
        "Export format",
        options=list(EXPORT_FORMATS),
        format_func=lambda f: labels[f],
        index=0,
        key="isolate_download_format",
        help=help_text,
    )
    return fmt or "wav"


def _render_downloads_panel(
    selected_stem_paths: dict[str, Path],
    *,
    base_name: str,
    run_dir: Path,
    stem_names: list[str],
    volumes_db: dict,
    muted: dict,
    soloed: dict,
    master_volume_db: float,
    playback_rate: float = PLAYBACK_RATE_DEFAULT,
    playback_semitones: int = PLAYBACK_SEMITONES_DEFAULT,
    pans: dict | None = None,
) -> None:
    """Save location and export to folder — equal-width button rows."""
    export_root = _resolved_export_dir()
    rate = clamp_playback_rate(playback_rate)
    semis = clamp_playback_semitones(playback_semitones)
    with st.container(border=True):
        st.caption(f"Save to: {export_root}")
        choose_col, open_col = st.columns(2)
        with choose_col:
            if st.button(
                "Choose folder",
                key="isolate_choose_export_dir",
                width="stretch",
            ):
                picked = choose_export_dir()
                if picked:
                    st.session_state[ISOLATE_EXPORT_DIR_KEY] = str(picked)
                    _persist_isolate_ui_state()
                else:
                    st.caption("Folder picker was cancelled or is not available.")
        with open_col:
            if st.button(
                "Open folder",
                key="isolate_open_export_dir",
                width="stretch",
            ):
                last = st.session_state.get("isolate_last_export_path")
                target = Path(str(last)) if last else export_root
                if not open_path_in_os(target):
                    st.warning("Could not open that folder.")
        st.caption(
            "If the folder window is hidden, Alt+Tab. Saving overwrites a same-named "
            "file. On Linux, a missing zenity picker is treated as cancel."
        )

        st.divider()
        fmt = _download_format_widget()
        if fmt == "wav":
            st.checkbox(
                "48 kHz / 24-bit WAV",
                key="isolate_wav_48_24",
                help="This saved file only. Listening speed and key stay as they are.",
            )
        if st.button("Chord chart", key="isolate_chord_chart"):
            from ui.library_song import chord_chart_text

            analysis_path = run_dir / "analysis.json"
            chords: list = []
            if analysis_path.is_file():
                try:
                    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
                    raw = payload.get("chords") if isinstance(payload, dict) else None
                    if isinstance(raw, list):
                        chords = raw
                except json.JSONDecodeError:
                    chords = []
            if not chords:
                st.caption("No chords in this run.")
            else:
                dest_dir = export_song_dir(export_root, str(base_name))
                dest_dir.mkdir(parents=True, exist_ok=True)
                chart = dest_dir / f"{base_name}_chord_chart.txt"
                chart.write_text(chord_chart_text(chords), encoding="utf-8")
                st.success(f"Chord chart saved to {chart}")
        st.caption(
            "Save current mix includes mute, solo, levels, pan, master, and Speed & Key. "
            "Save all tracks stays original speed/key."
        )
        save_col, mix_col = st.columns(2)
        with save_col:
            if st.button(
                "Save all tracks",
                type="primary",
                key="isolate_save_tracks",
                width="stretch",
            ):
                with st.spinner(f"Converting tracks to {fmt}..."):
                    _save_all_tracks(selected_stem_paths, export_root, str(base_name), fmt)
        with mix_col:
            if st.button(
                "Save current mix",
                key="isolate_save_mix",
                width="stretch",
            ):
                try:
                    transform = abs(rate - 1.0) >= 1e-6 or semis != 0
                    spinner = (
                        "Saving current mix (applying Speed & Key)…"
                        if transform
                        else "Saving current mix..."
                    )
                    with st.spinner(spinner):
                        _build_current_mix(
                            selected_stem_paths,
                            run_dir,
                            stem_names,
                            volumes_db,
                            muted,
                            soloed,
                            master_volume_db,
                            playback_rate=rate,
                            playback_semitones=semis,
                            pans=pans,
                        )
                        ready = st.session_state.get("isolate_mix_ready")
                        if not ready or not Path(str(ready)).exists():
                            raise RuntimeError("mix export was not written")
                        _save_current_mix(str(ready), export_root, str(base_name), fmt)
                        project_state = dict(
                            st.session_state.get("isolate_mixer_state") or {}
                        )
                        project_state.update(
                            {
                                "playbackRate": rate,
                                "playbackSemitones": semis,
                            }
                        )
                        _persist_project_from_mixer(
                            run_dir, project_state, force=True
                        )
                except Exception as exc:
                    st.session_state.pop("isolate_mix_ready", None)
                    st.session_state.pop("isolate_mix_fp", None)
                    st.warning("Could not build current mix export.")
                    with st.expander("Details"):
                        st.exception(exc)


def _resolve_audio_for_job(choice: dict) -> tuple[Path | None, str | None]:
    """Return (audio path, error). Error is None when a path is ready or there is no source."""
    uploaded = choice.get("uploaded")
    youtube_url = (choice.get("youtube_url") or "").strip()
    pending = st.session_state.get("isolate_pending_audio_path")
    carry_over_path = st.session_state.get("carry_over_audio_path")

    if youtube_url and is_direct_media_url(youtube_url):
        already = (
            pending
            and Path(pending).exists()
            and st.session_state.get("isolate_pending_fp") == f"direct:{youtube_url}"
        )
        if already:
            return Path(pending), None
        out = run_output_dir()
        try:
            path = download_direct_media(youtube_url, out)
        except Exception as exc:
            return None, f"Could not save that link: {exc}"
        st.session_state["isolate_pending_audio_path"] = str(path)
        st.session_state["isolate_pending_fp"] = f"direct:{youtube_url}"
        st.session_state["isolate_upload_fp"] = f"direct:{youtube_url}"
        if not str(choice.get("output_name") or "").strip():
            choice["output_name"] = path.stem
        return path, None

    if youtube_url:
        if not is_youtube_url(youtube_url):
            return None, "Paste a YouTube link or a direct audio or video file link."
        already = (
            pending
            and Path(pending).exists()
            and st.session_state.get("isolate_pending_fp") == f"youtube:{youtube_url}"
        )
        if already:
            audio = Path(pending)
            new_name = apply_youtube_output_name_sync(
                st.session_state,
                downloaded_stem=audio.stem,
                apply_now=False,
            )
            if new_name:
                choice["output_name"] = new_name
            elif not str(choice.get("output_name") or "").strip():
                choice["output_name"] = audio.stem
            return audio, None
        out = run_output_dir()
        try:
            with st.status("Downloading YouTube audio…", expanded=True) as status:
                st.write("Fetching audio from YouTube. This can take a minute.")
                path = download_youtube_audio(youtube_url, out)
                status.update(label=f"Ready: {path.stem}", state="complete")
            if pending and str(pending) != str(path):
                prev_fp = st.session_state.get("isolate_pending_fp") or st.session_state.get(
                    "isolate_upload_fp"
                )
                discard_youtube_staging(
                    str(pending),
                    fingerprint=str(prev_fp) if prev_fp else None,
                    retain_paths=(path,),
                )
            st.session_state["isolate_pending_audio_path"] = str(path)
            st.session_state["isolate_pending_fp"] = f"youtube:{youtube_url}"
            st.session_state["isolate_upload_fp"] = f"youtube:{youtube_url}"
            st.session_state.pop("isolate_pending_video_path", None)
            new_name = apply_youtube_output_name_sync(
                st.session_state,
                downloaded_stem=path.stem,
                apply_now=False,
            )
            if new_name:
                choice["output_name"] = new_name
            elif not str(choice.get("output_name") or "").strip():
                choice["output_name"] = path.stem
            return path, None
        except YouTubeDownloadError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"YouTube download failed: {exc}"

    if uploaded and pending and Path(pending).exists():
        return Path(pending), None
    if uploaded:
        path, err = _extract_uploaded_audio(uploaded)
        if err or path is None:
            return None, err or "Could not read this file."
        st.session_state["isolate_pending_audio_path"] = str(path)
        fp = upload_fingerprint(uploaded)
        if fp:
            st.session_state["isolate_pending_fp"] = fp
        return path, None
    if carry_over_path and Path(carry_over_path).exists():
        return Path(carry_over_path), None
    if choice.get("audio_path") and Path(choice["audio_path"]).exists():
        return Path(choice["audio_path"]), None
    return None, None


def _download_youtube_preview_into_run(output_dir: Path, youtube_url: str) -> None:
    """Save a capped mp4 beside the stems. Separation continues if this fails."""
    url = (youtube_url or "").strip()
    if not is_youtube_url(url):
        return
    dest = output_dir / "source_video.mp4"
    try:
        download_youtube_preview_video(url, dest)
    except Exception:
        logger.warning("YouTube preview video skipped for %s", url, exc_info=True)


def _copy_source_video_into_run(output_dir: Path) -> None:
    """Keep an uploaded video beside the stems so the mixer can play it muted."""
    fp = str(st.session_state.get("isolate_pending_fp") or "")
    if fp.startswith("youtube:"):
        return
    raw = st.session_state.get("isolate_pending_video_path")
    if not raw:
        return
    src = Path(str(raw))
    if not src.is_file() or src.suffix.lower() not in _VIDEO_UPLOAD_SUFFIXES:
        return
    dest = output_dir / f"source_video{src.suffix.lower()}"
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        logger.warning("Could not keep the source video for the mixer: %s", exc)


def _enqueue_confirmed_job(choice: dict, audio_path: Path) -> None:
    if not is_pro_mode(st.session_state) and not st.session_state.get(
        "isolate_lite_run_on_manual"
    ):
        # Final headroom check at enqueue (not a background timer).
        lite_auto = lite_auto_choice(get_desktop_probe())
        choice = dict(choice)
        choice["device"] = str(lite_auto["device"])
        if choice["device"] == "cpu":
            choice["quality"] = "fast"
        # Update job-facing session keys only — isolate_lite_run_on is a radio
        # widget key and must not be written after the widget is instantiated.
        st.session_state["isolate_device"] = choice["device"]
        st.session_state["isolate_quality"] = choice["quality"]

    start_sec = float(choice.get("start_sec") or 0.0)
    max_duration_sec = choice.get("max_duration_sec")
    region_label = choice.get("region_label")
    output_name = resolve_youtube_job_name(
        choice.get("output_name") or "",
        choice.get("youtube_url") or "",
        audio_path.stem,
    )

    file_dur: float | None = None
    try:
        file_dur = probe_duration_sec(audio_path)
    except Exception as exc:
        logger.warning("Audio duration probe failed; timing estimates unavailable: %s", exc)
        file_dur = None

    try:
        if max_duration_sec is not None:
            _, validated_length, _ = resolve_region(
                file_dur,
                start_sec,
                start_sec + float(max_duration_sec),
            )
            max_duration_sec = float(validated_length)
        job_audio_sec = (
            float(max_duration_sec)
            if max_duration_sec is not None
            else max(0.0, float(file_dur or 0.0) - start_sec)
        )
    except RegionError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        logger.warning("Audio region validation failed; timing estimates unavailable: %s", exc)
        job_audio_sec = float(max_duration_sec) if max_duration_sec is not None else None

    if job_requires_roformer_backend(choice["model"]) and not is_roformer_backend_available():
        st.error(f"RoFormer backend is not installed. {ROFORMER_INSTALL_HINT}")
        return

    if not is_pro_mode(st.session_state):
        probe = get_desktop_probe()
        cap_sec = roformer_max_audio_sec(probe)
        effective_sec = effective_clip_seconds(
            file_duration_sec=file_dur,
            start_sec=start_sec,
            max_duration_sec=max_duration_sec,
        )
        block = lite_roformer_enqueue_block_reason(
            choice["model"],
            effective_sec=effective_sec,
            cap_sec=cap_sec,
            duration_known=effective_sec is not None,
            roformer_available=is_roformer_backend_available(),
        )
        if block:
            st.error(block)
            return

    try:
        ensure_cuda_available(choice["device"], get_desktop_probe())
    except RuntimeError:
        st.error(CUDA_UNAVAILABLE_MESSAGE)
        return

    resolved_name = output_name
    if region_label and max_duration_sec is not None:
        clip_suffix = format_region_label_filename(start_sec, float(max_duration_sec))
        resolved_name = f"{resolved_name}_{clip_suffix}"
        st.session_state["isolate_region_label"] = region_label
        st.session_state["isolate_clip_length"] = max_duration_sec
    else:
        st.session_state.pop("isolate_region_label", None)
        st.session_state.pop("isolate_clip_length", None)

    output_dir = run_output_dir()
    _copy_source_video_into_run(output_dir)
    source_fp = st.session_state.get("isolate_pending_fp") or st.session_state.get("isolate_upload_fp")
    youtube_preview_url = (choice.get("youtube_url") or "").strip()
    if not youtube_preview_url and str(source_fp or "").startswith("youtube:"):
        youtube_preview_url = str(source_fp)[len("youtube:") :]
    _download_youtube_preview_into_run(output_dir, youtube_preview_url)
    staged_before = st.session_state.get("isolate_pending_audio_path")
    staged_fp = str(source_fp) if source_fp else None
    if staged_fp and staged_fp.startswith("youtube:"):
        try:
            audio_path = adopt_audio_into_run(Path(audio_path), output_dir)
        except Exception:
            # Keep the staged path so the job can still run; discard will skip
            # while in-flight jobs reference it.
            pass
    browser_id = st.session_state.get("isolate_user_id")
    source_kind = infer_source_kind(
        source_fingerprint=str(source_fp) if source_fp else None,
        youtube_url=choice.get("youtube_url"),
    )
    origin_tab = str(st.session_state.get(SHELL_TAB_KEY) or "")
    if not is_new_draft_tab(origin_tab):
        origin_tab = ""
    metro = metronome_render_from_session(st.session_state)
    spec = IsolateJobSpec(
        id=uuid.uuid4().hex,
        audio_path=str(audio_path),
        output_dir=str(output_dir),
        title=resolved_name,
        model=choice["model"],
        quality=choice["quality"],
        device=choice["device"],
        start_sec=start_sec,
        max_duration_sec=max_duration_sec,
        two_stems=choice.get("two_stems"),
        guitar_checkpoint=choice.get("guitar_checkpoint"),
        two_pass=bool(choice.get("two_pass")),
        guitar_refine=bool(choice.get("guitar_refine")),
        low_end_restore_db=float(choice.get("low_end_restore_db") or 0.0),
        sub_bass_debleed=bool(choice.get("sub_bass_debleed")),
        demucs_segment=choice.get("demucs_segment", 8),
        demucs_jobs=int(choice.get("demucs_jobs") or 1),
        demucs_shifts=choice.get("demucs_shifts"),
        demucs_overlap=choice.get("demucs_overlap"),
        adaptive_fold_gain=bool(choice.get("adaptive_fold_gain")),
        bleed_gate=bool(choice.get("bleed_gate")),
        bass_bleed_mitigation=bool(choice.get("bass_bleed_mitigation")),
        guitar_ensemble=bool(choice.get("guitar_ensemble")),
        emit_stems=list(choice["emit_stems"]) if choice.get("emit_stems") else None,
        fold_other_into_guitar=bool(choice.get("fold_other_into_guitar", True)),
        fold_other_mode=str(choice.get("fold_other_mode") or "best_effort"),
        custom_stems=list(choice.get("custom_stems") or []),
        track_options=list(choice.get("track_options") or []),
        source_fingerprint=str(source_fp) if source_fp else None,
        source_kind=source_kind,
        region_label=region_label,
        owner=browser_id if isinstance(browser_id, str) else None,
        created_at=time.time(),
        audio_duration_sec=job_audio_sec,
        prior_timing=st.session_state.get("isolate_last_job_timing"),
        origin_tab=origin_tab or None,
        metronome_accent=metro.accent,
        metronome_rate=metro.rate,
        metronome_sound=metro.sound,
        lead_rhythm=bool(choice.get("lead_rhythm")),
        lead_rhythm_mode=str(choice.get("lead_rhythm_mode") or "confident"),
    )
    st.session_state["isolate_last_custom_stems"] = list(choice.get("custom_stems") or [])
    st.session_state["isolate_results_source_fp"] = source_fp
    enqueue_job(spec)
    if staged_before:
        discard_youtube_staging(
            str(staged_before),
            fingerprint=staged_fp,
            retain_paths=(audio_path,),
        )
    reset_new_tab_source(st.session_state)
    if origin_tab:
        # Keep the same New tab focused with the song title + in-tab progress.
        mark_draft_tab_processing(
            st.session_state, origin_tab, title=resolved_name
        )
    st.session_state["isolate_flash"] = f"Separating **{resolved_name}**…"
    if job_audio_sec:
        st.session_state["isolate_last_job_timing"] = {
            "audio_sec": job_audio_sec,
            "quality": choice["quality"],
            "device": choice["device"],
            "model": choice["model"],
            "two_pass": bool(choice.get("two_pass")),
            "guitar_refine": bool(choice.get("guitar_refine")),
        }
    # Stay on the draft tab (or Home); open Queue on the next paint.
    st.session_state["_isolate_pending_queue"] = True
    st.session_state[ISOLATE_QUEUE_PANEL_OPEN_KEY] = True
    if origin_tab:
        # Do not jump to Home chrome — processing stays on this tab.
        st.session_state["_isolate_keep_draft_tab"] = origin_tab
    _rerun_scroll_top()


def _library_status_row(row: dict) -> dict:
    return {
        "id": row.get("id") or row.get("run_dir"),
        "status": "succeeded",
        "title": row.get("title") or "tracks",
        "run_dir": row.get("run_dir"),
        "artifacts": row.get("artifacts") or {},
        "source_audio_path": row.get("source_audio_path"),
        "source_fingerprint": row.get("source_fingerprint"),
        "source_kind": row.get("source_kind"),
        "region_label": row.get("region_label"),
        "clip_length": row.get("clip_length"),
        "custom_stems": row.get("custom_stems") or [],
        "created_at": row.get("created_at"),
    }


def _apply_library_row(row: dict, *, viewing_mode: str, reopen_name: bool) -> bool:
    status = _library_status_row(row)
    if not apply_succeeded_job_to_session(
        st.session_state,
        status,
        viewing_mode=viewing_mode,
    ):
        return False
    st.session_state.pop(ISOLATE_SKIP_REHYDRATE_KEY, None)
    run_dir = status.get("run_dir")
    if run_dir:
        st.session_state["isolate_listen_applied_dir"] = str(run_dir)
        st.session_state[LISTEN_PICKER_KEY] = str(run_dir)
        add_open_mix_tab(st.session_state, run_dir)
    if reopen_name:
        queue_reopen_output_name(st.session_state, status.get("title") or "tracks")
    _persist_isolate_ui_state()
    return True


def _focus_mix_tab(rows: list[dict], run_dir: str) -> None:
    row = next((r for r in rows if str(r.get("run_dir")) == str(run_dir)), None)
    if row is None:
        st.session_state["isolate_listen_missing"] = str(run_dir)
        close_open_mix_tab(st.session_state, run_dir)
        return
    _apply_library_row(row, viewing_mode=str(row.get("id") or str(run_dir)), reopen_name=True)


def _library_rows_available(browser_id: str | None) -> list[dict]:
    owner = browser_id if isinstance(browser_id, str) else None
    owned = list_recent_runs("isolate", owner=owner)
    unfiltered = owned if owned else list_recent_runs("isolate")
    recent = recent_runs_with_owner_fallback(owned, unfiltered)
    merged = merge_library_runs(recent, list_jobs(limit=30))
    usable: list[dict] = []
    for row in merged:
        paths = _stem_paths_from_artifacts(row.get("artifacts") or {})
        if paths:
            usable.append(row)
    return usable


def _clear_loaded_mixer(*, dismiss: bool = False) -> None:
    """Drop in-session mixer pointers.

    ``dismiss=True`` means the user closed the last mix tab — skip auto-rehydrate
    so the next paint does not revive that tab from the library.
    """
    _clear_loaded_mixer_keys(st.session_state, dismiss=dismiss)


def _dismiss_last_mix_tab() -> None:
    """Close the last mix without letting rehydrate reopen it."""
    _clear_loaded_mixer(dismiss=True)
    open_home_shell(st.session_state)


def _rename_listen_run() -> None:
    """Rename the currently-chosen mix from the editable "Listening to" name field.

    Runs only on Enter (on_change), so ordinary reruns never rewrite the title.
    """
    new_name = (st.session_state.get("isolate_listen_name") or "").strip()
    run_dir = st.session_state.get(LISTEN_PICKER_KEY) or st.session_state.get(
        "isolate_listen_applied_dir"
    )
    if not new_name or not run_dir:
        return
    current = st.session_state.get("isolate_base_name") or ""
    if new_name == current:
        return
    if rename_run_title(Path(run_dir), new_name):
        st.session_state["isolate_base_name"] = new_name
        _persist_isolate_ui_state()


def _render_listening_switcher(browser_id: str | None, rows: list[dict] | None = None) -> None:
    """Active mix header: rename + delete (open tabs live in the Moises strip)."""
    if rows is None:
        rows = _library_rows_available(browser_id)
    if not rows and not st.session_state.get("isolate_run_dir"):
        return
    options = [str(r["run_dir"]) for r in rows]
    option_set = set(options)
    apply_listen_picker_pending(st.session_state)

    loaded = st.session_state.get("isolate_run_dir") or st.session_state.get(
        "isolate_listen_applied_dir"
    )
    if loaded:
        add_open_mix_tab(st.session_state, loaded)
    open_mix_tabs_for_session(st.session_state, library_dirs=options or None)

    current = st.session_state.get(LISTEN_PICKER_KEY)
    if current not in option_set:
        default = listen_picker_default(
            options,
            str(loaded) if loaded else None,
            None,
        )
        if default:
            st.session_state[LISTEN_PICKER_KEY] = default
            current = default
    elif current:
        add_open_mix_tab(st.session_state, current)

    rename_dir = st.session_state.get(LISTEN_PICKER_KEY) or st.session_state.get(
        "isolate_listen_applied_dir"
    )
    if rename_dir:
        current_name = st.session_state.get("isolate_base_name") or "tracks"
        if st.session_state.get("isolate_listen_name_for") != rename_dir:
            st.session_state["isolate_listen_name"] = current_name
            st.session_state["isolate_listen_name_for"] = rename_dir

    closed_options = [
        d
        for d in options
        if d not in set(open_mix_tabs_for_session(st.session_state, library_dirs=options))
    ]
    if closed_options:
        labels = {
            str(r["run_dir"]): format_source_title(
                r.get("title") or "tracks",
                infer_source_kind(
                    source_kind=r.get("source_kind"),
                    source_fingerprint=r.get("source_fingerprint"),
                ),
            )
            for r in rows
        }
        open_choice = st.selectbox(
            "Open…",
            options=[""] + closed_options,
            format_func=lambda d: "Open a mix from library…" if not d else labels.get(d, d),
            key="isolate_open_mix_from_library",
        )
        if open_choice:
            add_open_mix_tab(st.session_state, open_choice)
            st.session_state["isolate_open_mix_from_library"] = ""
            _focus_mix_tab(rows, open_choice)
            open_mix_shell(st.session_state)
            _rerun_scroll_top()

    if "isolate_listen_name" not in st.session_state:
        st.session_state["isolate_listen_name"] = (
            st.session_state.get("isolate_base_name") or "tracks"
        )
    cols = st.columns([4, 1], vertical_alignment="bottom")
    with cols[0]:
        st.text_input(
            label="Listening to",
            key="isolate_listen_name",
            on_change=_rename_listen_run,
        )
    with cols[1]:
        delete_clicked = _confirm_button(
            "Delete this run",
            key="isolate_delete_listening",
            confirm_label="Delete from disk",
            confirm_help="Removes these separated tracks. No undo.",
            width="stretch",
        )

    chosen = str(st.session_state.get(LISTEN_PICKER_KEY) or "")
    if delete_clicked and chosen:
        deleted = delete_library_run(chosen)
        if deleted:
            neighbor = close_open_mix_tab(st.session_state, chosen)
            was_active = st.session_state.get("isolate_run_dir") == chosen
            st.session_state.pop("isolate_listen_applied_dir", None)
            st.session_state.pop(LISTEN_PICKER_KEY, None)
            st.session_state.pop(LISTEN_PICKER_NEXT_KEY, None)
            if was_active:
                if neighbor:
                    _clear_loaded_mixer()
                    row = next((r for r in rows if str(r.get("run_dir")) == neighbor), None)
                    if row is not None:
                        _apply_library_row(
                            row,
                            viewing_mode=str(row.get("id") or neighbor),
                            reopen_name=True,
                        )
                    else:
                        _dismiss_last_mix_tab()
                else:
                    _dismiss_last_mix_tab()
            _persist_isolate_ui_state()
            _rerun_scroll_top()
        else:
            st.error("Could not delete that separation.")
            return

    applied = st.session_state.get("isolate_listen_applied_dir")
    if chosen and chosen != applied:
        row = next((r for r in rows if str(r.get("run_dir")) == chosen), None)
        if row is None:
            st.caption("Those files are no longer available.")
            st.session_state["isolate_listen_missing"] = str(chosen)
            close_open_mix_tab(st.session_state, chosen)
            st.session_state.pop(LISTEN_PICKER_KEY, None)
            _rerun_scroll_top()
            return
        viewing_id = str(row.get("id") or chosen)
        if _apply_library_row(row, viewing_mode=viewing_id, reopen_name=True):
            _rerun_scroll_top()
        else:
            st.caption("Those files are no longer available.")
            st.session_state["isolate_listen_missing"] = str(chosen)
            close_open_mix_tab(st.session_state, chosen)
            st.session_state.pop(LISTEN_PICKER_KEY, None)
            _rerun_scroll_top()


def _render_moises_tab_strip(browser_id: str | None) -> None:
    """Top Home | mix tabs | + chrome (sticky while scrolling)."""
    rows = _library_rows_available(browser_id)
    labels = {
        str(r["run_dir"]): format_source_title(
            r.get("title") or "tracks",
            infer_source_kind(
                source_kind=r.get("source_kind"),
                source_fingerprint=r.get("source_fingerprint"),
            ),
        )
        for r in rows
    }
    options = [str(r["run_dir"]) for r in rows]
    loaded = st.session_state.get("isolate_run_dir") or st.session_state.get(
        "isolate_listen_applied_dir"
    )
    if loaded:
        add_open_mix_tab(st.session_state, loaded)
    open_tabs = open_mix_tabs_for_session(st.session_state, library_dirs=options)
    shell = apply_shell_view(st.session_state)
    shell_tab = str(st.session_state.get(SHELL_TAB_KEY) or "")
    draft_active = shell == "home" and is_new_draft_tab(shell_tab)
    active_id = str(st.session_state.get(LISTEN_PICKER_KEY) or loaded or "")
    jobs = list_jobs(limit=30)
    consumed_ids = (
        None
        if "isolate_consumed_job_ids" not in st.session_state
        else list(st.session_state.get("isolate_consumed_job_ids") or [])
    )
    overlays = draft_tab_job_overlays(
        jobs,
        pending_open_ids=pending_open_job_ids(jobs, consumed_ids),
    )
    tab_payload = []
    for d in open_tabs:
        overlay = overlays.get(d) or {}
        if is_new_draft_tab(d):
            title = str(overlay.get("title") or "") or draft_tab_title(
                st.session_state, d
            )
            active = draft_active and d == shell_tab
        else:
            title = labels.get(d, Path(d).name)
            active = shell == "mix" and d == active_id
        item: dict = {"id": d, "title": title, "active": active}
        if overlay.get("busy"):
            item["busy"] = True
            progress = overlay.get("progress")
            if progress is not None:
                item["progress"] = float(progress)
        tab_payload.append(item)

    result = None
    last_seq = int(st.session_state.get(MIX_TAB_EVENT_SEQ_KEY) or 0)
    with st.container(key="isolate_mix_tabs_strip"):
        try:
            from ui.mix_tabs_component import component_build_available, mix_tabs

            if component_build_available():
                theme_vars = get_iframe_theme_vars(st.session_state)
                result = mix_tabs(
                    tabs=tab_payload,
                    home_label="Home",
                    home_active=shell == "home" and not draft_active,
                    show_plus=True,
                    theme_vars=theme_vars,
                    key="isolate_moises_tabs",
                )
            else:
                st.caption("Mix tabs UI missing. Run `make mix-tabs-build`.")
        except Exception as exc:
            logger.warning("Mix tabs component failed: %s", exc)
            st.caption("Mix tabs unavailable.")

    last_seq, action, tab_id = consume_mix_tab_event(result, last_seq)
    if action is None:
        return
    st.session_state[MIX_TAB_EVENT_SEQ_KEY] = last_seq
    st.session_state.pop(MIX_TAB_NOTICE_KEY, None)
    if action == "home":
        save_active_draft_source(st.session_state)
        open_home_shell(st.session_state)
        _persist_isolate_ui_state()
        _rerun_scroll_top()
    elif action == "plus":
        open_new_draft_tab(st.session_state, fresh=True)
        _persist_isolate_ui_state()
        _rerun_scroll_top()
    elif action == "focus" and tab_id:
        if is_new_draft_tab(tab_id):
            focus_new_draft_tab(st.session_state, tab_id)
            _persist_isolate_ui_state()
            _rerun_scroll_top()
        else:
            save_active_draft_source(st.session_state)
            _focus_mix_tab(rows, tab_id)
            open_mix_shell(st.session_state)
            _rerun_scroll_top()
    elif action == "close" and tab_id:
        busy = overlays.get(tab_id) or {}
        if busy.get("busy"):
            st.session_state[BUSY_TAB_CLOSE_KEY] = {
                "id": tab_id,
                "title": str(busy.get("title") or "") or draft_tab_title(st.session_state, tab_id),
            }
            _rerun_scroll_top()
            return
        # Close the strip only. Queue → Delete removes the run from disk.
        _close_mix_tab(rows, tab_id)
    elif action == "reorder" and tab_id:
        if move_open_mix_tab(st.session_state, tab_id, mix_tab_event_index(result)):
            _persist_isolate_ui_state()
        # Order is strip-only: re-render this fragment, keep the page scroll.
        _rerun_preserve_scroll()


@st.fragment(run_every=1.0)
def _mix_tab_strip_fragment(browser_id: str | None) -> None:
    """Refresh tab busy/progress without remounting the mixer.

    Progress ticks stay inside this fragment. Tab clicks still call
    ``_rerun_scroll_top`` from the strip, which is a full-page rerun.
    A file opened from the OS while this window is up is claimed here, then
    the whole page remounts so Home shows the new tab.
    """
    if _consume_opened_audio():
        _rerun_scroll_top()
    _render_moises_tab_strip(browser_id)


def _render_busy_tab_close_prompt(browser_id: str | None) -> None:
    """Ask before closing a tab whose separation is still running."""
    pending = st.session_state.get(BUSY_TAB_CLOSE_KEY)
    if not isinstance(pending, dict) or not pending.get("id"):
        return
    tab_id = str(pending["id"])
    title = str(pending.get("title") or "this track")
    with st.container(border=True, key="isolate_busy_tab_close"):
        st.warning(
            f"**{title}** is still separating. Closing the tab does not stop the job; "
            "the tab comes back when it finishes."
        )
        cols = st.columns(2)
        with cols[0]:
            if st.button(
                "Close tab, keep separating",
                key="isolate_busy_close_yes",
                type="primary",
                width="stretch",
            ):
                st.session_state.pop(BUSY_TAB_CLOSE_KEY, None)
                _close_mix_tab(_library_rows_available(browser_id), tab_id)
        with cols[1]:
            if st.button("Keep tab open", key="isolate_busy_close_no", width="stretch"):
                st.session_state.pop(BUSY_TAB_CLOSE_KEY, None)
                _rerun_scroll_top()


def _close_mix_tab(rows: list[dict], tab_id: str) -> None:
    draft_was_focused = is_new_draft_tab(tab_id) and str(
        st.session_state.get(SHELL_TAB_KEY) or ""
    ) == str(tab_id)
    neighbor = close_open_mix_tab(st.session_state, tab_id)
    if is_new_draft_tab(tab_id):
        if draft_was_focused:
            if neighbor and is_new_draft_tab(neighbor):
                focus_new_draft_tab(st.session_state, neighbor)
            elif neighbor:
                _focus_mix_tab(rows, neighbor)
                open_mix_shell(st.session_state)
            else:
                open_home_shell(st.session_state)
        _persist_isolate_ui_state()
        _rerun_scroll_top()
        return
    # Always detach loaded mixer pointers for the closed run. Otherwise the
    # strip's ``if loaded: add_open_mix_tab(loaded)`` reopens the last tab.
    if is_loaded_mix_tab(st.session_state, tab_id):
        if neighbor and is_new_draft_tab(neighbor):
            focus_new_draft_tab(st.session_state, neighbor)
        elif neighbor:
            _focus_mix_tab(rows, neighbor)
            open_mix_shell(st.session_state)
        else:
            _dismiss_last_mix_tab()
    elif neighbor is None and not any(
        not is_new_draft_tab(t)
        for t in (st.session_state.get(OPEN_MIX_TABS_KEY) or [])
    ):
        _dismiss_last_mix_tab()
    _persist_isolate_ui_state()
    _rerun_scroll_top()


def _stage_batch_upload(uploaded: object) -> tuple[Path | None, str | None]:
    """Stage one batch file without replacing the single-file pending slot."""
    name = str(getattr(uploaded, "name", None) or "file")
    try:
        path = save_upload(uploaded)
    except Exception:
        return None, f"Could not read {name}."
    if path.suffix.lower() not in _VIDEO_UPLOAD_SUFFIXES:
        return path, None
    wav = path.with_name(f"{path.stem}.extracted.wav")
    try:
        audio = normalize_audio(path, wav)
    except Exception as exc:
        wav.unlink(missing_ok=True)
        logger.warning("Could not read audio from uploaded video: %s", exc)
        return None, "Could not read audio from this video."
    return audio, None


def _spec_for_batch_file(choice: dict, audio_path: Path) -> IsolateJobSpec:
    """One full-file job using the form's track, speed, and device choices."""
    browser_id = st.session_state.get("isolate_user_id")
    origin_tab = str(st.session_state.get(SHELL_TAB_KEY) or "")
    if not is_new_draft_tab(origin_tab):
        origin_tab = ""
    metro = metronome_render_from_session(st.session_state)
    return IsolateJobSpec(
        id=uuid.uuid4().hex,
        audio_path=str(audio_path),
        output_dir=str(run_output_dir()),
        title=audio_path.stem,
        model=choice["model"],
        quality=choice["quality"],
        device=choice["device"],
        two_stems=choice.get("two_stems"),
        guitar_checkpoint=choice.get("guitar_checkpoint"),
        two_pass=bool(choice.get("two_pass")),
        guitar_refine=bool(choice.get("guitar_refine")),
        low_end_restore_db=float(choice.get("low_end_restore_db") or 0.0),
        sub_bass_debleed=bool(choice.get("sub_bass_debleed")),
        demucs_segment=choice.get("demucs_segment", 8),
        demucs_jobs=int(choice.get("demucs_jobs") or 1),
        demucs_shifts=choice.get("demucs_shifts"),
        demucs_overlap=choice.get("demucs_overlap"),
        adaptive_fold_gain=bool(choice.get("adaptive_fold_gain")),
        bleed_gate=bool(choice.get("bleed_gate")),
        bass_bleed_mitigation=bool(choice.get("bass_bleed_mitigation")),
        guitar_ensemble=bool(choice.get("guitar_ensemble")),
        emit_stems=list(choice["emit_stems"]) if choice.get("emit_stems") else None,
        fold_other_into_guitar=bool(choice.get("fold_other_into_guitar", True)),
        fold_other_mode=str(choice.get("fold_other_mode") or "best_effort"),
        custom_stems=list(choice.get("custom_stems") or []),
        track_options=list(choice.get("track_options") or []),
        source_kind=infer_source_kind(source_fingerprint=None, youtube_url=""),
        owner=browser_id if isinstance(browser_id, str) else None,
        origin_tab=origin_tab or None,
        metronome_accent=metro.accent,
        metronome_rate=metro.rate,
        metronome_sound=metro.sound,
        lead_rhythm=bool(choice.get("lead_rhythm")),
        lead_rhythm_mode=str(choice.get("lead_rhythm_mode") or "confident"),
    )


def _enqueue_upload_batch(
    choice: dict,
    uploads: list,
    staged_paths: list[Path] | None = None,
) -> None:
    """Queue several local files. The serial worker runs the oldest first.

    ``staged_paths`` is set when screen 1 already copied the uploader. Passing
    it skips ``_stage_batch_upload`` so the same files are not saved twice.
    """
    section_selected = (
        choice.get("max_duration_sec") is not None
        or bool(choice.get("region_label"))
        or float(choice.get("start_sec") or 0.0) > 0.5
    )
    youtube_url = (choice.get("youtube_url") or "").strip()
    if section_selected or youtube_url:
        blocked = enqueue_local_batch(
            [],
            section_selected=section_selected,
            youtube_url=youtube_url,
            make_spec=lambda path: _spec_for_batch_file(choice, path),
        )
        st.error(blocked.blocked or "Could not queue these files.")
        return
    if not is_pro_mode(st.session_state) and not st.session_state.get(
        "isolate_lite_run_on_manual"
    ):
        lite_auto = lite_auto_choice(get_desktop_probe())
        choice = dict(choice)
        choice["device"] = str(lite_auto["device"])
        if choice["device"] == "cpu":
            choice["quality"] = "fast"
        st.session_state["isolate_device"] = choice["device"]
        st.session_state["isolate_quality"] = choice["quality"]
    if job_requires_roformer_backend(choice["model"]) and not is_roformer_backend_available():
        st.error(f"RoFormer backend is not installed. {ROFORMER_INSTALL_HINT}")
        return
    try:
        ensure_cuda_available(choice["device"], get_desktop_probe())
    except RuntimeError:
        st.error(CUDA_UNAVAILABLE_MESSAGE)
        return
    probe = get_desktop_probe()
    cap_sec = roformer_max_audio_sec(probe)
    staged: list[Path] = []
    notes: list[str] = []
    incoming: list[Path] = []
    if staged_paths is None:
        for item in uploads:
            path, err = _stage_batch_upload(item)
            if err or path is None:
                notes.append(err or f"Could not read {getattr(item, 'name', 'file')}.")
                continue
            incoming.append(path)
    else:
        incoming = list(staged_paths)
    for path in incoming:
        if not is_pro_mode(st.session_state):
            duration = _cached_probe_duration_sec(path)
            effective = effective_clip_seconds(
                file_duration_sec=duration,
                start_sec=0.0,
                max_duration_sec=None,
            )
            block = lite_roformer_enqueue_block_reason(
                choice["model"],
                effective_sec=effective,
                cap_sec=cap_sec,
                duration_known=effective is not None,
                roformer_available=is_roformer_backend_available(),
            )
            if block:
                notes.append(f"{path.stem}: {block}")
                continue
        staged.append(path)
    result = enqueue_local_batch(
        staged,
        section_selected=False,
        youtube_url="",
        make_spec=lambda path: _spec_for_batch_file(choice, path),
    )
    notes.extend(result.failed)
    if result.omitted == 1:
        notes.append("1 file was left out.")
    elif result.omitted > 1:
        notes.append(f"{result.omitted} files were left out.")
    if notes:
        st.warning(" ".join(notes))
    if not result.job_ids:
        if not notes:
            st.error("Could not queue these files.")
        return
    st.session_state["isolate_last_custom_stems"] = list(choice.get("custom_stems") or [])
    reset_new_tab_source(st.session_state)
    count = len(result.job_ids)
    st.session_state["isolate_flash"] = (
        "Separating **1** song…" if count == 1 else f"Queued **{count}** songs. One runs at a time."
    )
    st.session_state["_isolate_pending_queue"] = True
    st.session_state[ISOLATE_QUEUE_PANEL_OPEN_KEY] = True
    _rerun_scroll_top()


def _try_enqueue_choice(choice: dict) -> None:
    """Queue the current Home form. Same checks as Separate tracks."""
    uploads = choice.get("uploaded")
    live_batch = isinstance(uploads, list) and len(uploads) > 1
    batch_paths = _stored_batch_paths()
    if len(batch_paths) > 1 or live_batch:
        if choice.get("error"):
            st.error(choice["error"])
            return
        if len(batch_paths) > 1:
            _enqueue_upload_batch(choice, [], staged_paths=batch_paths)
        else:
            _enqueue_upload_batch(choice, list(uploads))
        return
    if choice.get("error"):
        st.error(choice["error"])
        return
    if choice.get("max_duration_sec") is not None and choice["max_duration_sec"] < MIN_REGION_SEC:
        st.error(f"Section must be at least {MIN_REGION_SEC:.0f} seconds.")
        return
    if not _has_source_for_job(choice):
        st.error("Upload an audio file or enter a YouTube URL.")
        return
    audio_path, resolve_error = _resolve_audio_for_job(choice)
    if resolve_error:
        st.error(resolve_error)
        return
    if audio_path is None:
        st.error("Upload an audio file or enter a YouTube URL.")
        return
    _enqueue_confirmed_job(choice, audio_path)


def _has_source_for_job(choice: dict) -> bool:
    uploaded = choice.get("uploaded")
    has_upload = (
        isinstance(uploaded, list) and len(uploaded) > 0
    ) or (uploaded is not None and not isinstance(uploaded, list))
    return bool(
        has_upload
        or len(_stored_batch_paths()) > 1
        or choice.get("youtube_url")
        or (
            st.session_state.get("carry_over_audio_path")
            and Path(st.session_state["carry_over_audio_path"]).exists()
        )
        or (
            st.session_state.get("isolate_pending_audio_path")
            and Path(st.session_state["isolate_pending_audio_path"]).exists()
        )
    )


def _render_new_workspace(demucs_ok: bool) -> None:
    choice = _render_separation_controls()
    if st.session_state.pop("isolate_search_enqueue", False):
        _try_enqueue_choice(choice)
    if current_setup_step(st.session_state) != SETUP_STEP_SEPARATE:
        return
    # Jobs run one at a time. Saying "Separate tracks" while one is already
    # running promises something immediate and then silently queues instead.
    busy = separation_in_progress()
    if busy:
        st.caption("A separation is already running. This one starts when that finishes.")
    has_source = _has_source_for_job(choice)
    if st.button(
        "Add to queue" if busy else "Separate tracks",
        type="primary",
        disabled=not demucs_ok or not has_source,
        key="isolate_separate",
    ):
        _try_enqueue_choice(choice)
    card = str(st.session_state.get(OUTCOME_CARD_KEY) or DEFAULT_OUTCOME_CARD)
    if card in OUTCOME_CARDS:
        outcome_label = str(OUTCOME_CARDS[card]["label"])
    else:
        stems = custom_stems_from_options(
            st.session_state.get("isolate_track_options") or []
        )
        names = [CUSTOM_STEM_CHOICES[s] for s in stems]
        outcome_label = f"Custom · {', '.join(names)}" if names else "Custom"
    source_note = "Source ready" if has_source else "Upload a file or a YouTube URL"
    st.caption(f"{outcome_label} · {source_note}")


def _render_mixer_region_caption(base_name: str) -> None:
    region_caption = st.session_state.get("isolate_region_label")
    source_audio_path = st.session_state.get("isolate_source_audio_path")
    clip_length = st.session_state.get("isolate_clip_length")
    filename = Path(source_audio_path).name if source_audio_path else None
    kind = infer_source_kind(
        source_kind=st.session_state.get("isolate_source_kind"),
        source_fingerprint=st.session_state.get("isolate_results_source_fp"),
    )
    st.caption(
        format_source_caption(
            base_name,
            kind,
            filename=filename,
            region_label=region_caption if isinstance(region_caption, str) else None,
            clip_length=float(clip_length) if clip_length is not None else None,
        )
    )


def _load_parent_config(run_dir: Path) -> dict:
    """Read the parent run's persisted separator config (Feature 2), else {}."""
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = meta.get("config")
    return dict(cfg) if isinstance(cfg, dict) else {}


def _enqueue_reseparate_job(
    stem_id: str,
    stem_path: Path,
    run_dir: Path,
    parent_config: dict,
) -> None:
    """Enqueue a re-separate of one existing stem, defaulting from the parent run."""
    output_dir = run_output_dir()
    source_fp = st.session_state.get("isolate_results_source_fp")
    source_kind = infer_source_kind(
        source_fingerprint=str(source_fp) if source_fp else None,
        source_kind=st.session_state.get("isolate_source_kind"),
    )
    spec = IsolateJobSpec(
        id=uuid.uuid4().hex,
        audio_path=str(stem_path),
        output_dir=str(output_dir),
        title=f"{stem_id} (re-separate)",
        model=str(parent_config.get("model") or "htdemucs_6s"),
        quality=str(parent_config.get("quality") or "balanced"),
        device=str(parent_config.get("device") or "cpu"),
        two_pass=bool(parent_config.get("two_pass", False)),
        guitar_refine=bool(parent_config.get("guitar_refine", False)),
        low_end_restore_db=float(parent_config.get("low_end_restore_db") or 0.0),
        sub_bass_debleed=bool(parent_config.get("sub_bass_debleed", False)),
        demucs_segment=parent_config.get("demucs_segment", 8),
        demucs_jobs=int(parent_config.get("demucs_jobs") or 1),
        demucs_shifts=parent_config.get("demucs_shifts"),
        demucs_overlap=parent_config.get("demucs_overlap"),
        adaptive_fold_gain=bool(parent_config.get("adaptive_fold_gain", False)),
        bleed_gate=bool(parent_config.get("bleed_gate", False)),
        bass_bleed_mitigation=bool(parent_config.get("bass_bleed_mitigation", False)),
        guitar_ensemble=bool(parent_config.get("guitar_ensemble", False)),
        # Fold-other is off by default for a breakdown.
        fold_other_into_guitar=False,
        fold_other_mode="best_effort",
        custom_stems=[stem_id],
        source_fingerprint=str(source_fp) if source_fp else None,
        source_kind=source_kind,
        owner=st.session_state.get("isolate_user_id"),
        created_at=time.time(),
        reseparate_from=stem_id,
        parent_run_dir=str(run_dir),
    )
    enqueue_job(spec)
    st.session_state["isolate_flash"] = f"Separating breakdown of **{stem_id}**…"
    # Stay on Mixer — do not jump to Queue for re-separate.
    st.rerun()


def _render_reseparate_panel(stem_paths: dict[str, Path], run_dir: Path) -> None:
    """'Break down a stem' control — re-separate one stem in place (section 6)."""
    if len(stem_paths) < 1:
        return
    parent_config = _load_parent_config(run_dir)
    run_fp = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:12]
    with st.expander(
        "Break down a stem (re-separate)",
        expanded=False,
    ):
        st.caption(
            "Re-run the separator on one existing stem to break out more "
            "instruments without reprocessing the whole track. On success the "
            "source stem is replaced in the mixer by its new sub-stems."
        )
        if not parent_config:
            st.caption("Parent run has no saved separator config. Using defaults.")
        for stem_id in sort_stem_names(stem_paths.keys()):
            if stem_id == "metronome":
                continue
            label = stem_label_for_id(stem_id)
            if st.button(
                f"Break down {label}",
                key=f"reseparate_{run_fp}_{stem_id}",
                help=f"Separate `{stem_id}` again into sub-stems.",
            ):
                _enqueue_reseparate_job(stem_id, stem_paths[stem_id], run_dir, parent_config)


def sidebar_chord_visible(shell: str, run_dir: str | None, analysis: dict | None) -> bool:
    """True only while a mix-shell run already has chords."""
    if str(shell or "") != "mix":
        return False
    if not str(run_dir or "").strip():
        return False
    if not isinstance(analysis, dict):
        return False
    chords = analysis.get("chords")
    return isinstance(chords, list) and len(chords) > 0


_SIDEBAR_CHORD_HTML = """
<style>
  body:has(.att-sidebar-chord) [data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    min-height: calc(100vh - 2rem);
  }
  body:has(.att-sidebar-chord) [data-testid="stElementContainer"]:has(.att-sidebar-chord) {
    margin-top: auto !important;
    padding-top: 8px;
    width: 100%;
  }
  .att-sidebar-chord {
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    width: 100%;
    max-width: 216px;
    margin-top: 8px;
    padding: 12px;
    border-top: 1px solid #A8B8CC;
    background: #192542;
    color: #F4F7FB;
    text-align: center;
  }
  .att-sidebar-chord-eyebrow {
    color: #B7C3D4;
    font-size: 0.75rem;
    line-height: 1.2;
  }
  .att-sidebar-chord-symbol {
    margin-top: 4px;
    color: #F6AD49;
    font-weight: 750;
    font-size: 1.8rem;
    line-height: 1;
  }
  .att-sidebar-chord-diagram {
    width: 168px;
  }
  .att-sidebar-chord-diagram .chord-diagram {
    width: 168px;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  .att-sidebar-chord .chord-string,
  .att-sidebar-chord .chord-fret {
    stroke: #B7C3D4;
    stroke-width: 1;
    opacity: 0.55;
  }
  .att-sidebar-chord .chord-fret { opacity: 0.28; }
  .att-sidebar-chord .chord-nut { fill: #F4F7FB; opacity: 0.75; }
  .att-sidebar-chord .chord-dot { fill: #F6AD49; }
  .att-sidebar-chord .chord-open { fill: none; stroke: #F4F7FB; stroke-width: 1.2; }
  .att-sidebar-chord .chord-mute,
  .att-sidebar-chord .chord-fret-label {
    fill: #F4F7FB;
    font-size: 10px;
    font-weight: 700;
  }
</style>
<div class="att-sidebar-chord">
  <div class="att-sidebar-chord-eyebrow">Chord</div>
  <div class="att-sidebar-chord-symbol" aria-live="polite"></div>
  <div class="att-sidebar-chord-diagram"></div>
</div>
"""


def _sidebar_chord_seed(analysis: dict) -> str:
    """First real chord name, escaped, so the card is not blank before playback."""
    chords = analysis.get("chords")
    if not isinstance(chords, list):
        return ""
    for chord in chords:
        if not isinstance(chord, dict):
            continue
        symbol = str(chord.get("symbol") or "").strip()
        if symbol and symbol.upper() != "N":
            return html.escape(symbol)
    return ""


def _render_sidebar_chord() -> None:
    """Current-chord card. Playback updates it by postMessage, not a rerun."""
    shell = str(st.session_state.get(SHELL_VIEW_KEY) or "")
    viewing = st.session_state.get("isolate_viewing_run_dir")
    listen = st.session_state.get(LISTEN_PICKER_KEY)
    tab = st.session_state.get(SHELL_TAB_KEY)
    run = mixer_focus_run_dir(
        shell_view=shell,
        viewing_run_dir=viewing if isinstance(viewing, str) else None,
        listen_picker=listen if isinstance(listen, str) else None,
        shell_tab=tab if isinstance(tab, str) else None,
    )
    analysis = load_analysis_for_run(run) if run else None
    if not isinstance(analysis, dict) or not sidebar_chord_visible(shell, run, analysis):
        return
    card = _SIDEBAR_CHORD_HTML.replace(
        '<div class="att-sidebar-chord-symbol" aria-live="polite"></div>',
        '<div class="att-sidebar-chord-symbol" aria-live="polite">'
        + _sidebar_chord_seed(analysis)
        + "</div>",
        1,
    )
    with st.sidebar:
        st.html(card, unsafe_allow_javascript=True)


def _render_mixer_workspace(browser_id: str | None) -> None:
    _render_sidebar_chord()
    owner = browser_id if isinstance(browser_id, str) else None
    rows = _library_rows_available(owner)
    if not _session_has_mixer_wavs():
        row = select_rehydrate_row(st.session_state, rows, wav_exists=_wav_exists)
        if row is not None and _apply_library_row(
            row, viewing_mode="latest", reopen_name=False
        ):
            _rerun_scroll_top()

    if rows:
        _render_listening_switcher(owner, rows)

    artifacts_map = st.session_state.get("isolate_artifacts")
    stem_paths = _stem_paths_from_artifacts(artifacts_map or {})
    if not stem_paths:
        if not rows:
            st.caption("No tracks yet.")
        else:
            st.caption("Pick a run above.")
        return

    pro = is_pro_mode(st.session_state)
    pending_fp = pending_upload_fp_for_stale(st.session_state)
    show_file_ready_banner = should_hide_stale_results(
        pending_upload_fp=pending_fp,
        has_artifacts=True,
        results_source_fp=st.session_state.get("isolate_results_source_fp"),
    )
    if pro and not show_file_ready_banner:
        st.caption(
            "New file selected. Mixer still shows the chosen run until you separate again."
        )

    base_name = st.session_state.get("isolate_base_name", "stems")
    st.session_state["isolate_results_fp"] = _artifact_fingerprint(stem_paths)
    run_dir = Path(
        st.session_state.get("isolate_run_dir", next(iter(stem_paths.values())).parent)
    )

    bass_bleed = _load_bass_bleed_diagnostics(artifacts_map)
    extra_uploads = st.file_uploader(
        "Add extra tracks",
        type=["mp3", "wav", "flac", "m4a"],
        accept_multiple_files=True,
        key="isolate_extra_tracks",
        help="Local files kept with this song. They are not separated.",
    )
    if extra_uploads:
        from ui.library_song import stage_extra_tracks

        saved = stage_extra_tracks(
            run_dir,
            [(str(item.name), item.getvalue()) for item in extra_uploads],
        )
        if saved:
            st.caption("Extra tracks: " + ", ".join(Path(path).name for path in saved))

    if pro:
        _render_mixer_region_caption(base_name)

    # Mixer first. Guitar fix-up and re-separate used to sit above it, so the
    # result people waited minutes for was below two panels of repair tooling.
    # NOTE: media URLs are generated once per session state, but Streamlit fragments
    # may re-render at different times. If URL generation becomes expensive or stale,
    # wrap with @st.cache_resource and a TTL.
    mixer_media_urls, mixer_download_urls = register_mixer_media(stem_paths)
    # Register the tab PDF during the full-page run. Fragment-scoped media can be
    # garbage-collected on a later fragment rerun, breaking the link.
    tab_pdf_url = None
    tab_pdf_path = run_dir / "tab" / "transcription.pdf"
    if tab_pdf_path.is_file():
        try:
            tab_pdf_url = media_url_for_file(
                tab_pdf_path, coordinates=f"isolate.tab_pdf.{run_dir.name}"
            )
        except Exception:
            tab_pdf_url = None
    _mixer_and_downloads_fragment(
        stem_paths,
        base_name=base_name,
        run_dir=run_dir,
        media_urls=mixer_media_urls,
        download_urls=mixer_download_urls,
        tab_pdf_url=tab_pdf_url,
    )

    with st.expander("Song studio", expanded=False):
        from ui.studio_panel import render_song_studio

        render_song_studio(run_dir)

    if pro and bass_bleed.get("flagged"):
        st.warning(
            "Guitar check: "
            f"{bass_bleed.get('reason', 'this track may contain extra bass bleed')} "
            "Use **Guitar fix-up** below to adjust without re-separating."
        )

    if pro:
        _render_guitar_fixup_panel(stem_paths, run_dir, artifacts_map or {}, bass_bleed)
        _render_reseparate_panel(stem_paths, run_dir)
    else:
        # Lite: single expander (the inner one from _render_guitar_fixup_panel)
        # Auto-opens when bass bleed is flagged. No re-separate panel.
        _render_guitar_fixup_panel(stem_paths, run_dir, artifacts_map or {}, bass_bleed)

    carry_path, carry_name, carry_is_stem = resolve_tab_pdf_carry_from_isolate(
        st.session_state
    )
    if carry_path:
        st.divider()
        st.caption("Other tools")
        if st.button("Make a tab PDF from this →"):
            st.session_state["carry_over_audio_path"] = carry_path
            st.session_state["carry_over_audio_name"] = carry_name
            st.session_state["carry_over_is_guitar_stem"] = carry_is_stem
            _request_loading_overlay()
            st.switch_page(str(Path(__file__).with_name("tab_pdf.py")))


def _render_file_ready_banner() -> None:
    pending_fp = pending_upload_fp_for_stale(st.session_state)
    if not should_hide_stale_results(
        pending_upload_fp=pending_fp,
        has_artifacts=bool(st.session_state.get("isolate_artifacts")),
        results_source_fp=st.session_state.get("isolate_results_source_fp"),
    ):
        return
    if is_pro_mode(st.session_state):
        st.info(
            f"**{_staged_source_name()}** is ready. Click **Separate tracks** on Home to "
            "replace the current results. Existing stems stay on disk until the new job finishes."
        )
    else:
        st.info(
            f"**{_staged_source_name()}** is ready. Click **Separate tracks** on Home to start."
        )


def main() -> None:
    # Title scrolls away. Home|mix|+ with Queue/Refresh stay pinned.
    # Status / flash / banners paint below so they never split the sticky surface.
    st.title("Audio Isolation", anchor=False, help=PAGE_TITLE_HELP)

    if not shutil.which("ffmpeg"):
        st.error(
            "ffmpeg is required for audio conversion but was not found on PATH. "
            f"{_ffmpeg_install_hint()}"
        )
        return

    demucs_ok = is_demucs_available()
    if not demucs_ok:
        st.error(
            "Audio separation requires Demucs, which is not installed. "
            f"{DEMUCS_INSTALL_HINT}"
        )
        return

    logger.debug("isolate paint demucs=%s", demucs_ok)
    _drain_pending_queue_delete()
    ensure_worker_started()
    browser_id = _get_browser_user_id()
    _restore_isolate_ui_state()
    _rehydrate_artifacts_from_disk(browser_id)

    _ensure_workspace_tab(
        has_artifacts=bool(st.session_state.get("isolate_artifacts"))
    )
    # Claim an OS open before selecting a draft, so an empty New tab is not
    # created and then joined by the file the user actually opened.
    _consume_opened_audio()
    ensure_upload_tab_focused(st.session_state)
    with st.container(key="isolate_sticky_chrome"):
        # The global overlay is nav-only (app.py). Job state belongs to the status
        # strip below, which stays non-blocking because the app remains usable.
        # Always mount this 0-height iframe so its slot never appears/disappears.
        _scroll_main_to_top(isolate_scroll_top_token(st.session_state))

        owner = browser_id if isinstance(browser_id, str) else None
        # Tabs flex; Queue/Refresh keep readable min widths via app.py CSS.
        tabs_col, queue_col, refresh_col = st.columns(
            [6, 1.35, 1.2], vertical_alignment="center"
        )
        with tabs_col:
            _mix_tab_strip_fragment(owner)
        with queue_col:
            _queue_header_fragment()
        with refresh_col:
            refresh_clicked = st.button(
                "Refresh",
                key="isolate_refresh",
                help="Reload this mix and the library from disk",
                width="stretch",
            )
        if refresh_clicked:
            _refresh_isolate_from_disk(browser_id)
        _render_busy_tab_close_prompt(owner)

    _poll_running_jobs()
    if flash := st.session_state.pop("isolate_flash", None):
        st.success(flash)
    else:
        _render_file_ready_banner()
    if notice := st.session_state.pop(MIX_TAB_NOTICE_KEY, None):
        st.info(str(notice))

    shell = apply_shell_view(st.session_state)

    if shell == "mix":
        _render_mixer_workspace(owner)
    else:
        try:
            _render_new_workspace(demucs_ok)
        except Exception as exc:
            logger.exception("Home view failed to draw")
            st.error(
                "Could not draw the home form. Try clicking Refresh, or check that the file is valid audio."
            )
            with st.expander("Technical details"):
                st.exception(exc)

    st.session_state["_isolate_form_drawn"] = True
    _persist_isolate_ui_state()
    # Separate tracks: stay on Home; keep the draft tab when one started the job.
    if st.session_state.pop("_isolate_pending_queue", False):
        keep_draft = str(st.session_state.pop("_isolate_keep_draft_tab", "") or "")
        if keep_draft and is_new_draft_tab(keep_draft):
            st.session_state[SHELL_TAB_KEY] = keep_draft
            st.session_state[SHELL_VIEW_NEXT_KEY] = "home"
            st.session_state[WORKSPACE_NEXT_KEY] = "New"
        else:
            open_home_shell(st.session_state)
        _rerun_scroll_top()


if __name__ == "__main__":
    main()

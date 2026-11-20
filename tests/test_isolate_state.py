"""Tests for ui.isolate_state helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from ui.isolate_state import (
    CUSTOM_STEM_CHOICES,
    CUSTOM_STEM_TILE_ORDER,
    DEFAULT_OUTCOME_CARD,
    DEFAULT_UI_MODE,
    FAILED_JOB_STRIP_TTL_SEC,
    MIXER_COMPONENT_KEY_PREFIX,
    OUTCOME_CARDS,
    OUTCOME_CARD_ORDER,
    dismiss_failed_job,
    outcome_card_for_options,
    custom_stems_from_options,
    outcome_icon_markdown,
    stem_icon_markdown,
    toggle_custom_stem_options,
    resolve_outcome_card,
    DEFAULT_LITE_GUITAR_ENGINE,
    LITE_GUITAR_ENGINE_LABELS,
    LITE_GUITAR_ENGINE_ORDER,
    apply_lite_guitar_engine,
    effective_clip_seconds,
    lite_guitar_engine_allowed,
    lite_guitar_engine_disabled_reason,
    lite_roformer_enqueue_block_reason,
    resolve_lite_guitar_engine,
    should_show_failed_job,
    UI_MODE_KEY,
    UI_MODES,
    apply_persisted_settings,
    is_pro_mode,
    load_ui_mode,
    persisted_settings_payload,
    resolve_ui_mode,
    roformer_speed_note,
    write_ui_mode,
    DEFAULT_SEPARATION_PRESET,
    DEFAULT_SPEED_PRESET,
    DEFAULT_TRACK_OPTIONS,
    default_isolate_device,
    CUSTOM_STEM_CHOICES,
    ISOLATE_OUTPUT_NAME_KEY,
    ISOLATE_OUTPUT_NAME_PENDING_KEY,
    ISOLATE_AUTO_OUTPUT_NAME_KEY,
    ISOLATE_NAMED_YOUTUBE_URL_KEY,
    ISOLATE_YOUTUBE_URL_KEY,
    ISOLATE_YOUTUBE_URL_PENDING_KEY,
    ISOLATE_YOUTUBE_TITLE_PENDING_KEY,
    SPEED_PRESETS,
    apply_listen_picker_pending,
    apply_pending_output_name,
    apply_pending_youtube_url,
    apply_stored_isolate_ui_state,
    apply_workspace_tab,
    apply_youtube_output_name_sync,
    checklist_items,
    LITE_MAX_DURATION_SEC,
    clamp_lite_clip,
    clamp_region_bounds,
    custom_selected_stems,
    default_guitar_track_option,
    default_region_end,
    estimate_job_seconds,
    estimate_remaining_seconds,
    estimated_stage_seconds,
    expects_guitar_stem,
    format_checklist_markdown,
    format_elapsed,
    format_eta_line,
    format_progress_label,
    format_source_caption,
    format_source_title,
    guitar_track_radio_ids,
    infer_source_kind,
    job_requires_roformer_backend,
    intra_stage_fraction,
    isolation_stages_for_job,
    isolate_poll_requires_full_rerun,
    isolate_ui_state_payload,
    listen_picker_default,
    load_persist_isolate_user_id,
    mixer_component_key,
    partition_queue_jobs,
    pending_upload_fp_for_stale,
    pick_library_row,
    plan_isolate_job_poll,
    promote_default_guitar_option,
    queue_clear_youtube_url,
    queue_output_name_if_empty,
    queue_reopen_output_name,
    queue_youtube_url,
    read_isolate_ui_state,
    recent_runs_with_owner_fallback,
    request_isolate_scroll_top,
    consume_isolate_scroll_top,
    consume_mix_tab_event,
    isolate_scroll_top_token,
    should_request_isolate_scroll_top,
    ISOLATE_SCROLL_TOP_KEY,
    reset_new_tab_source,
    resolve_tab_pdf_carry_from_isolate,
    adopt_audio_into_run,
    discard_youtube_staging,
    is_youtube_staging_path,
    resolve_custom_separation,
    resolve_isolate_user_id,
    resolve_separation_preset,
    resolve_speed_preset,
    resolve_track_selection,
    resolve_youtube_job_name,
    running_progress_view,
    normalize_guitar_track_selection,
    tracks_picker_help,
    seed_consumed_job_ids,
    seed_notified_job_ids,
    os_notify_message,
    jobs_needing_os_notify,
    select_rehydrate_row,
    refresh_open_mix_action,
    session_mixer_artifacts_ok,
    should_auto_apply_job,
    staged_audio_for_new_tab,
    should_hide_stale_results,
    stage_progress_percent,
    user_progress_hint,
    youtube_label_from_url,
    youtube_video_id,
    mixer_local_video_path,
    mixer_youtube_video_id,
    prefer_local_mixer_picture,
    sync_output_name_on_upload,
    sync_output_name_on_youtube,
    upload_fingerprint,
    write_isolate_ui_state,
    WORKSPACE_KEY,
    WORKSPACE_NEXT_KEY,
    LISTEN_PICKER_KEY,
    LISTEN_PICKER_NEXT_KEY,
    children_from_outputs,
    merge_reseparate,
    stem_label_for_id,
)


@pytest.fixture(autouse=True)
def _cpu_edition_by_default(monkeypatch):
    monkeypatch.delenv("AUDIO_TOOLS_EDITION", raising=False)


def test_failed_job_leaves_the_strip_once_dismissed():
    job = {"id": "job-1", "status": "failed", "updated_at": 1_000.0}
    assert should_show_failed_job(job, now=1_010.0) is True
    session: dict = {}
    dismiss_failed_job(session, "job-1")
    assert (
        should_show_failed_job(
            job, now=1_010.0, dismissed_ids=session["isolate_dismissed_job_ids"]
        )
        is False
    )


def test_failed_job_stops_owning_the_strip_after_its_ttl():
    job = {"id": "job-1", "status": "failed", "updated_at": 1_000.0}
    assert should_show_failed_job(job, now=1_000.0 + FAILED_JOB_STRIP_TTL_SEC - 1) is True
    assert should_show_failed_job(job, now=1_000.0 + FAILED_JOB_STRIP_TTL_SEC + 1) is False


def test_failed_job_without_a_timestamp_still_shows():
    """Missing metadata must not hide a real failure."""
    assert should_show_failed_job({"id": "j", "status": "failed"}, now=5.0) is True
    assert should_show_failed_job({"id": "j", "updated_at": "junk"}, now=5.0) is True
    assert should_show_failed_job(None, now=5.0) is False
    assert should_show_failed_job({"status": "failed"}, now=5.0) is False


def test_dismissing_twice_does_not_duplicate_the_id():
    session: dict = {}
    dismiss_failed_job(session, "job-1")
    dismiss_failed_job(session, "job-1")
    assert session["isolate_dismissed_job_ids"] == ["job-1"]


def test_lite_outcomes_use_the_best_engine_available():
    """Lite is presentation only — it must not quietly pick a weaker model."""
    with_ro = resolve_outcome_card("guitar", roformer_available=True)
    without_ro = resolve_outcome_card("guitar", roformer_available=False)
    assert with_ro == ["guitar_roformer"]
    assert without_ro == ["guitar_demucs_6s"]

    band = resolve_outcome_card("band", roformer_available=True)
    assert "guitar_roformer" in band
    assert {"vocals_demucs", "drums_demucs", "bass_demucs"} <= set(band)


def test_lite_outcomes_all_resolve_to_a_runnable_pipeline():
    for card in OUTCOME_CARD_ORDER:
        for roformer in (True, False):
            options = resolve_outcome_card(card, roformer_available=roformer)
            assert options, card
            resolved = resolve_track_selection(options)
            assert resolved["model"]
            assert resolved["emit_stems"]


def test_unknown_outcome_falls_back_to_the_default():
    assert resolve_outcome_card("nonsense", roformer_available=True) == resolve_outcome_card(
        DEFAULT_OUTCOME_CARD, roformer_available=True
    )


def test_outcome_card_round_trips_so_switching_modes_keeps_the_selection():
    for card in OUTCOME_CARD_ORDER:
        options = resolve_outcome_card(card, roformer_available=True)
        assert outcome_card_for_options(options, roformer_available=True) == card


def test_a_pro_only_selection_is_reported_as_custom_not_silently_replaced():
    # Piano has no Lite outcome; Lite must recognise it cannot describe this.
    custom = ["vocals_demucs", "piano_demucs"]
    assert outcome_card_for_options(custom, roformer_available=True) is None
    assert outcome_card_for_options([], roformer_available=True) is None


def test_outcome_cards_carry_stem_list_and_track_count():
    expected = {
        "band": ("Vocals · Drums · Bass · Guitar", 4),
        "karaoke": ("Vocals · Instrumental", 2),
        "guitar": ("Guitar", 1),
        "vocals": ("Vocals", 1),
    }
    assert set(OUTCOME_CARDS) == set(OUTCOME_CARD_ORDER)
    for card_id, (stems, n_tracks) in expected.items():
        assert OUTCOME_CARDS[card_id]["stems_line"] == stems
        assert OUTCOME_CARDS[card_id]["n_tracks"] == n_tracks
        assert OUTCOME_CARDS[card_id]["label"]
        assert OUTCOME_CARDS[card_id]["help"]


def test_toggle_custom_stem_exits_karaoke_and_uses_best_guitar():
    karaoke = resolve_outcome_card("karaoke", roformer_available=True)
    added = toggle_custom_stem_options(karaoke, "drums", roformer_available=True)
    assert added == ["drums_demucs"]
    assert custom_stems_from_options(karaoke) == []

    band = resolve_outcome_card("band", roformer_available=True)
    with_piano = toggle_custom_stem_options(band, "piano", roformer_available=True)
    assert "piano_demucs" in with_piano
    assert outcome_card_for_options(with_piano, roformer_available=True) is None

    no_guitar = toggle_custom_stem_options(band, "guitar", roformer_available=True)
    assert custom_stems_from_options(no_guitar) == ["vocals", "bass", "drums"]
    again = toggle_custom_stem_options([], "guitar", roformer_available=True)
    assert again == ["guitar_roformer"]
    demucs = toggle_custom_stem_options([], "guitar", roformer_available=False)
    assert demucs == ["guitar_demucs_6s"]


def test_effective_clip_seconds_prefers_section_then_file():
    assert effective_clip_seconds(
        file_duration_sec=280.0, start_sec=10.0, max_duration_sec=90.0
    ) == 90.0
    assert effective_clip_seconds(
        file_duration_sec=280.0, start_sec=10.0, max_duration_sec=None
    ) == 270.0
    assert (
        effective_clip_seconds(
            file_duration_sec=None, start_sec=0.0, max_duration_sec=None
        )
        is None
    )


def _allow_kwargs(**overrides):
    base = dict(
        effective_sec=120.0,
        cap_sec=180.0,
        duration_known=True,
        roformer_available=True,
    )
    base.update(overrides)
    return base


def test_lite_guitar_engine_demucs_always_allowed():
    for kwargs in (
        _allow_kwargs(),
        _allow_kwargs(effective_sec=280.0, cap_sec=180.0),
        _allow_kwargs(effective_sec=None, duration_known=False),
        _allow_kwargs(roformer_available=False),
    ):
        assert lite_guitar_engine_allowed(DEFAULT_LITE_GUITAR_ENGINE, **kwargs) is True
        assert lite_guitar_engine_allowed("guitar_demucs_6s", **kwargs) is True
        assert lite_guitar_engine_disabled_reason("guitar_demucs_6s", **kwargs) is None


def test_lite_guitar_engine_roformer_disabled_until_length_known_or_over_cap():
    unknown = _allow_kwargs(effective_sec=None, duration_known=False)
    assert lite_guitar_engine_allowed("guitar_roformer", **unknown) is False
    assert lite_guitar_engine_allowed("guitar_roformer_refine", **unknown) is False
    reason = lite_guitar_engine_disabled_reason("guitar_roformer", **unknown)
    assert reason is not None
    assert "length is known" in reason

    over = _allow_kwargs(effective_sec=277.0, cap_sec=180.0)
    assert lite_guitar_engine_allowed("guitar_roformer", **over) is False
    assert lite_guitar_engine_allowed("guitar_roformer_refine", **over) is False
    over_reason = lite_guitar_engine_disabled_reason("guitar_roformer", **over)
    assert over_reason is not None
    assert "180" in over_reason
    assert "277" in over_reason

    short = _allow_kwargs(effective_sec=90.0, cap_sec=180.0)
    assert lite_guitar_engine_allowed("guitar_roformer", **short) is True
    assert lite_guitar_engine_allowed("guitar_roformer_refine", **short) is True
    assert lite_guitar_engine_disabled_reason("guitar_roformer", **short) is None

    at_cap = _allow_kwargs(effective_sec=180.0, cap_sec=180.0)
    assert lite_guitar_engine_allowed("guitar_roformer", **at_cap) is True

    missing = _allow_kwargs(roformer_available=False, effective_sec=90.0)
    assert lite_guitar_engine_allowed("guitar_roformer", **missing) is False
    missing_reason = lite_guitar_engine_disabled_reason("guitar_roformer", **missing)
    assert missing_reason is not None
    assert "BS-RoFormer" in missing_reason


def test_lite_guitar_engine_falls_back_to_demucs_when_illegal():
    over = _allow_kwargs(effective_sec=282.0, cap_sec=180.0)
    engine, fell_back = resolve_lite_guitar_engine("guitar_roformer", **over)
    assert engine == DEFAULT_LITE_GUITAR_ENGINE
    assert fell_back is True

    refine, refine_fell = resolve_lite_guitar_engine("guitar_roformer_refine", **over)
    assert refine == DEFAULT_LITE_GUITAR_ENGINE
    assert refine_fell is True

    unknown = _allow_kwargs(effective_sec=None, duration_known=False)
    engine, fell_back = resolve_lite_guitar_engine("guitar_roformer", **unknown)
    assert engine == DEFAULT_LITE_GUITAR_ENGINE
    assert fell_back is True

    short = _allow_kwargs(effective_sec=90.0, cap_sec=180.0)
    engine, fell_back = resolve_lite_guitar_engine("guitar_roformer_refine", **short)
    assert engine == "guitar_roformer_refine"
    assert fell_back is False

    engine, fell_back = resolve_lite_guitar_engine("not-an-engine", **short)
    assert engine == DEFAULT_LITE_GUITAR_ENGINE
    assert fell_back is False

    engine, fell_back = resolve_lite_guitar_engine(None, **over)
    assert engine == DEFAULT_LITE_GUITAR_ENGINE
    assert fell_back is False


def test_apply_lite_guitar_engine_swaps_guitar_and_leaves_karaoke():
    band = ["vocals_demucs", "drums_demucs", "bass_demucs", "guitar_demucs_6s"]
    assert apply_lite_guitar_engine(band, "guitar_roformer") == [
        "vocals_demucs",
        "drums_demucs",
        "bass_demucs",
        "guitar_roformer",
    ]
    assert apply_lite_guitar_engine(["vocals_instrumental_demucs"], "guitar_roformer") == [
        "vocals_instrumental_demucs"
    ]
    assert apply_lite_guitar_engine(["guitar_roformer"], "guitar_demucs_6s") == [
        "guitar_demucs_6s"
    ]


def test_lite_outcomes_honor_explicit_guitar_engine():
    assert LITE_GUITAR_ENGINE_ORDER == (
        "guitar_demucs_6s",
        "guitar_roformer",
        "guitar_roformer_refine",
    )
    assert LITE_GUITAR_ENGINE_LABELS["guitar_demucs_6s"] == "Default (Demucs)"
    assert LITE_GUITAR_ENGINE_LABELS["guitar_roformer"] == "Better guitar (BS-RoFormer)"
    assert (
        LITE_GUITAR_ENGINE_LABELS["guitar_roformer_refine"]
        == "Best / slower (BS-RoFormer + MelBand)"
    )
    guitar = resolve_outcome_card(
        "guitar",
        roformer_available=True,
        guitar_option="guitar_demucs_6s",
    )
    assert guitar == ["guitar_demucs_6s"]
    band = resolve_outcome_card(
        "band",
        roformer_available=True,
        guitar_option="guitar_roformer_refine",
    )
    assert "guitar_roformer_refine" in band
    assert outcome_card_for_options(
        band,
        roformer_available=True,
        guitar_option="guitar_roformer_refine",
    ) == "band"
    added = toggle_custom_stem_options(
        [],
        "guitar",
        roformer_available=True,
        guitar_option="guitar_roformer_refine",
    )
    assert added == ["guitar_roformer_refine"]


def test_lite_roformer_enqueue_is_blocked_when_over_cap_or_unknown():
    over = _allow_kwargs(effective_sec=277.0, cap_sec=180.0)
    reason = lite_roformer_enqueue_block_reason("bs_roformer_sw", **over)
    assert reason is not None
    assert "180" in reason
    assert "277" in reason
    assert lite_roformer_enqueue_block_reason("htdemucs_6s", **over) is None
    assert lite_roformer_enqueue_block_reason("htdemucs", **over) is None

    short = _allow_kwargs(effective_sec=90.0, cap_sec=180.0)
    assert lite_roformer_enqueue_block_reason("bs_roformer_sw", **short) is None
    assert lite_roformer_enqueue_block_reason("melband_roformer_guitar", **short) is None

    unknown = _allow_kwargs(effective_sec=None, duration_known=False)
    unknown_reason = lite_roformer_enqueue_block_reason("bs_roformer_sw", **unknown)
    assert unknown_reason is not None
    assert "length is known" in unknown_reason


def test_custom_stem_tile_order_is_this_apps_stems_not_moises_extras():
    assert set(CUSTOM_STEM_TILE_ORDER) == set(CUSTOM_STEM_CHOICES)
    assert CUSTOM_STEM_TILE_ORDER == (
        "vocals",
        "guitar",
        "bass",
        "drums",
        "piano",
        "other",
    )
    for banned in ("keys", "wind", "strings", "multimedia", "dialogue"):
        assert banned not in CUSTOM_STEM_CHOICES


def test_lite_outcome_picker_uses_card_keys_not_pro_track_picker():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    tile = source[
        source.find("def _render_outcome_tile") : source.find("def _render_stem_tile")
    ]
    picker = source[
        source.find("def _render_outcome_picker") : source.find("def _render_engine_panel")
    ]
    grid = source[
        source.find("def _render_custom_stem_grid") : source.find("def _render_outcome_picker")
    ]
    track = source[
        source.find("def _render_track_picker") : source.find("def _outcome_tile_label")
    ]
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    assert "OUTCOME_CARD_KEY" in picker
    assert 'key=f"isolate_outcome_pick_{card_id}"' in tile
    for card_id in ("band", "karaoke"):
        assert f'"{card_id}"' in picker
    assert "**One track**" not in picker
    assert '"guitar"' not in picker.split("_render_custom_stem_grid")[0]
    assert '"vocals"' not in picker.split("_render_custom_stem_grid")[0]
    assert "CUSTOM_STEM_TILE_ORDER" in grid
    assert 'key=f"isolate_stem_pick_{stem_id}"' in source
    assert "icon=STEM_TILE_EMOJI" not in source
    assert "icon=OUTCOME_TILE_EMOJI" not in source
    assert "icon=" not in tile
    assert "stem_icon_markdown(stem_id, selected=selected, light=_tiles_use_light_icons())" in source
    assert "outcome_icon_markdown(card_id, selected=selected, light=_tiles_use_light_icons())" in source
    label_fn = source[
        source.find("def _outcome_tile_label") : source.find("def _render_outcome_tile")
    ]
    assert r"\u00a0track" in label_fn
    assert "stems_line']} · {tracks}" in label_fn
    assert "flex-direction: column" in (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "_render_custom_stem_grid" in picker
    assert "_render_track_picker(persist=persist)" in picker
    assert "st.checkbox" not in track
    assert 'key="isolate_guitar_track"' in track
    assert "resolve_outcome_card(" in tile
    branch = controls[controls.find("pro = is_pro_mode") : controls.find("custom_stems =")]
    assert "_render_section_and_outcomes(" in branch
    assert "prefer_roformer=prefer_roformer" in branch
    assert "get_desktop_probe()" in controls
    assert "lite_auto_choice(probe)" in controls
    assert "lite_auto_choice(probe)" in picker
    assert "isolate_lite_run_on_manual" in picker
    assert "lite_detected_caption(probe)" in picker
    assert "lite_using_caption(" in picker
    assert 'key="isolate_lite_run_on"' in picker
    assert 'st.session_state["isolate_lite_run_on"]' in picker
    # Form build must not rewrite the radio key after the widget exists.
    assert 'st.session_state["isolate_lite_run_on"]' not in controls
    assert "lite_device_choice_ids(probe)" in picker
    assert "caption_reason" in picker
    assert "GPU is the default" in picker
    assert "_render_machine_panel(" in picker
    assert "_render_machine_panel(" not in controls
    assert controls.count("_render_machine_panel(") == 0
    assert source.count("_render_machine_panel(") == 2  # def + one Pro call site
    assert "get_desktop_probe_without_torch" not in controls
    assert "_render_track_picker(persist=persist)" not in branch
    for banned in ("keys", "wind", "strings", "multimedia"):
        assert f"isolate_stem_pick_{banned}" not in source


def test_lite_enqueue_reprobes_memory_unless_manual():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    enqueue = source[
        source.find("def _enqueue_confirmed_job") : source.find("def _enqueue_reseparate_job")
        if "def _enqueue_reseparate_job" in source
        else source.find("def _render_new_workspace")
    ]
    assert "isolate_lite_run_on_manual" in enqueue
    assert "lite_auto_choice(" in enqueue
    assert 'st.session_state["isolate_device"]' in enqueue
    assert 'st.session_state["isolate_quality"]' in enqueue
    # Radio key must not be written after the widget is instantiated on this run.
    assert 'st.session_state["isolate_lite_run_on"]' not in enqueue


def test_pro_machine_panel_is_inside_outcome_fragment_only():
    """Device expander must not sit after the fragment — that duplicated it."""
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    picker = source[
        source.find("def _render_outcome_picker") : source.find("def _render_engine_panel")
    ]
    machine = source[
        source.find("def _render_machine_panel") : source.find("def _speed_preset_radio_label")
    ]
    assert "_render_machine_panel(" in picker
    assert "_render_engine_panel(" in picker
    assert picker.find("_render_engine_panel(") < picker.find("_render_machine_panel(")
    assert "_isolate_machine_panel_drawn" in machine
    assert 'key="isolate_machine_expanded"' in machine


def test_persisted_settings_include_lite_run_on():
    from ui.isolate_state import PERSISTED_SETTING_KEYS

    assert "isolate_lite_run_on" in PERSISTED_SETTING_KEYS
    assert "isolate_lite_guitar_engine" in PERSISTED_SETTING_KEYS
    assert "isolate_device" in PERSISTED_SETTING_KEYS
    assert "isolate_demucs_shifts" in PERSISTED_SETTING_KEYS
    assert "isolate_fold_other_mode" in PERSISTED_SETTING_KEYS
    assert "isolate_guitar_ensemble" in PERSISTED_SETTING_KEYS
    assert "isolate_bass_bleed_mitigation" in PERSISTED_SETTING_KEYS
    assert "isolate_htdemucs_ft" in PERSISTED_SETTING_KEYS
    assert "isolate_metro_accent" in PERSISTED_SETTING_KEYS
    assert "isolate_metro_rate" in PERSISTED_SETTING_KEYS
    assert "isolate_metro_sound" in PERSISTED_SETTING_KEYS
    assert "isolate_metro_follow" in PERSISTED_SETTING_KEYS


def _lite_engine_kwargs(**overrides):
    base = dict(
        is_pro=False,
        model="htdemucs_6s",
        quality="fast",
        two_stems=None,
        two_pass=False,
        fold_other_into_guitar=True,
        fold_other_mode="best_effort",
        roformer_available=True,
    )
    base.update(overrides)
    return base


def test_lite_engine_job_fields_ignore_pro_session_and_keep_defaults():
    from ui.isolate_state import resolve_pro_engine_job_fields

    session = {
        "isolate_demucs_shifts": "5",
        "isolate_demucs_overlap": "0.75",
        "isolate_demucs_segment": 0,
        "isolate_demucs_jobs": 4,
        "isolate_sub_bass_debleed": True,
        "isolate_bleed_gate": True,
        "isolate_bass_bleed_mitigation": True,
        "isolate_adaptive_fold_gain": True,
        "isolate_guitar_ensemble": True,
        "isolate_low_end_restore_db": 6.0,
        "isolate_fold_other_mode": "full",
        "isolate_htdemucs_ft": True,
    }
    fields = resolve_pro_engine_job_fields(session, **_lite_engine_kwargs())
    assert fields["demucs_shifts"] is None
    assert fields["demucs_overlap"] is None
    assert fields["demucs_segment"] == 8
    assert fields["demucs_jobs"] == 1
    assert fields["fold_other_mode"] == "best_effort"
    assert fields["adaptive_fold_gain"] is False
    assert fields["bleed_gate"] is False
    assert fields["bass_bleed_mitigation"] is False
    assert fields["sub_bass_debleed"] is False
    assert fields["low_end_restore_db"] == 0.0
    assert fields["guitar_ensemble"] is False
    assert fields["model"] == "htdemucs_6s"


def test_pro_engine_job_fields_map_compute_and_guitar_post():
    from ui.isolate_state import resolve_pro_engine_job_fields

    session = {
        "isolate_demucs_shifts": "5",
        "isolate_demucs_overlap": "0.75",
        "isolate_demucs_segment": 0,
        "isolate_demucs_jobs": 2,
        "isolate_fold_other_mode": "band_limited",
        "isolate_adaptive_fold_gain": True,
        "isolate_bleed_gate": True,
        "isolate_bass_bleed_mitigation": True,
        "isolate_sub_bass_debleed": True,
        "isolate_low_end_restore_db": 4.0,
        "isolate_guitar_ensemble": True,
    }
    fields = resolve_pro_engine_job_fields(
        session, **_lite_engine_kwargs(is_pro=True, quality="fast")
    )
    assert fields["demucs_shifts"] == 5
    assert fields["demucs_overlap"] == pytest.approx(0.75)
    assert fields["demucs_segment"] is None
    assert fields["demucs_jobs"] == 2
    assert fields["fold_other_mode"] == "band_limited"
    assert fields["adaptive_fold_gain"] is True
    assert fields["bleed_gate"] is True
    assert fields["bass_bleed_mitigation"] is True
    assert fields["sub_bass_debleed"] is True
    assert fields["low_end_restore_db"] == 4.0
    assert fields["guitar_ensemble"] is True


def test_pro_engine_matching_quality_map_does_not_send_shift_overrides():
    from ui.isolate_state import resolve_pro_engine_job_fields

    session = {"isolate_demucs_shifts": "0", "isolate_demucs_overlap": "0.25"}
    fields = resolve_pro_engine_job_fields(
        session, **_lite_engine_kwargs(is_pro=True, quality="fast")
    )
    assert fields["demucs_shifts"] is None
    assert fields["demucs_overlap"] is None


def test_sync_demucs_compute_defaults_resets_unless_overridden():
    from ui.isolate_state import sync_demucs_compute_defaults

    session: dict = {"isolate_demucs_shifts": "0", "isolate_demucs_overlap": "0.25"}
    sync_demucs_compute_defaults(session, quality="fast")
    assert session["isolate_demucs_shifts"] == "0"
    sync_demucs_compute_defaults(session, quality="extreme")
    assert session["isolate_demucs_shifts"] == "5"
    assert session["isolate_demucs_overlap"] == "0.75"

    session["isolate_demucs_shifts"] = "1"
    sync_demucs_compute_defaults(session, quality="fast")
    assert session["isolate_demucs_shifts"] == "1"
    assert session["isolate_demucs_overlap"] == "0.25"


def test_pro_htdemucs_ft_only_for_four_stem_and_ensemble_guards():
    from ui.isolate_state import resolve_pro_engine_job_fields

    session = {"isolate_htdemucs_ft": True, "isolate_guitar_ensemble": True}
    four = resolve_pro_engine_job_fields(
        session, **_lite_engine_kwargs(is_pro=True, model="htdemucs")
    )
    assert four["model"] == "htdemucs_ft"
    assert four["guitar_ensemble"] is True

    six = resolve_pro_engine_job_fields(
        session, **_lite_engine_kwargs(is_pro=True, model="htdemucs_6s")
    )
    assert six["model"] == "htdemucs_6s"

    blocked = resolve_pro_engine_job_fields(
        session,
        **_lite_engine_kwargs(is_pro=True, model="htdemucs_6s", two_pass=True),
    )
    assert blocked["guitar_ensemble"] is False

    no_ro = resolve_pro_engine_job_fields(
        session,
        **_lite_engine_kwargs(is_pro=True, model="htdemucs_6s", roformer_available=False),
    )
    assert no_ro["guitar_ensemble"] is False


def test_fold_mode_ignored_when_custom_keeps_other():
    from ui.isolate_state import resolve_pro_engine_job_fields

    session = {"isolate_fold_other_mode": "full", "isolate_adaptive_fold_gain": True}
    fields = resolve_pro_engine_job_fields(
        session,
        **_lite_engine_kwargs(
            is_pro=True, fold_other_into_guitar=False, fold_other_mode="best_effort"
        ),
    )
    assert fields["fold_other_mode"] == "best_effort"
    assert fields["adaptive_fold_gain"] is False


def test_isolate_job_spec_carries_pro_engine_fields_into_config_kwargs():
    from ui.isolate_jobs import IsolateJobSpec, isolate_config_kwargs_from_spec

    default = IsolateJobSpec(
        id="j", audio_path="a.wav", output_dir="out", title="t"
    )
    kwargs = isolate_config_kwargs_from_spec(default)
    assert kwargs["demucs_shifts"] is None
    assert kwargs["demucs_overlap"] is None
    assert kwargs["demucs_segment"] == 8
    assert kwargs["demucs_jobs"] == 1
    assert kwargs["bleed_gate"] is False
    assert kwargs["bass_bleed_mitigation"] is False
    assert kwargs["guitar_ensemble"] is False
    assert kwargs["lead_rhythm"] is False
    assert kwargs["metronome_accent"] is True
    assert kwargs["metronome_rate"] == 1.0
    assert kwargs["metronome_sound"] == "classic"

    spec = IsolateJobSpec(
        id="j2",
        audio_path="a.wav",
        output_dir="out",
        title="t",
        demucs_shifts=5,
        demucs_overlap=0.75,
        demucs_segment=None,
        demucs_jobs=2,
        bleed_gate=True,
        bass_bleed_mitigation=True,
        adaptive_fold_gain=True,
        guitar_ensemble=True,
        fold_other_mode="band_limited",
        low_end_restore_db=3.0,
        sub_bass_debleed=True,
        metronome_accent=False,
        metronome_rate=0.5,
        metronome_sound="wood",
    )
    restored = IsolateJobSpec.from_dict(spec.to_dict())
    mapped = isolate_config_kwargs_from_spec(restored)
    assert mapped["demucs_shifts"] == 5
    assert mapped["demucs_overlap"] == pytest.approx(0.75)
    assert mapped["demucs_segment"] is None
    assert mapped["demucs_jobs"] == 2
    assert mapped["bleed_gate"] is True
    assert mapped["bass_bleed_mitigation"] is True
    assert mapped["adaptive_fold_gain"] is True
    assert mapped["guitar_ensemble"] is True
    assert mapped["fold_other_mode"] == "band_limited"
    assert mapped["low_end_restore_db"] == pytest.approx(3.0)
    assert mapped["sub_bass_debleed"] is True
    assert mapped["metronome_accent"] is False
    assert mapped["metronome_rate"] == pytest.approx(0.5)
    assert mapped["metronome_sound"] == "wood"


def test_isolate_headings_disable_anchors_and_tiles_use_fragment_rerun():
    """Permalink chain-links scroll the page; help= + fragment-scope replace that."""
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    app_css = (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'anchor=False, help=PAGE_TITLE_HELP' in source
    assert "SEPARATE_TRACKS_HELP" in source
    assert 'st.subheader("Downloads", anchor=False, help=DOWNLOADS_HELP)' in source
    assert "SECTION_OPTIONAL_HELP" in source
    assert "@st.fragment" in source
    assert 'st.rerun(scope="fragment")' in source
    assert "def _rerun_after_tile_pick" in source
    assert "stHeaderActionElements" in app_css
    assert 'markdown("### Separate tracks")' not in source
    picker = source[
        source.find("def _render_outcome_picker") : source.find("def _render_engine_panel")
    ]
    assert "_render_engine_panel(" in picker
    assert 'key="isolate_speed_preset"' in picker
    section_head = source[
        max(0, source.find("def _render_section_and_outcomes") - 40) : source.find(
            "def _render_section_and_outcomes"
        )
        + 40
    ]
    assert "@st.fragment" in section_head
    rerun_fn = source[
        source.find("def _rerun_preserve_scroll") : source.find("def _render_custom_stem_grid")
    ]
    assert 'st.rerun(scope="fragment")' in rerun_fn
    assert "is_pro_mode" not in rerun_fn
    assert "def _rerun_scroll_top" in rerun_fn
    assert "_rerun_after_tile_pick()" in source[
        source.find("def _render_outcome_tile") : source.find("def _render_custom_stem_grid")
    ]


def test_lite_new_cta_captions_chosen_outcome_and_source():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    new_ws = source[
        source.find("def _render_new_workspace") : source.find("def _render_mixer_region_caption")
    ]
    assert 'key="isolate_separate"' in new_ws
    assert "SETUP_STEP_SEPARATE" in new_ws
    assert new_ws.find("SETUP_STEP_SEPARATE") < new_ws.find('key="isolate_separate"')
    assert "_try_enqueue_choice(choice)" in new_ws
    assert "OUTCOME_CARD_KEY" in new_ws
    assert "_has_source_for_job(choice)" in new_ws
    assert "disabled=not demucs_ok or not has_source" in new_ws
    assert "Custom ·" in new_ws
    css = (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "st-key-isolate_outcome_pick_" in css
    assert "st-key-isolate_stem_pick_" in css
    assert "white-space: pre-line" in css
    assert "flex-direction: column" in css
    assert "display: block !important" in css
    assert "margin-left: auto !important" in css
    assert "overflow-wrap: normal" in css
    assert "2.35rem" in css
    assert "Apple Color Emoji" not in css
    assert "stIconEmoji" not in css
    assert "mask-image" not in css
    for banned in ("keys", "wind", "strings", "multimedia"):
        assert f"st-key-isolate_stem_pick_{banned}" not in css


def test_tile_icons_are_markdown_data_uris_not_emoji():
    import base64

    def svg_body(md: str) -> str:
        uri = md[md.index("](") + 2 : -1]
        return base64.standard_b64decode(uri.split(",", 1)[1]).decode("utf-8")

    for stem_id in CUSTOM_STEM_TILE_ORDER:
        md = stem_icon_markdown(stem_id)
        assert md.startswith("![")
        assert "data:image/svg+xml;base64," in md
        assert not any(ord(ch) > 127 for ch in md)
        assert "#f6ad49" in svg_body(md)
        selected = stem_icon_markdown(stem_id, selected=True)
        assert "#192542" in svg_body(selected)
        assert "#f6ad49" not in svg_body(selected)
    for card_id in OUTCOME_CARD_ORDER:
        md = outcome_icon_markdown(card_id)
        assert md.startswith("![")
        assert "data:image/svg+xml;base64," in md
        assert not any(ord(ch) > 127 for ch in md)
        assert "#192542" in svg_body(outcome_icon_markdown(card_id, selected=True))
    # Unselected icons on the light page are Tyrian blue, same as a selected icon.
    assert "#192542" in svg_body(stem_icon_markdown("vocals", light=True))
    assert "#f6ad49" not in svg_body(stem_icon_markdown("vocals", light=True))


def test_roformer_speed_note_names_the_gpu_this_host_actually_has():
    mac = roformer_speed_note(platform="darwin")
    win = roformer_speed_note(platform="win32")
    linux = roformer_speed_note(platform="linux")
    assert "Apple GPU (MPS)" in mac
    # Windows users were previously told to look for an Apple setting.
    assert "Apple" not in win
    assert "NVIDIA GPU (CUDA)" in win
    assert "Apple" not in linux
    assert "CUDA" in linux
    for note in (mac, win, linux):
        assert "Speed only tunes Demucs" in note


def test_resolve_ui_mode_defaults_to_lite_and_rejects_junk():
    assert resolve_ui_mode("Pro") == "Pro"
    assert resolve_ui_mode("pro") == "Pro"
    assert resolve_ui_mode("Lite") == "Lite"
    assert resolve_ui_mode(None) == "Lite"
    assert resolve_ui_mode("") == "Lite"
    assert resolve_ui_mode("expert") == "Lite"
    assert DEFAULT_UI_MODE == "Lite"


def test_is_pro_mode_reads_the_session():
    assert is_pro_mode({}) is False
    assert is_pro_mode({UI_MODE_KEY: "Pro"}) is True
    assert is_pro_mode({UI_MODE_KEY: "Lite"}) is False


def test_settings_survive_a_restart_round_trip():
    session = {
        "isolate_quality": "extreme",
        "isolate_device": "cpu",
        "isolate_guitar_ft": True,
        "isolate_guitar_track": "guitar_roformer",
        "unrelated_key": "dropped",
    }
    payload = persisted_settings_payload(session)
    assert "unrelated_key" not in payload
    assert payload["isolate_quality"] == "extreme"

    fresh: dict = {}
    apply_persisted_settings(fresh, payload)
    assert fresh["isolate_quality"] == "extreme"
    assert fresh["isolate_guitar_ft"] is True
    assert fresh["isolate_guitar_track"] == "guitar_roformer"


def test_restoring_settings_never_overwrites_a_live_choice():
    live = {"isolate_quality": "fast"}
    apply_persisted_settings(live, {"isolate_quality": "extreme"})
    assert live["isolate_quality"] == "fast"


def test_ui_state_payload_carries_mode_and_settings(tmp_path):
    session = {
        UI_MODE_KEY: "Pro",
        "isolate_quality": "high",
        "isolate_setup_step": "separate",
    }
    payload = isolate_ui_state_payload(session)
    assert payload["mode"] == "Pro"
    assert payload["mode"] in UI_MODES
    assert payload["settings"]["isolate_quality"] == "high"
    assert "isolate_setup_step" not in payload

    restored: dict = {}
    apply_stored_isolate_ui_state(restored, payload)
    assert restored[UI_MODE_KEY] == "Pro"
    assert restored["isolate_quality"] == "high"


def test_mode_survives_relaunch_via_disk(tmp_path):
    path = tmp_path / "isolate_ui_state.json"
    assert load_ui_mode({}, path) == "Lite"
    write_ui_mode(path, "Pro")
    assert load_ui_mode({}, path) == "Pro"


def test_writing_mode_does_not_erase_the_rest_of_the_ui_state(tmp_path):
    path = tmp_path / "isolate_ui_state.json"
    write_isolate_ui_state(path, {"workspace": "Mixer", "run_dir": "/runs/abc"})
    write_ui_mode(path, "Pro")
    stored = read_isolate_ui_state(path)
    assert stored["mode"] == "Pro"
    assert stored["workspace"] == "Mixer"
    assert stored["run_dir"] == "/runs/abc"


def test_current_setup_step_defaults_to_source():
    from ui.isolate_state import (
        SETUP_STEP_KEY,
        SETUP_STEP_SEPARATE,
        SETUP_STEP_SOURCE,
        current_setup_step,
    )

    assert current_setup_step({}) == SETUP_STEP_SOURCE
    assert current_setup_step({SETUP_STEP_KEY: "nope"}) == SETUP_STEP_SOURCE
    assert current_setup_step({SETUP_STEP_KEY: SETUP_STEP_SEPARATE}) == SETUP_STEP_SEPARATE


def test_setup_continue_enabled_rules():
    from ui.isolate_state import setup_continue_enabled

    base = dict(
        download_in_progress=False,
        batch_count=0,
        live_upload_count=0,
        youtube_url="",
        pending_fp=None,
        pending_ready=False,
    )
    assert setup_continue_enabled(**base) is False
    assert setup_continue_enabled(**{**base, "download_in_progress": True, "pending_ready": True}) is False
    assert setup_continue_enabled(**{**base, "pending_ready": True}) is True
    youtube = "https://youtu.be/abc"
    assert (
        setup_continue_enabled(
            **{**base, "youtube_url": youtube, "pending_ready": True, "pending_fp": "song.wav:1"}
        )
        is False
    )
    assert (
        setup_continue_enabled(
            **{
                **base,
                "youtube_url": youtube,
                "pending_ready": True,
                "pending_fp": f"youtube:{youtube}",
            }
        )
        is True
    )
    assert (
        setup_continue_enabled(
            **{**base, "youtube_url": "https://example.com/song.mp3", "pending_ready": False}
        )
        is True
    )
    assert (
        setup_continue_enabled(
            **{
                **base,
                "batch_count": 2,
                "youtube_url": youtube,
                "pending_ready": False,
            }
        )
        is True
    )
    assert setup_continue_enabled(**{**base, "live_upload_count": 3}) is True
    assert (
        setup_continue_enabled(
            **{**base, "batch_count": 2, "download_in_progress": True}
        )
        is False
    )


def test_draft_snapshot_keeps_step_and_batch_paths():
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        OPEN_MIX_TABS_KEY,
        SETUP_STEP_KEY,
        SETUP_STEP_SEPARATE,
        SETUP_STEP_SOURCE,
        focus_new_draft_tab,
        open_new_draft_tab,
        reset_new_tab_source,
    )

    session: dict = {OPEN_MIX_TABS_KEY: []}
    first = open_new_draft_tab(session, fresh=True)
    session[SETUP_STEP_KEY] = SETUP_STEP_SEPARATE
    session["isolate_batch_paths"] = ["/tmp/a.wav", "/tmp/b.wav"]
    session["isolate_batch_fp"] = "a.wav:1|b.wav:2"
    second = open_new_draft_tab(session, fresh=True)
    assert session[SETUP_STEP_KEY] == SETUP_STEP_SOURCE
    assert "isolate_batch_paths" not in session
    saved = session[DRAFT_SOURCES_KEY][first]
    assert saved[SETUP_STEP_KEY] == SETUP_STEP_SEPARATE
    assert saved["isolate_batch_paths"] == ["/tmp/a.wav", "/tmp/b.wav"]
    assert saved["isolate_batch_fp"] == "a.wav:1|b.wav:2"
    focus_new_draft_tab(session, second)
    reset_new_tab_source(session)
    focus_new_draft_tab(session, first)
    assert session[SETUP_STEP_KEY] == SETUP_STEP_SEPARATE
    assert session["isolate_batch_paths"] == ["/tmp/a.wav", "/tmp/b.wav"]
    assert session["isolate_batch_fp"] == "a.wav:1|b.wav:2"


def test_ensure_upload_tab_focused_selects_existing_draft_on_home():
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        SHELL_VIEW_KEY,
        ensure_upload_tab_focused,
        open_new_draft_tab,
    )

    session: dict = {OPEN_MIX_TABS_KEY: [], SHELL_VIEW_KEY: "home"}
    draft = open_new_draft_tab(session, fresh=True)
    session[SHELL_TAB_KEY] = "home"
    session[SHELL_VIEW_KEY] = "home"
    focused = ensure_upload_tab_focused(session)
    assert focused == draft
    assert session[SHELL_TAB_KEY] == draft
    assert session[SHELL_VIEW_KEY] == "home" or session.get("_isolate_shell_view_next") == "home"


def test_ensure_upload_tab_focused_opens_a_draft_when_none_exists():
    from ui.isolate_state import (
        SHELL_TAB_KEY,
        SHELL_VIEW_KEY,
        ensure_upload_tab_focused,
        is_new_draft_tab,
    )

    session: dict = {SHELL_VIEW_KEY: "home", SHELL_TAB_KEY: "home"}
    focused = ensure_upload_tab_focused(session)
    assert is_new_draft_tab(focused)
    assert session[SHELL_TAB_KEY] == focused


def test_ensure_upload_tab_focused_leaves_mix_and_current_draft_alone():
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        SHELL_VIEW_KEY,
        ensure_upload_tab_focused,
        open_new_draft_tab,
    )

    mix: dict = {SHELL_VIEW_KEY: "mix", SHELL_TAB_KEY: "/runs/waltz", OPEN_MIX_TABS_KEY: ["/runs/waltz"]}
    assert ensure_upload_tab_focused(mix) is None
    assert mix[SHELL_TAB_KEY] == "/runs/waltz"

    session: dict = {OPEN_MIX_TABS_KEY: []}
    draft = open_new_draft_tab(session, fresh=True)
    session[SHELL_VIEW_KEY] = "home"
    assert ensure_upload_tab_focused(session) == draft
    assert session[SHELL_TAB_KEY] == draft


def test_ensure_upload_tab_focused_keeps_audio_already_on_the_home_form():
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        SHELL_VIEW_KEY,
        ensure_upload_tab_focused,
        is_new_draft_tab,
        open_new_draft_tab,
    )

    session: dict = {
        OPEN_MIX_TABS_KEY: [],
        SHELL_VIEW_KEY: "home",
        SHELL_TAB_KEY: "home",
    }
    empty = open_new_draft_tab(session, fresh=True)
    session[SHELL_TAB_KEY] = "home"
    session[SHELL_VIEW_KEY] = "home"
    session["carry_over_audio_path"] = "/tmp/song.wav"
    session["carry_over_audio_name"] = "song.wav"
    focused = ensure_upload_tab_focused(session)
    assert is_new_draft_tab(focused)
    assert focused != empty
    assert session["carry_over_audio_path"] == "/tmp/song.wav"
    assert session[SHELL_TAB_KEY] == focused


def test_upload_fingerprint():
    f = SimpleNamespace(name="song.mp3", size=12345)
    assert upload_fingerprint(f) == "song.mp3:12345"
    assert upload_fingerprint(None) is None


def test_sync_output_name_on_upload_new_file():
    f = SimpleNamespace(name="new_track.wav", size=999)
    fp, name, changed = sync_output_name_on_upload(
        f, last_fp="old.wav:1", output_name="old"
    )
    assert fp == "new_track.wav:999"
    assert name == "new_track"
    assert changed is True


def test_sync_output_name_on_upload_same_file():
    f = SimpleNamespace(name="same.wav", size=100)
    fp, name, changed = sync_output_name_on_upload(
        f, last_fp="same.wav:100", output_name="same"
    )
    assert fp == "same.wav:100"
    assert name == "same"
    assert changed is False


def test_reopen_queues_output_name_without_touching_widget_key():
    """Reopen must not write isolate_output_name after the text_input exists."""
    session: dict[str, object] = {ISOLATE_OUTPUT_NAME_KEY: "old-name"}
    queue_reopen_output_name(session, "Past run title")
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "old-name"
    assert session[ISOLATE_OUTPUT_NAME_PENDING_KEY] == "Past run title"
    applied = apply_pending_output_name(session)
    assert applied == "Past run title"
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "Past run title"
    assert ISOLATE_OUTPUT_NAME_PENDING_KEY not in session


def test_queue_output_name_if_empty_skips_when_name_set():
    session: dict[str, object] = {ISOLATE_OUTPUT_NAME_KEY: "keep-me"}
    queue_output_name_if_empty(session, "from-youtube")
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "keep-me"
    assert ISOLATE_OUTPUT_NAME_PENDING_KEY not in session


def test_queue_output_name_if_empty_queues_pending_when_blank():
    session: dict[str, object] = {ISOLATE_OUTPUT_NAME_KEY: "  "}
    queue_output_name_if_empty(session, "yt-stem")
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "  "
    assert session[ISOLATE_OUTPUT_NAME_PENDING_KEY] == "yt-stem"
    applied = apply_pending_output_name(session)
    assert applied == "yt-stem"
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "yt-stem"


def test_apply_pending_output_name_noop_without_pending():
    session: dict[str, object] = {ISOLATE_OUTPUT_NAME_KEY: "keep"}
    assert apply_pending_output_name(session) is None
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "keep"


def test_queue_clear_youtube_url_does_not_write_widget_key():
    session: dict[str, object] = {ISOLATE_YOUTUBE_URL_KEY: "https://youtu.be/abc"}
    queue_clear_youtube_url(session)
    assert session[ISOLATE_YOUTUBE_URL_KEY] == "https://youtu.be/abc"
    assert session[ISOLATE_YOUTUBE_URL_PENDING_KEY] == ""
    applied = apply_pending_youtube_url(session)
    assert applied == ""
    assert session[ISOLATE_YOUTUBE_URL_KEY] == ""
    assert ISOLATE_YOUTUBE_URL_PENDING_KEY not in session


def test_queue_youtube_url_fills_widget_on_apply():
    session: dict[str, object] = {}
    queue_youtube_url(session, "https://www.youtube.com/watch?v=BaW_jenozKc")
    assert ISOLATE_YOUTUBE_URL_KEY not in session
    applied = apply_pending_youtube_url(session)
    assert applied == "https://www.youtube.com/watch?v=BaW_jenozKc"
    assert session[ISOLATE_YOUTUBE_URL_KEY] == applied
    assert ISOLATE_YOUTUBE_URL_PENDING_KEY not in session


def test_apply_pending_youtube_url_noop_without_pending():
    session: dict[str, object] = {ISOLATE_YOUTUBE_URL_KEY: "keep"}
    assert apply_pending_youtube_url(session) is None
    assert session[ISOLATE_YOUTUBE_URL_KEY] == "keep"


def test_reset_new_tab_source_remounts_uploader_without_touching_widget_keys():
    session: dict[str, object] = {
        "isolate_upload_key": 3,
        "isolate_upload_fp": "a.wav:1",
        "isolate_pending_audio_path": "/tmp/a.wav",
        "isolate_pending_fp": "a.wav:1",
        "isolate_duration_sec": 12.0,
        "isolate_duration_fp": "a.wav:1",
        "carry_over_audio_path": "/tmp/c.wav",
        "carry_over_audio_name": "c.wav",
        "carry_over_is_guitar_stem": True,
        ISOLATE_OUTPUT_NAME_KEY: "Song A",
        ISOLATE_YOUTUBE_URL_KEY: "https://youtu.be/abc",
        "isolate_setup_step": "separate",
        "isolate_batch_paths": ["/tmp/a.wav", "/tmp/b.wav"],
        "isolate_batch_fp": "a.wav:1|b.wav:2",
    }
    reset_new_tab_source(session)
    assert session["isolate_upload_key"] == 4
    assert session["isolate_setup_step"] == "source"
    assert "isolate_batch_paths" not in session
    assert "isolate_batch_fp" not in session
    assert "isolate_pending_audio_path" not in session
    assert "isolate_pending_fp" not in session
    assert "isolate_upload_fp" not in session
    assert "isolate_duration_sec" not in session
    assert "carry_over_audio_path" not in session
    assert "carry_over_audio_name" not in session
    assert "carry_over_is_guitar_stem" not in session
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "Song A"
    assert session[ISOLATE_YOUTUBE_URL_KEY] == "https://youtu.be/abc"
    assert session[ISOLATE_YOUTUBE_URL_PENDING_KEY] == ""
    assert session[ISOLATE_OUTPUT_NAME_PENDING_KEY] == ""
    apply_pending_output_name(session)
    apply_pending_youtube_url(session)
    assert session[ISOLATE_OUTPUT_NAME_KEY] == ""
    assert session[ISOLATE_YOUTUBE_URL_KEY] == ""


def test_adopt_audio_into_run_copies_into_output_dir(tmp_path):
    src_dir = tmp_path / "stage"
    src_dir.mkdir()
    src = src_dir / "song.wav"
    src.write_bytes(b"RIFF")
    out = tmp_path / "run"
    out.mkdir()
    adopted = adopt_audio_into_run(src, out)
    assert adopted == out / "song.wav"
    assert adopted.read_bytes() == b"RIFF"
    assert src.exists()
    assert adopt_audio_into_run(adopted, out) == adopted


def test_is_youtube_staging_path_requires_data_dir_child(tmp_path, monkeypatch):
    import ui.isolate_state as state
    import ui.common as common

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    run = tmp_path / "abcd"
    run.mkdir()
    wav = run / "clip.wav"
    wav.write_bytes(b"x")
    assert is_youtube_staging_path(wav, "youtube:https://youtu.be/a")
    assert not is_youtube_staging_path(wav, "file.wav:1")
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"y")
    assert not is_youtube_staging_path(outside, "youtube:https://youtu.be/a")


def test_discard_youtube_staging_deletes_run_dir(tmp_path, monkeypatch):
    import ui.isolate_state as state
    import ui.common as common

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    run = tmp_path / "stage1"
    run.mkdir()
    wav = run / "clip.wav"
    wav.write_bytes(b"x")
    fp = "youtube:https://youtu.be/a"
    assert discard_youtube_staging(wav, fingerprint=fp) is True
    assert not run.exists()


def test_discard_youtube_staging_skips_retain_and_in_flight(tmp_path, monkeypatch):
    import ui.isolate_state as state
    import ui.common as common

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    run = tmp_path / "stage2"
    run.mkdir()
    wav = run / "clip.wav"
    wav.write_bytes(b"x")
    fp = "youtube:https://youtu.be/b"
    assert (
        discard_youtube_staging(wav, fingerprint=fp, retain_paths=(wav,)) is False
    )
    assert run.exists()

    monkeypatch.setattr(state, "staging_paths_still_needed", lambda _p: True)
    assert discard_youtube_staging(wav, fingerprint=fp) is False
    assert run.exists()


def test_reset_new_tab_source_discards_youtube_staging(tmp_path, monkeypatch):
    import ui.isolate_state as state
    import ui.common as common

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    run = tmp_path / "ytstage"
    run.mkdir()
    wav = run / "clip.wav"
    wav.write_bytes(b"x")
    session: dict[str, object] = {
        "isolate_upload_key": 1,
        "isolate_pending_audio_path": str(wav),
        "isolate_pending_fp": "youtube:https://youtu.be/z",
        "isolate_upload_fp": "youtube:https://youtu.be/z",
    }
    reset_new_tab_source(session)
    assert "isolate_pending_audio_path" not in session
    assert not run.exists()


def test_enqueue_confirmed_job_adopts_youtube_staging_before_enqueue():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    enqueue = source[
        source.find("def _enqueue_confirmed_job") : source.find("def _library_status_row")
    ]
    assert "adopt_audio_into_run(" in enqueue
    assert "discard_youtube_staging(" in enqueue
    assert 'startswith("youtube:")' in enqueue
    stage = source[
        source.find("def _stage_youtube_audio") : source.find("def _youtube_search_dialog")
    ]
    assert "discard_youtube_staging(" in stage


def test_youtube_video_id_from_common_urls():
    assert youtube_video_id("https://www.youtube.com/watch?v=BaW_jenozKc") == "BaW_jenozKc"
    assert youtube_video_id("https://youtu.be/abc123XYZ_-") == "abc123XYZ_-"
    assert youtube_video_id("https://www.youtube.com/shorts/shortId99") == "shortId99"
    assert youtube_video_id("https://example.com/watch?v=abc") is None
    assert youtube_label_from_url("https://youtu.be/abc") == "abc"
    assert youtube_label_from_url("not-a-url") == "youtube_audio"


def test_mixer_local_video_path_only_for_uploaded_source_video(tmp_path):
    assert mixer_local_video_path(None) is None
    assert mixer_local_video_path(tmp_path) is None
    extra = tmp_path / "vocals.mp4"
    extra.write_bytes(b"nope")
    assert mixer_local_video_path(tmp_path) is None
    video = tmp_path / "source_video.mp4"
    video.write_bytes(b"video")
    assert mixer_local_video_path(tmp_path) == video
    (tmp_path / "source_video.txt").write_text("notes", encoding="utf-8")
    assert mixer_local_video_path(tmp_path) == video


def test_home_upload_accepts_video_and_extracts_with_ffmpeg():
    page = (
        Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    ).read_text(encoding="utf-8")
    for suffix in (".mp4", ".mov", ".webm", ".mkv"):
        assert suffix in page
    assert "_HOME_UPLOAD_TYPES" in page
    assert "normalize_audio" in page
    assert 'output_dir / f"source_video{src.suffix.lower()}"' in page
    assert "st.error(err or \"Could not read this file.\")" in page


def test_prefer_local_mixer_picture_over_youtube_id(tmp_path):
    video = tmp_path / "source_video.mp4"
    video.write_bytes(b"video")
    assert prefer_local_mixer_picture("BaW_jenozKc", video) == ("BaW_jenozKc", video)
    assert prefer_local_mixer_picture("BaW_jenozKc", None) == ("BaW_jenozKc", None)
    assert prefer_local_mixer_picture("  ", None) == (None, None)


def test_live_mixer_passes_local_picture_and_offset():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    mixer = source[source.find("def _render_live_mixer") : source.find("def _init_track_picker_session")]
    assert "prefer_local_mixer_picture(" in mixer
    assert "youtube_video_id=youtube_id" in mixer
    assert "local_video_url=local_url" in mixer
    assert "video_offset_sec=video_offset_sec_for_run(run_dir)" in mixer
    enqueue = source[source.find("def _enqueue_confirmed_job") : source.find("def _library_status_row")]
    assert "_download_youtube_preview_into_run(" in enqueue


def test_mixer_youtube_video_id_only_for_loaded_youtube_mix():
    url = "https://www.youtube.com/watch?v=BaW_jenozKc"
    loaded = {
        "isolate_source_kind": "youtube",
        "isolate_results_source_fp": f"youtube:{url}",
    }
    assert mixer_youtube_video_id(loaded) == "BaW_jenozKc"
    assert mixer_youtube_video_id({"isolate_source_kind": "file", "isolate_results_source_fp": f"youtube:{url}"}) is None
    assert mixer_youtube_video_id({"isolate_source_kind": "youtube", "isolate_results_source_fp": "upload:abc"}) is None


def test_resolve_youtube_job_name_empty_prefers_downloaded_stem():
    assert (
        resolve_youtube_job_name("", "https://youtu.be/BaW_jenozKc", "ACDC - Back in Black")
        == "ACDC - Back in Black"
    )


def test_resolve_youtube_job_name_upgrades_video_id_default_to_real_title():
    assert (
        resolve_youtube_job_name("BaW_jenozKc", "https://youtu.be/BaW_jenozKc", "ACDC - Back in Black")
        == "ACDC - Back in Black"
    )


def test_resolve_youtube_job_name_keeps_user_edited_name():
    assert (
        resolve_youtube_job_name("My custom mix", "https://youtu.be/BaW_jenozKc", "ACDC - Back in Black")
        == "My custom mix"
    )


def test_resolve_youtube_job_name_non_youtube_falls_back_to_stem():
    assert resolve_youtube_job_name("", "https://example.com/song.wav", "song") == "song"
    assert resolve_youtube_job_name("my song", "", "song") == "my song"


def test_sync_output_name_on_youtube_new_url_replaces_old_title():
    name, named, auto, changed = sync_output_name_on_youtube(
        youtube_url="https://youtu.be/newvid",
        last_named_url="https://youtu.be/oldvid",
        output_name="Megadeth - Holy Wars",
        auto_output_name="Megadeth - Holy Wars",
    )
    assert changed is True
    assert name == "newvid"
    assert named == "https://youtu.be/newvid"
    assert auto == "newvid"


def test_sync_output_name_on_youtube_download_upgrades_auto_label():
    url = "https://youtu.be/newvid"
    name, named, auto, changed = sync_output_name_on_youtube(
        youtube_url=url,
        last_named_url=url,
        output_name="newvid",
        auto_output_name="newvid",
        downloaded_stem="Other Band - Other Song",
    )
    assert changed is True
    assert name == "Other Band - Other Song"
    assert named == url
    assert auto == "Other Band - Other Song"


def test_sync_output_name_on_youtube_keeps_user_edit_on_same_url():
    url = "https://youtu.be/newvid"
    name, named, auto, changed = sync_output_name_on_youtube(
        youtube_url=url,
        last_named_url=url,
        output_name="My custom mix",
        auto_output_name="newvid",
        downloaded_stem="Other Band - Other Song",
    )
    assert changed is False
    assert name == "My custom mix"
    assert named == url
    assert auto == "newvid"


def test_sync_output_name_on_youtube_preferred_label_beats_video_id():
    name, named, auto, changed = sync_output_name_on_youtube(
        youtube_url="https://youtu.be/YFMF4ZFmtnU",
        last_named_url="https://youtu.be/oldvid",
        output_name="old title",
        auto_output_name="old title",
        preferred_label="AC/DC - Girls Got Rhythm (Official Audio)",
    )
    assert changed is True
    assert name == "AC/DC - Girls Got Rhythm (Official Audio)"
    assert named == "https://youtu.be/YFMF4ZFmtnU"
    assert auto == name


def test_queue_youtube_url_with_title_syncs_output_name():
    session: dict[str, object] = {
        "isolate_youtube_enabled": True,
        ISOLATE_OUTPUT_NAME_KEY: "Megadeth - Holy Wars",
        ISOLATE_NAMED_YOUTUBE_URL_KEY: "https://youtu.be/oldvid",
        ISOLATE_AUTO_OUTPUT_NAME_KEY: "Megadeth - Holy Wars",
    }
    queue_youtube_url(
        session,
        "https://youtu.be/YFMF4ZFmtnU",
        title="AC/DC - Girls Got Rhythm (Official Audio)",
    )
    assert session[ISOLATE_YOUTUBE_TITLE_PENDING_KEY] == (
        "AC/DC - Girls Got Rhythm (Official Audio)"
    )
    apply_pending_youtube_url(session)
    applied = apply_youtube_output_name_sync(session)
    assert applied == "AC/DC - Girls Got Rhythm (Official Audio)"
    assert session[ISOLATE_OUTPUT_NAME_KEY] == applied
    assert ISOLATE_YOUTUBE_TITLE_PENDING_KEY not in session


def test_youtube_search_use_queues_hit_title():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    dialog = source[
        source.find("def _youtube_search_dialog") : source.find("def _stateful_expander")
    ]
    assert "queue_youtube_url(st.session_state, url, title=title)" in dialog
    assert "_download_youtube_with_status(" in dialog
    assert "st.status(" in source[
        source.find("def _download_youtube_with_status") : source.find(
            "def _youtube_search_dialog"
        )
    ]
    assert "Downloading" in dialog or "Downloading" in source[
        source.find("def _download_youtube_with_status") : source.find(
            "def _youtube_search_dialog"
        )
    ]
    # Use downloads in-dialog; do not close-first + silent auto-download.
    use_block = dialog[dialog.find('key=f"isolate_youtube_pick_') :]
    assert "_close_youtube_search_dialog()" in use_block
    assert use_block.find("_download_youtube_with_status(") < use_block.find(
        "_close_youtube_search_dialog()"
    )
    assert 'key=f"isolate_youtube_separate_p{page}_{vid}"' in dialog
    assert 'st.session_state["isolate_search_enqueue"] = True' in dialog
    workspace = source[
        source.find("def _render_new_workspace") : source.find("def _render_mixer_region_caption")
    ]
    assert 'st.session_state.pop("isolate_search_enqueue", False)' in workspace
    assert "_try_enqueue_choice(choice)" in workspace
    enqueue_helper = source[
        source.find("def _try_enqueue_choice") : source.find("def _has_source_for_job")
    ]
    assert "_enqueue_confirmed_job(choice, audio_path)" in enqueue_helper
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    # Screen 1 has no URL to download. Use/Separate still download inside the dialog.
    assert 'key="isolate_youtube_download"' not in controls
    assert "_download_youtube_with_status(" not in controls
    use_to_separate = dialog[
        dialog.find('key=f"isolate_youtube_pick_') : dialog.find(
            'key=f"isolate_youtube_separate_'
        )
    ]
    assert "SETUP_STEP_KEY" in use_to_separate
    assert use_to_separate.find("if err:") < use_to_separate.find("SETUP_STEP_KEY")
    assert use_to_separate.find("_download_youtube_with_status(") < use_to_separate.find(
        "_close_youtube_search_dialog()"
    )
    separate_block = dialog[dialog.find('key=f"isolate_youtube_separate_') :]
    separate_block = separate_block[: separate_block.find('key="isolate_youtube_search_prev"')]
    assert 'st.session_state["isolate_search_enqueue"] = True' in separate_block
    assert "isolate_setup_step" not in separate_block
    assert "SETUP_STEP_KEY" not in separate_block


def test_enqueue_confirmed_job_opens_queue_with_loading_overlay():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    enqueue = source[
        source.find("def _enqueue_confirmed_job") : source.find("def _library_status_row")
    ]
    # Queue opens on the next paint. Do not raise the nav overlay — jobs stay
    # interactive and the overlay contract is nav-only.
    assert '["_isolate_pending_queue"] = True' in enqueue
    assert "[ISOLATE_QUEUE_PANEL_OPEN_KEY] = True" in enqueue
    assert "_request_loading_overlay()" not in enqueue
    assert '["_nav_loading"]' not in enqueue
    assert "_rerun_scroll_top()" in enqueue
    assert "origin_tab=" in enqueue
    assert "mark_draft_tab_processing" in enqueue
    assert '["_isolate_keep_draft_tab"]' in enqueue
    assert 'WORKSPACE_NEXT_KEY] = "Queue"' not in enqueue
    main_tail = source[source.find('st.session_state["_isolate_form_drawn"]') :]
    assert 'pop("_isolate_pending_queue"' in main_tail
    assert 'pop("_isolate_keep_draft_tab"' in main_tail
    assert "open_home_shell" in main_tail
    assert "_rerun_scroll_top()" in main_tail
    assert "def _scroll_main_to_top" in source
    assert "isolate_scroll_top_token" in source
    assert "_scroll_main_to_top(" in source
    assert source.count("components.html") == 1
    poll = source[
        source.find("def _poll_running_jobs") : source.find("def _queue_tab_fragment")
    ]
    assert "apply_finished_job_poll_outcome" in poll
    assert "notify_only=bool(plan[\"notify_only\"])" in poll or "notify_only=" in poll
    assert "_open_mixer_workspace()" in poll
    assert "shell_view=" in poll
    assert "viewing_run_dir=" in poll
    assert "listen_picker=" in poll
    # Picker must not be retargeted before apply (desync bug).
    assert "LISTEN_PICKER_KEY] = str(run_dir)" not in poll
    assert 'isolate_listen_applied_dir"] = run_dir' not in poll
    tabs = source[
        source.find("def _render_moises_tab_strip") : source.find("def _has_source_for_job")
    ]
    assert "draft_tab_job_overlays" in tabs
    assert 'item["busy"]' in tabs or '"busy"' in tabs
    css = (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "pointer-events: auto !important" in css
    assert "audiotools-global-loading-root" in css


def test_apply_youtube_output_name_sync_queues_before_widget():
    session: dict[str, object] = {
        ISOLATE_YOUTUBE_URL_KEY: "https://youtu.be/newvid",
        ISOLATE_OUTPUT_NAME_KEY: "Megadeth - Holy Wars",
        ISOLATE_NAMED_YOUTUBE_URL_KEY: "https://youtu.be/oldvid",
        ISOLATE_AUTO_OUTPUT_NAME_KEY: "Megadeth - Holy Wars",
    }
    applied = apply_youtube_output_name_sync(session)
    assert applied == "newvid"
    assert session[ISOLATE_OUTPUT_NAME_KEY] == "newvid"
    assert ISOLATE_OUTPUT_NAME_PENDING_KEY not in session


def test_running_progress_view_keeps_raw_worker_text_out_of_the_label():
    view = running_progress_view(
        {
            "status": "running",
            "stage": "separate",
            "message": "demucs.apply: segment 4/9 shifts=1 device=cpu",
            "progress": 0.5,
        },
        now=0.0,
    )
    assert "demucs.apply" not in view["label"]
    assert "device=cpu" not in view["label"]
    assert view["hint"]
    assert "demucs.apply" not in view["hint"]


def test_running_progress_view_exposes_checklist_separately_from_the_label():
    view = running_progress_view(
        {"status": "running", "stage": "separate", "message": "", "progress": 0.5},
        now=0.0,
    )
    # The checklist is Pro-only detail: available on the view, never folded
    # into the one-line label that Lite shows.
    assert view["checklist_md"]
    assert view["checklist_md"] not in view["label"]
    assert view["checklist_md"].count("**") >= 2


def test_mixer_component_key_is_stable_for_the_same_run():
    first = mixer_component_key("/runs/abc")
    assert first == mixer_component_key("/runs/abc")
    assert first != mixer_component_key("/runs/def")
    assert first.startswith(MIXER_COMPONENT_KEY_PREFIX)


def test_mixer_component_key_survives_a_streamlit_key_roundtrip():
    # Streamlit keys must be plain identifiers; a path with separators would
    # otherwise leak into the DOM id and remount the iframe.
    key = mixer_component_key("C:\\Users\\me\\runs\\my song (live)")
    assert key.replace("_", "").isalnum()


def test_finishing_a_job_sends_the_user_to_the_mixer_not_the_queue():
    session: dict = {WORKSPACE_NEXT_KEY: "Mixer"}
    assert apply_workspace_tab(session, has_artifacts=True) == "Mixer"
    assert session[WORKSPACE_KEY] == "Mixer"
    assert WORKSPACE_NEXT_KEY not in session


def test_resolve_isolate_user_id_mints_without_stored():
    session: dict = {}
    first = resolve_isolate_user_id(session, None)
    assert len(first) == 32
    assert session["isolate_user_id"] == first
    assert resolve_isolate_user_id(session, "from-browser") == first


def test_resolve_isolate_user_id_uses_stored_when_session_empty():
    session: dict = {}
    assert resolve_isolate_user_id(session, "  abc123  ") == "abc123"
    assert session["isolate_user_id"] == "abc123"


def test_youtube_search_preview_does_not_use_st_video_embed():
    """Official uploads often refuse youtube.com/embed; st.video shows 'unavailable'."""
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    dialog = source[
        source.find("def _youtube_search_dialog") : source.find("def _stateful_expander")
    ]
    assert "st.video(" not in dialog
    assert "youtube.com/embed" not in dialog
    assert "Open on YouTube" in dialog
    assert "isolate_youtube_preview_" not in dialog
    assert "st.expander(" not in dialog


def test_staged_audio_for_new_tab_hides_preview_until_youtube_download(tmp_path):
    leftover = tmp_path / "old.wav"
    leftover.write_bytes(b"x")
    downloaded = tmp_path / "yt.wav"
    downloaded.write_bytes(b"y")
    url = "https://youtu.be/abc"
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url=url,
            pending_path=None,
            pending_fp=None,
            pending_exists=False,
            carry_path=None,
            carry_exists=False,
        )
        is None
    )
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url=url,
            pending_path=str(downloaded),
            pending_fp=f"youtube:{url}",
            pending_exists=True,
            carry_path=None,
            carry_exists=False,
        )
        == str(downloaded)
    )
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url=url,
            pending_path=str(leftover),
            pending_fp="old.wav:1",
            pending_exists=True,
            carry_path=None,
            carry_exists=False,
        )
        is None
    )
    direct_url = "https://example.com/song.mp3"
    direct_file = tmp_path / "direct.mp3"
    direct_file.write_bytes(b"d")
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url=direct_url,
            pending_path=str(direct_file),
            pending_fp=f"direct:{direct_url}",
            pending_exists=True,
            carry_path=None,
            carry_exists=False,
        )
        == str(direct_file)
    )
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url=direct_url,
            pending_path=str(leftover),
            pending_fp="other.wav:1",
            pending_exists=True,
            carry_path=None,
            carry_exists=False,
        )
        is None
    )


def test_staged_audio_for_new_tab_previews_upload_and_carry_over(tmp_path):
    uploaded = tmp_path / "song.wav"
    uploaded.write_bytes(b"x")
    carry = tmp_path / "from_tab.wav"
    carry.write_bytes(b"z")
    assert (
        staged_audio_for_new_tab(
            uploaded=True,
            youtube_url="",
            pending_path=str(uploaded),
            pending_fp="song.wav:1",
            pending_exists=True,
            carry_path=None,
            carry_exists=False,
        )
        == str(uploaded)
    )
    assert (
        staged_audio_for_new_tab(
            uploaded=False,
            youtube_url="",
            pending_path=None,
            pending_fp=None,
            pending_exists=False,
            carry_path=str(carry),
            carry_exists=True,
        )
        == str(carry)
    )


def test_pending_upload_fp_for_stale_uses_disk_not_uploader(tmp_path):
    staged = tmp_path / "song.wav"
    staged.write_bytes(b"x")
    session = {
        "isolate_pending_fp": "song.wav:1",
        "isolate_pending_audio_path": str(staged),
    }
    assert pending_upload_fp_for_stale(session) == "song.wav:1"
    session["isolate_pending_audio_path"] = str(tmp_path / "missing.wav")
    assert pending_upload_fp_for_stale(session) is None
    session["isolate_youtube_url"] = "https://youtu.be/abc"
    assert pending_upload_fp_for_stale(session) == "song.wav:1"


def test_infer_source_kind_from_fingerprint_and_url():
    assert infer_source_kind(source_fingerprint="youtube:https://youtu.be/abc") == "youtube"
    assert infer_source_kind(source_fingerprint="song.mp3:123") == "file"
    assert infer_source_kind(youtube_url="https://youtu.be/abc") == "youtube"
    assert infer_source_kind() == "file"


def test_infer_source_kind_explicit_wins():
    assert (
        infer_source_kind(
            source_kind="file",
            source_fingerprint="youtube:https://youtu.be/abc",
            youtube_url="https://youtu.be/abc",
        )
        == "file"
    )
    assert infer_source_kind(source_kind="youtube", source_fingerprint="song.mp3:1") == "youtube"


def test_format_source_title_and_caption():
    assert format_source_title("Party Wadokoni_", "youtube") == "YouTube · Party Wadokoni_"
    assert format_source_title("Calm Like You", "file") == "File · Calm Like You"
    assert format_source_title("  ", "youtube") == "YouTube · tracks"
    assert format_source_caption("Party Wadokoni_", "youtube") == "**Party Wadokoni_** · YouTube"
    assert (
        format_source_caption("Calm Like You", "file", filename="Calm Like You.wav")
        == "**Calm Like You** · File (`Calm Like You.wav`)"
    )
    assert (
        format_source_caption(
            "Party Wadokoni_",
            "youtube",
            region_label="0:10–0:40",
            clip_length=30,
        )
        == "**Party Wadokoni_** · 0:10–0:40 (30 s) · YouTube"
    )


def test_apply_workspace_tab_defaults_and_mixer_pin():
    from ui.isolate_state import apply_workspace_tab

    session: dict = {}
    assert apply_workspace_tab(session, has_artifacts=False) == "New"
    session = {}
    assert apply_workspace_tab(session, has_artifacts=True) == "New"
    session = {WORKSPACE_KEY: "New", WORKSPACE_NEXT_KEY: "Mixer"}
    assert apply_workspace_tab(session, has_artifacts=False) == "Mixer"
    assert session[WORKSPACE_KEY] == "Mixer"
    assert WORKSPACE_NEXT_KEY not in session
    session = {WORKSPACE_KEY: "Queue"}
    assert apply_workspace_tab(session, has_artifacts=True) == "Queue"


def test_apply_shell_view_home_and_mix():
    from ui.isolate_state import (
        SHELL_VIEW_KEY,
        SHELL_VIEW_NEXT_KEY,
        apply_shell_view,
        open_home_shell,
        open_mix_shell,
    )

    session: dict = {}
    assert apply_shell_view(session) == "home"
    open_mix_shell(session)
    assert apply_shell_view(session) == "mix"
    assert session[SHELL_VIEW_KEY] == "mix"
    assert SHELL_VIEW_NEXT_KEY not in session
    open_home_shell(session)
    assert apply_shell_view(session) == "home"


def test_partition_queue_jobs_includes_succeeded():
    jobs = [
        {"id": "s1", "status": "succeeded"},
        {"id": "r1", "status": "running"},
        {"id": "q1", "status": "queued"},
    ]
    parts = partition_queue_jobs(jobs)
    assert [j["id"] for j in parts["in_flight"]] == ["r1", "q1"]
    assert [j["id"] for j in parts["succeeded"]] == ["s1"]
    assert parts["empty"] is False
    assert partition_queue_jobs([])["empty"] is True


def test_recent_runs_with_owner_fallback():
    owned = [{"run_dir": "/mine"}]
    all_runs = [{"run_dir": "/mine"}, {"run_dir": "/other"}]
    assert recent_runs_with_owner_fallback(owned, all_runs) == owned
    assert recent_runs_with_owner_fallback([], all_runs) == all_runs


def test_pick_library_row_prefers_last_viewed():
    rows = [
        {"run_dir": "/runs/new", "title": "New"},
        {"run_dir": "/runs/old", "title": "Old"},
    ]
    assert pick_library_row(rows, "/runs/old")["title"] == "Old"
    assert pick_library_row(rows, None)["title"] == "New"
    assert pick_library_row([], "/runs/old") is None


def test_listen_picker_default_prefers_loaded_run():
    options = ["/runs/new", "/runs/loaded", "/runs/old"]
    assert listen_picker_default(options, "/runs/loaded", "/runs/new") == "/runs/loaded"
    assert listen_picker_default(options, "/runs/gone", "/runs/old") == "/runs/old"
    assert listen_picker_default(options, None, "/runs/missing") == "/runs/new"
    assert listen_picker_default([], "/runs/loaded", "/runs/old") is None


def test_apply_listen_picker_pending_sets_widget_key():
    session = {LISTEN_PICKER_NEXT_KEY: "/runs/keep"}
    apply_listen_picker_pending(session)
    assert session[LISTEN_PICKER_KEY] == "/runs/keep"
    assert LISTEN_PICKER_NEXT_KEY not in session
    apply_listen_picker_pending(session)
    assert session[LISTEN_PICKER_KEY] == "/runs/keep"


def test_closing_last_loaded_mix_tab_stays_closed(tmp_path: Path):
    """Closing the only mix tab must clear loaded pointers so the strip cannot re-add it.

    Regression: Home + one mix tab; close removed it from open tabs but left
    ``isolate_run_dir`` set, so the next ``add_open_mix_tab(loaded)`` revived it.
    """
    from ui.isolate_state import (
        ISOLATE_SKIP_REHYDRATE_KEY,
        LISTEN_PICKER_KEY,
        OPEN_MIX_TABS_KEY,
        add_open_mix_tab,
        close_open_mix_tab,
        is_loaded_mix_tab,
        mix_tab_ids_equal,
        open_mix_tabs_for_session,
        select_rehydrate_row,
    )

    run = tmp_path / "youtube_guns"
    run.mkdir()
    run_s = str(run)
    session: dict = {
        OPEN_MIX_TABS_KEY: [run_s],
        "isolate_run_dir": run_s,
        # Intentionally omit listen picker / applied dir — the old close path
        # only cleared when those matched, so run_dir alone reopened the tab.
    }

    assert is_loaded_mix_tab(session, run_s)
    assert mix_tab_ids_equal(run_s, run_s)

    neighbor = close_open_mix_tab(session, run_s)
    assert neighbor is None
    assert run_s not in session[OPEN_MIX_TABS_KEY]

    # Mimic the fixed page handler: clear loaded state when closing that mix.
    assert is_loaded_mix_tab(session, run_s)
    for key in (
        LISTEN_PICKER_KEY,
        "isolate_listen_applied_dir",
        "isolate_run_dir",
        "isolate_viewing_run_dir",
        "isolate_artifacts",
    ):
        session.pop(key, None)
    session[ISOLATE_SKIP_REHYDRATE_KEY] = True
    assert not is_loaded_mix_tab(session, run_s)

    loaded = session.get("isolate_run_dir") or session.get("isolate_listen_applied_dir")
    if loaded:
        add_open_mix_tab(session, loaded)
    tabs = open_mix_tabs_for_session(session, library_dirs=[run_s])
    assert tabs == []
    assert run_s not in session.get(OPEN_MIX_TABS_KEY, [])

    # Clearing artifacts must not let rehydrate revive the closed tab.
    rows = [{"run_dir": run_s, "title": "Guns"}]
    assert select_rehydrate_row(session, rows, wav_exists=lambda _: True) is None


def test_select_rehydrate_skips_after_last_mix_tab_dismissed():
    from ui.isolate_state import ISOLATE_SKIP_REHYDRATE_KEY, select_rehydrate_row

    rows = [{"run_dir": "/runs/a", "title": "A"}]
    session = {ISOLATE_SKIP_REHYDRATE_KEY: True}
    assert select_rehydrate_row(session, rows, wav_exists=lambda _: True) is None
    session.pop(ISOLATE_SKIP_REHYDRATE_KEY)
    assert select_rehydrate_row(session, rows, wav_exists=lambda _: True)["run_dir"] == "/runs/a"


def test_closing_last_mix_without_clearing_loaded_would_reopen(tmp_path: Path):
    """Document the failure mode: leaving isolate_run_dir set reopens the tab."""
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        add_open_mix_tab,
        close_open_mix_tab,
        open_mix_tabs_for_session,
    )

    run = tmp_path / "only_mix"
    run.mkdir()
    run_s = str(run)
    session: dict = {
        OPEN_MIX_TABS_KEY: [run_s],
        "isolate_run_dir": run_s,
    }
    close_open_mix_tab(session, run_s)
    assert session[OPEN_MIX_TABS_KEY] == []

    loaded = session.get("isolate_run_dir") or session.get("isolate_listen_applied_dir")
    if loaded:
        add_open_mix_tab(session, loaded)
    tabs = open_mix_tabs_for_session(session, library_dirs=[run_s])
    assert tabs == [run_s]


def test_open_mix_tabs_add_close_and_cap(tmp_path: Path):
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        OPEN_MIX_TABS_MAX,
        add_open_mix_tab,
        close_open_mix_tab,
        normalize_open_mix_tabs,
    )

    dirs = []
    for i in range(OPEN_MIX_TABS_MAX + 2):
        d = tmp_path / f"run{i}"
        d.mkdir()
        dirs.append(str(d))
    session: dict = {}
    for d in dirs:
        add_open_mix_tab(session, d, cap=OPEN_MIX_TABS_MAX)
    assert len(session[OPEN_MIX_TABS_KEY]) == OPEN_MIX_TABS_MAX
    assert session[OPEN_MIX_TABS_KEY][0] == dirs[2]
    assert session[OPEN_MIX_TABS_KEY][-1] == dirs[OPEN_MIX_TABS_MAX + 1]

    # Focusing an already-open tab must not move it to the end.
    before = list(session[OPEN_MIX_TABS_KEY])
    add_open_mix_tab(session, dirs[2], cap=OPEN_MIX_TABS_MAX)
    assert session[OPEN_MIX_TABS_KEY] == before
    assert session[OPEN_MIX_TABS_KEY].count(dirs[2]) == 1
    assert session[OPEN_MIX_TABS_KEY][0] == dirs[2]

    neighbor = close_open_mix_tab(session, dirs[2])
    assert dirs[2] not in session[OPEN_MIX_TABS_KEY]
    assert neighbor in session[OPEN_MIX_TABS_KEY]

    capped = normalize_open_mix_tabs(dirs, existing={dirs[0], dirs[1]}, cap=8)
    assert capped == [dirs[0], dirs[1]]


def test_plus_opens_distinct_new_draft_tabs(tmp_path: Path):
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        NEW_DRAFT_TAB_ID,
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        add_open_mix_tab,
        apply_shell_view,
        close_open_mix_tab,
        draft_tab_title,
        focus_new_draft_tab,
        is_new_draft_tab,
        open_home_shell,
        open_mix_tabs_for_session,
        open_new_draft_tab,
    )

    run = tmp_path / "mix"
    run.mkdir()
    session: dict = {OPEN_MIX_TABS_KEY: [str(run)]}
    first = open_new_draft_tab(session, fresh=False)
    assert is_new_draft_tab(first)
    assert first != NEW_DRAFT_TAB_ID
    assert session[OPEN_MIX_TABS_KEY][-1] == first
    assert session[SHELL_TAB_KEY] == first
    assert apply_shell_view(session) == "home"

    session["isolate_pending_audio_path"] = str(tmp_path / "song_a.wav")
    session["isolate_output_name"] = "Song A"
    second = open_new_draft_tab(session, fresh=True)
    assert second != first
    assert session[OPEN_MIX_TABS_KEY].count(first) == 1
    assert session[OPEN_MIX_TABS_KEY].count(second) == 1
    assert session[SHELL_TAB_KEY] == second
    assert not session.get("isolate_pending_audio_path")
    assert draft_tab_title(session, first) == "Song A"
    assert draft_tab_title(session, second) == "New"

    focus_new_draft_tab(session, first)
    assert session[SHELL_TAB_KEY] == first
    assert session.get("isolate_output_name") == "Song A"
    assert "song_a" in str(session.get("isolate_pending_audio_path") or "")

    tabs = open_mix_tabs_for_session(session, library_dirs=[str(run)])
    assert first in tabs and second in tabs and str(run) in tabs

    open_home_shell(session)
    assert session[SHELL_TAB_KEY] == "home"
    assert first in session[OPEN_MIX_TABS_KEY]

    neighbor = close_open_mix_tab(session, first)
    assert first not in session[OPEN_MIX_TABS_KEY]
    assert first not in (session.get(DRAFT_SOURCES_KEY) or {})
    assert neighbor in session[OPEN_MIX_TABS_KEY]

    # Legacy __new__ id still counts as a draft.
    add_open_mix_tab(session, NEW_DRAFT_TAB_ID)
    assert is_new_draft_tab(NEW_DRAFT_TAB_ID)
    assert NEW_DRAFT_TAB_ID in session[OPEN_MIX_TABS_KEY]


def test_promote_draft_tab_to_run_keeps_slot(tmp_path: Path):
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        draft_tab_job_overlays,
        pending_open_job_ids,
        mark_draft_tab_processing,
        open_new_draft_tab,
        promote_job_origin_tab,
        replace_open_mix_tab,
    )

    other = tmp_path / "other"
    other.mkdir()
    run = tmp_path / "finished"
    run.mkdir()
    session: dict = {OPEN_MIX_TABS_KEY: [str(other)]}
    draft = open_new_draft_tab(session, fresh=True)
    mark_draft_tab_processing(session, draft, title="My Song")
    assert session[SHELL_TAB_KEY] == draft
    assert session[DRAFT_SOURCES_KEY][draft]["isolate_output_name"] == "My Song"

    overlays = draft_tab_job_overlays(
        [
            {
                "id": "j1",
                "status": "running",
                "progress": 0.42,
                "title": "My Song",
                "origin_tab": draft,
            },
            {
                "id": "j2",
                "status": "queued",
                "progress": 0.0,
                "title": "Other",
                "origin_tab": "__new__:gone",
            },
            {"id": "j3", "status": "succeeded", "origin_tab": draft},
        ]
    )
    assert overlays[draft]["progress"] == pytest.approx(0.42)
    assert overlays[draft]["busy"] is True
    assert overlays[draft]["status"] == "running"
    assert "__new__:gone" in overlays
    assert overlays["__new__:gone"]["progress"] is None

    waiting = draft_tab_job_overlays(
        [{"id": "j3", "status": "succeeded", "title": "My Song", "origin_tab": draft}],
        pending_open_ids={"j3"},
    )
    assert waiting[draft]["busy"] is True
    assert waiting[draft]["progress"] == pytest.approx(1.0)
    assert waiting[draft]["status"] == "succeeded"

    omitted = draft_tab_job_overlays(
        [{"id": "j3", "status": "succeeded", "title": "My Song", "origin_tab": draft}]
    )
    assert draft not in omitted
    consumed = draft_tab_job_overlays(
        [{"id": "j3", "status": "succeeded", "title": "My Song", "origin_tab": draft}],
        pending_open_ids=set(),
    )
    assert draft not in consumed
    running_wins = draft_tab_job_overlays(
        [
            {
                "id": "j1",
                "status": "running",
                "progress": 0.42,
                "title": "My Song",
                "origin_tab": draft,
            },
            {"id": "j3", "status": "succeeded", "origin_tab": draft},
        ],
        pending_open_ids={"j3"},
    )
    assert running_wins[draft]["status"] == "running"
    assert running_wins[draft]["progress"] == pytest.approx(0.42)
    finished = [{"id": "j3", "status": "succeeded", "origin_tab": draft}]
    assert pending_open_job_ids(finished, None) == set()
    assert pending_open_job_ids(finished, []) == {"j3"}
    assert pending_open_job_ids(finished, ["j3"]) == set()

    promoted = promote_job_origin_tab(
        session,
        {"run_dir": str(run), "origin_tab": draft},
    )
    assert promoted == str(run)
    assert session[OPEN_MIX_TABS_KEY] == [str(other), str(run)]
    assert draft not in session[OPEN_MIX_TABS_KEY]
    assert draft not in session.get(DRAFT_SOURCES_KEY, {})
    assert session.get(SHELL_TAB_KEY) != draft

    session2: dict = {OPEN_MIX_TABS_KEY: [str(other)]}
    assert promote_job_origin_tab(session2, {"run_dir": str(run)}) == str(run)
    assert session2[OPEN_MIX_TABS_KEY] == [str(other), str(run)]

    session3: dict = {OPEN_MIX_TABS_KEY: [str(other)]}
    replace_open_mix_tab(session3, "__new__:missing", str(run))
    assert session3[OPEN_MIX_TABS_KEY] == [str(other), str(run)]


def test_normalize_prefers_dropping_mixes_before_drafts(tmp_path: Path):
    from ui.isolate_state import (
        NEW_DRAFT_TAB_PREFIX,
        normalize_open_mix_tabs,
    )

    mixes = []
    for i in range(10):
        d = tmp_path / f"m{i}"
        d.mkdir()
        mixes.append(str(d))
    drafts = [f"{NEW_DRAFT_TAB_PREFIX}:{i}" for i in range(3)]
    tabs = drafts + mixes
    out = normalize_open_mix_tabs(
        tabs, cap=8, require_dir=False, protect=[drafts[0]]
    )
    assert drafts[0] in out
    assert all(d in out for d in drafts)
    assert len(out) == 8
    assert sum(1 for p in out if not p.startswith(NEW_DRAFT_TAB_PREFIX)) == 5


def test_isolate_ui_state_roundtrips_open_mix_tabs():
    from ui.isolate_state import OPEN_MIX_TABS_KEY, apply_stored_isolate_ui_state, isolate_ui_state_payload

    session = {OPEN_MIX_TABS_KEY: ["/a", "/b"], "isolate_workspace": "Mixer"}
    payload = isolate_ui_state_payload(session)
    assert payload["open_mix_tabs"] == ["/a", "/b"]
    restored: dict = {}
    apply_stored_isolate_ui_state(restored, payload)
    assert restored[OPEN_MIX_TABS_KEY] == ["/a", "/b"]


def test_select_rehydrate_row_loads_latest_when_session_empty():
    rows = [{"run_dir": "/runs/new", "title": "New"}, {"run_dir": "/runs/old", "title": "Old"}]
    assert select_rehydrate_row({}, rows, wav_exists=lambda _: True)["title"] == "New"


def test_select_rehydrate_row_skips_when_session_wavs_ok():
    session = {"isolate_artifacts": {"vocals": "/a.wav"}}
    rows = [{"run_dir": "/runs/new"}]
    assert select_rehydrate_row(session, rows, wav_exists=lambda _: True) is None
    assert session_mixer_artifacts_ok(session, wav_exists=lambda _: True) is True


def test_refresh_open_mix_action_title_only_leaves_stem_set():
    session = {
        "isolate_artifacts": {"vocals": r"C:\runs\a\vocals.wav", "bass_bleed_diagnostics": "d.json"},
        "isolate_base_name": "Old",
        "isolate_volumes_db": {"vocals": -3},
    }
    row = {
        "title": "New",
        "artifacts": {"vocals": "C:/runs/a/vocals.wav", "bass_bleed_diagnostics": "other.json"},
    }
    assert refresh_open_mix_action(session, row) == "title"
    assert session["isolate_volumes_db"]["vocals"] == -3


def test_refresh_open_mix_action_stem_set_changed():
    session = {
        "isolate_artifacts": {"vocals": "/runs/a/vocals.wav"},
        "isolate_base_name": "Song",
    }
    row = {
        "title": "Song",
        "artifacts": {
            "vocals": "/runs/a/vocals.wav",
            "guitar": "/runs/a/guitar.wav",
        },
    }
    assert refresh_open_mix_action(session, row) == "stems"


def test_refresh_open_mix_action_unchanged():
    session = {
        "isolate_artifacts": {"guitar": "/runs/a/guitar.wav", "vocals": "/runs/a/vocals.wav"},
        "isolate_base_name": "Song",
    }
    row = {
        "title": "Song",
        "artifacts": {"vocals": "/runs/a/vocals.wav", "guitar": "/runs/a/guitar.wav"},
    }
    assert refresh_open_mix_action(session, row) == "unchanged"


def test_refresh_does_not_reopen_dismissed_home():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    refresh = source[
        source.find("def _refresh_isolate_from_disk") : source.find("def _open_mixer_workspace")
    ]
    assert "ISOLATE_SKIP_REHYDRATE_KEY" in refresh
    assert ".pop(ISOLATE_SKIP_REHYDRATE_KEY" not in refresh
    assert "refresh_open_mix_action(" in refresh
    assert "_apply_library_row(" in refresh
    assert 'action == "stems"' in refresh
    assert 'action == "title"' in refresh
    assert "Reloaded from disk." in refresh
    assert "Already up to date." in refresh
    assert "_rerun_scroll_top" not in refresh
    assert 'help="Reload this mix and the library from disk"' in source
    assert 'help="Re-scan the local run library"' not in source


def test_select_rehydrate_row_prefers_last_viewed():
    session = {"isolate_viewing_run_dir": "/runs/old"}
    rows = [{"run_dir": "/runs/new"}, {"run_dir": "/runs/old"}]
    assert select_rehydrate_row(session, rows, wav_exists=lambda _: False)["run_dir"] == "/runs/old"


def test_owner_filter_miss_rehydrates_via_jobs():
    from ui.isolate_jobs import merge_library_runs

    recent = recent_runs_with_owner_fallback([], [])
    jobs = [
        {
            "status": "succeeded",
            "run_dir": "/runs/job",
            "created_at": 2,
            "title": "From job",
        }
    ]
    merged = merge_library_runs(recent, jobs)
    row = select_rehydrate_row({}, merged, wav_exists=lambda _: True)
    assert row is not None
    assert row["title"] == "From job"


def test_isolate_ui_state_payload_prefers_next_tab():
    payload = isolate_ui_state_payload(
        {
            WORKSPACE_KEY: "New",
            WORKSPACE_NEXT_KEY: "Mixer",
            "isolate_viewing_run_dir": "/runs/a",
        }
    )
    assert payload["workspace"] == "Mixer"
    assert payload["viewing_run_dir"] == "/runs/a"


def test_isolate_ui_state_payload_includes_export_dir():
    payload = isolate_ui_state_payload(
        {
            WORKSPACE_KEY: "Mixer",
            "isolate_export_dir": "/Users/me/Downloads",
        }
    )
    assert payload["export_dir"] == "/Users/me/Downloads"


def test_isolate_ui_state_round_trip(tmp_path):
    path = tmp_path / "isolate_ui_state.json"
    session = {
        WORKSPACE_KEY: "Mixer",
        "isolate_viewing_run_dir": "/runs/a",
        "isolate_run_dir": "/runs/a",
    }
    payload = isolate_ui_state_payload(session)
    write_isolate_ui_state(path, payload)
    stored = read_isolate_ui_state(path)
    assert stored["workspace"] == "Mixer"
    assert stored["viewing_run_dir"] == "/runs/a"
    empty: dict = {}
    apply_stored_isolate_ui_state(empty, stored)
    assert empty[WORKSPACE_KEY] == "Mixer"
    assert empty["isolate_viewing_run_dir"] == "/runs/a"
    live = {WORKSPACE_KEY: "Queue"}
    apply_stored_isolate_ui_state(live, stored)
    assert live[WORKSPACE_KEY] == "Queue"


def test_fresh_session_poll_does_not_rerun_for_historical_jobs():
    jobs = [
        {"id": "a", "status": "succeeded"},
        {"id": "b", "status": "succeeded"},
        {"id": "c", "status": "succeeded"},
    ]
    assert seed_consumed_job_ids(jobs) == ["a", "b", "c"]
    plan = plan_isolate_job_poll(
        jobs, consumed_ids=None, viewing_id=None, form_drawn=False
    )
    assert plan["apply_job"] is None
    assert plan["rerun"] is False
    assert plan["consumed_ids"] == ["a", "b", "c"]
    again = plan_isolate_job_poll(
        jobs,
        consumed_ids=plan["consumed_ids"],
        viewing_id=None,
        form_drawn=True,
    )
    assert again["apply_job"] is None
    assert again["rerun"] is False


def test_poll_defers_new_success_until_form_is_drawn():
    jobs = [
        {"id": "old", "status": "succeeded"},
        {"id": "new", "status": "succeeded"},
    ]
    seeded = plan_isolate_job_poll(
        [{"id": "old", "status": "succeeded"}],
        consumed_ids=None,
        viewing_id=None,
        form_drawn=False,
    )["consumed_ids"]
    deferred = plan_isolate_job_poll(
        jobs, consumed_ids=seeded, viewing_id=None, form_drawn=False
    )
    assert deferred["apply_job"] is None
    assert deferred["rerun"] is False
    assert deferred["consumed_ids"] == ["old"]
    ready = plan_isolate_job_poll(
        jobs, consumed_ids=seeded, viewing_id=None, form_drawn=True
    )
    assert ready["apply_job"]["id"] == "new"
    assert ready["rerun"] is True
    assert ready["consumed_ids"] == ["old", "new"]
    settled = plan_isolate_job_poll(
        jobs, consumed_ids=ready["consumed_ids"], viewing_id=None, form_drawn=True
    )
    assert settled["apply_job"] is None
    assert settled["rerun"] is False


def test_scroll_policy_preserves_poll_and_jumps_on_nav():
    assert should_request_isolate_scroll_top("poll_progress") is False
    assert should_request_isolate_scroll_top("poll_failed") is False
    assert should_request_isolate_scroll_top("tile_pick") is False
    assert should_request_isolate_scroll_top("engine_widget") is False
    assert should_request_isolate_scroll_top("region") is False
    assert should_request_isolate_scroll_top("status_control") is False
    assert should_request_isolate_scroll_top("mixer_fader") is False
    assert should_request_isolate_scroll_top("unknown_widget") is False
    assert should_request_isolate_scroll_top("enqueue_separate") is True
    assert should_request_isolate_scroll_top("shell_nav") is True
    assert should_request_isolate_scroll_top("open_mixer") is True
    assert should_request_isolate_scroll_top("poll_apply_mixer") is True
    session: dict = {}
    request_isolate_scroll_top(session)
    assert session[ISOLATE_SCROLL_TOP_KEY] is True
    assert isolate_scroll_top_token(session) == 1
    request_isolate_scroll_top(session)
    assert isolate_scroll_top_token(session) == 2
    assert consume_isolate_scroll_top(session) is True
    assert consume_isolate_scroll_top(session) is False


def test_consume_mix_tab_event_applies_once_per_seq():
    last, action, tab_id = consume_mix_tab_event(None, 0)
    assert last == 0 and action is None and tab_id == ""
    last, action, tab_id = consume_mix_tab_event({"action": "plus"}, 0)
    assert last == 1 and action == "plus"
    last, action, tab_id = consume_mix_tab_event({"action": "plus", "seq": 10}, 0)
    assert last == 10 and action == "plus" and tab_id == ""
    last, action, tab_id = consume_mix_tab_event({"action": "plus", "seq": 10}, 10)
    assert last == 10 and action is None
    last, action, tab_id = consume_mix_tab_event(
        {"action": "focus", "id": "/runs/a", "seq": 11}, 10
    )
    assert last == 11 and action == "focus" and tab_id == "/runs/a"
    last, action, tab_id = consume_mix_tab_event({"action": "home", "seq": 12}, 11)
    assert action == "home"
    last, action, tab_id = consume_mix_tab_event(
        {"action": "close", "id": "__new__", "seq": 13}, 12
    )
    assert action == "close" and tab_id == "__new__"
    last, action, tab_id = consume_mix_tab_event({"action": "nope", "seq": 14}, 13)
    assert last == 13 and action is None
    # Missing seq still applies once (older iframe builds).
    last, action, tab_id = consume_mix_tab_event({"action": "plus"}, 20)
    assert last == 21 and action == "plus"


def test_mix_tab_x_closes_without_deleting():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    close = source[
        source.find('elif action == "close" and tab_id:') : source.find(
            "def _render_busy_tab_close_prompt"
        )
    ]
    assert "_close_mix_tab(rows, tab_id)" in close
    assert "MIX_TAB_DELETE_KEY" not in close
    assert "MIX_TAB_DELETE_KEY" not in source
    assert "_render_mix_tab_delete_prompt" not in source
    assert "Delete this mix from disk?" not in source
    assert "_request_queue_delete" in source


def test_running_poll_does_not_full_rerun_or_scroll_top():
    jobs = [
        {"id": "run", "status": "running"},
        {"id": "wait", "status": "queued"},
    ]
    seeded = plan_isolate_job_poll(
        jobs, consumed_ids=None, viewing_id=None, form_drawn=False
    )
    plan = plan_isolate_job_poll(
        jobs,
        consumed_ids=seeded["consumed_ids"],
        viewing_id=None,
        form_drawn=True,
    )
    assert plan["apply_job"] is None
    assert plan["rerun"] is False
    assert isolate_poll_requires_full_rerun(applied=False, plan_rerun=plan["rerun"]) is False
    assert should_request_isolate_scroll_top("poll_progress") is False
    assert isolate_poll_requires_full_rerun(applied=False, plan_rerun=True) is False
    assert isolate_poll_requires_full_rerun(applied=True, plan_rerun=True) is True


def test_os_notify_seeds_historical_jobs_and_fires_for_new_terminal():
    jobs = [
        {"id": "old-ok", "status": "succeeded", "title": "A"},
        {"id": "old-fail", "status": "failed", "title": "B"},
        {"id": "run", "status": "running", "title": "C"},
        {"id": "wait", "status": "queued", "title": "D"},
    ]
    assert seed_notified_job_ids(jobs) == ["old-ok", "old-fail"]
    ids, pending = jobs_needing_os_notify(jobs, None)
    assert pending == []
    assert ids == ["old-ok", "old-fail"]
    later = jobs + [{"id": "new-ok", "status": "succeeded", "title": "Hangar 18"}]
    ids, pending = jobs_needing_os_notify(later, ids)
    assert [row["id"] for row in pending] == ["new-ok"]
    later = later + [{"id": "new-fail", "status": "failed", "title": "Nope"}]
    ids, pending = jobs_needing_os_notify(later, ids)
    assert [row["id"] for row in pending] == ["new-fail"]
    later = later + [{"id": "gone", "status": "cancelled", "title": "Z"}]
    ids, pending = jobs_needing_os_notify(later, ids)
    assert pending == []


def test_os_notify_empty_seed_does_not_toast_stale_failures():
    """First poll with no jobs stored []; later history must not all re-toast."""
    now = 1_700_000_000.0
    stale = {
        "id": "old-fail",
        "status": "failed",
        "title": "Yesterday",
        "finished_at": now - 3600,
        "updated_at": now - 3600,
    }
    fresh_ok = {
        "id": "new-ok",
        "status": "succeeded",
        "title": "Hangar 18",
        "finished_at": now - 5,
        "updated_at": now - 5,
    }
    ids, pending = jobs_needing_os_notify([stale, fresh_ok], [], now=now)
    assert set(ids) == {"old-fail", "new-ok"}
    assert [row["id"] for row in pending] == ["new-ok"]
    ids, pending = jobs_needing_os_notify([stale, fresh_ok], ids, now=now)
    assert pending == []


def test_os_notify_message_copy():
    assert os_notify_message({"status": "succeeded", "title": "Hangar 18"}) == (
        "Audio Isolation",
        "Separated: Hangar 18",
    )
    assert os_notify_message({"status": "failed", "title": "Nope"}) == (
        "Audio Isolation",
        "Needs attention: Nope failed",
    )
    assert os_notify_message({"status": "queued", "title": "Soon"}) is None
    assert os_notify_message({"status": "running", "title": "Soon"}) is None


def test_load_persist_isolate_user_id_round_trip(tmp_path):
    path = tmp_path / "isolate_user_id"
    first = load_persist_isolate_user_id({}, path)
    assert path.read_text(encoding="utf-8") == first
    again = load_persist_isolate_user_id({}, path)
    assert again == first


def test_should_hide_stale_results():
    assert not should_hide_stale_results(pending_upload_fp=None, has_artifacts=True)
    assert not should_hide_stale_results(pending_upload_fp="x:1", has_artifacts=False)
    assert should_hide_stale_results(pending_upload_fp="new.mp3:500", has_artifacts=True)
    # Same fingerprint as the run that produced results → do not hide mixer.
    assert not should_hide_stale_results(
        pending_upload_fp="same.mp3:1",
        has_artifacts=True,
        results_source_fp="same.mp3:1",
    )
    assert should_hide_stale_results(
        pending_upload_fp="new.mp3:2",
        has_artifacts=True,
        results_source_fp="old.mp3:1",
    )


def test_pending_audio_needs_resave():
    from ui.isolate_state import pending_audio_needs_resave

    assert pending_audio_needs_resave(
        uploaded_fp="a.wav:1", pending_fp=None, pending_path_exists=False
    )
    assert not pending_audio_needs_resave(
        uploaded_fp="a.wav:1", pending_fp="a.wav:1", pending_path_exists=True
    )
    assert pending_audio_needs_resave(
        uploaded_fp="b.wav:2", pending_fp="a.wav:1", pending_path_exists=True
    )


def test_stage_progress_percent_weighted():
    # Current stage counts only via ``intra`` (0 = just started, 1 = finished).
    # Active weights total 1.09 on the full stage list.
    assert stage_progress_percent("ingest", intra=0.0) == pytest.approx(0.0)
    assert stage_progress_percent("ingest", intra=1.0) == pytest.approx(0.05 / 1.09)
    assert stage_progress_percent("separate", intra=0.0) == pytest.approx(0.05 / 1.09)
    assert stage_progress_percent("separate", intra=1.0) == pytest.approx(0.85 / 1.09)
    assert stage_progress_percent("done", intra=1.0) == pytest.approx(1.0)
    assert stage_progress_percent("unknown") == 0.0


def test_stage_progress_percent_monotonic_across_stages():
    stages = isolation_stages_for_job(expects_guitar=True)
    points = [
        stage_progress_percent(stage, stages=stages, intra=0.0) for stage in stages
    ]
    points.append(1.0)
    assert points == sorted(points)
    assert points[-2] < 1.0
    assert points[-1] == pytest.approx(1.0)


def test_stage_progress_first_mid_last_and_guitar_skipped():
    full = isolation_stages_for_job(expects_guitar=True)
    assert stage_progress_percent("ingest", stages=full, intra=0.0) == pytest.approx(0.0)

    mid = stage_progress_percent("separate", stages=full, intra=0.0)
    assert mid == pytest.approx(0.05 / 1.09)
    assert 0.0 < mid < 1.0

    assert stage_progress_percent("done", stages=full, intra=1.0) == pytest.approx(1.0)

    full = isolation_stages_for_job(expects_guitar=True)
    assert "guitar_split" not in full
    assert "bass_bleed" in full
    assert "analysis" in full
    assert "tab" in full
    assert full.index("analysis") < full.index("tab") < full.index("done")

    skipped = isolation_stages_for_job(expects_guitar=False, analyse=False, tab=False)
    assert "guitar_split" not in skipped
    assert "bass_bleed" not in skipped
    assert "analysis" not in skipped
    assert "tab" not in skipped
    # Weights without bass_bleed/analysis/tab: 0.05+0.80+0.05+0.03+0.02 = 0.95
    assert stage_progress_percent("separate", stages=skipped, intra=0.0) == pytest.approx(
        0.05 / 0.95
    )


def test_isolation_stages_can_omit_analysis_and_tab():
    with_analysis = isolation_stages_for_job(expects_guitar=True)
    without_analysis = isolation_stages_for_job(expects_guitar=True, analyse=False)
    without_tab = isolation_stages_for_job(expects_guitar=True, tab=False)

    assert "analysis" in with_analysis
    assert "analysis" not in without_analysis
    assert "tab" in without_analysis
    assert "tab" not in without_tab
    assert "analysis" in without_tab


def test_checklist_first_mid_last_and_guitar_skipped():
    full = isolation_stages_for_job(expects_guitar=True)
    first = checklist_items(full, "ingest")
    assert first[0]["state"] == "current"
    assert first[0]["id"] == "ingest"
    assert all(row["state"] == "pending" for row in first[1:])

    mid = checklist_items(full, "separate")
    assert mid[0]["state"] == "done"
    assert mid[1]["state"] == "current"
    assert mid[1]["id"] == "separate"
    assert all(row["state"] == "pending" for row in mid[2:])

    last = checklist_items(full, "done", complete=True)
    assert all(row["state"] == "done" for row in last)

    skipped = isolation_stages_for_job(expects_guitar=False)
    ids = [row["id"] for row in checklist_items(skipped, "ingest")]
    assert "guitar_split" not in ids
    assert "bass_bleed" not in ids
    text = format_checklist_markdown(checklist_items(skipped, "separate"))
    assert "[now]" in text
    assert "[done]" in text
    assert "[todo]" in text
    assert "Split lead / rhythm" not in text


def test_expects_guitar_stem():
    assert expects_guitar_stem(model="htdemucs_6s") is True
    assert expects_guitar_stem(model="htdemucs", two_stems="vocals") is False
    assert expects_guitar_stem(model="htdemucs") is False
    assert expects_guitar_stem(model="htdemucs", custom_stems=["guitar", "vocals"]) is True
    assert expects_guitar_stem(model="htdemucs_6s", custom_stems=["vocals"]) is False


def test_intra_stage_fraction_caps_and_unestimated():
    frac, estimated = intra_stage_fraction(30.0, 60.0)
    assert estimated is True
    assert frac == pytest.approx(0.5)
    capped, _ = intra_stage_fraction(1000.0, 60.0)
    assert capped == pytest.approx(0.98)
    none, is_est = intra_stage_fraction(10.0, None)
    assert none == 0.0
    assert is_est is False


def test_estimate_job_and_remaining_eta():
    stages = isolation_stages_for_job(expects_guitar=True)
    heuristic, conf = estimate_job_seconds(
        audio_duration_sec=60.0,
        quality="fast",
        device="cpu",
        stages=stages,
    )
    assert heuristic is not None and heuristic > 0
    assert conf == "low"

    scaled, high = estimate_job_seconds(
        audio_duration_sec=90.0,
        quality="fast",
        device="cpu",
        stages=stages,
        model="htdemucs_6s",
        expects_guitar=True,
        last_run={
            "wall_sec": 80.0,
            "audio_sec": 60.0,
            "quality": "fast",
            "device": "cpu",
            "model": "htdemucs_6s",
            "expects_guitar": True,
        },
    )
    assert high == "high"
    assert scaled == pytest.approx(120.0)

    remaining, _ = estimate_remaining_seconds(
        elapsed_sec=10.0,
        percent=0.25,
        total_estimate=80.0,
        confidence="low",
    )
    assert remaining is not None and remaining > 0

    none_left, none_conf = estimate_remaining_seconds(
        elapsed_sec=0.5,
        percent=0.0,
        total_estimate=None,
        confidence="low",
    )
    assert none_left is None
    assert none_conf == "low"

    stage_share = estimated_stage_seconds("separate", stages, 100.0)
    # separate is 0.80 of active weights (no guitar_split stage; 1.09 total with analysis/tab).
    assert stage_share == pytest.approx(100.0 * 0.80 / 1.09)

    assert format_eta_line(9, None).endswith("estimating…")
    assert "~2:40 left" in format_eta_line(9, 160)
    assert format_eta_line(80, 0.2) == "Elapsed 1:20"


def test_format_progress_label():
    assert format_progress_label(0.42, "Running Demucs") == "42% · Running Demucs"
    assert format_progress_label(0.42, "Running Demucs", estimated=True) == (
        "~42% · Running Demucs"
    )


def test_format_elapsed():
    assert format_elapsed(65) == "1:05"
    assert format_elapsed(0) == "0:00"


def test_running_progress_view_has_label_elapsed_eta_and_checklist():
    stages = isolation_stages_for_job(expects_guitar=True)
    started = 1_000.0
    view = running_progress_view(
        {
            "status": "running",
            "stage": "separate",
            "message": "Running Demucs…",
            "started_at": started,
            "stage_started_at": started,
            "job_estimate_sec": 120.0,
            "estimate_confidence": "low",
            "stages": list(stages),
            "progress": 0.1,
        },
        now=started + 12.0,
    )
    assert view["percent"] > 0
    assert view["percent"] < 1
    assert "Separate tracks" in view["label"] or "%" in view["label"]
    assert "Elapsed" in view["eta_line"]
    assert "[now]" in view["checklist_md"]
    assert "Prepare audio" in view["checklist_md"]
    assert "Analyse song" in view["checklist_md"]
    assert "Transcribe tab" in view["checklist_md"]
    assert view["hint"] == "Separating tracks. CPU may be slow"
    assert "Check guitar" not in view["checklist_md"]
    assert "Check which tracks have sound" not in view["checklist_md"]


def test_user_progress_hint_never_echoes_raw_worker_text():
    assert user_progress_hint("ingest", "libtorchcodec exploded") is None
    assert "libtorchcodec" not in (user_progress_hint("separate", "libtorchcodec exploded") or "")
    assert "NVIDIA GPU" in (user_progress_hint("separate", "Separating tracks on NVIDIA GPU") or "")


def test_running_progress_view_estimating_without_job_estimate():
    view = running_progress_view(
        {
            "status": "running",
            "stage": "ingest",
            "message": "Preparing audio…",
            "started_at": 10.0,
            "stage_started_at": 10.0,
            "job_estimate_sec": None,
            "stages": ["ingest", "separate", "collect", "presence", "done"],
        },
        now=11.0,
    )
    assert "estimating…" in view["eta_line"]


def test_should_auto_apply_job_respects_pin():
    assert should_auto_apply_job(None, "job-a") is True
    assert should_auto_apply_job("latest", "job-a") is True
    assert should_auto_apply_job("job-a", "job-a") is True
    assert should_auto_apply_job("job-old", "job-a") is False


def test_should_auto_apply_job_blocks_when_mixer_on_other_run():
    """Unpinned / latest still must not yank focus off another open mix."""
    assert (
        should_auto_apply_job(
            "latest",
            "job-new",
            shell_view="mix",
            viewing_run_dir="/runs/old",
            finished_run_dir="/runs/new",
        )
        is False
    )
    assert (
        should_auto_apply_job(
            None,
            "job-new",
            shell_view="mix",
            listen_picker="/runs/old",
            finished_run_dir="/runs/new",
        )
        is False
    )
    # Same mix finishing — allow auto-apply.
    assert (
        should_auto_apply_job(
            "latest",
            "job-new",
            shell_view="mix",
            viewing_run_dir="/runs/new",
            finished_run_dir="/runs/new",
        )
        is True
    )
    # Home / New shell — keep auto-open.
    assert (
        should_auto_apply_job(
            "latest",
            "job-new",
            shell_view="home",
            viewing_run_dir="/runs/old",
            finished_run_dir="/runs/new",
        )
        is True
    )
    # Draft tab focus is not "another song" — still auto-apply.
    assert (
        should_auto_apply_job(
            "latest",
            "job-new",
            shell_view="mix",
            shell_tab="__new__:abc",
            finished_run_dir="/runs/new",
        )
        is True
    )


def test_plan_isolate_job_poll_notify_only_when_mixer_on_other_run():
    jobs = [
        {
            "id": "job-new",
            "status": "succeeded",
            "run_dir": "/runs/new",
        }
    ]
    plan = plan_isolate_job_poll(
        jobs,
        consumed_ids=[],
        viewing_id="latest",
        form_drawn=True,
        shell_view="mix",
        viewing_run_dir="/runs/old",
    )
    assert plan["apply_job"]["id"] == "job-new"
    assert plan["notify_only"] is True
    assert plan["rerun"] is True

    auto = plan_isolate_job_poll(
        jobs,
        consumed_ids=[],
        viewing_id="latest",
        form_drawn=True,
        shell_view="home",
        viewing_run_dir="/runs/old",
    )
    assert auto["notify_only"] is False


def test_listening_picker_uses_moises_top_strip_not_inline_buttons():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    picker = source[source.find("def _render_listening_switcher") : source.find("def _has_source_for_job")]
    assert "Open mixes" not in picker
    assert "mix_tab_focus_" not in picker
    assert '"Change mix"' not in picker
    assert 'label="Listening to"' in picker
    assert "_render_moises_tab_strip" in source
    assert "st.tabs(" not in source[source.find("def main") :]
    assert "mix_tabs(" in source
    assert 'action == "plus"' in source
    assert "open_new_draft_tab" in source
    assert 'action == "home"' in source


def test_listening_switcher_renames_current_mix_from_editable_name():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    switcher = source[source.find("def _render_listening_switcher") : source.find("def _has_source_for_job")]
    assert 'label="Listening to"' in switcher
    assert 'key="isolate_listen_name"' in switcher
    assert "on_change=_rename_listen_run" in switcher
    assert "def _rename_listen_run" in source
    assert "rename_run_title(Path(run_dir), new_name)" in source
    assert 'label="Mix name"' not in switcher
    assert "isolate_rename_mix" not in switcher
    assert "isolate_mix_name_input" not in switcher


def test_poll_and_queue_fragments_run_every_one_second():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    assert source.count("@st.fragment(run_every=1.0)") == 3
    assert "run_every=2.0" not in source
    poll = source[
        source.find("def _poll_running_jobs") : source.find("def _queue_tab_fragment")
    ]
    assert "isolate_poll_requires_full_rerun" in poll
    assert "_rerun_scroll_top()" in poll
    assert "failed_flash" not in poll
    queue_row = source[
        source.find("def _render_queue_job_row") : source.find("def _render_failed_strip")
    ]
    assert "_rerun_preserve_scroll()" in queue_row
    assert "_rerun_scroll_top()" in queue_row
    assert "open_mixer_" in queue_row
    assert "_render_running_progress(" in queue_row
    assert "_retry_failed_job(" in queue_row
    assert "isolate_queue_retry_" in queue_row
    assert 'st.caption("Separating…")' not in queue_row
    tabs = source[
        source.find("def _render_moises_tab_strip") : source.find("def _has_source_for_job")
    ]
    assert tabs.count("_rerun_scroll_top()") >= 4
    assert "st.rerun()" not in tabs
    assert 'key="isolate_moises_tabs"' in tabs
    assert "_mix_tabs_nonce" not in tabs
    assert "isolate_moises_tabs_{nonce}" not in tabs
    assert "consume_mix_tab_event" in tabs
    assert "_dismiss_last_mix_tab()" in tabs
    assert "ISOLATE_SKIP_REHYDRATE_KEY" in source
    assert "dismiss=True" in source
    status = source[
        source.find("def _render_status_strip") : source.find("def _poll_running_jobs")
    ]
    assert 'key="isolate_status_strip"' in status
    assert 'key="isolate_status_running"' not in status
    assert "_render_running_progress(" not in status
    assert "_render_compact_running_hint(" in status
    assert "ISOLATE_QUEUE_PANEL_OPEN_KEY" in status
    assert 'Separating **' not in status
    assert "if not (failed or show_running_hint or stopping_previous or show_queued):" in status
    idle = status[
        status.find("show_queued =") : status.find(
            'with st.container(key="isolate_status_strip")'
        )
    ]
    assert "return" in idle
    assert "st.empty()" not in idle
    compact = source[
        source.find("def _render_compact_running_hint") : source.find(
            "def _retry_failed_job"
        )
    ]
    assert "Open Queue for Pause/Stop" in compact
    retry = source[
        source.find("def _retry_failed_job") : source.find("def _job_source_title")
    ]
    assert "requeue_job(job_id)" in retry
    assert "Add the file again on Home." in retry
    assert "if not (running or failed or show_queued or stopping_previous):" not in status
    main = source[source.find("def main") :]
    assert "isolate_scroll_top_token" in main
    assert "consume_isolate_scroll_top" not in main
    css = (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "st-key-isolate_status_strip" in css
    assert "st-key-isolate_status_running" not in css
    assert "min-height: 6.5rem" not in css
    assert ":not(:last-of-type)" in css
    assert 'iframe[title*="mix_tabs"]' not in css
    assert "st-key-isolate_sticky_chrome" in css
    assert (
        '[data-testid="stLayoutWrapper"]:has(.st-key-isolate_sticky_chrome)'
        in css
    )
    assert (
        '[data-testid="stElementContainer"]:has(.st-key-isolate_sticky_chrome)'
        in css
    )
    assert 'key="isolate_sticky_chrome"' in main
    tab_frag = source[
        source.find("def _mix_tab_strip_fragment") : source.find(
            "def _render_busy_tab_close_prompt"
        )
    ]
    assert "_render_moises_tab_strip(" in tab_frag
    assert "st.rerun(" not in tab_frag
    assert "pending_open_ids" in source[
        source.find("def _render_moises_tab_strip") : source.find(
            "def _mix_tab_strip_fragment"
        )
    ]
    assert main.find("_mix_tab_strip_fragment") < main.find("_poll_running_jobs()")


def test_delete_confirm_first_click_is_labeled_as_arming():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    arm = "Click to arm confirm. Nothing is deleted yet."
    confirm = source[source.find("def _confirm_button") : source.find("def _queue_confirm_pair")]
    pair = source[
        source.find("def _queue_confirm_pair") : source.find("def _render_job_queue_panel")
    ]
    assert arm in confirm
    assert arm in pair
    assert source.count('confirm_label="Delete from disk"') == 3


def test_home_queue_is_floating_panel_not_bottom_section():
    """Queue is a header-toggled floating card on Home or mix, painted in a fragment."""
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    main = source[source.find("def main") :]
    fragment = source[
        source.find("def _queue_header_fragment") : source.find(
            "def _render_queue_float_panel"
        )
    ]
    assert 'st.subheader("Queue")' not in main
    assert 'st.divider()' not in main
    assert 'key="isolate_queue_toggle"' in fragment
    assert "ISOLATE_QUEUE_PANEL_OPEN_KEY" in fragment
    assert 'key="isolate_queue_float"' in source
    assert "_render_queue_float_panel()" in fragment
    assert "_queue_header_fragment()" in main
    assert "_queue_tab_fragment()" in source
    assert 'shell != "mix"' not in main
    assert "if st.session_state.get(ISOLATE_QUEUE_PANEL_OPEN_KEY):" in fragment
    assert 'key="isolate_queue_toggle_slot"' in fragment
    assert "[6, 1.35, 1.2]" in main
    assert main.find("_mix_tab_strip_fragment") < main.find("_queue_header_fragment()")
    assert main.find("_queue_header_fragment()") < main.find('key="isolate_refresh"')
    assert main.find("_queue_header_fragment()") < main.find('if shell == "mix":')
    assert main.find("_queue_header_fragment()") < main.find("_render_mixer_workspace")
    assert main.find("_queue_header_fragment()") < main.find("_render_new_workspace")
    assert "open Queue for jobs" in source
    assert "See Queue" not in source
    assert "scroll to Queue" not in source
    assert "Open Queue." in source
    css = (
        Path(__file__).resolve().parents[1] / "ui" / "app.py"
    ).read_text(encoding="utf-8")
    assert "st-key-isolate_queue_float" in css
    assert "st-key-isolate_queue_toggle" in css
    assert "st-key-isolate_queue_toggle_slot" in css
    assert "right: 6.75rem" not in css
    assert "overflow: visible !important" in css
    assert 'stVerticalBlock"]:has(.st-key-isolate_queue_float)' in css
    assert "right: 1.15rem" in css
    assert "position: fixed !important" in css
    assert "420px" in css
    assert "55vh" in css
    assert "not a sidebar" in css.lower() or "not a sidebar / drawer" in css
    chrome_css = css[
        css.find("Refresh + Queue sit on the mix-tabs row") : css.find("Isolate Queue:")
    ]
    assert "white-space: nowrap !important" in chrome_css
    assert "min-width: 6.25rem !important" in chrome_css
    assert "flex: 0 0 auto !important" in chrome_css
    assert "letter-wrap" in chrome_css
    # Match the mix-tabs iframe (48px) so Queue/Refresh top edges do not step.
    assert "align-items: center !important" in chrome_css
    btn_block = chrome_css[
        chrome_css.find(".st-key-isolate_refresh button") : chrome_css.find(
            ".st-key-isolate_refresh button p"
        )
    ]
    assert "height: 48px !important" in btn_block
    assert "min-height: 48px !important" in btn_block
    assert "2.5rem" not in btn_block
    queue_css = css[css.find("Isolate Queue:") : css.find("New-tab outcome")]
    assert "flex: 1 1 100%" not in queue_css
    assert "min-width: 5.5rem" not in queue_css
    assert "st-key-queue_confirm_row" in queue_css
    assert "flex: 1 1 0" in queue_css
    assert "white-space: normal" in queue_css
    panel = source[
        source.find("def _render_job_queue_panel") : source.find("def _request_queue_delete")
    ]
    assert "_queue_confirm_pair(" in panel
    assert "_confirm_button(" not in panel
    pair = source[
        source.find("def _queue_confirm_pair") : source.find("def _render_job_queue_panel")
    ]
    assert 'key=f"queue_confirm_row_{key}"' in pair
    assert "st.columns(2)" in pair
    assert 'width="stretch"' in pair
    queue_row = source[
        source.find("def _render_queue_job_row") : source.find("def _render_failed_strip")
    ]
    title_at = queue_row.find('st.write(f"**{title}**')
    cols_at = queue_row.find("st.columns(")
    assert title_at != -1 and cols_at != -1 and title_at < cols_at
    assert "[4, *([1] * n_actions)]" not in queue_row
    armed = queue_row.find("delete_finished_{job_id}__confirm")
    mixer = queue_row.find("Open in Mixer")
    assert armed != -1 and mixer != -1 and armed < mixer
    assert "return" in queue_row[armed:mixer]
    assert "_queue_confirm_pair(" in queue_row
    assert "Open in Mixer" in queue_row
    assert "pause_job_" in queue_row
    assert "isolate_stop_ask_" in queue_row
    assert "delete_finished_" in queue_row
    assert "_render_running_progress(" in queue_row
    assert "isolate_queue_retry_" in queue_row
    assert "_retry_failed_job(" in queue_row
    assert 'st.caption("Separating…")' not in queue_row
    fragment = source[
        source.find("def _queue_tab_fragment") : source.find("def _mixer_and_downloads_fragment")
    ]
    assert "@st.fragment(run_every=1.0)" in source[
        source.find("def _queue_tab_fragment") - 80 : source.find("def _mixer_and_downloads_fragment")
    ]
    assert "_render_job_queue_panel()" in fragment
    assert "on_click=_toggle_queue_panel" in source
    assert "isolate_queue_panel_open" in source


def test_new_tab_section_and_output_name_prominent_before_advanced():
    """Name and trim stay above Advanced, so the fields people edit aren't buried.

    Anchored on widget keys and call sites rather than label copy, so rewording
    does not fail the test but reordering does.
    """
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    block_at = controls.find("_render_section_and_outcomes(")
    output_at = controls.find('key="isolate_output_name"')
    assert block_at != -1 and output_at != -1
    assert output_at < block_at
    section = source[
        source.find("def _render_section_and_outcomes") : source.find("def _persist_kwargs")
    ]
    region_at = section.find("_render_region_controls(")
    picker_at = section.find("_render_outcome_picker(")
    assert region_at != -1 and picker_at != -1
    assert region_at < picker_at
    picker = source[
        source.find("def _render_outcome_picker") : source.find("def _render_engine_panel")
    ]
    assert "_render_engine_panel(" in picker


def test_isolate_youtube_search_dialog_pages_nine_in_a_three_by_three_grid():
    """Isolation search is a 3×3 page. A full page of 9 still offers Next.

    YouTube to MP3 keeps PAGE_SIZE 10. This lock is the isolate call sites only.
    """
    from ui.youtube_search import PAGE_SIZE, has_next_page, page_offset, result_range_caption

    assert PAGE_SIZE == 10
    assert has_next_page(9, page_size=9) is True
    assert has_next_page(8, page_size=9) is False
    assert page_offset(4, page_size=9) == 27
    assert result_range_caption(4, 9, page_size=9) == "Page 4 · results 28–36"
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    search = source[
        source.find("def _run_youtube_search") : source.find("def _stage_youtube_audio")
    ]
    dialog = source[
        source.find("def _youtube_search_dialog") : source.find("def _stateful_expander")
    ]
    assert "ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE = 9" in source
    assert "max_results=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE" in search
    assert "page_offset(page, page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE)" in search
    assert "max_results=YT_SEARCH_PAGE_SIZE" not in search
    assert "st.columns([1.5, 3.2, 1.3])" not in dialog
    assert "st.expander(" not in dialog
    assert 'key=f"isolate_youtube_preview_' not in dialog
    assert "youtube.com/embed" not in dialog
    assert "st.columns(3)" in dialog
    assert "range(3)" in dialog
    guard = dialog.find("if slot >= len(")
    pick = dialog.find('key=f"isolate_youtube_pick_')
    assert guard != -1 and pick != -1 and guard < pick
    assert "continue" in dialog[guard : guard + 120]
    assert dialog.count("result_range_caption(") >= 2
    assert dialog.count("page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE") >= 2
    assert (
        "has_next_page(len(hits_state), page_size=ISOLATE_YOUTUBE_SEARCH_PAGE_SIZE)"
        in dialog
    )
    assert 'key="isolate_youtube_search_prev"' in dialog
    assert 'key="isolate_youtube_search_next"' in dialog
    assert 'key=f"isolate_youtube_separate_p{page}_{vid}"' in dialog
    assert "isolate_youtube_search_form" in dialog


def test_isolate_source_screen_is_upload_then_youtube_search_box():
    """Screen 1 is two stacked boxes: the existing uploader, then click-to-search.

    A URL field or Download audio button on this screen would bring paste-and-fetch
    back. The search button keeps its key and is the whole YouTube hit target.
    """
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    source_step = controls[
        controls.find("if step == SETUP_STEP_SOURCE") : controls.find(
            "batch_paths = _stored_batch_paths"
        )
    ]
    assert 'key="isolate_youtube_enabled"' not in controls
    assert 'key="isolate_youtube_url"' not in controls
    assert "st.text_input(" not in source_step
    assert 'key="isolate_youtube_download"' not in source_step
    assert "st.columns([4, 1]" not in source_step
    assert "queue_youtube_url" not in source_step
    assert 'drew_youtube, "isolate_youtube_url", ISOLATE_YOUTUBE_URL_PENDING_KEY' in controls
    assert "apply_pending_youtube_url(st.session_state)" in controls
    open_at = source_step.find('key="isolate_youtube_search_open_btn"')
    assert open_at != -1
    button_at = source_step.rfind("st.button(", 0, open_at)
    assert button_at != -1
    assert "st.columns" not in source_step[max(0, button_at - 400) : button_at]
    button_src = source_step[button_at : open_at + 180]
    assert "Search YouTube" in button_src
    assert 'width="stretch"' in button_src
    handler = source_step[open_at : open_at + 520]
    assert "ISOLATE_YOUTUBE_SEARCH_OPEN_KEY] = True" in handler
    assert "_youtube_search_dialog()" in handler
    assert "accept_multiple_files=True" in source_step
    assert "_HOME_UPLOAD_TYPES" in source_step
    assert 'key=f"isolate_upload_{st.session_state.get(\'isolate_upload_key\', 0)}"' in source_step
    assert "_remember_batch_uploads(uploads)" in source_step
    assert "_render_batch_caption(batch_count)" in source_step
    assert source_step.find("st.file_uploader(") < open_at
    assert 'key="isolate_setup_continue"' in source_step
    assert 'type="primary"' in source_step[source_step.find('key="isolate_setup_continue"') - 80 :]
    assert "is_direct_media_url(direct_url)" in source_step
    assert "_resolve_audio_for_job" in source_step
    assert "YOUTUBE_DISCLAIMER" in source_step
    disc = source_step[source_step.find("st.caption(YOUTUBE_DISCLAIMER)") - 80 :]
    assert "is_pro_mode" not in disc[:80]
    assert "off in this installer build" not in source
    assert 'key="isolate_source_entry"' in source_step
    css = (Path(__file__).resolve().parents[1] / "ui" / "app.py").read_text(encoding="utf-8")
    tile = css[css.find("Isolation Screen 1:") :]
    assert tile
    tile = tile[: tile.find("Isolation search 3x3")]
    assert "st-key-isolate_upload_" in tile
    assert "st-key-isolate_source_entry" in tile
    assert "min-height: 7.25rem" in tile
    assert "100vh" in tile
    assert "overflow: hidden" in tile
    assert "justify-content: center" in tile
    assert "1px dashed" in tile
    assert "Drop or select files here" in tile
    assert "Find a public video by song or artist" in tile
    assert "st-key-isolate_youtube_search_open_btn" in tile
    assert "st-key-yt_audio_search_open_btn" not in tile
    assert "up to 5" not in tile
    assert "WMA" not in tile
    assert "Cloud storage" not in tile
    grid_css = css[css.find("Isolation search 3x3") : css.find("Isolation search 3x3") + 700]
    assert "st-key-isolate_youtube_pick_" in grid_css
    assert "st-key-yt_audio_search_open_btn" not in grid_css


def test_mixer_builds_current_mix_on_save_click():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    fragment = source[
        source.find("def _mixer_and_downloads_fragment") : source.find("def _save_all_tracks")
    ]
    assert "_build_current_mix(" not in fragment
    panel = source[
        source.find("def _render_downloads_panel") : source.find("def _resolve_audio_for_job")
    ]
    assert "_build_current_mix(" in panel
    assert 'key="isolate_save_mix"' in panel
    assert "Alt+Tab" in panel
    assert "overwrites a same-named" in panel
    assert "Save current mix includes mute, solo, levels, pan, master, and Speed & Key" in panel
    assert "playback_rate" in panel
    assert "playback_semitones" in panel
    assert "apply_playback_transform" in source
    hint_at = panel.find("If the folder window is hidden")
    assert "is_pro_mode" not in panel[max(0, hint_at - 80) : hint_at]


def test_mixer_workspace_lite_shares_fixup_and_tab_pdf_carry_over():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    mixer_ws = source[
        source.find("def _render_mixer_workspace") : source.find("def _render_file_ready_banner")
    ]
    assert "Fix the guitar track" not in mixer_ws
    assert "_render_guitar_fixup_panel(" in mixer_ws
    assert "Make a tab PDF from this" in mixer_ws
    assert "resolve_tab_pdf_carry_from_isolate" in mixer_ws
    assert "carry_over_is_guitar_stem" in mixer_ws
    assert "_request_loading_overlay()" in mixer_ws
    # Guitar fix-up panel called for both Lite (else branch) and Pro (if branch)
    assert mixer_ws.count("_render_guitar_fixup_panel(") == 2


def test_resolve_tab_pdf_carry_prefers_guitar_stem(tmp_path):
    mix = tmp_path / "mix.wav"
    guitar = tmp_path / "guitar.wav"
    mix.write_bytes(b"RIFF")
    guitar.write_bytes(b"RIFF")
    session = {
        "isolate_source_audio_path": str(mix),
        "isolate_artifacts": {"guitar": str(guitar)},
    }
    path, name, is_stem = resolve_tab_pdf_carry_from_isolate(session)
    assert path == str(guitar)
    assert name == "guitar.wav"
    assert is_stem is True


def test_resolve_tab_pdf_carry_falls_back_to_source_mix(tmp_path):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"RIFF")
    session = {
        "isolate_source_audio_path": str(mix),
        "isolate_artifacts": {"vocals": str(tmp_path / "missing.wav")},
    }
    path, name, is_stem = resolve_tab_pdf_carry_from_isolate(session)
    assert path == str(mix)
    assert name == "mix.wav"
    assert is_stem is False


def test_resolve_tab_pdf_carry_empty_when_nothing_usable():
    path, name, is_stem = resolve_tab_pdf_carry_from_isolate({})
    assert path is None
    assert name is None
    assert is_stem is False


def test_file_ready_banner_points_at_home_not_new():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    banner = source[
        source.find("def _render_file_ready_banner") : source.find("def main")
    ]
    assert "on Home to" in banner
    assert "on New to" not in banner


def test_desktop_mixer_renders_stem_hint_captions():
    frontend = (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "stem_mixer_component"
        / "frontend"
        / "src"
        / "main.ts"
    )
    source = frontend.read_text(encoding="utf-8")
    assert 'class="stem-hint"' in source
    assert "stem.hint" in source


def test_lite_region_defaults_full_file_with_safer_warn():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    region = source[
        source.find("def _render_region_controls") : source.find("def _render_section_preview")
    ]
    enqueue = source[source.find("def _enqueue_confirmed_job") :]
    assert "Safer: first" in region
    assert "LITE_MAX_DURATION_SEC" in region
    assert "float(duration)" in region
    assert "clamp_lite_clip" not in enqueue
    choice = source[
        source.find("def _separation_choice_from_session") : source.find(
            "def _render_separation_controls"
        )
    ]
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    assert "def _separation_choice_from_session" in source
    assert "clamp_lite_clip" not in choice
    assert "clamp_lite_clip" not in controls
    assert "is_pro_mode" in region
    assert "no silent clamp" in region or "never a silent clamp" in region


def test_setup_step_split_keeps_download_and_batch_guards():
    page = Path(__file__).resolve().parents[1] / "ui" / "pages" / "isolate.py"
    source = page.read_text(encoding="utf-8")
    assert source.count("_render_section_preview(") == 1
    controls = source[
        source.find("def _render_separation_controls") : source.find("def _ffmpeg_install_hint")
    ]
    source_step = controls[
        controls.find("if step == SETUP_STEP_SOURCE") : controls.find(
            "batch_paths = _stored_batch_paths"
        )
    ]
    assert 'key="isolate_youtube_download"' not in source_step
    assert 'key="isolate_youtube_search_open_btn"' in source_step
    assert source_step.find('key="isolate_youtube_search_open_btn"') < source_step.find(
        "carry_over_path"
    )
    back = controls[controls.find('key="isolate_setup_back"') : controls.find('key="isolate_setup_back"') + 400]
    assert "reset_new_tab_source" not in back
    assert "isolate_upload_key" not in back
    enqueue = source[
        source.find("def _try_enqueue_choice") : source.find("def _has_source_for_job")
    ]
    assert "staged_paths=" in enqueue
    batch = source[
        source.find("def _enqueue_upload_batch") : source.find("def _try_enqueue_choice")
    ]
    assert "staged_paths" in batch
    assert "_stage_batch_upload" in batch


def test_resolve_separation_preset_full_band():
    resolved = resolve_separation_preset("full_band")
    assert resolved["model"] == "bs_roformer_sw"
    assert resolved["guitar_refine"] is False
    assert resolved["two_stems"] is None
    assert resolved["track_count"] == 4
    assert "Guitar" in resolved["tracks"]
    assert "Piano" not in resolved["tracks"]
    assert "Other" not in resolved["tracks"]
    assert resolved["emit_stems"] == ("vocals", "drums", "bass", "guitar")
    assert resolved["fold_other_into_guitar"] is True


def test_resolve_separation_preset_best_guitar_uses_refine():
    legacy = resolve_separation_preset("best_guitar")
    current = resolve_separation_preset("full_band")
    assert legacy["model"] == "bs_roformer_sw"
    assert legacy["guitar_refine"] is True
    assert current["guitar_refine"] is False
    assert legacy["emit_stems"] == current["emit_stems"]
    assert legacy["id"] == "custom"


def test_resolve_separation_preset_essential():
    resolved = resolve_separation_preset("essential")
    assert resolved["model"] == "htdemucs"
    assert resolved["two_stems"] is None
    assert resolved["track_count"] == 4
    assert resolved["emit_stems"] == ("vocals", "drums", "bass", "other")


def test_resolve_separation_preset_vocals_music():
    resolved = resolve_separation_preset("vocals_music")
    assert resolved["model"] == "htdemucs"
    assert resolved["two_stems"] == "vocals"
    assert resolved["track_count"] == 1
    assert "instrumental" in resolved["tracks"].lower()


def test_resolve_separation_preset_unknown_falls_back():
    resolved = resolve_separation_preset("not_a_real_preset")
    assert resolved["id"] == "custom"
    assert resolved["model"] == "htdemucs_6s"
    assert resolved["track_options"] == list(DEFAULT_TRACK_OPTIONS)


def test_resolve_track_selection_default_demucs_guitar():
    resolved = resolve_track_selection(DEFAULT_TRACK_OPTIONS)
    assert resolved["model"] == "htdemucs_6s"
    assert resolved["guitar_refine"] is False
    assert resolved["emit_stems"] == ("vocals", "guitar")
    assert resolved["caveat"] == ""


def test_resolve_track_selection_roformer_guitar_refine():
    resolved = resolve_track_selection(
        ["vocals_demucs", "drums_demucs", "guitar_roformer_refine"]
    )
    assert resolved["model"] == "bs_roformer_sw"
    assert resolved["guitar_refine"] is True
    assert resolved["caveat"]
    assert "BS-RoFormer" in resolved["caveat"]


def test_resolve_track_selection_roformer_with_demucs_stems_notes_mixed_pass():
    resolved = resolve_track_selection(["vocals_demucs", "guitar_roformer"])
    assert resolved["model"] == "bs_roformer_sw"
    assert "one BS-RoFormer pass" in resolved["caveat"]


def test_resolve_track_selection_rejects_two_guitar_options():
    with pytest.raises(ValueError, match="one guitar"):
        resolve_track_selection(["guitar_demucs_6s", "guitar_roformer"])


def test_guitar_track_radio_ids_without_roformer():
    assert guitar_track_radio_ids(roformer_available=False) == ("none", "guitar_demucs_6s")
    assert guitar_track_radio_ids(roformer_available=True) == (
        "none",
        "guitar_demucs_6s",
        "guitar_roformer",
        "guitar_roformer_refine",
    )


def test_normalize_guitar_track_selection_downgrades_roformer_when_unavailable():
    assert (
        normalize_guitar_track_selection("guitar_roformer_refine", roformer_available=False)
        == "guitar_demucs_6s"
    )
    assert normalize_guitar_track_selection("guitar_roformer", roformer_available=True) == "guitar_roformer"
    assert normalize_guitar_track_selection("none", roformer_available=False) == "none"


def test_tracks_picker_help_mentions_install_when_roformer_missing():
    missing = tracks_picker_help(roformer_available=False)
    assert "bs-roformer-infer" in missing
    assert tracks_picker_help(roformer_available=True) != missing


def test_default_guitar_track_option_promotes_roformer_when_available():
    assert default_guitar_track_option(roformer_available=True) == "guitar_roformer"
    assert default_guitar_track_option(roformer_available=False) == "guitar_demucs_6s"
    assert (
        default_guitar_track_option(
            roformer_available=True, prefer_roformer=False
        )
        == "guitar_demucs_6s"
    )


def test_promote_default_guitar_option_respects_prefer_roformer():
    base = ["vocals_demucs", "guitar_demucs_6s"]
    assert promote_default_guitar_option(
        base, roformer_available=True, prefer_roformer=False
    ) == ["vocals_demucs", "guitar_demucs_6s"]
    assert promote_default_guitar_option(
        base, roformer_available=True, prefer_roformer=True
    ) == ["vocals_demucs", "guitar_roformer"]


def test_resolve_outcome_card_lite_cpu_keeps_demucs_when_roformer_installed():
    band = resolve_outcome_card(
        "band", roformer_available=True, prefer_roformer=False
    )
    assert "guitar_demucs_6s" in band
    assert "guitar_roformer" not in band
    guitar = resolve_outcome_card(
        "guitar", roformer_available=True, prefer_roformer=False
    )
    assert guitar == ["guitar_demucs_6s"]
    strong = resolve_outcome_card(
        "guitar", roformer_available=True, prefer_roformer=True
    )
    assert strong == ["guitar_roformer"]


def test_promote_default_guitar_option_swaps_only_fresh_defaults():
    base = ["vocals_demucs", "guitar_demucs_6s"]
    assert promote_default_guitar_option(
        base, roformer_available=True
    ) == ["vocals_demucs", "guitar_roformer"]
    assert promote_default_guitar_option(
        base, roformer_available=False
    ) == ["vocals_demucs", "guitar_demucs_6s"]
    kept = ["guitar_demucs_6s"]
    assert promote_default_guitar_option(kept, roformer_available=False) == [
        "guitar_demucs_6s"
    ]
    explicit = ["guitar_roformer_refine"]
    assert promote_default_guitar_option(
        explicit, roformer_available=True
    ) == ["guitar_roformer_refine"]


def test_job_requires_roformer_backend():
    assert job_requires_roformer_backend("bs_roformer_sw") is True
    assert job_requires_roformer_backend("melband_roformer_guitar") is True
    assert job_requires_roformer_backend("htdemucs_6s") is False


def test_resolve_track_selection_vocals_instrumental_exclusive():
    resolved = resolve_track_selection(["vocals_instrumental_demucs"])
    assert resolved["model"] == "htdemucs"
    assert resolved["two_stems"] == "vocals"
    with pytest.raises(ValueError, match="cannot combine"):
        resolve_track_selection(["vocals_instrumental_demucs", "drums_demucs"])


def test_resolve_custom_separation_vocals_only_uses_four_stem_demucs():
    resolved = resolve_custom_separation(["vocals"])
    assert resolved["model"] == "htdemucs"
    assert resolved["two_stems"] is None
    assert resolved["stems"] == ["vocals"]


def test_resolve_custom_separation_piano_and_vocals_needs_six_stem_model():
    resolved = resolve_custom_separation(["piano", "vocals"])
    assert resolved["model"] == "htdemucs_6s"
    assert resolved["two_stems"] is None
    assert resolved["stems"] == ["vocals", "piano"]
    assert "Vocals (Demucs)" in resolved["tracks"]
    assert "Piano (Demucs 6-stem" in resolved["tracks"]


def test_resolve_custom_separation_guitar_needs_six_stem_model():
    resolved = resolve_custom_separation(["guitar"])
    assert resolved["model"] == "htdemucs_6s"
    assert resolved["two_stems"] is None
    assert resolved["emit_stems"] == ("guitar",)
    assert resolved["fold_other_into_guitar"] is True


def test_resolve_custom_separation_guitar_and_other_keeps_other():
    resolved = resolve_custom_separation(["guitar", "other"])
    assert resolved["model"] == "htdemucs_6s"
    assert resolved["emit_stems"] == ("guitar", "other")
    assert resolved["fold_other_into_guitar"] is False


def test_resolve_custom_separation_drums_and_bass_use_four_stem_model():
    resolved = resolve_custom_separation(["bass", "drums"])
    assert resolved["model"] == "htdemucs"
    assert resolved["two_stems"] is None
    assert resolved["stems"] == ["drums", "bass"]


def test_resolve_custom_separation_ignores_unknown_picks():
    resolved = resolve_custom_separation(["drums", "kazoo", "drums"])
    assert resolved["stems"] == ["drums"]


def test_resolve_custom_separation_requires_a_pick():
    with pytest.raises(ValueError):
        resolve_custom_separation([])
    with pytest.raises(ValueError):
        resolve_custom_separation(["kazoo"])


def test_resolve_custom_separation_has_no_caveat():
    assert resolve_custom_separation(["vocals"])["caveat"] == ""


def test_custom_stem_choices_include_other():
    assert "other" in CUSTOM_STEM_CHOICES
    assert "guitar" in CUSTOM_STEM_CHOICES
    assert "piano" in CUSTOM_STEM_CHOICES


def test_custom_selected_stems_only_turns_on_requested_instruments():
    selected = custom_selected_stems(
        ["vocals", "drums", "bass", "guitar", "piano", "other"],
        ["vocals", "piano"],
    )
    assert selected == {
        "vocals": True,
        "drums": False,
        "bass": False,
        "guitar": False,
        "piano": True,
        "other": False,
    }


def test_custom_selected_stems_combined_guitar_only():
    selected = custom_selected_stems(
        ["guitar", "lead_guitar", "rhythm_guitar", "vocals"],
        ["guitar"],
    )
    # Default path maps guitar → guitar only (no lead/rhythm ids).
    assert selected["guitar"] is True
    assert selected["lead_guitar"] is False
    assert selected["rhythm_guitar"] is False
    assert selected["vocals"] is False


def test_custom_selected_stems_keeps_combined_guitar_without_a_split():
    selected = custom_selected_stems(["guitar", "vocals"], ["guitar"])
    assert selected["guitar"] is True


def test_custom_selected_stems_falls_back_to_all_when_nothing_matches():
    selected = custom_selected_stems(["vocals", "no_vocals"], ["drums"])
    assert selected == {"vocals": True, "no_vocals": True}


def test_custom_selected_stems_keeps_metronome_on():
    selected = custom_selected_stems(
        ["vocals", "guitar", "metronome"],
        ["vocals"],
    )
    assert selected["vocals"] is True
    assert selected["guitar"] is False
    assert selected["metronome"] is True


def test_resolve_speed_preset_unknown_and_auto_become_balanced():
    from audio_to_tab.hardware import HostProbe

    probe = HostProbe(cuda=False, mps=False, ram_gb=16.0)
    six = resolve_speed_preset("auto", probe, platform="darwin", model="htdemucs_6s")
    four = resolve_speed_preset("auto", probe, platform="darwin", model="htdemucs")
    faster = resolve_speed_preset("faster", probe, platform="darwin", model="htdemucs_6s")
    assert six["id"] == "balanced"
    assert six["quality"] == "balanced"
    assert four["id"] == "balanced"
    assert four["quality"] == "balanced"
    assert faster["quality"] == "fast"


def test_estimate_job_seconds_two_pass_is_double_separate():
    stages = isolation_stages_for_job(expects_guitar=True)
    one, _ = estimate_job_seconds(
        audio_duration_sec=60.0,
        quality="fast",
        device="cpu",
        stages=stages,
    )
    two, _ = estimate_job_seconds(
        audio_duration_sec=60.0,
        quality="fast",
        device="cpu",
        stages=stages,
        two_pass=True,
    )
    assert one is not None and two is not None
    assert two == pytest.approx(one * 2.0)


def test_estimate_roformer_ignores_demucs_quality():
    stages = isolation_stages_for_job(expects_guitar=True)
    fast, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="fast",
        device="cpu",
        stages=stages,
        model="bs_roformer_sw",
    )
    balanced, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="balanced",
        device="cpu",
        stages=stages,
        model="bs_roformer_sw",
    )
    assert fast is not None and balanced is not None
    assert fast == pytest.approx(balanced)
    assert estimated_stage_seconds("separate", stages, fast) == pytest.approx(
        100.0 * 2.2, rel=0.02
    )


def test_estimate_roformer_scales_by_cores_and_device():
    stages = isolation_stages_for_job(expects_guitar=True)
    m1, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="balanced",
        device="cpu",
        stages=stages,
        model="bs_roformer_sw",
        cpu_threads=4,
    )
    assert estimated_stage_seconds("separate", stages, m1) == pytest.approx(
        100.0 * 2.2 * (8.0 / 4.0), rel=0.02
    )
    mps, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="balanced",
        device="mps",
        stages=stages,
        model="bs_roformer_sw",
    )
    assert estimated_stage_seconds("separate", stages, mps) == pytest.approx(
        100.0 * 1.0, rel=0.02
    )


def test_estimate_roformer_refine_adds_a_pass():
    stages = isolation_stages_for_job(expects_guitar=True)
    plain, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="balanced",
        device="cpu",
        stages=stages,
        model="bs_roformer_sw",
    )
    refined, _ = estimate_job_seconds(
        audio_duration_sec=100.0,
        quality="balanced",
        device="cpu",
        stages=stages,
        model="bs_roformer_sw",
        guitar_refine=True,
    )
    assert plain is not None and refined is not None
    assert refined > plain
    assert estimated_stage_seconds("separate", stages, refined) == pytest.approx(
        100.0 * (2.2 + 1.0), rel=0.02
    )


def test_resolve_speed_preset_faster():
    resolved = resolve_speed_preset("faster")
    assert resolved["quality"] == "fast"
    assert resolved["device"] == "cpu"
    assert resolved["id"] == "faster"


def test_resolve_speed_preset_balanced():
    resolved = resolve_speed_preset("balanced")
    assert resolved["quality"] == "balanced"
    assert resolved["device"] == "cpu"


def test_resolve_speed_preset_best():
    resolved = resolve_speed_preset("best")
    assert resolved["quality"] == "high"
    assert resolved["device"] == "cpu"


def test_resolve_speed_preset_unknown_falls_back():
    resolved = resolve_speed_preset("unknown")
    assert resolved["id"] == DEFAULT_SPEED_PRESET


def test_default_speed_preset_is_balanced():
    assert DEFAULT_SPEED_PRESET == "balanced"
    assert "auto" not in SPEED_PRESETS
    assert "balanced" in SPEED_PRESETS


def test_resolve_speed_preset_balanced_uses_cuda_probe_on_windows():
    from audio_to_tab.hardware import HostProbe

    probe = HostProbe(cuda=True, mps=False, ram_gb=24.0)
    resolved = resolve_speed_preset("balanced", probe, platform="win32")
    assert resolved["id"] == "balanced"
    assert resolved["device"] == "cuda"
    assert resolved["quality"] == "balanced"


def test_resolve_speed_preset_best_uses_cuda_probe_on_windows():
    from audio_to_tab.hardware import HostProbe

    probe = HostProbe(cuda=True, mps=False, ram_gb=24.0)
    resolved = resolve_speed_preset("best", probe, platform="win32")
    assert resolved["device"] == "cuda"
    assert resolved["quality"] == "high"


def test_default_region_end():
    assert default_region_end(120.0) == 30.0
    assert default_region_end(10.0) == 10.0
    assert default_region_end(120.0, min_length=90) == 90.0
    assert default_region_end(60.0, min_length=90) == 60.0
    assert default_region_end(90.0, min_length=LITE_MAX_DURATION_SEC) == 90.0


def test_clamp_lite_clip():
    start, length, clamped = clamp_lite_clip(0.0, None, 200.0)
    assert start == 0.0
    assert length == LITE_MAX_DURATION_SEC
    assert clamped is True

    start, length, clamped = clamp_lite_clip(0.0, None, 60.0)
    assert length == 60.0
    assert clamped is False

    start, length, clamped = clamp_lite_clip(10.0, 45.0, 200.0)
    assert start == 10.0
    assert length == 45.0
    assert clamped is False

    start, length, clamped = clamp_lite_clip(0.0, 200.0, 200.0)
    assert length == LITE_MAX_DURATION_SEC
    assert clamped is True

    start, length, clamped = clamp_lite_clip(0.0, None, None)
    assert length == LITE_MAX_DURATION_SEC
    assert clamped is False


def test_clamp_region_bounds():
    start, end = clamp_region_bounds(0, 3, 120.0, min_length=5.0)
    assert end - start >= 5.0


def test_stem_label_for_id():
    assert stem_label_for_id("other::guitar") == "Guitar (from Other)"
    assert stem_label_for_id("guitar") == "Guitar"
    assert stem_label_for_id("lead_guitar") == "Lead Guitar"


def _write_wav(path: Path, amplitude: float):
    t = np.linspace(0, 0.5, 22050, endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), tone, 44100)


def test_children_from_outputs_filters_silent(tmp_path):
    _write_wav(tmp_path / "guitar.wav", amplitude=0.5)
    _write_wav(tmp_path / "empty.wav", amplitude=0.0)
    out = children_from_outputs(tmp_path)
    assert set(out.keys()) == {"guitar"}
    assert out["guitar"] == tmp_path / "guitar.wav"


def test_children_from_outputs_skips_diagnostics(tmp_path):
    _write_wav(tmp_path / "guitar.wav", amplitude=0.5)
    _write_wav(tmp_path / "guitar_diagnostic.wav", amplitude=0.5)
    out = children_from_outputs(tmp_path)
    assert set(out.keys()) == {"guitar"}


def test_children_from_outputs_skips_metronome(tmp_path):
    _write_wav(tmp_path / "guitar.wav", amplitude=0.5)
    _write_wav(tmp_path / "metronome.wav", amplitude=0.5)
    out = children_from_outputs(tmp_path)
    assert set(out.keys()) == {"guitar"}


def test_children_from_outputs_empty_or_missing(tmp_path):
    assert children_from_outputs(tmp_path / "nope") == {}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert children_from_outputs(empty) == {}


def test_merge_reseparate_removes_source_and_adds_children():
    artifacts = {"other": Path("/a/other.wav"), "vocals": Path("/a/vocals.wav")}
    children = {"other::guitar": Path("/a/other::guitar.wav"), "other::synth": Path("/a/other::synth.wav")}
    merged = merge_reseparate(artifacts, "other", children)
    assert artifacts == {"other": Path("/a/other.wav"), "vocals": Path("/a/vocals.wav")}
    assert "other" not in merged
    assert merged["other::guitar"] == Path("/a/other::guitar.wav")
    assert merged["other::synth"] == Path("/a/other::synth.wav")
    assert merged["vocals"] == Path("/a/vocals.wav")


def test_seed_metronome_widgets_uses_diagnostics_render():
    from ui.isolate_state import (
        ISOLATE_METRO_ACCENT_KEY,
        ISOLATE_METRO_APPLIED_KEY,
        ISOLATE_METRO_RATE_KEY,
        ISOLATE_METRO_SOUND_KEY,
        seed_metronome_widgets_for_run,
    )

    session: dict = {}
    seed_metronome_widgets_for_run(
        session,
        "/runs/song-a",
        {"render": {"accent": False, "rate": 2.0, "sound": "wood"}},
    )
    assert session[ISOLATE_METRO_ACCENT_KEY] is False
    assert session[ISOLATE_METRO_RATE_KEY] == 2.0
    assert session[ISOLATE_METRO_SOUND_KEY] == "wood"
    assert session[ISOLATE_METRO_APPLIED_KEY]["sound"] == "wood"
    seed_metronome_widgets_for_run(
        session,
        "/runs/song-a",
        {"render": {"accent": True, "rate": 1.0, "sound": "classic"}},
    )
    assert session[ISOLATE_METRO_SOUND_KEY] == "wood"


def test_seed_metronome_widgets_without_render_keeps_prefs():
    from ui.isolate_state import (
        ISOLATE_METRO_ACCENT_KEY,
        ISOLATE_METRO_APPLIED_KEY,
        ISOLATE_METRO_SOUND_KEY,
        seed_metronome_widgets_for_run,
    )

    session = {ISOLATE_METRO_ACCENT_KEY: False, ISOLATE_METRO_SOUND_KEY: "soft"}
    seed_metronome_widgets_for_run(session, "/runs/old", {})
    assert session[ISOLATE_METRO_SOUND_KEY] == "soft"
    assert session[ISOLATE_METRO_ACCENT_KEY] is False
    assert ISOLATE_METRO_APPLIED_KEY not in session


def test_queue_delete_clears_loaded_mix_and_normalized_path(tmp_path: Path):
    from ui.isolate_state import ISOLATE_SKIP_REHYDRATE_KEY, OPEN_MIX_TABS_KEY
    from ui.pages.isolate import accept_confirm_click, detach_deleted_mix_runs

    run = tmp_path / "Brianstorm"
    run.mkdir()
    run_s = str(run)
    aliased = str(run / ".." / run.name)
    session = {
        OPEN_MIX_TABS_KEY: [run_s],
        "isolate_run_dir": run_s,
        "isolate_artifacts": {"vocals": str(run / "vocals.wav")},
        "isolate_viewing_run_dir": run_s,
    }
    detach_deleted_mix_runs(session, [aliased])
    assert session.get(OPEN_MIX_TABS_KEY) == []
    assert "isolate_artifacts" not in session
    assert "isolate_run_dir" not in session
    assert session.get(ISOLATE_SKIP_REHYDRATE_KEY) is True

    flag = "delete__confirm"
    armed = {flag: True}
    assert accept_confirm_click(armed, flag, False) is False
    assert flag in armed
    assert accept_confirm_click(armed, flag, True) is True
    assert flag not in armed


def test_failed_delete_run_does_not_detach_mix(tmp_path: Path, monkeypatch):
    import streamlit as st
    from ui.pages import isolate as page

    run = tmp_path / "outside" / "run"
    run.mkdir(parents=True)
    calls: list = []
    monkeypatch.setattr(page, "detach_deleted_mix_runs", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(page, "_persist_isolate_ui_state", lambda: None)
    monkeypatch.setattr(
        page,
        "delete_finished_job",
        lambda job: {"ok": False, "run_dir": str(run)},
    )
    page._execute_pending_queue_delete(
        [{"id": "x", "status": "succeeded", "run_dir": str(run)}]
    )
    assert calls == []
    assert st.session_state["isolate_flash"] == page._DELETE_FAILED_FLASH



def test_default_isolate_device_prefers_recommendation_over_position(monkeypatch):
    from audio_to_tab.hardware import HostProbe

    monkeypatch.setenv("AUDIO_TOOLS_EDITION", "both")
    cpu_only = HostProbe(cuda=False, mps=False, ram_gb=12.0)
    assert (
        default_isolate_device(cpu_only, ["cuda", "cpu"], None, platform="win32")
        == "cpu"
    )


def test_default_isolate_device_keeps_persisted_choice(monkeypatch):
    from audio_to_tab.hardware import HostProbe

    monkeypatch.setenv("AUDIO_TOOLS_EDITION", "both")
    cpu_only = HostProbe(cuda=False, mps=False, ram_gb=12.0)
    assert (
        default_isolate_device(cpu_only, ["cuda", "cpu"], "cpu", platform="win32")
        == "cpu"
    )


def test_default_isolate_device_respects_stale_persisted_choice(monkeypatch):
    """A persisted device no longer offered must be replaced, not kept."""
    from audio_to_tab.hardware import HostProbe

    monkeypatch.setenv("AUDIO_TOOLS_EDITION", "both")
    cpu_only = HostProbe(cuda=False, mps=False, ram_gb=12.0)
    assert (
        default_isolate_device(cpu_only, ["cpu"], "cuda", platform="win32") == "cpu"
    )


def test_default_isolate_device_with_nvidia_prefers_cuda(monkeypatch):
    from audio_to_tab.hardware import HostProbe

    monkeypatch.setenv("AUDIO_TOOLS_EDITION", "both")
    nvidia = HostProbe(cuda=True, mps=False, ram_gb=24.0)
    assert (
        default_isolate_device(nvidia, ["cuda", "cpu"], None, platform="win32")
        == "cuda"
    )


def test_default_isolate_device_empty_options_returns_current():
    assert default_isolate_device(None, [], None) is None


def test_save_theme_state_round_trips_preset_mode_and_tokens(tmp_path, monkeypatch):
    import ui.common as common
    import ui.isolate_state as state

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    session = {
        state.THEME_PRESET_KEY: "custom",
        state.THEME_MODE_KEY: "light",
        state.THEME_CUSTOM_TOKENS_KEY: {"--custom-bg-deep": "#111111"},
    }
    state.save_theme_state(session)
    loaded: dict = {}
    state.load_theme_state(loaded)
    assert loaded[state.THEME_PRESET_KEY] == "custom"
    assert loaded[state.THEME_MODE_KEY] == "light"
    assert loaded[state.THEME_CUSTOM_TOKENS_KEY]["--custom-bg-deep"] == "#111111"


def test_load_theme_state_replaces_retired_preset_with_tyrian(tmp_path, monkeypatch):
    import json

    import ui.common as common
    import ui.isolate_state as state

    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "DATA_DIR", tmp_path)
    (tmp_path / "theme_config.json").write_text(
        json.dumps({"preset": "forest-dark", "mode": "system", "custom_tokens": {}}),
        encoding="utf-8",
    )
    loaded: dict = {}
    state.load_theme_state(loaded)
    assert loaded[state.THEME_PRESET_KEY] == "tyrian"
    saved = json.loads((tmp_path / "theme_config.json").read_text(encoding="utf-8"))
    assert saved["preset"] == "tyrian"
    assert saved["mode"] == "system"


def test_light_and_dark_themes_use_the_same_two_hues():
    import ui.isolate_state as state

    dark = state.build_theme_css_vars({state.THEME_MODE_KEY: "dark", state.THEME_PRESET_KEY: "tyrian"})
    light = state.build_theme_css_vars({state.THEME_MODE_KEY: "light", state.THEME_PRESET_KEY: "tyrian"})
    assert "color-scheme: dark" in dark
    assert "color-scheme: light" in light
    assert "#192542" in dark and "#f6ad49" in dark
    assert "#FFF6DC" in light and "#FFFFFF" in light
    assert "#192542" in light and "#f6ad49" in light
    for css in (dark, light):
        assert 'button[kind="primary"]:disabled' in css
        assert '[data-testid="stBaseButton-primary"]:disabled' in css
        assert "opacity: 0.45" in css
        assert "cursor: not-allowed" in css


def test_system_theme_follows_os(monkeypatch):
    import ui.isolate_state as state

    monkeypatch.setattr(state, "_os_theme_mode", lambda: "light")
    assert state.get_effective_theme_mode({state.THEME_MODE_KEY: "system"}) == "light"
    assert state.get_effective_theme_mode({state.THEME_MODE_KEY: "dark"}) == "dark"


def test_theme_widgets_use_zero_arg_persist_callback():
    root = Path(__file__).resolve().parents[1]
    app = (root / "ui" / "app.py").read_text(encoding="utf-8")
    page = (root / "ui" / "pages" / "settings.py").read_text(encoding="utf-8")
    assert "def _persist_theme()" in page
    assert "save_theme_state(st.session_state)" in page
    assert "on_change=_persist_theme" in page
    assert "on_change=save_theme_state" not in page
    assert "on_change=save_theme_state" not in app
    assert 'subheader("🎨 Theme")' not in app
    assert "_render_custom_colors" not in page
    assert "Use default colors" not in page
    assert "The palette is dark Tyrian blue and yellow-orange." in page


def _open_runs(tmp_path: Path, count: int) -> tuple[dict, list[str]]:
    from ui.isolate_state import OPEN_MIX_TABS_KEY

    dirs = []
    for i in range(count):
        d = tmp_path / f"run{i}"
        d.mkdir()
        dirs.append(str(d))
    return {OPEN_MIX_TABS_KEY: list(dirs)}, dirs


def test_move_open_mix_tab_reorders_and_clamps(tmp_path: Path):
    from ui.isolate_state import OPEN_MIX_TABS_KEY, move_open_mix_tab

    session, d = _open_runs(tmp_path, 4)
    assert move_open_mix_tab(session, d[0], 2) is True
    assert session[OPEN_MIX_TABS_KEY] == [d[1], d[2], d[0], d[3]]
    # Drop past the end means last; before the start means first.
    assert move_open_mix_tab(session, d[1], 99) is True
    assert session[OPEN_MIX_TABS_KEY] == [d[2], d[0], d[3], d[1]]
    assert move_open_mix_tab(session, d[1], -5) is True
    assert session[OPEN_MIX_TABS_KEY] == [d[1], d[2], d[0], d[3]]
    # Same slot, unknown id, and junk targets are no-ops that keep the order.
    before = list(session[OPEN_MIX_TABS_KEY])
    assert move_open_mix_tab(session, d[1], 0) is False
    assert move_open_mix_tab(session, "/nope", 1) is False
    assert move_open_mix_tab(session, d[2], "x") is False
    assert move_open_mix_tab(session, d[2], None) is False
    assert session[OPEN_MIX_TABS_KEY] == before
    # Order survives the normalizer the strip runs on every paint.
    from ui.isolate_state import open_mix_tabs_for_session

    assert open_mix_tabs_for_session(session, library_dirs=d) == before


def test_move_open_mix_tab_moves_drafts_like_mixes(tmp_path: Path):
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        is_new_draft_tab,
        move_open_mix_tab,
        open_new_draft_tab,
    )

    session, d = _open_runs(tmp_path, 2)
    draft = open_new_draft_tab(session, fresh=False)
    assert is_new_draft_tab(draft)
    assert session[OPEN_MIX_TABS_KEY] == [d[0], d[1], draft]
    assert move_open_mix_tab(session, draft, 0) is True
    assert session[OPEN_MIX_TABS_KEY] == [draft, d[0], d[1]]


def test_close_other_and_right_keep_anchor_and_spare_busy_tabs(tmp_path: Path):
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        OPEN_MIX_TABS_KEY,
        close_other_open_mix_tabs,
        close_right_open_mix_tabs,
        open_new_draft_tab,
    )

    session, d = _open_runs(tmp_path, 4)
    draft = open_new_draft_tab(session, fresh=False)
    assert draft in session[DRAFT_SOURCES_KEY]

    closed, spared = close_right_open_mix_tabs(session, d[1], protect=[d[3]])
    assert closed == [d[2], draft]
    assert spared == [d[3]]
    assert session[OPEN_MIX_TABS_KEY] == [d[0], d[1], d[3]]
    # A closed draft drops its saved source; closing never touches disk.
    assert draft not in session[DRAFT_SOURCES_KEY]
    assert all(Path(p).is_dir() for p in d)

    closed, spared = close_other_open_mix_tabs(session, d[1], protect=[d[3]])
    assert closed == [d[0]]
    assert spared == [d[3]]
    assert session[OPEN_MIX_TABS_KEY] == [d[1], d[3]]

    # Rightmost anchor, unknown anchor: nothing to close, nothing changes.
    assert close_right_open_mix_tabs(session, d[3]) == ([], [])
    assert close_other_open_mix_tabs(session, "/nope") == ([], [])
    assert session[OPEN_MIX_TABS_KEY] == [d[1], d[3]]

    closed, spared = close_other_open_mix_tabs(session, d[3])
    assert closed == [d[1]] and spared == []
    assert session[OPEN_MIX_TABS_KEY] == [d[3]]


def test_duplicate_draft_copies_source_and_sits_next_to_original(tmp_path: Path):
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        draft_tab_title,
        duplicate_open_mix_tab,
        is_new_draft_tab,
        open_new_draft_tab,
    )

    session, d = _open_runs(tmp_path, 2)
    song = tmp_path / "song_a.wav"
    song.write_bytes(b"RIFF")
    first = open_new_draft_tab(session, fresh=False)
    session["isolate_pending_audio_path"] = str(song)
    session["isolate_pending_fp"] = "upload:song_a"
    session["isolate_output_name"] = "Song A"
    # Another mix opened after the draft, so "next to" is not "at the end".
    session[OPEN_MIX_TABS_KEY].append(str(tmp_path / "run_late"))
    (tmp_path / "run_late").mkdir()

    copy = duplicate_open_mix_tab(session, first)
    assert copy and is_new_draft_tab(copy) and copy != first
    tabs = session[OPEN_MIX_TABS_KEY]
    assert tabs.index(copy) == tabs.index(first) + 1
    assert session[SHELL_TAB_KEY] == copy
    assert draft_tab_title(session, copy) == "Song A copy"
    assert draft_tab_title(session, first) == "Song A"
    assert session["isolate_pending_audio_path"] == str(song)
    # The original keeps its own saved source.
    assert session[DRAFT_SOURCES_KEY][first]["isolate_pending_audio_path"] == str(song)


def test_duplicate_youtube_draft_keeps_url_but_not_staging_file(tmp_path: Path):
    from ui.isolate_state import (
        DRAFT_SOURCES_KEY,
        duplicate_open_mix_tab,
        open_new_draft_tab,
    )

    session: dict = {}
    first = open_new_draft_tab(session, fresh=False)
    staged = tmp_path / "yt.wav"
    staged.write_bytes(b"RIFF")
    session["isolate_pending_audio_path"] = str(staged)
    session["isolate_pending_fp"] = "youtube:https://youtu.be/abc"
    session["isolate_upload_fp"] = "youtube:https://youtu.be/abc"
    session["isolate_youtube_url"] = "https://youtu.be/abc"

    copy = duplicate_open_mix_tab(session, first)
    saved = session[DRAFT_SOURCES_KEY][copy]
    assert saved["isolate_youtube_url"] == "https://youtu.be/abc"
    for key in ("isolate_pending_audio_path", "isolate_pending_fp", "isolate_upload_fp"):
        assert key not in saved
    assert not session.get("isolate_pending_audio_path")
    # The original draft still owns its staging file.
    assert session[DRAFT_SOURCES_KEY][first]["isolate_pending_audio_path"] == str(staged)


def test_duplicate_mix_opens_draft_with_same_song_and_never_touches_the_run(tmp_path: Path):
    from ui.isolate_state import (
        MIX_TAB_NOTICE_KEY,
        OPEN_MIX_TABS_KEY,
        SHELL_TAB_KEY,
        draft_tab_title,
        duplicate_open_mix_tab,
        is_new_draft_tab,
    )

    session, d = _open_runs(tmp_path, 2)
    stem = Path(d[0]) / "vocals.wav"
    stem.write_bytes(b"RIFF")
    source = tmp_path / "original.wav"
    source.write_bytes(b"RIFF")

    copy = duplicate_open_mix_tab(
        session, d[0], title="My Song", source_audio_path=str(source)
    )
    assert copy and is_new_draft_tab(copy)
    assert session[OPEN_MIX_TABS_KEY] == [d[0], copy, d[1]]
    assert session[SHELL_TAB_KEY] == copy
    assert session["carry_over_audio_path"] == str(source)
    assert draft_tab_title(session, copy) == "My Song copy"
    assert MIX_TAB_NOTICE_KEY not in session
    assert stem.is_file() and source.is_file()
    assert Path(d[0]).is_dir()


def test_duplicate_mix_without_source_says_so_and_unknown_tab_is_noop(tmp_path: Path):
    from ui.isolate_state import (
        MIX_TAB_NOTICE_KEY,
        OPEN_MIX_TABS_KEY,
        duplicate_open_mix_tab,
    )

    session, d = _open_runs(tmp_path, 1)
    copy = duplicate_open_mix_tab(
        session, d[0], title="Gone", source_audio_path=str(tmp_path / "missing.wav")
    )
    assert copy
    assert "no longer on disk" in session[MIX_TAB_NOTICE_KEY]
    assert not session.get("carry_over_audio_path")

    before = list(session[OPEN_MIX_TABS_KEY])
    assert duplicate_open_mix_tab(session, "/nope") is None
    assert session[OPEN_MIX_TABS_KEY] == before


def test_duplicate_at_the_cap_keeps_the_original_tab(tmp_path: Path):
    from ui.isolate_state import (
        OPEN_MIX_TABS_KEY,
        OPEN_MIX_TABS_MAX,
        duplicate_open_mix_tab,
    )

    session, d = _open_runs(tmp_path, OPEN_MIX_TABS_MAX)
    copy = duplicate_open_mix_tab(session, d[0], title="Oldest")
    tabs = session[OPEN_MIX_TABS_KEY]
    assert len(tabs) == OPEN_MIX_TABS_MAX
    assert d[0] in tabs and copy in tabs


def test_consume_mix_tab_event_accepts_reorder_and_rejects_removed_toolbar_actions():
    from ui.isolate_state import consume_mix_tab_event, mix_tab_event_index

    last, got, tab_id = consume_mix_tab_event(
        {"action": "reorder", "id": "/runs/a", "to": 1, "seq": 1}, 0
    )
    assert got == "reorder" and tab_id == "/runs/a" and last == 1
    for action in ("duplicate", "close_others", "close_right"):
        last, got, tab_id = consume_mix_tab_event(
            {"action": action, "id": "/runs/a", "seq": 2}, 1
        )
        assert got is None
        assert not tab_id
    assert mix_tab_event_index({"action": "reorder", "id": "/a", "to": 3, "seq": 1}) == 3
    assert mix_tab_event_index({"to": "2"}) == 2
    assert mix_tab_event_index({"to": True}) is None
    assert mix_tab_event_index({"to": "x"}) is None
    assert mix_tab_event_index({"action": "reorder"}) is None
    assert mix_tab_event_index(None) is None

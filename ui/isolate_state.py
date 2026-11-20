"""Pure helpers for isolate page upload/result state and separation progress."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from audio_to_tab.hardware import HostProbe, desktop_recommend, resolve_desktop_speed
from audio_to_tab.ingest import is_direct_media_url, is_youtube_url
from audio_to_tab.isolate import (
    DEMUCS_MODELS,
    FOLD_OTHER_MODES,
    QUALITY_OVERLAP,
    QUALITY_SHIFTS,
    effective_isolation_quality,
)
from audio_to_tab.metronome import (
    MetronomeRenderOptions,
    coerce_metronome_render_options,
)
from audio_to_tab.mixer import stem_display_name, stem_energy_db
from ui.common import DATA_DIR, delete_run
from ui.stem_icons import outcome_icon_markdown, stem_icon_markdown
from ui.theme_presets import (
    ACCENT_TOKENS,
    BASE_TOKENS_DARK,
    BASE_TOKENS_LIGHT,
    SIGNAL_TOKENS,
    THEME_PRESETS,
    TYRIAN_BASE_DARK,
    TYRIAN_BASE_LIGHT,
    TYRIAN_BLUE,
)

logger = logging.getLogger(__name__)

ISOLATION_STAGE_ORDER = (
    "ingest",
    "separate",
    "collect",
    "bass_bleed",
    "presence",
    "analysis",
    "tab",
    "done",
)

STAGE_WEIGHTS: dict[str, float] = {
    "ingest": 0.05,
    "separate": 0.80,
    "collect": 0.05,
    "guitar_refine": 0.15,
    "bass_bleed": 0.05,
    "presence": 0.03,
    "analysis": 0.03,
    "tab": 0.06,
    "done": 0.02,
}

# One-line user labels. Behind-the-scenes stages reuse "Separate tracks".
STAGE_CHECKLIST_LABELS: dict[str, str] = {
    "ingest": "Prepare audio",
    "separate": "Separate tracks",
    "collect": "Separate tracks",
    "guitar_refine": "Refine guitar",
    "bass_bleed": "Separate tracks",
    "presence": "Separate tracks",
    "analysis": "Analyse song",
    "tab": "Transcribe tab",
    "done": "Finish",
}

_GUITAR_ONLY_STAGES = frozenset({"bass_bleed"})

# Wall-clock vs audio length for the Demucs `separate` stage at quality=fast.
# Fast CPU is roughly 1× the clip (docs: about as long as the song).
_DEVICE_REALTIME = {"cpu": 1.0, "cuda": 0.25, "mps": 0.8}
# Extra work from Demucs --shifts (QUALITY_SHIFTS in isolate.py).
_QUALITY_TIME_FACTOR = {"fast": 1.0, "balanced": 2.0, "high": 4.0, "extreme": 6.0}
# BS-RoFormer-SW is ~2.2× realtime on an 8 P-core Apple Silicon CPU (measured
# 2.13× on an M2 Pro); MPS roughly halves that. The Demucs speed/quality preset
# does NOT affect RoFormer, so these estimates ignore _QUALITY_TIME_FACTOR.
_REFERENCE_P_CORES = 8.0
_ROFORMER_REALTIME = {"cpu": 2.2, "cuda": 0.12, "mps": 1.0}
_ROFORMER_REFINE_REALTIME = {"cpu": 1.0, "cuda": 0.06, "mps": 0.45}
_INTRA_STAGE_CAP = 0.98
_GUITAR_MODEL_MARKERS = ("6s", "guitar", "roformer")

# Labeled track picks → one resolved pipeline (see resolve_track_selection).
TRACK_OPTIONS: dict[str, dict[str, Any]] = {
    "vocals_demucs": {
        "label": "Vocals (Demucs)",
        "stem": "vocals",
    },
    "drums_demucs": {
        "label": "Drums (Demucs)",
        "stem": "drums",
    },
    "bass_demucs": {
        "label": "Bass (Demucs)",
        "stem": "bass",
    },
    "other_demucs": {
        "label": "Other (Demucs)",
        "stem": "other",
    },
    "piano_demucs": {
        "label": "Piano (Demucs 6-stem). Heavy bleed.",
        "stem": "piano",
    },
    "guitar_demucs_6s": {
        "label": "Guitar (Demucs 6-stem, weaker). Other instruments still bleed in.",
        "stem": "guitar",
    },
    "guitar_roformer": {
        "label": "Guitar (BS-RoFormer, better, slower). Some bleed remains.",
        "stem": "guitar",
    },
    "guitar_roformer_refine": {
        "label": "Guitar (BS-RoFormer + MelBand refine, best, slowest). Some bleed remains.",
        "stem": "guitar",
    },
    "vocals_instrumental_demucs": {
        "label": "Vocals & instrumental (Demucs 2-stem)",
        "stem": "vocals",
    },
}

GUITAR_TRACK_OPTION_IDS = frozenset(
    {"guitar_demucs_6s", "guitar_roformer", "guitar_roformer_refine"}
)
GUITAR_TRACK_RADIO_ORDER = (
    "guitar_demucs_6s",
    "guitar_roformer",
    "guitar_roformer_refine",
)
DEMUCS_STEM_CHECKBOX_IDS = (
    "vocals_demucs",
    "drums_demucs",
    "bass_demucs",
    "piano_demucs",
    "other_demucs",
)
VOCALS_INSTRUMENTAL_OPTION_ID = "vocals_instrumental_demucs"
DEFAULT_TRACK_OPTIONS = ("vocals_demucs", "guitar_demucs_6s")

ROFORMER_DOWNLOAD_CAVEAT = (
    "Downloads ~700 MB BS-RoFormer-SW weights on first use, then a guitar "
    "specialist (~45 MB). Much slower on CPU. Residual bleed remains. "
    "Community weights have no stated license. Use accordingly."
)
ROFORMER_MIXED_STEMS_NOTE = (
    "All stems are separated in one BS-RoFormer pass; unselected stems are discarded."
)
TRACKS_PICKER_HELP = (
    "Pick the stems you want. Guitar can use Demucs (faster) or BS-RoFormer "
    "(cleaner on a GPU, much slower on CPU). Neither is bleed-free, and piano "
    "from Demucs is unreliable."
)

ROFORMER_GUITAR_OPTION_IDS = frozenset({"guitar_roformer", "guitar_roformer_refine"})
ROFORMER_MODEL_IDS = frozenset({"bs_roformer_sw", "melband_roformer_guitar"})
ROFORMER_BACKEND_UI_HINT = (
    "BS-RoFormer guitar options need the optional runtime. "
    "In your app Python environment run: `pip install bs-roformer-infer`."
)
def roformer_speed_note(*, platform: str | None = None) -> str:
    """Why the Speed control does nothing for RoFormer, in host-correct terms.

    The GPU escape hatch differs per platform, so naming the wrong one (this note
    used to say "Apple GPU (MPS)" on every host) sends Windows users looking for
    a setting that isn't there.
    """
    plat = sys.platform if platform is None else platform
    if plat == "darwin":
        gpu_hint = "pick Apple GPU (MPS) in Advanced when this Mac has ≥12 GB RAM"
    elif plat.startswith("win"):
        gpu_hint = "pick NVIDIA GPU (CUDA) in Advanced if this PC has one"
    else:
        gpu_hint = "pick a CUDA GPU in Advanced if this machine has one"
    return (
        "Speed only tunes Demucs (--shifts). BS-RoFormer always runs its full "
        "transformer pass, so Faster/Balanced/Best do not change its time. Expect "
        "roughly 2× the song length on CPU. To go faster: shorten the Section, "
        f"use the Demucs guitar option, or {gpu_hint}."
    )


def guitar_track_radio_ids(*, roformer_available: bool) -> tuple[str, ...]:
    """Radio keys for the Guitar row: ``none`` plus engines available on this host."""
    if roformer_available:
        return ("none", *GUITAR_TRACK_RADIO_ORDER)
    return ("none", "guitar_demucs_6s")


def normalize_guitar_track_selection(guitar: str, *, roformer_available: bool) -> str:
    """Map persisted RoFormer picks to Demucs when the backend is missing."""
    pick = (guitar or "none").strip()
    if pick in ROFORMER_GUITAR_OPTION_IDS and not roformer_available:
        return "guitar_demucs_6s"
    if pick == "none" or pick in GUITAR_TRACK_OPTION_IDS:
        return pick
    return "none"


def default_guitar_track_option(
    *,
    roformer_available: bool,
    prefer_roformer: bool = True,
) -> str:
    """Preferred guitar engine for a fresh session / Lite outcome card.

    When ``prefer_roformer`` is False (Lite on CPU-only hosts), keep Demucs even
    if BS-RoFormer is installed. Pro seeding keeps ``prefer_roformer=True``.
    """
    if prefer_roformer and roformer_available:
        return "guitar_roformer"
    return "guitar_demucs_6s"


def promote_default_guitar_option(
    options: list[str] | tuple[str, ...],
    *,
    roformer_available: bool,
    prefer_roformer: bool = True,
) -> list[str]:
    """Swap the default Demucs guitar pick for BS-RoFormer when preferred and present.

    Used only when seeding UI state for a brand-new session; persisted user
    picks are never rewritten by this helper (``normalize_guitar_track_selection``
    is the downgrade path, this is the upgrade path).
    """
    promoted = default_guitar_track_option(
        roformer_available=roformer_available,
        prefer_roformer=prefer_roformer,
    )
    out = [str(x) for x in options]
    if "guitar_demucs_6s" in out and promoted == "guitar_roformer":
        out[out.index("guitar_demucs_6s")] = promoted
    return out


# Lite guitar-engine chooser. Options stay visible; RoFormer is disabled when
# the clip is longer than ``roformer_max_audio_sec`` (or length is unknown).
LITE_GUITAR_ENGINE_KEY = "isolate_lite_guitar_engine"
LITE_GUITAR_ENGINE_ORDER: tuple[str, ...] = (
    "guitar_demucs_6s",
    "guitar_roformer",
    "guitar_roformer_refine",
)
DEFAULT_LITE_GUITAR_ENGINE = "guitar_demucs_6s"
LITE_GUITAR_ENGINE_LABELS: dict[str, str] = {
    "guitar_demucs_6s": "Default (Demucs)",
    "guitar_roformer": "Better guitar (BS-RoFormer)",
    "guitar_roformer_refine": "Best / slower (BS-RoFormer + MelBand)",
}


def effective_clip_seconds(
    *,
    file_duration_sec: float | None,
    start_sec: float = 0.0,
    max_duration_sec: float | None,
) -> float | None:
    """Length that will actually be separated, or None if unknown."""
    if max_duration_sec is not None:
        try:
            return max(0.0, float(max_duration_sec))
        except (TypeError, ValueError):
            return None
    if file_duration_sec is None:
        return None
    try:
        return max(0.0, float(file_duration_sec) - float(start_sec or 0.0))
    except (TypeError, ValueError):
        return None


def lite_guitar_engine_allowed(
    engine_id: str,
    *,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
    roformer_available: bool,
) -> bool:
    """Whether Lite may run this guitar engine on the current clip."""
    pick = str(engine_id or "").strip()
    if pick == DEFAULT_LITE_GUITAR_ENGINE or pick == "guitar_demucs_6s":
        return True
    if pick not in ROFORMER_GUITAR_OPTION_IDS:
        return False
    if not roformer_available:
        return False
    if not duration_known or effective_sec is None:
        return False
    try:
        return float(effective_sec) <= float(cap_sec)
    except (TypeError, ValueError):
        return False


def lite_guitar_engine_disabled_reason(
    engine_id: str,
    *,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
    roformer_available: bool,
) -> str | None:
    """User-facing why a Lite engine button is disabled, or None if allowed."""
    if lite_guitar_engine_allowed(
        engine_id,
        effective_sec=effective_sec,
        cap_sec=cap_sec,
        duration_known=duration_known,
        roformer_available=roformer_available,
    ):
        return None
    pick = str(engine_id or "").strip()
    if pick not in ROFORMER_GUITAR_OPTION_IDS:
        return None
    if not roformer_available:
        return (
            "Needs the BS-RoFormer runtime "
            "(pip install bs-roformer-infer in your app environment)."
        )
    if not duration_known or effective_sec is None:
        return (
            "Available after the track length is known. "
            "Upload a file or download YouTube audio."
        )
    return (
        f"Needs a section ≤ {float(cap_sec):.0f} s on this PC "
        f"(this track is {float(effective_sec):.0f} s)."
    )


def resolve_lite_guitar_engine(
    selected: str | None,
    *,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
    roformer_available: bool,
) -> tuple[str, bool]:
    """Legal Lite guitar engine. Falls back to Demucs when the pick cannot run.

    Returns ``(engine_id, fell_back)``.
    """
    pick = str(selected or "").strip()
    if pick not in LITE_GUITAR_ENGINE_ORDER:
        pick = DEFAULT_LITE_GUITAR_ENGINE
    if lite_guitar_engine_allowed(
        pick,
        effective_sec=effective_sec,
        cap_sec=cap_sec,
        duration_known=duration_known,
        roformer_available=roformer_available,
    ):
        return pick, False
    return DEFAULT_LITE_GUITAR_ENGINE, pick != DEFAULT_LITE_GUITAR_ENGINE


def apply_lite_guitar_engine(
    option_ids: Iterable[str],
    engine_id: str,
) -> list[str]:
    """Swap any guitar track option for the Lite engine choice."""
    engine = str(engine_id or "").strip()
    if engine not in LITE_GUITAR_ENGINE_ORDER:
        engine = DEFAULT_LITE_GUITAR_ENGINE
    had_guitar = False
    out: list[str] = []
    for oid in option_ids or ():
        if str(oid) in GUITAR_TRACK_OPTION_IDS:
            had_guitar = True
            continue
        out.append(str(oid))
    if had_guitar:
        out.append(engine)
    return out


def lite_roformer_enqueue_block_reason(
    model: str,
    *,
    effective_sec: float | None,
    cap_sec: float,
    duration_known: bool,
    roformer_available: bool,
) -> str | None:
    """Why Lite must not start this RoFormer job, or None if it may run."""
    if not job_requires_roformer_backend(model):
        return None
    if lite_guitar_engine_allowed(
        "guitar_roformer",
        effective_sec=effective_sec,
        cap_sec=cap_sec,
        duration_known=duration_known,
        roformer_available=roformer_available,
    ):
        return None
    return lite_guitar_engine_disabled_reason(
        "guitar_roformer",
        effective_sec=effective_sec,
        cap_sec=cap_sec,
        duration_known=duration_known,
        roformer_available=roformer_available,
    )


# Lite goal picker. Outcomes are goals; guitar engine is a separate Lite chooser.
# Lite auto-picks speed/device. Pro exposes every stem/engine control.
OUTCOME_CARD_ORDER: tuple[str, ...] = ("band", "guitar", "karaoke", "vocals")
OUTCOME_CARDS: dict[str, dict[str, str | int]] = {
    "band": {
        "label": "Every part separately",
        "help": (
            "Vocals, drums, bass and guitar as separate tracks. Guitar can still "
            "pick up other instruments. Lite Faster is quicker than Balanced, not "
            "the same quality."
        ),
        "stems_line": "Vocals · Drums · Bass · Guitar",
        "n_tracks": 4,
    },
    "guitar": {
        "label": "Guitar on its own",
        "help": (
            "Just the guitar, using the strongest model this computer can run. "
            "Expect bleed from bass, drums, or vocals; lead and rhythm stay on one track."
        ),
        "stems_line": "Guitar",
        "n_tracks": 1,
    },
    "karaoke": {
        "label": "Vocals + backing track",
        "help": "Two tracks: the singer, and everything else. Quickest option.",
        "stems_line": "Vocals · Instrumental",
        "n_tracks": 2,
    },
    "vocals": {
        "label": "Vocals on its own",
        "help": "Just the singer.",
        "stems_line": "Vocals",
        "n_tracks": 1,
    },
}
DEFAULT_OUTCOME_CARD = "band"
OUTCOME_CARD_KEY = "isolate_outcome_card"
CUSTOM_OUTCOME_CARD = "custom"


def resolve_outcome_card(
    card_id: str,
    *,
    roformer_available: bool,
    prefer_roformer: bool = True,
    guitar_option: str | None = None,
) -> list[str]:
    """Track option ids behind a Lite outcome.

    ``guitar_option`` (Lite chooser) wins when set. Otherwise guitar follows
    ``prefer_roformer`` (Lite historically gated on accelerator; Pro prefers
    RoFormer when installed).
    """
    if guitar_option:
        guitar = normalize_guitar_track_selection(
            guitar_option, roformer_available=roformer_available
        )
        if guitar not in GUITAR_TRACK_OPTION_IDS:
            guitar = default_guitar_track_option(
                roformer_available=roformer_available,
                prefer_roformer=prefer_roformer,
            )
    else:
        guitar = default_guitar_track_option(
            roformer_available=roformer_available,
            prefer_roformer=prefer_roformer,
        )
    card = card_id if card_id in OUTCOME_CARDS else DEFAULT_OUTCOME_CARD
    if card == "karaoke":
        return [VOCALS_INSTRUMENTAL_OPTION_ID]
    if card == "guitar":
        return [guitar]
    if card == "vocals":
        return ["vocals_demucs"]
    return ["vocals_demucs", "drums_demucs", "bass_demucs", guitar]


def outcome_card_for_options(
    option_ids: Iterable[str],
    *,
    roformer_available: bool,
    prefer_roformer: bool = True,
    guitar_option: str | None = None,
) -> str | None:
    """Which outcome describes this exact selection, or None for a custom one.

    Used when switching Pro→Lite: without this, Lite would overwrite a deliberate
    Pro selection with its own default and quietly change what gets separated.
    """
    wanted = {str(x) for x in (option_ids or ())}
    if not wanted:
        return None
    for card in OUTCOME_CARD_ORDER:
        if wanted == set(
            resolve_outcome_card(
                card,
                roformer_available=roformer_available,
                prefer_roformer=prefer_roformer,
                guitar_option=guitar_option,
            )
        ):
            return card
    return None


def custom_stems_from_options(option_ids: Iterable[str]) -> list[str]:
    """Which custom-grid stems are on for this option list.

    Karaoke (vocals + instrumental) is a 2-stem path, not the six-stem grid.
    """
    ids = [str(x) for x in (option_ids or ())]
    if ids == [VOCALS_INSTRUMENTAL_OPTION_ID]:
        return []
    present: set[str] = set()
    for oid in ids:
        spec = TRACK_OPTIONS.get(oid)
        if not spec:
            continue
        stem = str(spec["stem"])
        if stem in CUSTOM_STEM_CHOICES:
            present.add(stem)
    return [stem for stem in CUSTOM_STEM_TILE_ORDER if stem in present]


def toggle_custom_stem_options(
    option_ids: Iterable[str],
    stem_id: str,
    *,
    roformer_available: bool,
    prefer_roformer: bool = True,
    guitar_option: str | None = None,
) -> list[str]:
    """Turn one custom-grid stem on or off. Exits karaoke 2-stem mode."""
    if stem_id not in CUSTOM_STEM_CHOICES:
        return [str(x) for x in (option_ids or ())]
    current = [str(x) for x in (option_ids or ())]
    if current == [VOCALS_INSTRUMENTAL_OPTION_ID]:
        current = []
    selected = set(custom_stems_from_options(current))
    if stem_id in selected:
        if stem_id == "guitar":
            current = [oid for oid in current if oid not in GUITAR_TRACK_OPTION_IDS]
        else:
            drop = STEM_TO_TRACK_OPTION[stem_id]
            current = [oid for oid in current if oid != drop]
    elif stem_id == "guitar":
        if not any(oid in GUITAR_TRACK_OPTION_IDS for oid in current):
            if guitar_option:
                pick = normalize_guitar_track_selection(
                    guitar_option, roformer_available=roformer_available
                )
                if pick not in GUITAR_TRACK_OPTION_IDS:
                    pick = default_guitar_track_option(
                        roformer_available=roformer_available,
                        prefer_roformer=prefer_roformer,
                    )
            else:
                pick = default_guitar_track_option(
                    roformer_available=roformer_available,
                    prefer_roformer=prefer_roformer,
                )
            current.append(pick)
    else:
        current.append(STEM_TO_TRACK_OPTION[stem_id])
    return current


def tracks_picker_help(*, roformer_available: bool) -> str:
    if roformer_available:
        return TRACKS_PICKER_HELP
    return (
        "Pick the stems you want. Guitar uses Demucs only until BS-RoFormer is installed "
        "(pip install bs-roformer-infer in your app environment)."
    )


def job_requires_roformer_backend(model: str) -> bool:
    return (model or "").strip().lower() in ROFORMER_MODEL_IDS

# Legacy preset ids → track option ids (session migration / tests only).
LEGACY_PRESET_TRACK_OPTIONS: dict[str, tuple[str, ...]] = {
    "full_band": (
        "vocals_demucs",
        "drums_demucs",
        "bass_demucs",
        "guitar_roformer",
    ),
    "best_guitar": (
        "vocals_demucs",
        "drums_demucs",
        "bass_demucs",
        "guitar_roformer_refine",
    ),
    "essential": (
        "vocals_demucs",
        "drums_demucs",
        "bass_demucs",
        "other_demucs",
    ),
    "vocals_music": (VOCALS_INSTRUMENTAL_OPTION_ID,),
}

# Internal migration only — not shown in UI.
SEPARATION_PRESETS: dict[str, dict[str, Any]] = {
    "full_band": {"label": "Full band"},
    "essential": {"label": "Essential tracks"},
    "vocals_music": {"label": "Vocals & music"},
    "custom": {"label": "Custom"},
}

DEFAULT_SEPARATION_PRESET = "full_band"

# Instruments a Custom run can ask for → the stem files that satisfy each pick.
CUSTOM_STEM_CHOICES: dict[str, str] = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other",
}
# Display order for the New-tab custom grid (not Moises’ extra locked row).
CUSTOM_STEM_TILE_ORDER: tuple[str, ...] = (
    "vocals",
    "guitar",
    "bass",
    "drums",
    "piano",
    "other",
)
STEM_TO_TRACK_OPTION: dict[str, str] = {
    "vocals": "vocals_demucs",
    "drums": "drums_demucs",
    "bass": "bass_demucs",
    "piano": "piano_demucs",
    "other": "other_demucs",
}

CUSTOM_STEM_OUTPUTS: dict[str, tuple[str, ...]] = {
    "vocals": ("vocals",),
    "drums": ("drums",),
    "bass": ("bass",),
    "guitar": ("guitar",),
    "piano": ("piano",),
    "other": ("other",),
}

DEFAULT_CUSTOM_STEMS = ("vocals", "guitar")

SPEED_PRESETS: dict[str, dict[str, Any]] = {
    "faster": {
        "label": "Faster",
        "quality": "fast",
        "device": "cpu",
        "help": "",
    },
    "balanced": {
        "label": "Balanced",
        "quality": "balanced",
        "device": "cpu",
        "help": "",
    },
    "best": {
        "label": "Best",
        "quality": "high",
        "device": "cpu",
        "help": "",
    },
}

DEFAULT_SPEED_PRESET = "balanced"


def resolve_isolate_user_id(session: MutableMapping[str, Any], stored: str | None) -> str:
    """Pick a library owner id without waiting on browser localStorage.

    Prefers an existing session id, then a stored browser id, else mints one.
    Never sleeps or reruns.
    """
    cached = session.get("isolate_user_id")
    if isinstance(cached, str) and cached.strip():
        return cached.strip()
    if isinstance(stored, str) and stored.strip():
        user_id = stored.strip()
        session["isolate_user_id"] = user_id
        return user_id
    user_id = uuid4().hex
    session["isolate_user_id"] = user_id
    return user_id


def resolve_speed_preset(
    preset_id: str,
    probe=None,
    *,
    platform: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Map a UI speed preset id to Demucs quality + device for this host.

    Unknown ids fall back to Balanced. Faster keeps ``fast`` quality even on
    ``htdemucs_6s``; other speeds floor 6-stem jobs to Balanced.
    """
    host = probe if probe is not None else HostProbe(cuda=False, mps=False, ram_gb=None)
    resolved = resolve_desktop_speed(preset_id, host, platform=platform)
    known = resolved["id"] if resolved["id"] in SPEED_PRESETS else DEFAULT_SPEED_PRESET
    label = SPEED_PRESETS[known]["label"]
    quality = resolved["quality"]
    if model:
        quality = effective_isolation_quality(
            quality, model=model, speed_id=resolved["id"]
        )
    return {
        "id": known,
        "label": label,
        "quality": quality,
        "device": resolved["device"],
        "help": resolved.get("help", ""),
    }


def default_isolate_device(
    probe: HostProbe | None,
    allowed_devices: Sequence[str],
    current: str | None = None,
    *,
    platform: str | None = None,
) -> str | None:
    """Seed the persisted ``isolate_device`` key from the host recommendation.

    Positional order is wrong on the ``both`` edition: ``cuda`` is listed first even when
    the machine has no NVIDIA GPU, so ``allowed_devices[0]`` would persist a CUDA default
    that the run gate then rejects. An existing, still-valid choice always wins so a
    deliberate user pick survives a restart.
    """
    if current and current in allowed_devices:
        return current
    if probe is not None and allowed_devices:
        recommended = str(desktop_recommend(probe, platform=platform)["device"])
        if recommended in allowed_devices:
            return recommended
    return allowed_devices[0] if allowed_devices else current


LITE_MAX_DURATION_SEC = 90.0


def default_region_end(duration_sec: float, *, min_length: float = 30.0) -> float:
    """Pick a sensible default section end for the region slider."""
    if duration_sec <= min_length:
        return duration_sec
    return min(duration_sec, min_length)


def clamp_lite_clip(
    start_sec: float,
    max_duration_sec: float | None,
    file_duration_sec: float | None,
    *,
    cap: float = LITE_MAX_DURATION_SEC,
) -> tuple[float, float, bool]:
    """Optional helper: shorten a clip to ``cap`` (Safer: first 90 s logic).

    Desktop Lite enqueue no longer calls this — full-file is allowed with a
    visible warning. Returns ``(start_sec, length_sec, clamped)``.
    """
    start = max(0.0, float(start_sec or 0.0))
    remaining: float | None = None
    if file_duration_sec is not None:
        remaining = max(0.0, float(file_duration_sec) - start)
    if max_duration_sec is None:
        requested = remaining if remaining is not None else cap
    else:
        requested = float(max_duration_sec)
        if remaining is not None:
            requested = min(requested, remaining)
    requested = max(0.0, requested)
    length = min(requested, cap)
    clamped = length + 1e-6 < requested
    return start, length, clamped


def clamp_region_bounds(
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    *,
    min_length: float = 5.0,
) -> tuple[float, float]:
    """Clamp start/end to file bounds with minimum region length."""
    start = max(0.0, min(start_sec, duration_sec))
    end = max(start + min_length, min(end_sec, duration_sec))
    if end > duration_sec:
        end = duration_sec
        start = max(0.0, end - min_length)
    return start, end


def custom_stems_to_track_options(stems: Any) -> list[str]:
    """Map legacy instrument ids to default track option ids."""
    wanted = {s for s in (stems or ()) if s in CUSTOM_STEM_CHOICES}
    if not wanted:
        raise ValueError("Pick at least one instrument to separate.")
    options: list[str] = []
    for stem_id in CUSTOM_STEM_CHOICES:
        if stem_id not in wanted:
            continue
        if stem_id == "guitar":
            options.append("guitar_demucs_6s")
        else:
            options.append(f"{stem_id}_demucs")
    return options or list(DEFAULT_TRACK_OPTIONS)


def migrate_track_options(session: MutableMapping[str, Any]) -> list[str]:
    """Resolve track option ids from session, migrating legacy preset keys."""
    current = session.get("isolate_track_options")
    if isinstance(current, (list, tuple)) and current:
        known = [str(x) for x in current if str(x) in TRACK_OPTIONS]
        if known:
            return known
    legacy_preset = session.get("isolate_separation_preset")
    if isinstance(legacy_preset, str):
        if legacy_preset == "best_guitar":
            legacy_preset = "full_band"
        mapped = LEGACY_PRESET_TRACK_OPTIONS.get(legacy_preset)
        if mapped:
            return list(mapped)
        if legacy_preset == "custom":
            return custom_stems_to_track_options(session.get("isolate_last_custom_stems"))
    return list(DEFAULT_TRACK_OPTIONS)


def _track_selection_caveat(option_ids: list[str], *, uses_roformer: bool) -> str:
    parts: list[str] = []
    if uses_roformer:
        parts.append(ROFORMER_DOWNLOAD_CAVEAT)
        non_guitar = [oid for oid in option_ids if oid not in GUITAR_TRACK_OPTION_IDS]
        if non_guitar:
            parts.append(ROFORMER_MIXED_STEMS_NOTE)
    return " ".join(parts)


def resolve_track_selection(option_ids: Any) -> dict[str, Any]:
    """Map labeled track picks to one engine config (model, emit_stems, guitar_refine)."""
    ids = [str(x) for x in (option_ids or ()) if str(x) in TRACK_OPTIONS]
    if not ids:
        raise ValueError("Pick at least one track to separate.")

    guitar_picks = [oid for oid in ids if oid in GUITAR_TRACK_OPTION_IDS]
    if len(guitar_picks) > 1:
        raise ValueError("Pick only one guitar option.")

    if VOCALS_INSTRUMENTAL_OPTION_ID in ids:
        if len(ids) > 1:
            raise ValueError("Vocals & instrumental cannot combine with other tracks.")
        return {
            "id": "custom",
            "label": "Custom",
            "tracks": TRACK_OPTIONS[VOCALS_INSTRUMENTAL_OPTION_ID]["label"],
            "track_count": 1,
            "model": "htdemucs",
            "two_stems": "vocals",
            "stems": ["vocals"],
            "emit_stems": ("vocals",),
            "track_options": list(ids),
            "fold_other_into_guitar": True,
            "fold_other_mode": "best_effort",
            "guitar_refine": False,
            "caveat": "",
        }

    emit: list[str] = []
    custom_stems: list[str] = []
    for oid in ids:
        stem = str(TRACK_OPTIONS[oid]["stem"])
        if stem not in emit:
            emit.append(stem)
        if stem in CUSTOM_STEM_CHOICES and stem not in custom_stems:
            custom_stems.append(stem)

    uses_roformer = bool(guitar_picks and guitar_picks[0] in {"guitar_roformer", "guitar_roformer_refine"})
    guitar_refine = guitar_picks == ["guitar_roformer_refine"]

    if uses_roformer:
        model = "bs_roformer_sw"
        two_stems = None
    elif {"guitar", "piano"} & set(custom_stems):
        model = "htdemucs_6s"
        two_stems = None
    else:
        model = "htdemucs"
        two_stems = None

    return {
        "id": "custom",
        "label": "Custom",
        "tracks": ", ".join(TRACK_OPTIONS[oid]["label"] for oid in ids),
        "track_count": len(ids),
        "model": model,
        "two_stems": two_stems,
        "stems": custom_stems,
        "emit_stems": tuple(emit),
        "track_options": list(ids),
        "fold_other_into_guitar": "other" not in custom_stems,
        "fold_other_mode": "best_effort",
        "guitar_refine": guitar_refine,
        "caveat": _track_selection_caveat(ids, uses_roformer=uses_roformer),
    }


def resolve_separation_preset(preset_id: str) -> dict[str, Any]:
    """Legacy preset id → engine settings via track option mapping."""
    option_ids = LEGACY_PRESET_TRACK_OPTIONS.get(preset_id)
    if option_ids is None:
        option_ids = DEFAULT_TRACK_OPTIONS
    resolved = resolve_track_selection(option_ids)
    resolved["id"] = preset_id if preset_id in SEPARATION_PRESETS else "custom"
    if preset_id in SEPARATION_PRESETS:
        resolved["label"] = SEPARATION_PRESETS[preset_id]["label"]
    return resolved


def resolve_custom_separation(stems: Any) -> dict[str, Any]:
    """Map legacy Custom instrument picks to engine settings."""
    return resolve_track_selection(custom_stems_to_track_options(stems))


def custom_selected_stems(
    produced_stem_names: Any,
    picked_stems: Any,
) -> dict[str, bool]:
    """
    Mixer/download selection after a Custom run: on for what the user asked for.

    Default path emits a single combined Guitar stem (no lead/rhythm).
    """
    produced = list(produced_stem_names)
    wanted: set[str] = set()
    for pick in picked_stems:
        wanted.update(CUSTOM_STEM_OUTPUTS.get(pick, ()))

    selected = {name: name in wanted for name in produced}
    if "metronome" in selected:
        selected["metronome"] = True
    if not any(selected.values()):
        return {name: True for name in produced}
    return selected


def upload_fingerprint(uploaded: Any) -> str | None:
    """Fingerprint a Streamlit UploadedFile (name + size), or None if absent."""
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", None)
    size = getattr(uploaded, "size", None)
    if not name:
        return None
    return f"{name}:{size}"


def sync_output_name_on_upload(
    uploaded: Any,
    *,
    last_fp: str | None,
    output_name: str,
) -> tuple[str | None, str, bool]:
    """
    When the user picks a new file, return updated fingerprint and output name.

    Returns ``(new_fp, output_name, changed)``.
    """
    fp = upload_fingerprint(uploaded)
    if fp is None or fp == last_fp:
        return fp, output_name, False
    new_name = Path(uploaded.name).stem
    return fp, new_name, True


ISOLATE_OUTPUT_NAME_KEY = "isolate_output_name"
ISOLATE_OUTPUT_NAME_PENDING_KEY = "isolate_output_name_pending"
ISOLATE_YOUTUBE_URL_KEY = "isolate_youtube_url"
ISOLATE_YOUTUBE_URL_PENDING_KEY = "isolate_youtube_url_pending"
ISOLATE_YOUTUBE_TITLE_PENDING_KEY = "isolate_youtube_title_pending"
ISOLATE_NAMED_YOUTUBE_URL_KEY = "isolate_named_youtube_url"
ISOLATE_AUTO_OUTPUT_NAME_KEY = "isolate_auto_output_name"
SETUP_STEP_KEY = "isolate_setup_step"
SETUP_STEP_SOURCE = "source"
SETUP_STEP_SEPARATE = "separate"
ISOLATE_BATCH_PATHS_KEY = "isolate_batch_paths"
ISOLATE_BATCH_FP_KEY = "isolate_batch_fp"


def current_setup_step(session: Mapping[str, Any]) -> str:
    """Home form screen. Anything other than separate is the source screen."""
    if session.get(SETUP_STEP_KEY) == SETUP_STEP_SEPARATE:
        return SETUP_STEP_SEPARATE
    return SETUP_STEP_SOURCE


def setup_continue_enabled(
    *,
    download_in_progress: bool,
    batch_count: int,
    live_upload_count: int,
    youtube_url: str,
    pending_fp: str | None,
    pending_ready: bool,
) -> bool:
    """Whether Continue can leave the source screen.

    A batch can continue while a URL is still pasted, so submit can show the
    existing batch block. An unfinished YouTube URL cannot. A direct media
    URL can, because Continue fetches it.
    """
    if download_in_progress:
        return False
    if batch_count > 1 or live_upload_count > 1:
        return True
    url = (youtube_url or "").strip()
    if url and is_youtube_url(url):
        return bool(pending_ready and pending_fp == f"youtube:{url}")
    if url and is_direct_media_url(url):
        return True
    if url:
        return False
    return bool(pending_ready)


def youtube_video_id(url: str) -> str | None:
    """Video id from a watch / youtu.be / shorts / embed URL, else None."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in {"youtu.be", "www.youtu.be"}:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        qs = parse_qs(parsed.query)
        if qs.get("v") and qs["v"][0]:
            return qs["v"][0]
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]
    return None


_LOCAL_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv"})


def mixer_local_video_path(run_dir: Path | None) -> Path | None:
    """Uploaded source video copied beside the stems, if this run has one."""
    if run_dir is None or not run_dir.is_dir():
        return None
    found = [
        path
        for path in run_dir.iterdir()
        if path.is_file()
        and path.stem == "source_video"
        and path.suffix.lower() in _LOCAL_VIDEO_SUFFIXES
    ]
    return found[0] if found else None


def mixer_youtube_video_id(session: Mapping[str, Any]) -> str | None:
    """YouTube id for the loaded mix, or None for a file or a missing link."""
    kind = infer_source_kind(
        source_kind=session.get("isolate_source_kind"),
        source_fingerprint=session.get("isolate_results_source_fp"),
    )
    if kind != SOURCE_KIND_YOUTUBE:
        return None
    fingerprint = str(session.get("isolate_results_source_fp") or "")
    prefix = "youtube:"
    if not fingerprint.startswith(prefix):
        return None
    return youtube_video_id(fingerprint[len(prefix) :])


def prefer_local_mixer_picture(
    youtube_id: str | None,
    local_video: Path | None,
) -> tuple[str | None, Path | None]:
    """Keep both. The mixer tries the file first and can fall back to the id."""
    vid = (youtube_id or "").strip() or None
    if local_video is not None:
        return vid, local_video
    return vid, None


def youtube_label_from_url(url: str) -> str:
    """Default Output name from a pasted URL (id until the file is downloaded)."""
    return youtube_video_id(url) or "youtube_audio"


def resolve_youtube_job_name(output_name: str, youtube_url: str, audio_stem: str) -> str:
    """Resolve the human-readable title used for a job and its run.

    Empty names fall back to the stem of the audio file on disk (for YouTube
    that is the real video title). For a YouTube source, a name that is still
    the video-id auto default is replaced with that title; user-edited names
    are preserved.
    """
    name = (output_name or "").strip()
    url = (youtube_url or "").strip()
    if url and (not name or name == youtube_label_from_url(url)):
        return audio_stem or name
    return name or audio_stem


def sync_output_name_on_youtube(
    *,
    youtube_url: str,
    last_named_url: str | None,
    output_name: str,
    auto_output_name: str | None,
    downloaded_stem: str | None = None,
    preferred_label: str | None = None,
) -> tuple[str, str, str, bool]:
    """When the YouTube URL changes, replace an auto Output name.

    Returns ``(name, named_url, auto_name, changed)``. A user-edited name is
    kept for the same URL; a new URL always takes the new label.
    ``preferred_label`` is the search-hit title (beats video-id fallback).
    """
    url = (youtube_url or "").strip()
    if not url:
        return output_name, last_named_url or "", auto_output_name or "", False
    label = (
        (downloaded_stem or "").strip()
        or (preferred_label or "").strip()
        or youtube_label_from_url(url)
    )
    last = (last_named_url or "").strip()
    current = (output_name or "").strip()
    auto = (auto_output_name or "").strip()
    if last != url:
        return label, url, label, True
    if downloaded_stem and (not current or current == auto):
        stem = downloaded_stem.strip()
        return stem, url, stem, stem != current
    return output_name, last or url, auto, False


def apply_youtube_output_name_sync(
    session: MutableMapping[str, object],
    *,
    downloaded_stem: str | None = None,
    apply_now: bool = True,
) -> str | None:
    """Queue (and optionally apply) a YouTube Output name before the text_input."""
    url = str(session.get(ISOLATE_YOUTUBE_URL_KEY) or "").strip()
    if not url:
        return None
    preferred = None
    if ISOLATE_YOUTUBE_TITLE_PENDING_KEY in session:
        preferred = str(session.pop(ISOLATE_YOUTUBE_TITLE_PENDING_KEY) or "").strip() or None
    name, named_url, auto, changed = sync_output_name_on_youtube(
        youtube_url=url,
        last_named_url=(
            str(session.get(ISOLATE_NAMED_YOUTUBE_URL_KEY) or "")
            or None
        ),
        output_name=str(session.get(ISOLATE_OUTPUT_NAME_KEY) or ""),
        auto_output_name=(
            str(session.get(ISOLATE_AUTO_OUTPUT_NAME_KEY) or "") or None
        ),
        downloaded_stem=downloaded_stem,
        preferred_label=preferred,
    )
    session[ISOLATE_NAMED_YOUTUBE_URL_KEY] = named_url
    session[ISOLATE_AUTO_OUTPUT_NAME_KEY] = auto
    if not changed:
        return None
    queue_reopen_output_name(session, name)
    if apply_now:
        apply_pending_output_name(session)
    return name


def queue_reopen_output_name(session: MutableMapping[str, object], title: str) -> None:
    """Stash a Reopen title for the next run. MUST NOT write the widget-bound key."""
    session[ISOLATE_OUTPUT_NAME_PENDING_KEY] = title


def queue_output_name_if_empty(session: MutableMapping[str, object], title: str) -> None:
    """Queue a default name only when the Output name field is blank."""
    current = str(session.get(ISOLATE_OUTPUT_NAME_KEY) or "").strip()
    if current:
        return
    queue_reopen_output_name(session, title)


def apply_pending_output_name(session: MutableMapping[str, object]) -> str | None:
    """Copy pending title onto the widget key. Call before ``st.text_input`` exists."""
    pending = session.pop(ISOLATE_OUTPUT_NAME_PENDING_KEY, None)
    if pending is None:
        return None
    name = str(pending)
    session[ISOLATE_OUTPUT_NAME_KEY] = name
    return name


def queue_clear_youtube_url(session: MutableMapping[str, object]) -> None:
    """Clear the YouTube field on the next run. MUST NOT write the widget-bound key."""
    session[ISOLATE_YOUTUBE_URL_PENDING_KEY] = ""
    session.pop(ISOLATE_YOUTUBE_TITLE_PENDING_KEY, None)


def queue_youtube_url(
    session: MutableMapping[str, object],
    url: str,
    *,
    title: str | None = None,
) -> None:
    """Set the YouTube URL on the next run (e.g. after picking a search hit)."""
    session[ISOLATE_YOUTUBE_URL_PENDING_KEY] = (url or "").strip()
    cleaned = (title or "").strip()
    if cleaned:
        session[ISOLATE_YOUTUBE_TITLE_PENDING_KEY] = cleaned
    else:
        session.pop(ISOLATE_YOUTUBE_TITLE_PENDING_KEY, None)


def apply_pending_youtube_url(session: MutableMapping[str, object]) -> str | None:
    """Copy pending URL onto the widget key. Call before ``st.text_input`` exists."""
    if ISOLATE_YOUTUBE_URL_PENDING_KEY not in session:
        return None
    pending = session.pop(ISOLATE_YOUTUBE_URL_PENDING_KEY)
    value = str(pending)
    session[ISOLATE_YOUTUBE_URL_KEY] = value
    return value


def adopt_audio_into_run(audio_path: Path, output_dir: Path) -> Path:
    """Copy ``audio_path`` into ``output_dir`` (same basename) for job ownership.

    No-op when the file already lives inside ``output_dir``.
    """
    src = Path(audio_path)
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        if src.resolve() == dest.resolve():
            return dest
        if src.parent.resolve() == dest_dir.resolve():
            return src
    except OSError:
        pass
    shutil.copy2(src, dest)
    return dest


def is_youtube_staging_path(path: str | Path | None, fingerprint: str | None) -> bool:
    """True when ``path`` is a DATA_DIR run-child file staged from YouTube."""
    if not path or not fingerprint:
        return False
    if not str(fingerprint).startswith("youtube:"):
        return False
    try:
        file_path = Path(path).resolve()
        data_root = DATA_DIR.resolve()
    except OSError:
        return False
    if not file_path.is_file():
        return False
    run_dir = file_path.parent
    try:
        run_dir.relative_to(data_root)
    except ValueError:
        return False
    if run_dir == data_root or run_dir.parent != data_root:
        return False
    return True


def staging_paths_still_needed(path: str | Path | None) -> bool:
    """True when a queued/running job still points at this staging folder."""
    if not path:
        return False
    try:
        staging_dir = Path(path).resolve()
        if staging_dir.is_file():
            staging_dir = staging_dir.parent
    except OSError:
        return False
    try:
        from ui.isolate_jobs import list_jobs, read_spec
    except Exception:
        return False
    try:
        jobs = list_jobs(limit=50)
    except Exception:
        return False
    for job in jobs:
        status = str(job.get("status") or "")
        if status not in {"queued", "running", "pausing", "paused"}:
            continue
        job_id = str(job.get("id") or "")
        if not job_id:
            continue
        spec = read_spec(job_id)
        if spec is None or not spec.audio_path:
            continue
        try:
            audio = Path(spec.audio_path).resolve()
        except OSError:
            continue
        if audio == staging_dir or staging_dir in audio.parents:
            return True
    return False


def discard_youtube_staging(
    path: str | Path | None,
    *,
    fingerprint: str | None,
    retain_paths: Sequence[str | Path] = (),
) -> bool:
    """Delete a YouTube staging run folder when safe. Never raises."""
    if not is_youtube_staging_path(path, fingerprint):
        return False
    try:
        file_path = Path(path).resolve()
        staging_dir = file_path.parent
    except OSError:
        return False
    retain: set[Path] = set()
    for raw in retain_paths:
        try:
            retain.add(Path(raw).resolve())
        except OSError:
            continue
    if file_path in retain or staging_dir in retain:
        return False
    for kept in retain:
        try:
            if staging_dir in kept.parents or kept == staging_dir:
                return False
        except Exception:
            continue
    if staging_paths_still_needed(staging_dir):
        return False
    try:
        return bool(delete_run(staging_dir))
    except Exception:
        logger.debug("discard_youtube_staging failed for %s", staging_dir, exc_info=True)
        return False


def resolve_tab_pdf_carry_from_isolate(
    session: Mapping[str, object],
) -> tuple[str | None, str | None, bool]:
    """Pick Tab PDF carry-over audio from Isolation results.

    Prefers ``isolate_artifacts["guitar"]`` when the file exists; otherwise the
    source mix. Returns ``(path, display_name, is_guitar_stem)``.
    """
    artifacts = session.get("isolate_artifacts")
    if isinstance(artifacts, dict):
        raw = artifacts.get("guitar")
        if raw:
            guitar = Path(str(raw))
            if guitar.is_file():
                return str(guitar), guitar.name, True
    source = session.get("isolate_source_audio_path")
    if source:
        mix = Path(str(source))
        if mix.is_file():
            return str(mix), mix.name, False
    return None, None, False


def reset_new_tab_source(session: MutableMapping[str, object]) -> None:
    """Remount the uploader and clear New-tab source after a job is queued.

    May delete a YouTube staging folder under ``DATA_DIR`` when the pending
    fingerprint is ``youtube:…``. Must not write widget-bound
    ``isolate_output_name`` / ``isolate_youtube_url`` keys — those go through
    pending keys so the next run can apply them before the widgets mount.
    """
    pending_path = session.get("isolate_pending_audio_path")
    pending_fp = session.get("isolate_pending_fp") or session.get("isolate_upload_fp")
    discard_youtube_staging(
        str(pending_path) if pending_path else None,
        fingerprint=str(pending_fp) if pending_fp else None,
    )
    try:
        current = int(session.get("isolate_upload_key") or 0)
    except (TypeError, ValueError):
        current = 0
    session["isolate_upload_key"] = current + 1
    for key in (
        "isolate_upload_fp",
        "isolate_pending_audio_path",
        "isolate_pending_fp",
        "isolate_duration_sec",
        "isolate_duration_fp",
        "carry_over_audio_path",
        "carry_over_audio_name",
        "carry_over_is_guitar_stem",
    ):
        session.pop(key, None)
    queue_clear_youtube_url(session)
    queue_reopen_output_name(session, "")
    session[SETUP_STEP_KEY] = SETUP_STEP_SOURCE
    session.pop(ISOLATE_BATCH_PATHS_KEY, None)
    session.pop(ISOLATE_BATCH_FP_KEY, None)
    tab = str(session.get(SHELL_TAB_KEY) or "")
    if is_new_draft_tab(tab):
        # Keep the empty draft slot in sync after Separate / +.
        raw = session.get(DRAFT_SOURCES_KEY)
        if not isinstance(raw, dict):
            raw = {}
            session[DRAFT_SOURCES_KEY] = raw
        raw[tab] = {
            "isolate_upload_key": session.get("isolate_upload_key"),
        }


def should_hide_stale_results(
    *,
    pending_upload_fp: str | None,
    has_artifacts: bool,
    results_source_fp: str | None = None,
) -> bool:
    """True when a *new* source is queued that differs from the loaded run's source.

    Having any upload fingerprint while artifacts exist is not enough — after a
    finished run the fingerprint may still be set. Hide only when the pending
    source is different from the run that produced the current mixer.
    """
    if not has_artifacts or pending_upload_fp is None:
        return False
    if results_source_fp is None:
        # Legacy callers: treat any pending fp as a new source.
        return True
    return pending_upload_fp != results_source_fp


WORKSPACE_KEY = "isolate_workspace"
WORKSPACE_TABS = ("New", "Mixer", "Queue")
WORKSPACE_NEXT_KEY = "_isolate_workspace_next"
LISTEN_PICKER_KEY = "isolate_listen_picker"
LISTEN_PICKER_NEXT_KEY = "_isolate_listen_picker_next"
OPEN_MIX_TABS_KEY = "isolate_open_mix_tabs"
OPEN_MIX_TABS_MAX = 12
# Set when the user closes the last mix tab so ``select_rehydrate_row`` does not
# immediately reload the latest run (which would re-open the tab via
# ``add_open_mix_tab``). Cleared when a mix is opened intentionally.
ISOLATE_SKIP_REHYDRATE_KEY = "isolate_skip_rehydrate"
NEW_DRAFT_TAB_ID = "__new__"
NEW_DRAFT_TAB_PREFIX = "__new__"
DRAFT_SOURCES_KEY = "isolate_draft_sources"
SHELL_TAB_KEY = "isolate_shell_tab"
SHELL_VIEW_KEY = "isolate_shell_view"
SHELL_VIEW_NEXT_KEY = "_isolate_shell_view_next"
SHELL_VIEWS = ("home", "mix")
ISOLATE_EXPORT_DIR_KEY = "isolate_export_dir"

# Live New-tab source fields snapshotted when switching between draft tabs.
_DRAFT_SOURCE_LIVE_KEYS = (
    "isolate_upload_key",
    "isolate_upload_fp",
    "isolate_pending_audio_path",
    "isolate_pending_fp",
    "isolate_duration_sec",
    "isolate_duration_fp",
    "carry_over_audio_path",
    "carry_over_audio_name",
    "carry_over_is_guitar_stem",
    "isolate_pending_video_path",
    ISOLATE_OUTPUT_NAME_KEY,
    ISOLATE_OUTPUT_NAME_PENDING_KEY,
    ISOLATE_YOUTUBE_URL_KEY,
    ISOLATE_YOUTUBE_URL_PENDING_KEY,
    ISOLATE_YOUTUBE_TITLE_PENDING_KEY,
    ISOLATE_NAMED_YOUTUBE_URL_KEY,
    ISOLATE_AUTO_OUTPUT_NAME_KEY,
    SETUP_STEP_KEY,
    ISOLATE_BATCH_PATHS_KEY,
    ISOLATE_BATCH_FP_KEY,
)


def is_new_draft_tab(tab_id: object) -> bool:
    text = str(tab_id or "")
    return text == NEW_DRAFT_TAB_ID or text.startswith(f"{NEW_DRAFT_TAB_PREFIX}:")


def make_new_draft_tab_id() -> str:
    """Unique New-draft id so + can open one staging tab per song."""
    return f"{NEW_DRAFT_TAB_PREFIX}:{uuid4().hex[:10]}"


def _draft_sources_map(session: MutableMapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = session.get(DRAFT_SOURCES_KEY)
    if not isinstance(raw, dict):
        raw = {}
        session[DRAFT_SOURCES_KEY] = raw
    return raw


def snapshot_draft_source(session: MutableMapping[str, Any]) -> dict[str, Any]:
    """Capture the New-tab source fields currently in session."""
    out: dict[str, Any] = {}
    for key in _DRAFT_SOURCE_LIVE_KEYS:
        if key in session:
            out[key] = session.get(key)
    return out


def save_active_draft_source(session: MutableMapping[str, Any]) -> None:
    """Persist the focused draft's source so switching tabs does not clobber it."""
    tab = str(session.get(SHELL_TAB_KEY) or "")
    if not is_new_draft_tab(tab):
        return
    _draft_sources_map(session)[tab] = snapshot_draft_source(session)


def apply_draft_source_snapshot(
    session: MutableMapping[str, Any],
    snap: Mapping[str, Any] | None,
) -> None:
    """Replace live New-tab source fields from a draft snapshot (or clear)."""
    data = dict(snap or {})
    for key in _DRAFT_SOURCE_LIVE_KEYS:
        if key in data:
            session[key] = data[key]
        else:
            session.pop(key, None)
    # Remount the uploader so Streamlit does not keep the previous file widget.
    try:
        current = int(session.get("isolate_upload_key") or 0)
    except (TypeError, ValueError):
        current = 0
    session["isolate_upload_key"] = current + 1


def draft_tab_title(session: Mapping[str, Any], tab_id: object) -> str:
    """Strip label for a New draft: song name when known, else New."""
    tid = str(tab_id or "")
    live = is_new_draft_tab(session.get(SHELL_TAB_KEY)) and str(
        session.get(SHELL_TAB_KEY) or ""
    ) == tid
    src: Mapping[str, Any]
    if live:
        src = session
    else:
        raw = session.get(DRAFT_SOURCES_KEY)
        stored = raw.get(tid) if isinstance(raw, dict) else None
        src = stored if isinstance(stored, dict) else {}
    # Pending rename/clear wins so strip labels match post-+ empty drafts before
    # apply_pending_output_name runs later in the same paint.
    if ISOLATE_OUTPUT_NAME_PENDING_KEY in src:
        name = str(src.get(ISOLATE_OUTPUT_NAME_PENDING_KEY) or "").strip()
    else:
        name = str(src.get(ISOLATE_OUTPUT_NAME_KEY) or "").strip()
    if not name:
        pending = src.get("isolate_pending_audio_path") or src.get("carry_over_audio_path")
        if pending:
            name = Path(str(pending)).stem
    if not name:
        name = str(src.get("carry_over_audio_name") or "").strip()
        if name:
            name = Path(name).stem
    return name or "New"


def normalize_open_mix_tabs(
    tabs: object,
    *,
    existing: set[str] | None = None,
    cap: int = OPEN_MIX_TABS_MAX,
    require_dir: bool = True,
    protect: Iterable[str] | None = None,
) -> list[str]:
    """Dedupe run_dir paths / New draft ids, optionally drop missing dirs, enforce cap.

    When over cap, drop oldest mix run dirs first so New drafts and ``protect``
    ids (active shell tab) stay available.
    """
    if not isinstance(tabs, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in tabs:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        if is_new_draft_tab(path):
            seen.add(path)
            out.append(path)
            continue
        if existing is not None and path not in existing:
            continue
        if require_dir and existing is None and not Path(path).is_dir():
            continue
        seen.add(path)
        out.append(path)
    if len(out) <= cap:
        return out
    keep = {str(p) for p in (protect or ()) if p}
    drafts = [p for p in out if is_new_draft_tab(p)]
    mixes = [p for p in out if not is_new_draft_tab(p)]
    # Drop oldest mixes first, then oldest drafts, never drop protected ids last.
    while len(drafts) + len(mixes) > cap:
        if mixes:
            victim = next((m for m in mixes if m not in keep), None)
            if victim is None:
                victim = mixes[0]
            mixes = [m for m in mixes if m != victim]
            continue
        if drafts:
            victim = next((d for d in drafts if d not in keep), None)
            if victim is None:
                break
            drafts = [d for d in drafts if d != victim]
            continue
        break
    # Preserve relative order from ``out``.
    allowed = set(drafts) | set(mixes)
    return [p for p in out if p in allowed]


def add_open_mix_tab(
    session: MutableMapping[str, Any],
    run_dir: str | Path | None,
    *,
    cap: int = OPEN_MIX_TABS_MAX,
) -> list[str]:
    """Ensure a mix/draft tab is open without reordering existing tabs.

    New tabs append at the end (Chrome-style). Focusing an already-open tab
    leaves strip order alone so only the active indicator moves.
    """
    path = str(run_dir or "").strip()
    protect = [str(session.get(SHELL_TAB_KEY) or "")]
    if not path:
        return normalize_open_mix_tabs(
            session.get(OPEN_MIX_TABS_KEY),
            cap=cap,
            require_dir=False,
            protect=protect,
        )
    current = normalize_open_mix_tabs(
        session.get(OPEN_MIX_TABS_KEY),
        cap=max(cap * 2, cap + 4),
        require_dir=False,
        protect=protect + [path],
    )
    if path in current:
        session[OPEN_MIX_TABS_KEY] = normalize_open_mix_tabs(
            current, cap=cap, require_dir=False, protect=protect + [path]
        )
        return list(session[OPEN_MIX_TABS_KEY])
    current.append(path)
    session[OPEN_MIX_TABS_KEY] = normalize_open_mix_tabs(
        current, cap=cap, require_dir=False, protect=protect + [path]
    )
    return list(session[OPEN_MIX_TABS_KEY])


def replace_open_mix_tab(
    session: MutableMapping[str, Any],
    old_id: str | Path | None,
    new_id: str | Path | None,
    *,
    cap: int = OPEN_MIX_TABS_MAX,
) -> list[str]:
    """Swap ``old_id`` for ``new_id`` in the same strip slot (draft → finished mix).

    When ``old_id`` is missing, falls back to ``add_open_mix_tab``. Draft source
    state for ``old_id`` is dropped.
    """
    old = str(old_id or "").strip()
    new = str(new_id or "").strip()
    if not new:
        return add_open_mix_tab(session, None, cap=cap)
    protect = [str(session.get(SHELL_TAB_KEY) or ""), new]
    tabs = normalize_open_mix_tabs(
        session.get(OPEN_MIX_TABS_KEY),
        cap=max(cap * 2, cap + 4),
        require_dir=False,
        protect=protect + ([old] if old else []),
    )
    if not old or old not in tabs:
        return add_open_mix_tab(session, new, cap=cap)
    replaced: list[str] = []
    seen: set[str] = set()
    for path in tabs:
        candidate = new if path == old else path
        if candidate in seen:
            continue
        seen.add(candidate)
        replaced.append(candidate)
    if is_new_draft_tab(old):
        _draft_sources_map(session).pop(old, None)
    if str(session.get(SHELL_TAB_KEY) or "") == old:
        session.pop(SHELL_TAB_KEY, None)
    session[OPEN_MIX_TABS_KEY] = normalize_open_mix_tabs(
        replaced, cap=cap, require_dir=False, protect=protect
    )
    return list(session[OPEN_MIX_TABS_KEY])


def promote_job_origin_tab(
    session: MutableMapping[str, Any],
    status: Mapping[str, Any],
) -> str | None:
    """Turn a job's origin draft tab into its finished ``run_dir`` tab.

    Returns the run_dir when a tab was opened or promoted, else None.
    """
    run_dir = str(status.get("run_dir") or "").strip()
    if not run_dir:
        return None
    origin = str(status.get("origin_tab") or "").strip()
    if origin and is_new_draft_tab(origin):
        replace_open_mix_tab(session, origin, run_dir)
    else:
        add_open_mix_tab(session, run_dir)
    return run_dir


_DRAFT_JOB_BUSY_STATUSES = frozenset({"queued", "running", "pausing", "paused"})


def pending_open_job_ids(
    jobs: Iterable[Mapping[str, Any]],
    consumed_ids: Iterable[str] | None,
) -> set[str]:
    """Succeeded job ids the tab strip should keep busy until the poll consumes them.

    ``consumed_ids is None`` is a fresh session. The poll seeds those historical
    successes and does not open them, so they are not waiting on the mixer.
    """
    if consumed_ids is None:
        return set()
    consumed = {str(item) for item in consumed_ids if str(item or "").strip()}
    pending: set[str] = set()
    for job in jobs:
        if str(job.get("status") or "") != "succeeded":
            continue
        job_id = str(job.get("id") or "").strip()
        if job_id and job_id not in consumed:
            pending.add(job_id)
    return pending


def draft_tab_job_overlays(
    jobs: Iterable[Mapping[str, Any]],
    pending_open_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map draft tab id → in-flight job overlay for the mix-tabs strip.

    Used to paint a per-tab progress bar while separation runs in that slot.
    Succeeded ids in ``pending_open_ids`` stay busy at 100% until the mixer
    poll consumes them. A running or queued job on the same tab still wins.
    """
    pending = {str(item) for item in (pending_open_ids or ()) if str(item or "").strip()}
    out: dict[str, dict[str, Any]] = {}
    for job in jobs:
        origin = str(job.get("origin_tab") or "").strip()
        if not is_new_draft_tab(origin):
            continue
        status = str(job.get("status") or "")
        job_id = str(job.get("id") or "")
        opening = status == "succeeded" and job_id in pending
        if status not in _DRAFT_JOB_BUSY_STATUSES and not opening:
            continue
        progress: float | None = None
        if status == "running":
            try:
                progress = float(job.get("progress") or 0.0)
            except (TypeError, ValueError):
                progress = 0.0
            progress = min(1.0, max(0.0, progress))
        elif opening:
            progress = 1.0
        # Prefer a running job over a queued one if both somehow share a tab.
        # An in-flight job also wins over a succeeded mix that is not open yet.
        existing = out.get(origin)
        if existing and existing.get("status") == "running" and status != "running":
            continue
        if existing and existing.get("status") in _DRAFT_JOB_BUSY_STATUSES and opening:
            continue
        out[origin] = {
            "job_id": job_id,
            "status": status,
            "title": str(job.get("title") or "").strip() or None,
            "busy": True,
            "progress": progress,
        }
    return out


def mark_draft_tab_processing(
    session: MutableMapping[str, Any],
    tab_id: str,
    *,
    title: str,
) -> None:
    """Keep the draft strip label after source reset while a job is bound to it."""
    if not is_new_draft_tab(tab_id):
        return
    name = str(title or "").strip()
    _draft_sources_map(session)[tab_id] = {
        "isolate_upload_key": session.get("isolate_upload_key"),
        ISOLATE_OUTPUT_NAME_KEY: name,
        ISOLATE_OUTPUT_NAME_PENDING_KEY: name,
    }
    session[SHELL_TAB_KEY] = tab_id


def mix_tab_ids_equal(a: str | Path | None, b: str | Path | None) -> bool:
    """True when two strip ids refer to the same draft or run_dir path."""
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right:
        return False
    if left == right:
        return True
    if is_new_draft_tab(left) or is_new_draft_tab(right):
        return False
    return os.path.normcase(os.path.normpath(left)) == os.path.normcase(
        os.path.normpath(right)
    )


def is_loaded_mix_tab(
    session: Mapping[str, Any],
    run_dir: str | Path | None,
) -> bool:
    """True when ``run_dir`` is the session's loaded / listening mix.

    Closing that tab must clear these pointers; otherwise
    ``add_open_mix_tab(loaded)`` on the next strip render reopens it and the
    last remaining mix tab looks uncloseable.
    """
    path = str(run_dir or "").strip()
    if not path or is_new_draft_tab(path):
        return False
    for key in (
        LISTEN_PICKER_KEY,
        "isolate_listen_applied_dir",
        "isolate_run_dir",
        "isolate_viewing_run_dir",
    ):
        if mix_tab_ids_equal(session.get(key), path):
            return True
    return False


def close_open_mix_tab(
    session: MutableMapping[str, Any],
    run_dir: str | Path | None,
) -> str | None:
    """Remove a tab. Returns a neighbor to focus, or None if none remain."""
    path = str(run_dir or "").strip()
    tabs = [
        p
        for p in normalize_open_mix_tabs(
            session.get(OPEN_MIX_TABS_KEY), require_dir=False
        )
        if is_new_draft_tab(p) or Path(p).is_dir()
    ]
    if not path or path not in tabs:
        # Path may differ only by slash/case on Windows — still try to match.
        match = next((p for p in tabs if mix_tab_ids_equal(p, path)), None)
        if match is None:
            session[OPEN_MIX_TABS_KEY] = tabs
            return tabs[-1] if tabs else None
        path = match
    idx = tabs.index(path)
    tabs = [p for p in tabs if p != path]
    session[OPEN_MIX_TABS_KEY] = tabs
    if is_new_draft_tab(path):
        sources = _draft_sources_map(session)
        sources.pop(path, None)
    if not tabs:
        return None
    if idx >= len(tabs):
        return tabs[-1]
    return tabs[idx]


def open_mix_tabs_for_session(
    session: MutableMapping[str, Any],
    *,
    library_dirs: Iterable[str] | None = None,
) -> list[str]:
    current = normalize_open_mix_tabs(
        session.get(OPEN_MIX_TABS_KEY), require_dir=False
    )
    draft_ids = {p for p in current if is_new_draft_tab(p)}
    protect = [str(session.get(SHELL_TAB_KEY) or "")]
    if library_dirs is not None:
        keep = {str(d) for d in library_dirs} | draft_ids
        tabs = normalize_open_mix_tabs(
            current,
            existing=keep,
            require_dir=False,
            protect=protect,
        )
        tabs = [p for p in tabs if p in keep]
    else:
        tabs = [
            p
            for p in normalize_open_mix_tabs(
                current, require_dir=False, protect=protect
            )
            if is_new_draft_tab(p) or Path(p).is_dir()
        ]
    session[OPEN_MIX_TABS_KEY] = tabs
    return tabs


# One-shot message for the next Isolate paint (rendered once, then dropped).
MIX_TAB_NOTICE_KEY = "_isolate_mix_tab_notice"
_YOUTUBE_STAGING_DRAFT_KEYS = (
    "isolate_pending_audio_path",
    "isolate_pending_fp",
    "isolate_upload_fp",
    "isolate_duration_sec",
    "isolate_duration_fp",
)


def _open_tabs_in_order(session: Mapping[str, Any]) -> list[str]:
    return normalize_open_mix_tabs(session.get(OPEN_MIX_TABS_KEY), require_dir=False)


def _match_open_tab(tabs: list[str], tab_id: object) -> str | None:
    path = str(tab_id or "").strip()
    if not path:
        return None
    if path in tabs:
        return path
    return next((p for p in tabs if mix_tab_ids_equal(p, path)), None)


def _drop_open_tabs(
    session: MutableMapping[str, Any], tabs: list[str], drop: set[str]
) -> None:
    session[OPEN_MIX_TABS_KEY] = [p for p in tabs if p not in drop]
    sources = _draft_sources_map(session)
    for path in drop:
        if is_new_draft_tab(path):
            sources.pop(path, None)


def move_open_mix_tab(
    session: MutableMapping[str, Any],
    tab_id: object,
    to_index: int | str | None,
) -> bool:
    """Move one tab so it ends at ``to_index`` (0-based). Home is not a slot.

    Returns True only when the order changed. Out-of-range targets clamp to the
    ends, so a drop past the last tab means "last".
    """
    tabs = _open_tabs_in_order(session)
    path = _match_open_tab(tabs, tab_id)
    if path is None:
        return False
    try:
        target = int(to_index)
    except (TypeError, ValueError):
        return False
    target = max(0, min(target, len(tabs) - 1))
    current = tabs.index(path)
    if target == current:
        return False
    tabs.pop(current)
    tabs.insert(target, path)
    session[OPEN_MIX_TABS_KEY] = tabs
    return True


def close_other_open_mix_tabs(
    session: MutableMapping[str, Any],
    keep_id: object,
    *,
    protect: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Close every tab except ``keep_id``. Returns ``(closed, spared)``.

    ``protect`` ids (tabs whose separation is still running) stay open and come
    back as ``spared``. Strip only: nothing is deleted from disk.
    """
    tabs = _open_tabs_in_order(session)
    keep = _match_open_tab(tabs, keep_id)
    if keep is None:
        return [], []
    guarded = {str(p) for p in protect}
    others = [p for p in tabs if p != keep]
    spared = [p for p in others if p in guarded]
    closed = [p for p in others if p not in guarded]
    _drop_open_tabs(session, tabs, set(closed))
    return closed, spared


def close_right_open_mix_tabs(
    session: MutableMapping[str, Any],
    anchor_id: object,
    *,
    protect: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Close every tab to the right of ``anchor_id``. Returns ``(closed, spared)``."""
    tabs = _open_tabs_in_order(session)
    anchor = _match_open_tab(tabs, anchor_id)
    if anchor is None:
        return [], []
    guarded = {str(p) for p in protect}
    right = tabs[tabs.index(anchor) + 1 :]
    spared = [p for p in right if p in guarded]
    closed = [p for p in right if p not in guarded]
    _drop_open_tabs(session, tabs, set(closed))
    return closed, spared


def duplicate_open_mix_tab(
    session: MutableMapping[str, Any],
    tab_id: object,
    *,
    title: str | None = None,
    source_audio_path: str | Path | None = None,
) -> str | None:
    """Open a second tab next to ``tab_id`` and focus it. Returns the new draft id.

    Duplicate never copies stems and never touches a run on disk:

    * a New draft is copied (its staged source and settings);
    * a finished mix opens a New draft with the same song staged, named
      "<title> copy", so the musician can separate it again another way.

    A YouTube-staged draft keeps its URL but not its staging file (that folder
    belongs to the original), so the copy downloads its own. When a mix's source
    audio is gone, the draft opens empty and ``MIX_TAB_NOTICE_KEY`` says why.
    """
    tabs = _open_tabs_in_order(session)
    origin = _match_open_tab(tabs, tab_id)
    if origin is None:
        return None
    save_active_draft_source(session)

    if is_new_draft_tab(origin):
        stored = _draft_sources_map(session).get(origin)
        snap = dict(stored) if isinstance(stored, dict) else {}
        fp = str(snap.get("isolate_pending_fp") or snap.get("isolate_upload_fp") or "")
        if fp.startswith("youtube:"):
            for key in _YOUTUBE_STAGING_DRAFT_KEYS:
                snap.pop(key, None)
        base = title if title is not None else draft_tab_title(session, origin)
    else:
        snap = {}
        base = str(title or "").strip()
        source = str(source_audio_path or "").strip()
        if source and Path(source).is_file():
            snap["carry_over_audio_path"] = source
            snap["carry_over_audio_name"] = Path(source).name
        else:
            session[MIX_TAB_NOTICE_KEY] = (
                "The original audio for this mix is no longer on disk. "
                "Upload it again to separate it a second time."
            )

    base = str(base or "").strip()
    if base == "New":
        base = ""
    # The strip title reads the pending name first; never write the widget key.
    snap.pop(ISOLATE_OUTPUT_NAME_KEY, None)
    snap.pop(ISOLATE_OUTPUT_NAME_PENDING_KEY, None)
    if base:
        snap[ISOLATE_OUTPUT_NAME_PENDING_KEY] = f"{base} copy"

    draft_id = make_new_draft_tab_id()
    ordered = list(tabs)
    ordered.insert(ordered.index(origin) + 1, draft_id)
    session[OPEN_MIX_TABS_KEY] = normalize_open_mix_tabs(
        ordered, require_dir=False, protect=[origin, draft_id]
    )
    session[SHELL_TAB_KEY] = draft_id
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    apply_draft_source_snapshot(session, snap)
    _draft_sources_map(session)[draft_id] = snapshot_draft_source(session)
    return draft_id


def apply_workspace_tab(session: MutableMapping[str, Any], *, has_artifacts: bool) -> str:
    """Legacy New/Mixer/Queue resolver — maps onto Home/mix shell views.

    Kept so older persisted UI state and tests keep working. Prefer
    ``apply_shell_view`` for new code.
    """
    _ = has_artifacts
    nxt = session.pop(WORKSPACE_NEXT_KEY, None)
    if nxt in WORKSPACE_TABS:
        session[WORKSPACE_KEY] = nxt
    elif session.get(WORKSPACE_KEY) not in WORKSPACE_TABS:
        session[WORKSPACE_KEY] = "New"
    tab = str(session.get(WORKSPACE_KEY) or "New")
    if tab == "Mixer":
        session[SHELL_VIEW_KEY] = "mix"
    else:
        session[SHELL_VIEW_KEY] = "home"
    return tab


def apply_shell_view(session: MutableMapping[str, Any]) -> str:
    """Resolve Home vs mix before the Moises tab strip is painted."""
    nxt = session.pop(SHELL_VIEW_NEXT_KEY, None)
    if nxt in SHELL_VIEWS:
        session[SHELL_VIEW_KEY] = nxt
    # Migrate legacy workspace tabs once.
    if session.get(SHELL_VIEW_KEY) not in SHELL_VIEWS:
        legacy = session.get(WORKSPACE_KEY)
        if legacy == "Mixer" or session.get(WORKSPACE_NEXT_KEY) == "Mixer":
            session[SHELL_VIEW_KEY] = "mix"
        else:
            session[SHELL_VIEW_KEY] = "home"
    view = str(session.get(SHELL_VIEW_KEY) or "home")
    if view not in SHELL_VIEWS:
        view = "home"
        session[SHELL_VIEW_KEY] = view
    # Mirror into legacy workspace key for persisted snapshots.
    session[WORKSPACE_KEY] = "Mixer" if view == "mix" else "New"
    return view


def open_mix_shell(session: MutableMapping[str, Any]) -> None:
    """Request the mix shell on the next full run."""
    session[SHELL_VIEW_NEXT_KEY] = "mix"
    session[WORKSPACE_NEXT_KEY] = "Mixer"
    session.pop(SHELL_TAB_KEY, None)


def open_home_shell(session: MutableMapping[str, Any]) -> None:
    """Request the Home shell on the next full run (Home chrome, not New tab)."""
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    session[SHELL_TAB_KEY] = "home"


def open_new_draft_tab(
    session: MutableMapping[str, Any],
    *,
    fresh: bool = True,
) -> str:
    """Open a new New-draft tab (Chrome-style +) with its own source state.

    Returns the draft tab id. Each + creates a distinct tab so one song can
    stay staged while another is prepared.
    """
    save_active_draft_source(session)
    draft_id = make_new_draft_tab_id()
    add_open_mix_tab(session, draft_id)
    session[SHELL_TAB_KEY] = draft_id
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    if fresh:
        reset_new_tab_source(session)
        _draft_sources_map(session)[draft_id] = snapshot_draft_source(session)
    else:
        apply_draft_source_snapshot(session, None)
        _draft_sources_map(session)[draft_id] = snapshot_draft_source(session)
    return draft_id


def open_opened_audio_tab(
    session: MutableMapping[str, Any],
    staged_path: str | Path,
    *,
    title: str,
    video_path: str | Path | None = None,
) -> str:
    """Open a new Home tab whose source is already on disk.

    ``staged_path`` is a copy in the app's data dir, never the file the user
    opened. The original run, if any, stays untouched.
    """
    save_active_draft_source(session)
    draft_id = make_new_draft_tab_id()
    add_open_mix_tab(session, draft_id)
    staged = str(staged_path)
    fingerprint = f"open:{staged}"
    snap: dict[str, Any] = {
        "isolate_pending_audio_path": staged,
        "isolate_pending_fp": fingerprint,
        "isolate_upload_fp": fingerprint,
        ISOLATE_OUTPUT_NAME_PENDING_KEY: str(title or "").strip(),
    }
    if video_path:
        snap["isolate_pending_video_path"] = str(video_path)
    apply_draft_source_snapshot(session, snap)
    session[SHELL_TAB_KEY] = draft_id
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    _draft_sources_map(session)[draft_id] = snapshot_draft_source(session)
    return draft_id


def focus_new_draft_tab(
    session: MutableMapping[str, Any],
    tab_id: str | None = None,
) -> None:
    """Focus an existing New draft without wiping another draft's source."""
    save_active_draft_source(session)
    target = str(tab_id or "").strip()
    if not is_new_draft_tab(target):
        # Legacy callers (no id): prefer the focused draft, else the last open one.
        current = str(session.get(SHELL_TAB_KEY) or "")
        if is_new_draft_tab(current):
            target = current
        else:
            open_tabs = normalize_open_mix_tabs(
                session.get(OPEN_MIX_TABS_KEY), require_dir=False
            )
            drafts = [p for p in open_tabs if is_new_draft_tab(p)]
            target = drafts[-1] if drafts else make_new_draft_tab_id()
    add_open_mix_tab(session, target)
    snap = _draft_sources_map(session).get(target)
    apply_draft_source_snapshot(session, snap)
    session[SHELL_TAB_KEY] = target
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    _draft_sources_map(session)[target] = snapshot_draft_source(session)


def _live_upload_has_source(session: Mapping[str, Any]) -> bool:
    """True when the home form already has a file, link, or batch to keep."""
    for key in (
        "isolate_pending_audio_path",
        "carry_over_audio_path",
        "isolate_pending_video_path",
        ISOLATE_YOUTUBE_URL_KEY,
        ISOLATE_YOUTUBE_URL_PENDING_KEY,
    ):
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return True
    batch = session.get(ISOLATE_BATCH_PATHS_KEY)
    return isinstance(batch, (list, tuple)) and bool(batch)


def _adopt_live_upload_as_draft(session: MutableMapping[str, Any]) -> str:
    """Give the current home form its own New tab without clearing the source."""
    save_active_draft_source(session)
    draft_id = make_new_draft_tab_id()
    add_open_mix_tab(session, draft_id)
    session[SHELL_TAB_KEY] = draft_id
    session[SHELL_VIEW_NEXT_KEY] = "home"
    session[WORKSPACE_NEXT_KEY] = "New"
    _draft_sources_map(session)[draft_id] = snapshot_draft_source(session)
    return draft_id


def ensure_upload_tab_focused(session: MutableMapping[str, Any]) -> str | None:
    """Select a New draft when the upload form is the page.

    The mix shell is left alone. A home shell whose tab id is not a draft
    keeps a file already on the form, otherwise focuses the last open draft,
    or opens one, so the strip underlines New.
    """
    if str(session.get(SHELL_VIEW_KEY) or "") == "mix":
        return None
    current = str(session.get(SHELL_TAB_KEY) or "")
    if is_new_draft_tab(current):
        return current
    if _live_upload_has_source(session):
        return _adopt_live_upload_as_draft(session)
    open_tabs = normalize_open_mix_tabs(
        session.get(OPEN_MIX_TABS_KEY), require_dir=False
    )
    drafts = [p for p in open_tabs if is_new_draft_tab(p)]
    if drafts:
        focus_new_draft_tab(session, drafts[-1])
        return drafts[-1]
    return open_new_draft_tab(session, fresh=True)


QUEUE_IN_FLIGHT_STATUSES = frozenset(
    {"queued", "running", "failed", "cancelled", "paused", "pausing"}
)
ISOLATE_UI_STATE_FILENAME = "isolate_ui_state.json"


def stopping_previous_caption() -> str:
    return "Stopping previous job… next track starts when ready"


def is_stopping_previous_job(active_job_id: str | None, job_status: str | None) -> bool:
    """True when the worker is tearing down a cancelled/pausing job."""
    if not active_job_id:
        return False
    return job_status in {"cancelled", "pausing"}


def queued_wait_caption(
    job_id: str,
    queued_ids: list[str],
    *,
    stopping_previous: bool = False,
) -> str:
    """Waiting line with queue position; never a fake percent."""
    if stopping_previous and queued_ids and job_id in queued_ids:
        return "Up next. Waiting for the previous job to stop."
    try:
        n = queued_ids.index(job_id) + 1
    except ValueError:
        return "Waiting…"
    return f"Waiting. Position {n} of {len(queued_ids)}"


def status_strip_waiting_caption(
    queued_ids: list[str],
    *,
    stopping_previous: bool,
) -> str:
    if stopping_previous:
        return stopping_previous_caption()
    if not queued_ids:
        return "Waiting…"
    return queued_wait_caption(queued_ids[0], queued_ids)


def paused_job_caption(completed_stages: Any = None) -> str:
    stages = list(completed_stages or [])
    if stages:
        return f"Paused. Resume continues after {stages[-1]}"
    return "Paused. Resume restarts from the beginning"


def partition_queue_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Split jobs into in-flight first, then succeeded. Empty only when both are empty."""
    in_flight: list[dict[str, Any]] = []
    succeeded: list[dict[str, Any]] = []
    for job in jobs:
        status = job.get("status")
        if status in QUEUE_IN_FLIGHT_STATUSES:
            in_flight.append(job)
        elif status == "succeeded":
            succeeded.append(job)
    return {
        "in_flight": in_flight,
        "succeeded": succeeded,
        "empty": not in_flight and not succeeded,
    }


def recent_runs_with_owner_fallback(
    owned: list[dict[str, Any]],
    unfiltered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Desktop library: use owner-scoped runs, else every isolate run on disk."""
    return owned if owned else unfiltered


def pick_library_row(
    rows: list[dict[str, Any]],
    preferred_run_dir: str | None,
) -> dict[str, Any] | None:
    """Newest row, unless preferred_run_dir still exists in the library."""
    if not rows:
        return None
    if preferred_run_dir:
        wanted = str(preferred_run_dir)
        for row in rows:
            if str(row.get("run_dir") or "") == wanted:
                return row
    return rows[0]


def listen_picker_default(
    options: list[str],
    loaded_run_dir: str | None,
    current: str | None,
) -> str | None:
    """Selectbox value: loaded mixer run, else current if still listed, else first option."""
    if not options:
        return None
    if loaded_run_dir and str(loaded_run_dir) in options:
        return str(loaded_run_dir)
    if current and str(current) in options:
        return str(current)
    return options[0]


def apply_listen_picker_pending(session: MutableMapping[str, Any]) -> None:
    """Apply a deferred picker reset before the selectbox is instantiated."""
    nxt = session.pop(LISTEN_PICKER_NEXT_KEY, None)
    if nxt:
        session[LISTEN_PICKER_KEY] = str(nxt)


def wav_artifact_paths(artifacts: object) -> frozenset[str]:
    """Normalized wav paths from a session or library artifact map.

    Diagnostic JSON sidecars are ignored. Path case and separators are
    normalized so a Windows rescan does not look like a stem-set change.
    """
    if not isinstance(artifacts, dict):
        return frozenset()
    found: set[str] = set()
    for name, path in artifacts.items():
        if str(name).endswith("_diagnostics"):
            continue
        text = str(path or "").strip()
        if not text.lower().endswith(".wav"):
            continue
        found.add(os.path.normcase(text.replace("\\", "/")))
    return frozenset(found)


def refresh_open_mix_action(session: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """How Refresh should update the loaded mix from its library row.

    ``stems`` — wav path set changed; the caller may re-apply the row, which
    resets mixer faders. ``title`` — only the display name changed; update
    ``isolate_base_name`` and leave faders. ``unchanged`` — leave the mixer.
    """
    if wav_artifact_paths(session.get("isolate_artifacts")) != wav_artifact_paths(
        row.get("artifacts")
    ):
        return "stems"
    title = str(row.get("title") or "tracks")
    if title != str(session.get("isolate_base_name") or ""):
        return "title"
    return "unchanged"


def select_rehydrate_row(
    session: MutableMapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    wav_exists,
) -> dict[str, Any] | None:
    """Library row to load when the session has no usable mixer wavs.

    Skips when the user just closed the last mix tab — otherwise clearing
    artifacts for that close would immediately revive the tab via rehydrate.
    """
    if session.get(ISOLATE_SKIP_REHYDRATE_KEY):
        return None
    if session_mixer_artifacts_ok(session, wav_exists=wav_exists):
        return None
    preferred = session.get("isolate_viewing_run_dir") or session.get("isolate_run_dir")
    return pick_library_row(rows, str(preferred) if preferred else None)


def session_mixer_artifacts_ok(session: MutableMapping[str, Any], *, wav_exists) -> bool:
    """True when session already points at mixer wavs that still exist."""
    arts = session.get("isolate_artifacts")
    if not isinstance(arts, dict) or not arts:
        return False
    return any(
        str(path).endswith(".wav") and wav_exists(str(path))
        for path in arts.values()
    )


UI_MODE_KEY = "ui_mode"
UI_MODES: tuple[str, ...] = ("Lite", "Pro")
DEFAULT_UI_MODE = "Lite"


def resolve_ui_mode(value: Any) -> str:
    """Normalise a stored or session mode value to a known mode.

    Anything unrecognised falls back to Lite, so a corrupt state file cannot
    strand someone in a UI they did not ask for.
    """
    text = str(value or "").strip().title()
    return text if text in UI_MODES else DEFAULT_UI_MODE


def is_pro_mode(session: MutableMapping[str, Any]) -> bool:
    """Whether Pro controls should render. Lite auto-profiles; Pro is manual."""
    return resolve_ui_mode(session.get(UI_MODE_KEY)) == "Pro"


# Widget keys mirrored to disk so choices survive an app restart. Streamlit's
# persist_state only spans page switches inside one session, so re-launching the
# desktop app used to silently reset every Advanced choice back to default.
PERSISTED_SETTING_KEYS: tuple[str, ...] = (
    "isolate_speed_preset",
    "isolate_speed_applied",
    "isolate_quality",
    "isolate_device",
    "isolate_lite_run_on",
    "isolate_lite_guitar_engine",
    "isolate_guitar_ft",
    "isolate_two_pass",
    "isolate_guitar_refine",
    "isolate_demucs_shifts",
    "isolate_demucs_overlap",
    "isolate_demucs_segment",
    "isolate_demucs_jobs",
    "isolate_demucs_compute_source_quality",
    "isolate_fold_other_mode",
    "isolate_adaptive_fold_gain",
    "isolate_bleed_gate",
    "isolate_bass_bleed_mitigation",
    "isolate_sub_bass_debleed",
    "isolate_low_end_restore_db",
    "isolate_guitar_ensemble",
    "isolate_htdemucs_ft",
    "isolate_lead_rhythm",
    "isolate_options_expanded",
    "isolate_vocals_instrumental_only",
    "isolate_guitar_track",
    "isolate_track_options",
    "isolate_track_picker_initialized",
    "isolate_metro_accent",
    "isolate_metro_rate",
    "isolate_metro_sound",
    "isolate_metro_follow",
    *(f"isolate_track_{oid}" for oid in DEMUCS_STEM_CHECKBOX_IDS),
)

DEMUCS_SHIFT_OPTIONS: tuple[str, ...] = ("0", "1", "3", "5")
DEMUCS_OVERLAP_OPTIONS: tuple[str, ...] = ("0.25", "0.5", "0.75")
DEFAULT_DEMUCS_SEGMENT = 8
DEFAULT_DEMUCS_JOBS = 1
FOLD_OTHER_MODE_ORDER: tuple[str, ...] = ("best_effort", "band_limited", "full")
PRO_LOW_END_RESTORE_MAX_DB = 6.0


def quality_demucs_shifts(quality: str) -> str:
    return QUALITY_SHIFTS.get(str(quality or "").strip(), "0")


def quality_demucs_overlap(quality: str) -> str:
    return QUALITY_OVERLAP.get(str(quality or "").strip(), "0.25")


def parse_demucs_segment(value: Any) -> int | None:
    """Widget 0 / missing → model default (None). Otherwise a positive int."""
    if value is None or value == "":
        return DEFAULT_DEMUCS_SEGMENT
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_DEMUCS_SEGMENT
    if n <= 0:
        return None
    return n


def parse_demucs_jobs(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_DEMUCS_JOBS


def parse_demucs_shifts(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_demucs_overlap(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_fold_other_mode(value: Any) -> str:
    pick = str(value or "").strip()
    if pick in FOLD_OTHER_MODES:
        return pick
    return "best_effort"


def demucs_compute_is_overridden(*, quality: str, shifts: Any, overlap: Any) -> bool:
    return (
        str(shifts) != quality_demucs_shifts(quality)
        or str(overlap) != quality_demucs_overlap(quality)
    )


def sync_demucs_compute_defaults(
    session: MutableMapping[str, Any],
    *,
    quality: str,
) -> None:
    """Reset shift/overlap to the Quality map unless the user overrode them."""
    q = str(quality or "fast")
    if q not in QUALITY_SHIFTS:
        q = "fast"
    new_shifts = quality_demucs_shifts(q)
    new_overlap = quality_demucs_overlap(q)
    last_q = session.get("isolate_demucs_compute_source_quality")
    last_q = str(last_q) if last_q in QUALITY_SHIFTS else None
    cur_shifts = session.get("isolate_demucs_shifts")
    cur_overlap = session.get("isolate_demucs_overlap")
    if last_q is None:
        if cur_shifts not in DEMUCS_SHIFT_OPTIONS:
            session["isolate_demucs_shifts"] = new_shifts
        if cur_overlap not in DEMUCS_OVERLAP_OPTIONS:
            session["isolate_demucs_overlap"] = new_overlap
    elif last_q != q:
        if cur_shifts not in DEMUCS_SHIFT_OPTIONS or str(cur_shifts) == quality_demucs_shifts(last_q):
            session["isolate_demucs_shifts"] = new_shifts
        if cur_overlap not in DEMUCS_OVERLAP_OPTIONS or str(cur_overlap) == quality_demucs_overlap(last_q):
            session["isolate_demucs_overlap"] = new_overlap
    session["isolate_demucs_compute_source_quality"] = q
    if "isolate_demucs_segment" not in session:
        session["isolate_demucs_segment"] = DEFAULT_DEMUCS_SEGMENT
    if "isolate_demucs_jobs" not in session:
        session["isolate_demucs_jobs"] = DEFAULT_DEMUCS_JOBS
    if session.get("isolate_fold_other_mode") not in FOLD_OTHER_MODES:
        session["isolate_fold_other_mode"] = "best_effort"
    if "isolate_low_end_restore_db" not in session:
        session["isolate_low_end_restore_db"] = 0.0


def resolve_pro_engine_job_fields(
    session: Mapping[str, Any],
    *,
    is_pro: bool,
    model: str,
    quality: str,
    two_stems: str | None,
    two_pass: bool,
    fold_other_into_guitar: bool,
    fold_other_mode: str,
    roformer_available: bool = True,
) -> dict[str, Any]:
    """Lite keeps IsolateConfig defaults. Pro reads Engine widget keys.

    Untouched Pro session still matches prior behaviour (quality→shifts/overlap,
    segment 8, jobs 1, fold best_effort, post knobs off).
    """
    resolved_model = str(model or "htdemucs_6s")
    fold_mode = normalize_fold_other_mode(fold_other_mode)
    out: dict[str, Any] = {
        "model": resolved_model,
        "demucs_shifts": None,
        "demucs_overlap": None,
        "demucs_segment": DEFAULT_DEMUCS_SEGMENT,
        "demucs_jobs": DEFAULT_DEMUCS_JOBS,
        "fold_other_mode": fold_mode,
        "adaptive_fold_gain": False,
        "bleed_gate": False,
        "bass_bleed_mitigation": False,
        "sub_bass_debleed": False,
        "low_end_restore_db": 0.0,
        "guitar_ensemble": False,
        "lead_rhythm": False,
        "lead_rhythm_mode": "confident",
    }
    if not is_pro:
        return out
    if resolved_model == "htdemucs_6s" and bool(session.get("isolate_lead_rhythm")):
        out["lead_rhythm"] = True
    if resolved_model == "htdemucs" and bool(session.get("isolate_htdemucs_ft")):
        out["model"] = "htdemucs_ft"
    out["demucs_shifts"] = parse_demucs_shifts(session.get("isolate_demucs_shifts"))
    if out["demucs_shifts"] is not None and str(out["demucs_shifts"]) == quality_demucs_shifts(quality):
        out["demucs_shifts"] = None
    out["demucs_overlap"] = parse_demucs_overlap(session.get("isolate_demucs_overlap"))
    if out["demucs_overlap"] is not None:
        default_overlap = float(quality_demucs_overlap(quality))
        if abs(out["demucs_overlap"] - default_overlap) < 1e-9:
            out["demucs_overlap"] = None
    out["demucs_segment"] = parse_demucs_segment(
        session.get("isolate_demucs_segment", DEFAULT_DEMUCS_SEGMENT)
    )
    out["demucs_jobs"] = parse_demucs_jobs(
        session.get("isolate_demucs_jobs", DEFAULT_DEMUCS_JOBS)
    )
    if fold_other_into_guitar:
        out["fold_other_mode"] = normalize_fold_other_mode(
            session.get("isolate_fold_other_mode") or fold_mode
        )
        out["adaptive_fold_gain"] = bool(session.get("isolate_adaptive_fold_gain"))
    out["bleed_gate"] = bool(session.get("isolate_bleed_gate"))
    out["bass_bleed_mitigation"] = bool(session.get("isolate_bass_bleed_mitigation"))
    out["sub_bass_debleed"] = bool(session.get("isolate_sub_bass_debleed"))
    try:
        boost = float(session.get("isolate_low_end_restore_db") or 0.0)
    except (TypeError, ValueError):
        boost = 0.0
    out["low_end_restore_db"] = max(0.0, min(PRO_LOW_END_RESTORE_MAX_DB, boost))
    ensemble_ok = (
        bool(roformer_available)
        and not two_stems
        and not two_pass
        and out["model"] in DEMUCS_MODELS
    )
    out["guitar_ensemble"] = bool(session.get("isolate_guitar_ensemble")) and ensemble_ok
    return out


def persisted_settings_payload(session: MutableMapping[str, Any]) -> dict[str, Any]:
    """Snapshot of the setting keys worth carrying across restarts."""
    return {key: session[key] for key in PERSISTED_SETTING_KEYS if key in session}


def apply_persisted_settings(
    session: MutableMapping[str, Any],
    stored: dict[str, Any] | None,
) -> None:
    """Restore last-used settings without ever clobbering a live session value.

    Must run before the widgets are instantiated, otherwise Streamlit treats the
    write as a post-instantiation mutation and raises.
    """
    for key, value in (stored or {}).items():
        if key in PERSISTED_SETTING_KEYS and key not in session:
            session[key] = value


def isolate_ui_state_payload(session: MutableMapping[str, Any]) -> dict[str, Any]:
    view = session.get(SHELL_VIEW_KEY)
    if view not in SHELL_VIEWS:
        nxt = session.get(WORKSPACE_NEXT_KEY)
        tab = nxt if nxt in WORKSPACE_TABS else session.get(WORKSPACE_KEY)
        view = "mix" if tab == "Mixer" else "home"
    viewing = session.get("isolate_viewing_run_dir")
    run_dir = session.get("isolate_run_dir")
    export_dir = session.get(ISOLATE_EXPORT_DIR_KEY)
    open_tabs = normalize_open_mix_tabs(session.get(OPEN_MIX_TABS_KEY), require_dir=False)
    shell_tab = session.get(SHELL_TAB_KEY)
    return {
        "workspace": "Mixer" if view == "mix" else "New",
        "shell_view": view,
        "shell_tab": str(shell_tab) if shell_tab else None,
        "viewing_run_dir": str(viewing) if viewing else None,
        "run_dir": str(run_dir) if run_dir else None,
        "export_dir": str(export_dir) if export_dir else None,
        "library_folder": str(session["library_folder"]) if session.get("library_folder") else None,
        "sync_folder": str(session["sync_folder"]) if session.get("sync_folder") else None,
        "open_mix_tabs": open_tabs,
        "mode": resolve_ui_mode(session.get(UI_MODE_KEY)),
        "settings": persisted_settings_payload(session),
    }


def apply_stored_isolate_ui_state(
    session: MutableMapping[str, Any],
    stored: dict[str, Any] | None,
) -> None:
    """Fill missing session tab/run pointers from disk. Live session keys win."""
    if not stored:
        return
    shell = stored.get("shell_view")
    if session.get(SHELL_VIEW_KEY) not in SHELL_VIEWS:
        if shell in SHELL_VIEWS:
            session[SHELL_VIEW_KEY] = shell
        else:
            tab = stored.get("workspace")
            if tab == "Mixer":
                session[SHELL_VIEW_KEY] = "mix"
            elif tab in WORKSPACE_TABS:
                session[SHELL_VIEW_KEY] = "home"
    tab = stored.get("workspace")
    if session.get(WORKSPACE_KEY) not in WORKSPACE_TABS and tab in WORKSPACE_TABS:
        session[WORKSPACE_KEY] = tab
    if not session.get("isolate_viewing_run_dir") and stored.get("viewing_run_dir"):
        session["isolate_viewing_run_dir"] = str(stored["viewing_run_dir"])
    if not session.get("isolate_run_dir") and stored.get("run_dir"):
        session["isolate_run_dir"] = str(stored["run_dir"])
    if not session.get(ISOLATE_EXPORT_DIR_KEY) and stored.get("export_dir"):
        session[ISOLATE_EXPORT_DIR_KEY] = str(stored["export_dir"])
    if not session.get("library_folder") and stored.get("library_folder"):
        session["library_folder"] = str(stored["library_folder"])
    if not session.get("sync_folder") and stored.get("sync_folder"):
        session["sync_folder"] = str(stored["sync_folder"])
    if OPEN_MIX_TABS_KEY not in session and stored.get("open_mix_tabs") is not None:
        session[OPEN_MIX_TABS_KEY] = normalize_open_mix_tabs(
            stored.get("open_mix_tabs"), require_dir=False
        )
    if SHELL_TAB_KEY not in session and stored.get("shell_tab"):
        session[SHELL_TAB_KEY] = str(stored["shell_tab"])
    if UI_MODE_KEY not in session and stored.get("mode"):
        session[UI_MODE_KEY] = resolve_ui_mode(stored["mode"])
    apply_persisted_settings(session, stored.get("settings"))


def read_isolate_ui_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_isolate_ui_state(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


# How long a failure keeps the top strip before it becomes Queue-only history.
FAILED_JOB_STRIP_TTL_SEC = 15 * 60.0
DISMISSED_JOB_IDS_KEY = "isolate_dismissed_job_ids"


def should_show_failed_job(
    job: Mapping[str, Any] | None,
    *,
    now: float,
    dismissed_ids: Iterable[str] = (),
    ttl_sec: float = FAILED_JOB_STRIP_TTL_SEC,
) -> bool:
    """Whether a failed job still earns the top strip.

    A failure used to sit there until the job was manually removed, so the strip
    became permanent furniture and every later success was reported underneath a
    red box. It clears when dismissed, or on its own after ``ttl_sec``.
    """
    if not job:
        return False
    job_id = str(job.get("id") or "").strip()
    if not job_id or job_id in {str(d) for d in dismissed_ids}:
        return False
    try:
        failed_at = float(job.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        return True
    if failed_at <= 0.0:
        return True
    return (now - failed_at) < ttl_sec


def dismiss_failed_job(session: MutableMapping[str, Any], job_id: str) -> None:
    """Hide one failed job from the strip for the rest of this session."""
    current = [str(j) for j in (session.get(DISMISSED_JOB_IDS_KEY) or [])]
    if job_id and job_id not in current:
        current.append(str(job_id))
    session[DISMISSED_JOB_IDS_KEY] = current


def load_ui_mode(session: MutableMapping[str, Any], path: Path) -> str:
    """Resolve the interface mode for this run, seeding session from disk once."""
    if UI_MODE_KEY not in session:
        session[UI_MODE_KEY] = resolve_ui_mode(read_isolate_ui_state(path).get("mode"))
    return resolve_ui_mode(session.get(UI_MODE_KEY))


def write_ui_mode(path: Path, mode: str) -> None:
    """Persist only the mode, merging into whatever else is already stored.

    The shell writes this from the sidebar, where the rest of the isolate state
    is not in scope — a full payload write from there would erase it.
    """
    payload = read_isolate_ui_state(path)
    payload["mode"] = resolve_ui_mode(mode)
    write_isolate_ui_state(path, payload)


def seed_consumed_job_ids(jobs: list[dict[str, Any]]) -> list[str]:
    """Succeeded job ids already on disk — treat as seen so launch does not rerun."""
    ids: list[str] = []
    for job in jobs:
        if job.get("status") != "succeeded":
            continue
        jid = str(job.get("id") or "").strip()
        if jid:
            ids.append(jid)
    return ids


_OS_NOTIFY_STATUSES = frozenset({"succeeded", "failed"})
_OS_NOTIFY_APP_TITLE = "Audio Isolation"
# When notified_ids was seeded against an empty job list, the next poll must not
# toast the entire history. Only toast terminal jobs that finished this recently.
_OS_NOTIFY_CATCHUP_SEC = 180.0


def seed_notified_job_ids(jobs: list[dict[str, Any]]) -> list[str]:
    """Succeeded/failed ids already on disk — treat as seen so launch does not toast."""
    ids: list[str] = []
    for job in jobs:
        if job.get("status") not in _OS_NOTIFY_STATUSES:
            continue
        jid = str(job.get("id") or "").strip()
        if jid:
            ids.append(jid)
    return ids


def os_notify_message(job: dict[str, Any]) -> tuple[str, str] | None:
    """App title and body for a terminal job, or None if it should not toast."""
    status = job.get("status")
    title = str(job.get("title") or "track").strip() or "track"
    if len(title) > 80:
        title = title[:77] + "..."
    if status == "succeeded":
        return _OS_NOTIFY_APP_TITLE, f"Separated: {title}"
    if status == "failed":
        return _OS_NOTIFY_APP_TITLE, f"Needs attention: {title} failed"
    return None


def jobs_needing_os_notify(
    jobs: list[dict[str, Any]],
    notified_ids: list[str] | None,
    *,
    now: float | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(updated_ids, jobs_to_toast)``. First call (``None``) seeds without toasting.

    An empty ``notified_ids`` list is not "toast everyone". That happens when the
    first poll saw no jobs yet; the next poll would otherwise re-notify every
    historical failure when a new separation finishes.
    """
    if notified_ids is None:
        return seed_notified_job_ids(jobs), []
    clock = time.time() if now is None else now
    if not notified_ids:
        seeded = seed_notified_job_ids(jobs)
        pending: list[dict[str, Any]] = []
        for job in jobs:
            if job.get("status") not in _OS_NOTIFY_STATUSES:
                continue
            finished = float(job.get("finished_at") or job.get("updated_at") or 0.0)
            if finished > 0.0 and (clock - finished) <= _OS_NOTIFY_CATCHUP_SEC:
                pending.append(job)
        return seeded, pending
    seen = {str(item) for item in notified_ids if item}
    updated = [str(item) for item in notified_ids if item]
    pending = []
    for job in jobs:
        if job.get("status") not in _OS_NOTIFY_STATUSES:
            continue
        jid = str(job.get("id") or "").strip()
        if not jid or jid in seen:
            continue
        pending.append(job)
        seen.add(jid)
        updated.append(jid)
    return updated, pending


def next_unconsumed_succeeded_job(
    jobs: list[dict[str, Any]],
    consumed_ids: list[str],
    viewing_id: str | None,
    *,
    shell_view: str | None = None,
    viewing_run_dir: str | None = None,
    listen_picker: str | None = None,
    shell_tab: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """First succeeded job not yet consumed. Bool is True when the mixer may auto-load it."""
    seen = {str(item) for item in consumed_ids if item}
    for job in jobs:
        jid = str(job.get("id") or "").strip()
        if not jid or jid in seen or job.get("status") != "succeeded":
            continue
        return job, should_auto_apply_job(
            viewing_id,
            jid,
            shell_view=shell_view,
            viewing_run_dir=viewing_run_dir,
            listen_picker=listen_picker,
            shell_tab=shell_tab,
            finished_run_dir=str(job.get("run_dir") or "") or None,
        )
    return None, False


def plan_isolate_job_poll(
    jobs: list[dict[str, Any]],
    *,
    consumed_ids: list[str] | None,
    viewing_id: str | None,
    form_drawn: bool,
    shell_view: str | None = None,
    viewing_run_dir: str | None = None,
    listen_picker: str | None = None,
    shell_tab: str | None = None,
) -> dict[str, Any]:
    """Decide whether a job poll may apply a result or rerun the page.

    A fresh session (``consumed_ids is None``) records historical successes and
    never reruns — otherwise two finished jobs ping-pong ``st.rerun`` and the
    Isolate form never paints.
    """
    if consumed_ids is None:
        return {
            "consumed_ids": seed_consumed_job_ids(jobs),
            "apply_job": None,
            "notify_only": False,
            "rerun": False,
        }
    consumed = [str(item) for item in consumed_ids if item]
    job, auto = next_unconsumed_succeeded_job(
        jobs,
        consumed,
        viewing_id,
        shell_view=shell_view,
        viewing_run_dir=viewing_run_dir,
        listen_picker=listen_picker,
        shell_tab=shell_tab,
    )
    if job is None or not form_drawn:
        return {
            "consumed_ids": consumed,
            "apply_job": None,
            "notify_only": False,
            "rerun": False,
        }
    jid = str(job.get("id") or "")
    return {
        "consumed_ids": consumed + [jid],
        "apply_job": job,
        "notify_only": not auto,
        "rerun": True,
    }


ISOLATE_SCROLL_TOP_KEY = "_isolate_scroll_top"
ISOLATE_SCROLL_TOP_TOKEN_KEY = "_isolate_scroll_top_token"
MIX_TAB_EVENT_SEQ_KEY = "_mix_tabs_last_seq"
MIX_TAB_ACTIONS = frozenset(
    {
        "home",
        "plus",
        "focus",
        "close",
        "reorder",
    }
)

# Navigation-level Isolate actions: full remount, then scroll the main pane up.
ISOLATE_SCROLL_TOP_ACTIONS = frozenset(
    {
        "enqueue_separate",
        "shell_nav",
        "open_mixer",
        "poll_apply_mixer",
    }
)

# In-place controls and 1s job polls: keep the current main-pane scroll.
ISOLATE_PRESERVE_SCROLL_ACTIONS = frozenset(
    {
        "tile_pick",
        "engine_widget",
        "region",
        "upload_youtube",
        "status_control",
        "queue_control",
        "mixer_fader",
        "poll_progress",
        "poll_failed",
    }
)


def request_isolate_scroll_top(session: MutableMapping[str, Any]) -> None:
    """Ask the Isolate page to scroll ``section.main`` / ``stMain`` on the next paint."""
    session[ISOLATE_SCROLL_TOP_KEY] = True
    try:
        current = int(session.get(ISOLATE_SCROLL_TOP_TOKEN_KEY) or 0)
    except (TypeError, ValueError):
        current = 0
    session[ISOLATE_SCROLL_TOP_TOKEN_KEY] = current + 1


def consume_isolate_scroll_top(session: MutableMapping[str, Any]) -> bool:
    """True once per requested jump. The scroll iframe slot is always mounted."""
    return bool(session.pop(ISOLATE_SCROLL_TOP_KEY, False))


def isolate_scroll_top_token(session: Mapping[str, Any]) -> int:
    """Monotonic token embedded in the always-on scroll iframe script."""
    try:
        return int(session.get(ISOLATE_SCROLL_TOP_TOKEN_KEY) or 0)
    except (TypeError, ValueError):
        return 0


def consume_mix_tab_event(
    result: object, last_seq: int
) -> tuple[int, str | None, str]:
    """Apply a mix-tabs click once. Replay of the same ``seq`` is ignored.

    Returns ``(new_seq, action, tab_id)``. ``action`` is None when the payload
    should not change the shell. Payloads without ``seq`` (older builds) still
    apply once by synthesizing a monotonic seq from ``last_seq``.
    """
    try:
        prev = int(last_seq or 0)
    except (TypeError, ValueError):
        prev = 0
    if not isinstance(result, dict):
        return prev, None, ""
    action = str(result.get("action") or "")
    if action not in MIX_TAB_ACTIONS:
        return prev, None, ""
    raw_seq = result.get("seq")
    try:
        seq = int(raw_seq) if raw_seq is not None else 0
    except (TypeError, ValueError):
        return prev, None, ""
    if seq <= 0:
        # Legacy iframe without seq: still accept the click once.
        seq = prev + 1
    if seq <= prev:
        return prev, None, ""
    return seq, action, str(result.get("id") or "")


def mix_tab_event_index(result: object) -> int | None:
    """Target slot from a ``reorder`` payload, else None."""
    if not isinstance(result, dict):
        return None
    raw = result.get("to")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def should_request_isolate_scroll_top(action: str) -> bool:
    """Mixed scroll policy: nav/enqueue/Mixer-open vs in-place clicks and polls.

    Unknown actions preserve scroll (do not jump the main pane).
    """
    if action in ISOLATE_SCROLL_TOP_ACTIONS:
        return True
    return False


def isolate_poll_requires_full_rerun(*, applied: bool, plan_rerun: bool) -> bool:
    """Full-app remount only when a finished job is applied and Mixer must open.

    Running/queued progress ticks and failure toasts stay fragment-local so the
    New-tab scroll position does not reset every second.
    """
    return bool(applied and plan_rerun)


ISOLATE_USER_ID_FILENAME = "isolate_user_id"

SOURCE_KIND_FILE = "file"
SOURCE_KIND_YOUTUBE = "youtube"
_YOUTUBE_FINGERPRINT_PREFIX = "youtube:"


def infer_source_kind(
    *,
    source_kind: Any = None,
    source_fingerprint: Any = None,
    youtube_url: Any = None,
) -> str:
    """``youtube`` or ``file``. Explicit kind wins; else fingerprint/URL; else file."""
    kind = str(source_kind or "").strip().lower()
    if kind in {SOURCE_KIND_FILE, SOURCE_KIND_YOUTUBE}:
        return kind
    if str(youtube_url or "").strip():
        return SOURCE_KIND_YOUTUBE
    fingerprint = str(source_fingerprint or "")
    if fingerprint.startswith(_YOUTUBE_FINGERPRINT_PREFIX):
        return SOURCE_KIND_YOUTUBE
    return SOURCE_KIND_FILE


def format_source_title(title: str | None, kind: str) -> str:
    """Queue / Listening-to label, e.g. ``YouTube · Party Wadokoni_``."""
    name = (title or "tracks").strip() or "tracks"
    label = "YouTube" if infer_source_kind(source_kind=kind) == SOURCE_KIND_YOUTUBE else "File"
    return f"{label} · {name}"


def format_source_caption(
    title: str | None,
    kind: str,
    *,
    filename: str | None = None,
    region_label: str | None = None,
    clip_length: float | None = None,
) -> str:
    """Mixer caption, e.g. ``**Calm Like You** · File (`Calm Like You.wav`)``."""
    name = (title or "tracks").strip() or "tracks"
    parts = [f"**{name}**"]
    if region_label:
        length_note = f" ({float(clip_length):.0f} s)" if clip_length else ""
        parts.append(f"{region_label}{length_note}")
    if infer_source_kind(source_kind=kind) == SOURCE_KIND_YOUTUBE:
        parts.append("YouTube")
    elif filename:
        parts.append(f"File (`{filename}`)")
    else:
        parts.append("File")
    return " · ".join(parts)


def read_stored_user_id(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_stored_user_id(path: Path, user_id: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(user_id, encoding="utf-8")
    except OSError:
        pass


def load_persist_isolate_user_id(session: MutableMapping[str, Any], path: Path) -> str:
    """Session UUID, persisted to a local file (never a blocking browser component)."""
    stored = read_stored_user_id(path)
    user_id = resolve_isolate_user_id(session, stored)
    if stored != user_id:
        write_stored_user_id(path, user_id)
    return user_id


def staged_audio_for_new_tab(
    *,
    uploaded: bool,
    youtube_url: str,
    pending_path: str | None,
    pending_fp: str | None,
    pending_exists: bool,
    carry_path: str | None,
    carry_exists: bool,
) -> str | None:
    """Path to preview on New, or None when there is nothing ready to play.

    A pasted YouTube URL does not preview until that URL has been downloaded
    (pending fingerprint ``youtube:{url}``). A direct media link previews
    once Continue has saved it (``direct:{url}``). Uploads and Tab PDF
    carry-over preview immediately because the file is already on disk.
    """
    url = (youtube_url or "").strip()
    if url:
        if pending_exists and pending_fp in {f"youtube:{url}", f"direct:{url}"}:
            return pending_path
        return None
    if pending_exists:
        return pending_path
    if not uploaded and carry_exists:
        return carry_path
    return None


def pending_upload_fp_for_stale(session: MutableMapping[str, Any]) -> str | None:
    """Fingerprint of a staged isolate source, independent of the file_uploader widget.

    Used when the New tab is not mounted so ``uploaded`` is unavailable.
    """
    pending_fp = session.get("isolate_pending_fp") or session.get("isolate_upload_fp")
    pending_path = session.get("isolate_pending_audio_path")
    has_file = bool(pending_path and Path(str(pending_path)).exists())
    youtube_url = (session.get("isolate_youtube_url") or "").strip()
    if (has_file or youtube_url) and pending_fp:
        return str(pending_fp)
    return None


def pending_audio_needs_resave(
    *,
    uploaded_fp: str | None,
    pending_fp: str | None,
    pending_path_exists: bool,
) -> bool:
    """True when the uploader's file is not yet mirrored on the pending path."""
    if uploaded_fp is None:
        return False
    if not pending_path_exists:
        return True
    return uploaded_fp != pending_fp


def expects_guitar_stem(
    *,
    model: str,
    two_stems: str | None = None,
    custom_stems: Any = None,
) -> bool:
    """True when this job is expected to produce a guitar stem (bass/split stages)."""
    if two_stems:
        return False
    picked = [s for s in (custom_stems or []) if s]
    if picked:
        return "guitar" in picked
    lowered = (model or "").lower()
    return any(marker in lowered for marker in _GUITAR_MODEL_MARKERS)


def isolation_stages_for_job(
    *,
    expects_guitar: bool,
    guitar_refine: bool = False,
    analyse: bool = True,
    tab: bool = True,
) -> tuple[str, ...]:
    """Stages that will actually run for this job.

    ``analyse``/``tab`` let the worker omit optional post-processing (for example, a
    re-separate child that must not clobber its parent's analysis artifacts).
    """
    stages = ISOLATION_STAGE_ORDER
    if not expects_guitar:
        stages = tuple(s for s in stages if s not in _GUITAR_ONLY_STAGES)
    if not analyse:
        stages = tuple(s for s in stages if s != "analysis")
    if not tab:
        stages = tuple(s for s in stages if s != "tab")
    if guitar_refine and expects_guitar:
        out: list[str] = []
        for stage in stages:
            out.append(stage)
            if stage == "collect":
                out.append("guitar_refine")
        return tuple(out)
    return stages


def resolve_active_stage(stage: str, stages: tuple[str, ...]) -> str:
    """Map a pipeline stage onto the job's visible stage list."""
    if stage in stages:
        return stage
    if not stages:
        return "ingest"
    if stage not in ISOLATION_STAGE_ORDER:
        return stages[0]
    idx = ISOLATION_STAGE_ORDER.index(stage)
    for later in ISOLATION_STAGE_ORDER[idx + 1 :]:
        if later in stages:
            return later
    return stages[-1]


def stage_progress_percent(
    stage: str,
    *,
    weights: dict[str, float] | None = None,
    stages: tuple[str, ...] | None = None,
    intra: float = 0.0,
) -> float:
    """Weighted progress (0.0–1.0): prior stages + ``intra`` of the current stage.

    ``intra`` is 0 at the start of a stage and 1 when that stage has finished.
    """
    order = stages or ISOLATION_STAGE_ORDER
    w = weights or STAGE_WEIGHTS
    total = sum(w.get(s, 0.0) for s in order)
    if total <= 0:
        return 0.0
    active = resolve_active_stage(stage, order)
    if active not in order:
        return 0.0
    idx = order.index(active)
    prior = sum(w.get(order[i], 0.0) for i in range(idx))
    current = w.get(active, 0.0)
    frac = min(1.0, max(0.0, intra))
    return min(1.0, (prior + current * frac) / total)


def intra_stage_fraction(
    elapsed_in_stage_sec: float,
    estimated_stage_sec: float | None,
) -> tuple[float, bool]:
    """Progress within the current stage. Caps below 1 so we never invent completion.

    Returns ``(fraction, is_estimated)``. ``is_estimated`` is True when a duration
    guess was used. Without a guess, fraction stays 0.
    """
    if estimated_stage_sec is None or estimated_stage_sec <= 0:
        return 0.0, False
    if elapsed_in_stage_sec <= 0:
        return 0.0, True
    return min(_INTRA_STAGE_CAP, elapsed_in_stage_sec / estimated_stage_sec), True


def estimated_stage_seconds(
    stage: str,
    stages: tuple[str, ...],
    total_job_sec: float | None,
    *,
    weights: dict[str, float] | None = None,
) -> float | None:
    """Share of the job estimate that belongs to ``stage``."""
    if total_job_sec is None or total_job_sec <= 0:
        return None
    w = weights or STAGE_WEIGHTS
    total_w = sum(w.get(s, 0.0) for s in stages)
    if total_w <= 0:
        return None
    return total_job_sec * (w.get(stage, 0.0) / total_w)


def estimate_job_seconds(
    *,
    audio_duration_sec: float | None,
    quality: str,
    device: str,
    stages: tuple[str, ...],
    last_run: dict[str, Any] | None = None,
    model: str | None = None,
    expects_guitar: bool | None = None,
    two_pass: bool = False,
    guitar_refine: bool = False,
    cpu_threads: int | None = None,
) -> tuple[float | None, str]:
    """Predicted wall time for the whole job.

    Uses a prior run when quality/device/model match (confidence ``high``),
    otherwise a clip-length × quality × device heuristic (``low``).
    Two-pass isolation is about 2× the Demucs ``separate`` stage.
    Guitar refine and BS-RoFormer-SW add extra wall time.
    """
    if last_run:
        same_setup = (
            last_run.get("quality") == quality
            and last_run.get("device") == device
            and (model is None or last_run.get("model") == model)
            and (expects_guitar is None or bool(last_run.get("expects_guitar")) == bool(expects_guitar))
            and bool(last_run.get("two_pass")) == bool(two_pass)
            and bool(last_run.get("guitar_refine")) == bool(guitar_refine)
        )
        last_audio = float(last_run.get("audio_sec") or 0.0)
        last_wall = float(last_run.get("wall_sec") or 0.0)
        if (
            same_setup
            and last_audio > 0.5
            and last_wall > 0.5
            and audio_duration_sec
            and audio_duration_sec > 0
        ):
            return last_wall * (audio_duration_sec / last_audio), "high"

    if not audio_duration_sec or audio_duration_sec <= 0:
        return None, "low"

    model_id = (model or "").lower()
    if model_id in ROFORMER_MODEL_IDS:
        d = _ROFORMER_REALTIME.get(device, _ROFORMER_REALTIME["cpu"])
        if device == "cpu" and cpu_threads:
            d = d * (_REFERENCE_P_CORES / max(1, cpu_threads))
        separate_sec = audio_duration_sec * d
        if guitar_refine:
            ref = _ROFORMER_REFINE_REALTIME.get(device, _ROFORMER_REFINE_REALTIME["cpu"])
            separate_sec += audio_duration_sec * ref
    else:
        q = _QUALITY_TIME_FACTOR.get(quality, 1.0)
        d = _DEVICE_REALTIME.get(device, _DEVICE_REALTIME["cpu"])
        pass_factor = 2.0 if two_pass else 1.0
        separate_sec = audio_duration_sec * q * d * pass_factor
    sep_w = STAGE_WEIGHTS["separate"]
    active_w = sum(STAGE_WEIGHTS.get(s, 0.0) for s in stages)
    if sep_w <= 0 or active_w <= 0:
        return separate_sec, "low"
    return separate_sec * (active_w / sep_w), "low"


def estimate_remaining_seconds(
    *,
    elapsed_sec: float,
    percent: float,
    total_estimate: float | None,
    confidence: str,
) -> tuple[float | None, str]:
    """Seconds left. Prefers observed rate once enough progress exists."""
    if percent >= 0.995:
        return 0.0, "high"
    observed: float | None = None
    if elapsed_sec >= 3.0 and percent >= 0.03:
        observed = elapsed_sec * (1.0 - percent) / percent
    planned: float | None = None
    if total_estimate is not None:
        planned = max(0.0, total_estimate - elapsed_sec)
    if observed is None and planned is None:
        return None, "low"
    if observed is None:
        return planned, confidence
    if planned is None:
        return observed, "high" if elapsed_sec >= 8 else "low"
    blend = min(1.0, elapsed_sec / 20.0)
    mixed = (1.0 - blend) * planned + blend * observed
    out_conf = "high" if confidence == "high" or elapsed_sec >= 8 else "low"
    return max(0.0, mixed), out_conf


def checklist_items(
    stages: tuple[str, ...],
    current_stage: str,
    *,
    labels: dict[str, str] | None = None,
    complete: bool = False,
) -> list[dict[str, str]]:
    """Per-stage rows with state ``done``, ``current``, or ``pending``."""
    names = labels or STAGE_CHECKLIST_LABELS
    active = resolve_active_stage(current_stage, stages)
    idx = stages.index(active) if active in stages else 0
    items: list[dict[str, str]] = []
    for i, stage in enumerate(stages):
        if complete or i < idx:
            state = "done"
        elif i == idx:
            state = "current"
        else:
            state = "pending"
        items.append(
            {
                "id": stage,
                "label": names.get(stage, stage),
                "state": state,
            }
        )
    return items


def format_checklist_markdown(items: list[dict[str, str]]) -> str:
    """Plain-text checklist that does not rely on clickable Streamlit widgets."""
    marks = {"done": "[done]", "current": "[now]", "pending": "[todo]"}
    lines = []
    for item in items:
        mark = marks.get(item["state"], "[todo]")
        label = item["label"]
        if item["state"] == "current":
            lines.append(f"{mark} **{label}**")
        else:
            lines.append(f"{mark} {label}")
    return "\n\n".join(lines)


def user_progress_hint(stage: str, message: str = "") -> str | None:
    """Short user-facing caption for the status strip. Never the raw worker message."""
    if stage != "separate":
        return None
    if "NVIDIA GPU" in (message or ""):
        return "Separating tracks on NVIDIA GPU"
    return "Separating tracks. CPU may be slow"


def format_progress_label(
    percent: float, message: str, *, estimated: bool = False
) -> str:
    """Human-readable progress line, e.g. ``42% · Running Demucs…``."""
    pct = int(round(percent * 100))
    prefix = f"~{pct}%" if estimated else f"{pct}%"
    return f"{prefix} · {message}"


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as M:SS."""
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def format_eta_line(
    elapsed_sec: float,
    remaining_sec: float | None,
    *,
    confidence: str = "low",
) -> str:
    """Elapsed plus remaining, e.g. ``Elapsed 0:09 · ~2:40 left``."""
    elapsed = format_elapsed(elapsed_sec)
    if remaining_sec is None:
        return f"Elapsed {elapsed} · estimating…"
    if remaining_sec <= 0.5:
        return f"Elapsed {elapsed}"
    left = format_elapsed(remaining_sec)
    _ = confidence  # always ~ ; low confidence uses estimating… when remaining is None
    return f"Elapsed {elapsed} · ~{left} left"


def mixer_focus_run_dir(
    *,
    shell_view: str | None,
    viewing_run_dir: str | None = None,
    listen_picker: str | None = None,
    shell_tab: str | None = None,
) -> str | None:
    """Library mix the user is immersed in, if Mixer shell is open on a real run.

    Draft / Home tab ids are ignored so waiting on a New-tab progress slot still
    allows auto-apply. Returns None when not in Mixer or no library mix is focused.
    """
    if str(shell_view or "").strip() != "mix":
        return None
    for candidate in (viewing_run_dir, listen_picker, shell_tab):
        text = str(candidate or "").strip()
        if not text or text == "home" or is_new_draft_tab(text):
            continue
        return text
    return None


def should_auto_apply_job(
    viewing_id: str | None,
    job_id: str,
    *,
    shell_view: str | None = None,
    viewing_run_dir: str | None = None,
    listen_picker: str | None = None,
    shell_tab: str | None = None,
    finished_run_dir: str | None = None,
) -> bool:
    """True when a finished job may replace the mixer (latest / unpinned / same id).

    Also False when Mixer is open on a *different* library mix than the job's
    ``run_dir`` — keep listening; flash + tab promote only (notify_only).
    """
    if not job_id:
        return False
    if viewing_id not in (None, "", "latest") and viewing_id != job_id:
        return False
    focused = mixer_focus_run_dir(
        shell_view=shell_view,
        viewing_run_dir=viewing_run_dir,
        listen_picker=listen_picker,
        shell_tab=shell_tab,
    )
    finished = str(finished_run_dir or "").strip()
    if focused and finished and focused != finished:
        return False
    return True


def running_progress_view(status: dict[str, Any], now: float) -> dict[str, Any]:
    """Pure snapshot of running-job progress for the queue banner and row."""
    stages_raw = status.get("stages") or ISOLATION_STAGE_ORDER
    stages = tuple(str(s) for s in stages_raw)
    stage = str(status.get("stage") or "ingest")
    message = str(status.get("message") or "")
    started_at = float(status.get("started_at") or now)
    stage_started = float(status.get("stage_started_at") or started_at)
    try:
        total_est = float(status["job_estimate_sec"]) if status.get("job_estimate_sec") is not None else None
    except (TypeError, ValueError):
        total_est = None
    confidence = str(status.get("estimate_confidence") or "low")
    elapsed_in = max(0.0, now - stage_started)
    stage_est = estimated_stage_seconds(stage, stages, total_est)
    intra, intra_estimated = intra_stage_fraction(elapsed_in, stage_est)
    percent = stage_progress_percent(stage, stages=stages, intra=intra)
    elapsed = max(0.0, now - started_at)
    remaining, rem_conf = estimate_remaining_seconds(
        elapsed_sec=elapsed,
        percent=percent,
        total_estimate=total_est,
        confidence=confidence,
    )
    human = STAGE_CHECKLIST_LABELS.get(stage, "Separate tracks")
    label = format_progress_label(percent, human, estimated=intra_estimated and confidence == "low")
    items = checklist_items(stages, stage)
    return {
        "percent": percent,
        "label": label,
        "eta_line": format_eta_line(elapsed, remaining, confidence=rem_conf),
        "checklist_md": format_checklist_markdown(items),
        "hint": user_progress_hint(stage, message),
        "estimated": intra_estimated,
    }


# Near-silence threshold (dBFS) below which a re-separate child is discarded.
RESEPARATE_SILENCE_DBFS = -40.0
# Diagnostic marker in a filename; such files are never treated as child stems.
_DIAGNOSTIC_FILENAME_MARKER = "diagnostic"


def children_from_outputs(output_dir: Path) -> dict[str, Path]:
    """Scan a re-separate output dir for audible child wav stems.

    Returns ``{child_stem_id: Path}`` where ``child_stem_id`` is the raw filename
    stem (e.g. ``guitar`` from ``guitar.wav``). Skips diagnostic-named files and
    near-silent children (``stem_energy_db <= RESEPARATE_SILENCE_DBFS``). Returns
    ``{}`` when the dir is missing or no audible children remain.
    """
    folder = Path(output_dir)
    if not folder.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in folder.iterdir():
        if path.suffix.lower() != ".wav":
            continue
        name = path.stem
        if _DIAGNOSTIC_FILENAME_MARKER in name:
            continue
        if name == "metronome":
            continue
        if stem_energy_db(path) <= RESEPARATE_SILENCE_DBFS:
            continue
        out[str(name)] = path
    return out


def merge_reseparate(
    artifacts: dict[str, Any],
    source_stem: str,
    children: dict[str, Path],
) -> dict[str, Any]:
    """Return a new artifacts dict with ``source_stem`` removed and children added.

    ``children`` keys must already be composite-prefixed by the caller
    (e.g. ``other::guitar``). Never mutates the input ``artifacts`` dict.
    """
    merged = {k: v for k, v in artifacts.items() if k != source_stem}
    merged.update(children)
    return merged


def stem_label_for_id(stem_id: str) -> str:
    """Human label for a stem id, including composite ``Child (from Parent)`` ids."""
    return stem_display_name(stem_id)


def stem_group_fields(stem_id: str) -> dict[str, str]:
    """Mixer payload fields that let the iframe fold "Break down a stem" children.

    ``other::guitar`` becomes ``{"group": "other", "groupLabel": "Other",
    "shortLabel": "Guitar"}``. Plain stems return ``{}``. The separator is the one
    ``audio_to_tab.mixer`` joins with (``tests`` pin that they match).
    """
    parent, sep, child = str(stem_id).partition("::")
    if not sep or not parent or not child:
        return {}
    return {
        "group": parent,
        "groupLabel": stem_display_name(parent),
        "shortLabel": stem_display_name(child),
    }


MIXER_COMPONENT_KEY_PREFIX = "stem_mixer_run_"
ISOLATE_METRO_ACCENT_KEY = "isolate_metro_accent"
ISOLATE_METRO_RATE_KEY = "isolate_metro_rate"
ISOLATE_METRO_SOUND_KEY = "isolate_metro_sound"
ISOLATE_METRO_FOLLOW_KEY = "isolate_metro_follow"
ISOLATE_METRO_APPLIED_KEY = "isolate_metro_applied"
_METRO_WIDGET_RUN_KEY = "_metro_widget_run"


MIXER_PAN_MIN = -100
MIXER_PAN_MAX = 100


def clamp_pan(value: object) -> int:
    """Stem pan as a whole step in [-100, 100]; anything unreadable is center (0).

    Mirrors ``clampPan`` in ``ui/stem_mixer_component/frontend/src/pan.ts``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number or number in (float("inf"), float("-inf")):
        return 0
    return max(MIXER_PAN_MIN, min(MIXER_PAN_MAX, int(round(number))))


def mixer_component_key(key_source: str) -> str:
    """Streamlit key for the mixer iframe: same run in, same key out.

    The key must not change between reruns of the same run, or Streamlit
    remounts the iframe and playback restarts mid-listen.
    """
    digest = hashlib.sha256(str(key_source).encode()).hexdigest()[:16]
    return f"{MIXER_COMPONENT_KEY_PREFIX}{digest}"


def metronome_render_from_session(session: Mapping[str, Any]) -> MetronomeRenderOptions:
    """Coerce persisted/widget metronome options to engine defaults."""
    return coerce_metronome_render_options(
        accent=session.get(ISOLATE_METRO_ACCENT_KEY, True),
        rate=session.get(ISOLATE_METRO_RATE_KEY, 1.0),
        sound=session.get(ISOLATE_METRO_SOUND_KEY, "classic"),
        follow=session.get(ISOLATE_METRO_FOLLOW_KEY, "smart"),
    )


def metronome_applied_stamp(run_dir: Path | str, options: MetronomeRenderOptions) -> dict[str, Any]:
    return {
        "run": str(Path(run_dir).resolve()),
        "accent": bool(options.accent),
        "rate": float(options.rate),
        "sound": str(options.sound),
        "follow": str(options.follow),
    }


def seed_metronome_widgets_for_run(
    session: MutableMapping[str, Any],
    run_dir: Path | str,
    diagnostics: Mapping[str, Any] | None,
) -> None:
    """Seed click-track widgets from this run's diagnostics, else leave prefs.

    Runs once per ``run_dir`` so later widget edits are not overwritten.
    """
    token = str(Path(run_dir).resolve())
    if session.get(_METRO_WIDGET_RUN_KEY) == token:
        return
    session[_METRO_WIDGET_RUN_KEY] = token
    render = diagnostics.get("render") if isinstance(diagnostics, Mapping) else None
    if isinstance(render, Mapping):
        options = coerce_metronome_render_options(
            accent=render.get("accent", True),
            rate=render.get("rate", 1.0),
            sound=render.get("sound", "classic"),
            follow=render.get("follow", "smart"),
        )
        session[ISOLATE_METRO_ACCENT_KEY] = options.accent
        session[ISOLATE_METRO_RATE_KEY] = options.rate
        session[ISOLATE_METRO_SOUND_KEY] = options.sound
        session[ISOLATE_METRO_FOLLOW_KEY] = options.follow
        session[ISOLATE_METRO_APPLIED_KEY] = metronome_applied_stamp(token, options)
        return
    session.setdefault(ISOLATE_METRO_ACCENT_KEY, True)
    session.setdefault(ISOLATE_METRO_RATE_KEY, 1.0)
    session.setdefault(ISOLATE_METRO_SOUND_KEY, "classic")
    session.setdefault(ISOLATE_METRO_FOLLOW_KEY, "smart")
    # Unknown baked options: let the Mixer rebake to the current prefs.
    session.pop(ISOLATE_METRO_APPLIED_KEY, None)


# ═══════════════════════════════════════════════════════════════
# THEME STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

THEME_PRESET_KEY = "theme_preset_id"
THEME_MODE_KEY = "theme_mode"  # light, dark, system
THEME_CUSTOM_TOKENS_KEY = "theme_custom_tokens"

DEFAULT_THEME_PRESET = "tyrian"
DEFAULT_THEME_MODE = "dark"

_THEME_CONFIG_FILENAME = "theme_config.json"


def _theme_config_path() -> Path:
    return DATA_DIR / _THEME_CONFIG_FILENAME


def load_theme_state(session: MutableMapping[str, Any]) -> None:
    """Load theme from persisted config file."""
    config_path = _theme_config_path()
    if config_path.is_file():
        try:
            import json
            config = json.loads(config_path.read_text(encoding="utf-8"))
            session[THEME_PRESET_KEY] = config.get("preset", DEFAULT_THEME_PRESET)
            session[THEME_MODE_KEY] = config.get("mode", DEFAULT_THEME_MODE)
            session[THEME_CUSTOM_TOKENS_KEY] = config.get("custom_tokens", {})
        except Exception:
            pass
    # Set defaults if not present
    session.setdefault(THEME_PRESET_KEY, DEFAULT_THEME_PRESET)
    session.setdefault(THEME_MODE_KEY, DEFAULT_THEME_MODE)
    session.setdefault(THEME_CUSTOM_TOKENS_KEY, {})
    # Settings only offers Tyrian (or a saved custom palette). An older
    # forest/teal/purple/ocean file would keep painting the green theme.
    if session.get(THEME_PRESET_KEY) not in ("tyrian", "custom"):
        session[THEME_PRESET_KEY] = DEFAULT_THEME_PRESET
        session[THEME_CUSTOM_TOKENS_KEY] = {}
        save_theme_state(session)


def save_theme_state(session: MutableMapping[str, Any]) -> None:
    """Persist theme to config file."""
    config_path = _theme_config_path()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import json
    config = {
        "preset": session.get(THEME_PRESET_KEY, DEFAULT_THEME_PRESET),
        "mode": session.get(THEME_MODE_KEY, DEFAULT_THEME_MODE),
        "custom_tokens": session.get(THEME_CUSTOM_TOKENS_KEY, {}),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _os_theme_mode() -> str:
    """Light or dark from the operating system. Dark when it cannot be read."""
    import sys

    if sys.platform.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if int(value) == 1 else "dark"
        except OSError:
            return "dark"
    if sys.platform == "darwin":
        import subprocess

        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "dark"
        return "dark" if "dark" in (result.stdout or "").lower() else "light"
    return "dark"


def get_effective_theme_mode(session: MutableMapping[str, Any]) -> str:
    """Resolve 'system' to the operating system's light or dark setting."""
    mode = session.get(THEME_MODE_KEY, DEFAULT_THEME_MODE)
    if mode == "light":
        return "light"
    if mode == "dark":
        return "dark"
    return _os_theme_mode()


def _resolved_theme_tokens(session: Mapping[str, Any]) -> dict[str, str]:
    """Token map for the saved preset. Tyrian stays on its two hues."""
    preset_id = session.get(THEME_PRESET_KEY, DEFAULT_THEME_PRESET)
    mode = get_effective_theme_mode(session)
    custom = session.get(THEME_CUSTOM_TOKENS_KEY, {})
    preset = next((p for p in THEME_PRESETS if p.id == preset_id), THEME_PRESETS[0])
    if preset.accent_id == "tyrian":
        base = TYRIAN_BASE_LIGHT if mode == "light" else TYRIAN_BASE_DARK
        return {**base, **ACCENT_TOKENS["tyrian"]}
    base = BASE_TOKENS_DARK if mode == "dark" else BASE_TOKENS_LIGHT
    accent = ACCENT_TOKENS.get(preset.accent_id, {})
    if preset.accent_id == "custom":
        # Custom tokens use --custom- prefix for base layers
        accent = {
            f"--custom-{k.lstrip('--')}": v
            for k, v in custom.items()
            if k.startswith("--custom-")
        }
        accent.update({k: v for k, v in custom.items() if not k.startswith("--custom-")})
    return {**base, **accent, **SIGNAL_TOKENS}


_THEME_SHELL_CSS = """
:root {
  --background-color: var(--color-bg-deepest) !important;
  --secondary-background-color: var(--color-bg-raised) !important;
  --text-color: var(--color-text-primary) !important;
  --secondary-text-color: var(--color-text-primary) !important;
  --primary-color: var(--color-accent-primary) !important;
  --link-color: var(--color-link) !important;
  --border-color: var(--color-border-subtle) !important;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
  background-color: var(--color-bg-deepest) !important;
  color: var(--color-text-primary) !important;
}
.stApp a {
  color: var(--color-link);
}
[data-testid="stSidebarNavLink"] {
  color: var(--color-text-primary) !important;
  background-color: transparent !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background-color: var(--color-bg-highlight) !important;
  color: var(--color-text-primary) !important;
}
[data-testid="stMainMenuPopover"] {
  background-color: var(--color-bg-deepest) !important;
  color: var(--color-text-primary) !important;
}
[data-testid^="stMainMenuItem-"] {
  color: var(--color-text-primary) !important;
}
[data-testid="stMainMenuItem-theme-System"],
[data-testid="stMainMenuItem-theme-Light"],
[data-testid="stMainMenuItem-theme-Dark"] {
  display: none !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-baseweb="select"] > div,
[data-baseweb="popover"] [data-baseweb="menu"] {
  background-color: var(--color-bg-raised) !important;
  color: var(--color-text-primary) !important;
  border-color: var(--color-border-subtle) !important;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stHeading"],
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
  color: var(--color-text-primary) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  background-color: var(--color-bg-raised) !important;
  color: var(--color-text-primary) !important;
}
button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
  background-color: var(--color-bg-raised) !important;
  color: var(--color-text-primary) !important;
  border-color: var(--color-border-subtle) !important;
}
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background-color: var(--color-accent-primary) !important;
  color: """ + TYRIAN_BLUE + """ !important;
  border-color: var(--color-accent-primary-hover) !important;
}
button[kind="primary"] [data-testid="stMarkdownContainer"],
button[kind="primary"] [data-testid="stMarkdownContainer"] p,
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p {
  color: """ + TYRIAN_BLUE + """ !important;
}
button[kind="primary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled {
  opacity: 0.45 !important;
  cursor: not-allowed !important;
}
"""


def get_iframe_theme_vars(session: MutableMapping[str, Any]) -> dict[str, str]:
    """Extract CSS custom properties for iframe components."""
    return _resolved_theme_tokens(session)


def build_theme_css_vars(session: MutableMapping[str, Any]) -> str:
    """Generate CSS custom properties string for injection via st.html."""
    preset_id = session.get(THEME_PRESET_KEY, DEFAULT_THEME_PRESET)
    mode = get_effective_theme_mode(session)
    all_tokens = _resolved_theme_tokens(session)
    lines = [":root {"]
    for key, value in all_tokens.items():
        lines.append(f"  {key}: {value};")
    lines.append("}")
    lines.append(
        f"html {{ --theme-preset: '{preset_id}'; --theme-mode: '{mode}'; color-scheme: {mode}; }}"
    )
    lines.append(_THEME_SHELL_CSS)
    return "\n".join(lines)

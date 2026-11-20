"""Multi-stem audio isolation via Demucs (standalone feature).

Stage-1 guitar model (2026-08-24, updated 2026-08-29): keep ``htdemucs_6s`` as
the **default** first-stage separator whenever a dedicated ``guitar`` stem is
required and no optional extra is installed. Do not swap the default to
``htdemucs`` / ``htdemucs_ft`` (no guitar stem). Lead vs rhythm remains a
*post-process* on that single guitar stem — see ``eval/lead_rhythm/README.md``.

2026-08-29 guitar-quality decision: ``htdemucs_6s`` has no dedicated guitar
head — its guitar stem bleeds bass/drums/cymbals/vocals and loses clean
content, and folding the full ``other`` stem back into guitar reintroduces
piano/keys. Opt-in engines that beat it on guitar SDR (MVSep: BS-RoFormer-SW
guitar ~9.05 dB vs guitar "not established" for htdemucs_6s):

- ``bs_roformer_sw`` — jarredou BS-RoFormer-SW 6-stem, mirrored at
  ``enerjazzer/BS-ROFO-SW-Fixed`` after the original HF account was deleted.
  Weights have **no stated license**. Requires optional ``.[roformer]``
  (``bs-roformer-infer``) or ``.[separator]`` (``audio-separator``).
- ``melband_roformer_guitar`` / ``IsolateConfig.guitar_refine`` — becruily
  MelBand-Roformer Guitar 2-stem specialist as a second-pass refine.
  Requires optional ``.[separator]``.
- ``guitar_scnet`` — SCNet 4-stem (MUSDB18, ~10.6M params) via optional
  ``.[scnet]``; a dedicated-guitar engine is not what it outputs, so guitar is
  mapped from its ``other`` stem (fuzzier than ``bs_roformer_sw`` but a real
  SDR jump over stock 6s). Weights MIT-friendly (UVR mirror, SHA256-pinned).
- ``htdemucs_6s_guitar_ft`` stays Advanced opt-in (smallest gain).

Default path stays Demucs-only and backwards-compatible on time: smarter
``fold_other_mode="best_effort"`` (band-limited / skip piano-like ``other``)
instead of dumping the residual mix into guitar. New ML is never the default.

Bass/guitar bleed (2026-07-17): distorted/overdriven bass and electric guitar
share heavily overlapping spectral and timbral content, so Demucs (including
``htdemucs_6s``) does not always keep them cleanly separated — the ``guitar``
stem can carry audible bass energy. This is a documented, model-level
limitation (see facebookresearch/demucs issue #291) — not a bug in this
repo's post-processing. ``htdemucs_ft`` does not have this specific failure
mode because it never produces a ``guitar`` stem. See
``eval/lead_rhythm/README.md``, ``BassBleedDiagnostics``, and
``IsolateConfig.bass_bleed_mitigation``.

Two-pass (opt-in ``IsolateConfig.two_pass`` / ``--two-pass``): 4-stem
``htdemucs`` then ``htdemucs_6s`` on the leftover mix. About 2× time. Not
the default. Guitar refine is a separate opt-in stage.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_to_tab.hardware import ensure_cuda_available, separate_progress_message
from audio_to_tab.ingest import normalize_audio
from audio_to_tab.lead_rhythm import LeadRhythmThresholds, resolve_lead_rhythm_mode
from audio_to_tab.mixer import apply_true_peak_ceiling, mix_stems_to_wav
from audio_to_tab.roformer import (
    BS_ROFORMER_SW_ID,
    MELBAND_GUITAR_ID,
    ROFORMER_INSTALL_HINT,
    ROFORMER_MODELS,
    is_guitar_refine_available,
    is_roformer_backend_available,
    run_guitar_refine,
    run_roformer_model,
)
from audio_to_tab.scnet import (
    SCNET_INSTALL_HINT,
    SCNET_MODELS,
    is_scnet_available,
    run_scnet_model,
)
from audio_to_tab.separate import (
    GUITAR_FT_CHECKPOINT_ID,
    SUPPORTED_GUITAR_CHECKPOINTS,
    is_demucs_available,
    run_demucs,
    run_demucs_guitar_ft_inprocess,
)
from audio_to_tab.subprocess_util import JobAborted, subprocess_run_kwargs

logger = logging.getLogger(__name__)

DEMUCS_INSTALL_HINT = (
    "Demucs is required for stem separation. Install with: make install-demucs "
    "or pip install -e \".[demucs]\""
)

ProgressCallback = Callable[[str, str], None]
StageCompleteCallback = Callable[[str], None]


def _abort_if(should_abort: Callable[[], bool] | None) -> None:
    if should_abort and should_abort():
        raise JobAborted("Job aborted")


def _stage_done(completed: set[str], stage: str) -> bool:
    return stage in completed


def _mark_stage_complete(stage: str, on_stage_complete: StageCompleteCallback | None) -> None:
    if on_stage_complete:
        on_stage_complete(stage)

DEMUCS_MODELS = ("htdemucs_6s", "htdemucs", "htdemucs_ft")
SUPPORTED_MODELS = DEMUCS_MODELS + ROFORMER_MODELS + SCNET_MODELS
# Models that emit a dedicated guitar stem (quality floor, refine, fold).
GUITAR_PRODUCING_MODELS = frozenset(
    {"htdemucs_6s", BS_ROFORMER_SW_ID, MELBAND_GUITAR_ID}
)
FOLD_OTHER_MODES = ("full", "best_effort", "band_limited")
# Guitar-typical recovery band when folding leftover ``other`` (open low E ≈82 Hz).
FOLD_GUITAR_BAND_LOW_HZ = 82.0
FOLD_GUITAR_BAND_HIGH_HZ = 8000.0
# Skip mixing ``other`` when it spectrally matches the piano stem this closely.
FOLD_SKIP_PIANO_OVERLAP = 0.40

QUALITY_SHIFTS = {"fast": "0", "balanced": "1", "high": "3", "extreme": "5"}
QUALITY_OVERLAP = {"fast": "0.25", "balanced": "0.25", "high": "0.5", "extreme": "0.75"}
QUALITY_RANK = {"fast": 0, "balanced": 1, "high": 2, "extreme": 3}
# Desktop Faster keeps fast; other speeds floor 6-stem guitar jobs to this quality.
GUITAR_QUALITY_FLOOR = "balanced"
# High-end preservation band for guitar-stem diagnostics (vs. bass-bleed <250 Hz).
GUITAR_HIGH_END_HZ = 4000.0
# Opt-in low-E presence restore (fundamental ≈82 Hz, 2nd harmonic ≈165 Hz).
LOW_END_RESTORE_LOW_HZ = 60.0
LOW_END_RESTORE_HIGH_HZ = 200.0
LOW_END_RESTORE_MAX_DB = 12.0
# Subtractive bass de-bleed: remove bled bass/drum energy below this band (opt-in).
SUB_BASS_DEBLEED_CUTOFF_HZ = 150.0
SUB_BASS_DEBLEED_MIX = 0.85
SUB_BASS_DRUMS_MIX = 0.35
# Linear gain on ``other`` when folding into guitar (reserve headroom vs peak ceiling).
FOLD_OTHER_MIX_GAIN = 0.5
# Opt-in fold-gain search (Phase 1b) — DSL gains sized for band-limited Other.
FOLD_OTHER_GAIN_CANDIDATES = (0.25, 0.5, 0.75)
# Opt-in spectral bleed gate (flutter/cymbal/bass residue scrub).
# A time-frequency soft gate: when competing stems strongly dominate the
# guitar's own magnitude in a bin, the guitar is attenuated (that energy is
# bleed the separator left behind). Competing-dominated solo sections keep
# gutier content because the score only trips when the competitor wins.
BLEED_GATE_NPERSEG = 2048
BLEED_GATE_OVERLAP = 1536  # 75% — Hann is COLA at this hop.
BLEED_GATE_DOMINANCE_THRESHOLD = 0.90  # competitor share of bin energy to trip.
BLEED_GATE_MAX_ATTENUATION = 0.25  # linear floor (~−12 dB) while fully dominated.
BLEED_GATE_STRENGTH = 1.0
BLEED_GATE_HIGH_HZ = 16000.0  # protect only up to top of instrument range.
# Cross-model guitar ensemble (Phase 2): per-band energy soft-max blend.
# Bands are log-spaced over the guitar's playable range; gamma≈3 ≈ soft max,
# so whichever separator captured more energy in a band wins that band.
ENSEMBLE_BLEND_BANDS = 24
ENSEMBLE_BLEND_GAMMA = 3.0
ENSEMBLE_BLEND_LOW_HZ = 40.0
GUITAR_PREREFINE_NAME = "guitar_prerefine.wav"
GUITAR_REFINED_NAME = "guitar_refined.wav"
TWO_PASS_KEEP_FROM_FIRST = ("vocals", "drums", "bass")
TWO_PASS_KEEP_FROM_SECOND = ("guitar", "piano")
TWO_PASS_FOLD_INTO_OTHER = ("other", "vocals", "drums", "bass")

# Transformer Demucs checkpoints reject --segment longer than they were trained on.
# CLI --segment is an int; HTDemucs max is 7.8s so 7 is the largest valid value.
DEMUCS_MAX_SEGMENT_SEC = {
    "htdemucs_6s": 7,
    "htdemucs": 7,
    "htdemucs_ft": 7,
}

SUBPROCESS_TIMEOUT = 300


def isolate_timeout_multiplier(
    *,
    model: str,
    two_pass: bool = False,
    guitar_refine: bool = False,
    guitar_ensemble: bool = False,
) -> int:
    """Wall-clock timeout scale vs a single Demucs pass."""
    n = 1
    if two_pass:
        n += 1
    if guitar_refine:
        n += 1
    if guitar_ensemble:
        n += 1
    if model in ROFORMER_MODELS or model in SCNET_MODELS:
        n += 1
    return n


def effective_demucs_segment(model: str, requested: float | None) -> int | None:
    """Clamp Demucs --segment to the model training limit, or None to omit the flag."""
    if requested is None:
        return None
    try:
        value = round(float(requested))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    cap = DEMUCS_MAX_SEGMENT_SEC.get(model, 7)
    return min(value, cap)


def effective_isolation_quality(
    quality: str,
    *,
    model: str,
    speed_id: str | None = None,
    floor: str = GUITAR_QUALITY_FLOOR,
) -> str:
    """Floor quality for dedicated-guitar models unless the user picked Faster.

    IsolateConfig / CLI keep an explicit ``quality`` (default ``fast``) so eval
    A/B stays apples-to-apples. Desktop Faster keeps ``fast``; other speeds
    floor dedicated-guitar models to Balanced.
    """
    current = quality if quality in QUALITY_SHIFTS else "fast"
    if model not in GUITAR_PRODUCING_MODELS:
        return current
    if speed_id == "faster":
        return current
    floor_q = floor if floor in QUALITY_SHIFTS else GUITAR_QUALITY_FLOOR
    if QUALITY_RANK.get(current, 0) < QUALITY_RANK.get(floor_q, 1):
        return floor_q
    return current


def resolve_demucs_shifts(quality: str, override: int | None = None) -> str:
    """Demucs ``--shifts``: explicit override wins, else the Quality map."""
    if override is not None:
        try:
            return str(int(override))
        except (TypeError, ValueError):
            pass
    return QUALITY_SHIFTS.get(quality, "0")


def resolve_demucs_overlap(quality: str, override: float | None = None) -> str:
    """Demucs ``--overlap``: explicit override wins, else the Quality map."""
    if override is not None:
        try:
            return str(float(override))
        except (TypeError, ValueError):
            pass
    return QUALITY_OVERLAP.get(quality, "0.25")


# Demucs always writes the full stem set for a model, even when an instrument
# isn't actually present in the mix. These gate the auto-detection heuristic
# used to flag which produced stems are audibly real vs. near-silent filler.
STEM_PRESENCE_FLOOR_DB = -45.0
STEM_PRESENCE_ENERGY_SHARE_MIN = 0.02
_STEM_PRESENCE_MIN_RMS = 10.0 ** (-100.0 / 20.0)  # floor to avoid log(0)

# Bass-bleed diagnostic (see module docstring): a "guitar" stem where too much
# RMS energy sits below this band is unusually bass-heavy for a guitar and
# likely carries bled-in bass content, per the known Demucs limitation.
BASS_BLEED_LOW_BAND_HZ = 250.0
# Calibrated so a synthetic "clean guitar chord" fixture (~0.54 share, mostly
# 82-370 Hz content with a modest low-E fundamental) stays well under the
# floor while a fixture with added distorted-bass content (~0.85 share) is
# clearly flagged — see tests/test_isolate.py.
BASS_BLEED_ENERGY_SHARE_FLOOR = 0.65
# Opt-in mitigation cutoff: below the lowest fundamental of standard-tuned
# guitar (open low E ≈ 82.4 Hz) and Drop D (≈73.4 Hz), so it only removes
# sub-guitar-range rumble/bleed. Tunings dropped lower than this (Drop C
# ≈65.4 Hz, 7-string low B ≈61.7 Hz, Drop A, etc.) will have their own
# fundamental attenuated too — there is no cutoff that is safe for every
# tuning, so this is a deliberate, documented tradeoff, not a fix.
BASS_BLEED_HPF_CUTOFF_HZ = 60.0
BASS_BLEED_HPF_ORDER = 4

MIN_REGION_SEC = 5.0

_FFMPEG_DURATION_RE = re.compile(
    r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)",
    re.IGNORECASE,
)


class RegionError(ValueError):
    """Invalid audio region selection."""


def format_time_sec(sec: float) -> str:
    """Format seconds as M:SS."""
    sec = max(sec, 0)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def format_region_label(start_sec: float, length_sec: float) -> str:
    """Human-readable region label, e.g. ``0:32–1:15``."""
    end_sec = start_sec + length_sec
    return f"{format_time_sec(start_sec)}–{format_time_sec(end_sec)}"


def format_region_label_filename(start_sec: float, length_sec: float) -> str:
    """Filename-safe region suffix, e.g. ``0m32-1m15``."""

    def _part(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m{s:02d}"

    end_sec = start_sec + length_sec
    return f"{_part(start_sec)}-{_part(end_sec)}"


def probe_duration_sec(path: str | Path) -> float | None:
    """
    Return audio duration in seconds, or None if it cannot be determined.

    Tries ffprobe, then ffmpeg stderr parsing, then soundfile.
    """
    src = Path(path)
    if not src.exists():
        return None

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(src),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=SUBPROCESS_TIMEOUT,
                **subprocess_run_kwargs(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out after %ss probing %s", SUBPROCESS_TIMEOUT, path)
            result = None
        if result is not None and result.returncode == 0:
            try:
                dur = float(result.stdout.strip())
                if dur > 0:
                    return dur
            except ValueError:
                pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-i", str(src)],
                capture_output=True,
                text=True,
                check=False,
                timeout=SUBPROCESS_TIMEOUT,
                **subprocess_run_kwargs(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out after %ss probing %s", SUBPROCESS_TIMEOUT, path)
            result = None
        if result is not None:
            match = _FFMPEG_DURATION_RE.search(result.stderr or "")
            if match:
                hours, minutes, seconds = match.groups()
                dur = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                if dur > 0:
                    return dur

    try:
        import soundfile as sf

        info = sf.info(str(src))
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    logger.warning("All probe methods failed for %s; duration unknown", path)
    return None


def resolve_region(
    duration_sec: float | None,
    start_sec: float,
    end_sec: float | None,
    *,
    cap_sec: float | None = None,
) -> tuple[float, float, str | None]:
    """
    Validate and resolve a trim region.

    Returns ``(start_sec, length_sec, note)`` where ``note`` explains any cap
    shortening. Raises ``RegionError`` when the range is invalid.
    """
    if start_sec < 0:
        raise RegionError("start_sec must be >= 0")

    if end_sec is None:
        if duration_sec is None:
            raise RegionError("end_sec required when file duration is unknown")
        end_sec = duration_sec

    if start_sec >= end_sec:
        raise RegionError("start_sec must be less than end_sec")

    length = end_sec - start_sec
    if length < MIN_REGION_SEC:
        raise RegionError(f"region must be at least {MIN_REGION_SEC:.0f} seconds")

    if duration_sec is not None and end_sec > duration_sec + 0.05:
        raise RegionError("end_sec exceeds file duration")

    note: str | None = None
    if cap_sec is not None and length > cap_sec:
        length = cap_sec
        note = (
            f"Region shortened to {cap_sec:.0f}s for processing mode limit "
            f"({format_region_label(start_sec, length)})"
        )

    return start_sec, length, note


@dataclass
class DualGuitarDiagnostics:
    """Legacy stereo heuristic diagnostics (not used by live isolate path).

    Prefer ``LeadRhythmDiagnostics`` from ``audio_to_tab.lead_rhythm``.
    """

    attempted: bool
    split: bool
    reason: str
    correlation: float | None = None
    balance_ratio: float | None = None


@dataclass
class FoldOtherDiagnostics:
    """Whether leftover ``other`` was mixed into guitar (and how)."""

    attempted: bool
    folded: bool
    mode: str
    reason: str
    piano_overlap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass
class BleedGateDiagnostics:
    """Whether the spectral flutter/bleed gate ran and how much it removed.

    Fraction of time-frequency bins where competing stems dominated the
    bin's energy (``dominated_bin_fraction``) is the "how dirty was it"
    signal; ``mean_attenuation_db`` quantifies how much was scrubbed. Both
    are driven by competitor stems (bass/drums/other), never the full mix.
    """

    attempted: bool
    reason: str
    gated: bool = False
    dominated_bin_fraction: float | None = None
    mean_attenuation_db: float | None = None
    threshold: float = BLEED_GATE_DOMINANCE_THRESHOLD
    max_attenuation_linear: float = BLEED_GATE_MAX_ATTENUATION
    high_hz: float = BLEED_GATE_HIGH_HZ
    competitors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass
class AdaptiveFoldGainDiagnostics:
    """Result of the opt-in fold-gain search (which gain won and by how much)."""

    attempted: bool
    reason: str
    chosen_gain: float | None = None
    scores: dict[str, float] | None = None
    wins_by: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _bandpass_audio(
    data: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    lo = min(0.99, max(1e-6, low_hz / nyquist))
    hi = min(0.99, max(lo + 1e-4, high_hz / nyquist))
    if lo >= hi:
        return data.astype(np.float32)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


def _write_band_limited_other(other: Path, dest: Path) -> Path:
    import soundfile as sf

    data, sr = sf.read(str(other), always_2d=True)
    filtered = _bandpass_audio(data, int(sr), FOLD_GUITAR_BAND_LOW_HZ, FOLD_GUITAR_BAND_HIGH_HZ)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), filtered, int(sr), subtype="PCM_16")
    return dest


def _drop_other_stem(artifacts: dict[str, Path]) -> None:
    """Remove Other only after it has been folded into an existing guitar stem."""
    guitar = artifacts.get("guitar")
    if guitar is None or not Path(guitar).is_file():
        return
    other = artifacts.pop("other", None)
    if other is None:
        return
    with suppress(OSError):
        Path(other).unlink(missing_ok=True)


def fold_other_into_guitar(
    artifacts: dict[str, Path],
    *,
    mode: str = "full",
) -> dict[str, Path]:
    """Mix leftover Other into Guitar and drop Other from 6-stem artifacts.

    ``mode="full"`` is the legacy mix-everything path. ``band_limited`` mixes
    only the guitar-typical band of Other. ``best_effort`` skips the mix when
    Other looks piano-like (or is near-silent) so keys/synths are not dumped
    into Guitar — Other is still dropped. Essential 4-stem runs have no
    guitar stem; Other stays. Mutates ``artifacts`` in place.
    """
    artifacts, _diag = apply_fold_other_into_guitar(artifacts, mode=mode)
    return artifacts


def apply_fold_other_into_guitar(
    artifacts: dict[str, Path],
    *,
    mode: str = "full",
    gain: float | None = None,
    adaptive_gain: bool = False,
) -> tuple[dict[str, Path], FoldOtherDiagnostics]:
    """Like ``fold_other_into_guitar`` but returns diagnostics for tests/eval.

    ``gain`` overrides the fixed ``FOLD_OTHER_MIX_GAIN`` (0.5). When
    ``adaptive_gain`` is True a small gain search is run first and the best
    candidate (scored by ``analyze_guitar_stem_quality`` vs the competing
    bass/drum stems) is used; the search result is returned alongside via
    ``apply_fold_other_gain_search``.
    """
    other = artifacts.get("other")
    guitar = artifacts.get("guitar")
    resolved_mode = mode if mode in FOLD_OTHER_MODES else "full"
    missing = FoldOtherDiagnostics(
        attempted=False,
        folded=False,
        mode=resolved_mode,
        reason="other or guitar stem missing",
    )
    not_a_file = FoldOtherDiagnostics(
        attempted=False,
        folded=False,
        mode=resolved_mode,
        reason="other or guitar stem is not a file",
    )
    if other is None or guitar is None:
        return artifacts, missing
    try:
        if not other.is_file() or not guitar.is_file():
            return artifacts, not_a_file

        piano_overlap: float | None = None
        skip_reason: str | None = None
        if resolved_mode == "best_effort":
            other_rms = _stem_rms(other)
            if other_rms < _STEM_PRESENCE_MIN_RMS * 10:
                skip_reason = "other stem is near-silent; skip mix"
            else:
                piano = artifacts.get("piano")
                if piano is not None and Path(piano).is_file():
                    try:
                        import soundfile as sf

                        other_data, other_sr = sf.read(str(other), always_2d=True)
                        piano_data, piano_sr = sf.read(str(piano), always_2d=True)
                        if other_sr == piano_sr and other_data.size and piano_data.size:
                            piano_overlap = _spectral_overlap(
                                _to_mono(other_data), _to_mono(piano_data)
                            )
                            if piano_overlap >= FOLD_SKIP_PIANO_OVERLAP:
                                skip_reason = (
                                    f"other overlaps piano stem ({piano_overlap:.2f} "
                                    f">= {FOLD_SKIP_PIANO_OVERLAP:.2f}); skip mix to "
                                    "avoid keys bleed"
                                )
                    except FileNotFoundError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Piano overlap check failed; defaulting to fold: %s", exc
                        )
                        piano_overlap = None

        if skip_reason:
            _drop_other_stem(artifacts)
            return artifacts, FoldOtherDiagnostics(
                attempted=True,
                folded=False,
                mode=resolved_mode,
                reason=skip_reason,
                piano_overlap=piano_overlap,
            )

        mix_src = other
        tmp_band: Path | None = None
        resolved_gain = FOLD_OTHER_MIX_GAIN if gain is None else float(gain)
        if resolved_mode in ("best_effort", "band_limited"):
            tmp_band = other.parent / f"{other.stem}_guitar_band.wav"
            other_size = other.stat().st_size
            if other_size > 500 * 1024 * 1024:
                logger.warning(
                    "Band-limited other fold on a large file (%s MB); may cause memory pressure",
                    other_size // (1024 * 1024),
                )
            try:
                mix_src = _write_band_limited_other(other, tmp_band)
            except FileNotFoundError:
                raise
            except Exception as exc:
                logger.warning("band-limited other fold failed; using full other: %s", exc)
                mix_src = other
                resolved_mode = "full"
                tmp_band = None

        try:
            scoring_competitors = {
                name: artifacts[name]
                for name in ("bass", "drums")
                if name in artifacts and artifacts[name] is not None
            }
            search = apply_fold_other_gain_search(
                guitar,
                mix_src,
                competitors=scoring_competitors,
                mode=resolved_mode,
                adaptive=adaptive_gain,
                fixed_gain=resolved_gain,
            )
            if search.attempted:
                search_path = Path(guitar).parent / "adaptive_fold_gain_diagnostics.json"
                search_path = search.write_json(search_path)
                artifacts["adaptive_fold_gain_diagnostics"] = search_path

            mix_stems_to_wav(
                {"guitar": guitar, "other": mix_src},
                output_path=guitar,
                gains={"guitar": 1.0, "other": search.chosen_gain},
            )
        finally:
            if tmp_band is not None:
                with suppress(OSError):
                    tmp_band.unlink(missing_ok=True)
        _drop_other_stem(artifacts)
        band_note = (
            f"mixed guitar-band other ({FOLD_GUITAR_BAND_LOW_HZ:.0f}–"
            f"{FOLD_GUITAR_BAND_HIGH_HZ:.0f} Hz)"
            if mix_src != other
            else "mixed full other into guitar"
        )
        if search.attempted:
            band_note += f" @ {search.chosen_gain:.2f} gain (adaptive)"
        return artifacts, FoldOtherDiagnostics(
            attempted=True,
            folded=True,
            mode=resolved_mode,
            reason=band_note,
            piano_overlap=piano_overlap,
        )
    except FileNotFoundError:
        return artifacts, not_a_file


def apply_fold_other_gain_search(
    guitar_path: Path,
    other_path: Path,
    *,
    competitors: dict[str, Path] | None = None,
    mode: str = "full",
    adaptive: bool = False,
    fixed_gain: float = FOLD_OTHER_MIX_GAIN,
) -> AdaptiveFoldGainDiagnostics:
    """Pick the fold mix gain (fixed default or search over candidates).

    The objective favors candidates that raise the guitar stem's high-end
    presence while lowering overlap with the (still-present) bass/drum
    competitors — i.e. more guitar content, less leftover bleed. Candidates
    are scored with ``analyze_guitar_stem_quality`` on a scratch mix, so the
    guitar file is only re-written once with the winning gain.
    """
    guitar = Path(guitar_path)
    other = Path(other_path)
    if not adaptive:
        return AdaptiveFoldGainDiagnostics(
            attempted=False,
            reason="adaptive off; fixed gain",
            chosen_gain=max(0.0, min(1.0, float(fixed_gain))),
            scores=None,
            wins_by=0.0,
        )
    if not guitar.is_file() or not other.is_file():
        return AdaptiveFoldGainDiagnostics(
            attempted=False,
            reason="guitar or other stem missing",
            chosen_gain=max(0.0, min(1.0, float(fixed_gain))),
        )

    import soundfile as sf

    guitar_data, sr = sf.read(str(guitar), always_2d=True)
    other_data, other_sr = sf.read(str(other), always_2d=True)
    if guitar_data.size == 0 or other_data.size == 0 or int(sr) != int(other_sr):
        return AdaptiveFoldGainDiagnostics(
            attempted=False,
            reason="incompatible stems for band-limited scoring",
            chosen_gain=max(0.0, min(1.0, float(fixed_gain))),
        )

    from audio_to_tab.mixer import mix_stems_to_wav

    work = guitar.parent / "_fold_gain_search"
    work.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    best_gain = float(fixed_gain)
    best_score = float("-inf")
    try:
        for cand in FOLD_OTHER_GAIN_CANDIDATES:
            probe = work / f"fold_{cand!s}.wav"
            mix_stems_to_wav(
                {"guitar": guitar, "other": other},
                output_path=probe,
                gains={"guitar": 1.0, "other": cand},
            )
            diag = analyze_guitar_stem_quality(probe, competing_stems=competitors or None)
            overlap = 1.0
            if diag.competitor_overlap:
                values = [v for v in diag.competitor_overlap.values() if v is not None]
                if values:
                    overlap = float(sum(values)) / len(values)
            score = (diag.high_end_energy_share or 0.0) - 0.5 * overlap
            scores[f"{cand:.2f}"] = float(score)
            if score > best_score:
                best_score = score
                best_gain = cand
    finally:
        with suppress(OSError):
            import shutil

            shutil.rmtree(work, ignore_errors=True)

    wins_by = 0.0
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) >= 2:
        wins_by = float(ranked[0] - ranked[1])
    return AdaptiveFoldGainDiagnostics(
        attempted=True,
        reason=f"search over {len(scores)} gains; picked {best_gain:.2f}",
        chosen_gain=best_gain,
        scores=scores,
        wins_by=wins_by,
    )


def apply_emit_stems(
    artifacts: dict[str, Path],
    emit: tuple[str, ...] | list[str] | None,
) -> dict[str, Path]:
    """Keep only named WAV stems; leave diagnostics JSON keys in place.

    Mutates ``artifacts`` in place and unlinks dropped ``.wav`` files.
    """
    if not emit:
        return artifacts
    keep = set(emit)
    drop = [
        name
        for name, path in list(artifacts.items())
        if not name.endswith("_diagnostics")
        and Path(path).suffix.lower() == ".wav"
        and name not in keep
    ]
    for name in drop:
        path = artifacts.pop(name)
        with suppress(OSError):
            Path(path).unlink(missing_ok=True)
    return artifacts


@dataclass
class IsolateConfig:
    model: str = "htdemucs_6s"
    quality: str = "fast"
    device: str = "cpu"
    # Seconds from file start where isolation begins (default: 0 = beginning).
    start_sec: float = 0.0
    # Length of the section to process from start_sec. None = through end of file.
    max_duration_sec: float | None = None
    two_stems: str | None = None  # e.g. "vocals" for karaoke-style split
    # Opt-in: 4-stem first, then htdemucs_6s on the leftover ``other`` mix.
    # About 2× slower. Ignored for karaoke / non-6s models.
    two_pass: bool = False
    # Emit policy for Lead/Rhythm post-process. Default path never runs the split
    # unless lead_rhythm / dual_guitar is True (CLI/eval opt-in).
    lead_rhythm_mode: str = "confident"  # confident | best_effort
    # Opt-in: True → run Lead/Rhythm post-process (eval/CLI). Default isolate = False.
    lead_rhythm: bool = False
    dual_guitar: bool = False  # deprecated alias for lead_rhythm
    # Optional stage-1 guitar checkpoint (htdemucs_6s only). None = stock Demucs.
    guitar_checkpoint: str | None = None
    # Optional Lead/Rhythm gate overrides (CLI/eval); None → env + defaults.
    lead_rhythm_thresholds: LeadRhythmThresholds | None = None
    # Opt-in, partial mitigation only: high-pass the guitar stem below
    # BASS_BLEED_HPF_CUTOFF_HZ when BassBleedDiagnostics flags it. Default off
    # because it is a lossy tradeoff (see BASS_BLEED_HPF_CUTOFF_HZ docstring),
    # not a correction of the underlying Demucs separation.
    bass_bleed_mitigation: bool = False
    # Demucs --jobs (keep 1 on 8–16 GB desktop to avoid OOM from parallel chunks).
    demucs_jobs: int = 1
    # Demucs --segment length in seconds. None = model default (full-track tensors).
    # ~8s reduces peak RAM on CPU; clamped per model in effective_demucs_segment
    # because HTDemucs was trained at 7.8s max. Must be int — Demucs argparse
    # rejects '8.0'.
    demucs_segment: int | None = 8
    # Optional Demucs --shifts / --overlap. None = QUALITY_SHIFTS / QUALITY_OVERLAP.
    demucs_shifts: int | None = None
    demucs_overlap: float | None = None
    # Keep only these WAV stems after Demucs. None = keep the model's full set
    # (after optional Other→Guitar fold). Built-in UI presets cap at 4 stems.
    emit_stems: tuple[str, ...] | None = None
    # Mix leftover Other into Guitar and drop Other. False when Custom asks for Other.
    fold_other_into_guitar: bool = True
    # How to fold: full (legacy mix-all), band_limited (guitar-band only),
    # best_effort (skip piano-like/silent other, else band-limited). Default
    # best_effort avoids dumping keys/synths into guitar.
    fold_other_mode: str = "best_effort"
    # Opt-in second-pass MelBand guitar specialist (becruily). Default off.
    guitar_refine: bool = False
    # Opt-in low-shelf boost on the final guitar stem (60–200 Hz). 0 = off.
    low_end_restore_db: float = 0.0
    # Opt-in subtractive bass/drum de-bleed below ~150 Hz (default off).
    sub_bass_debleed: bool = False
    # Opt-in competitive spectral scrub of bass/drum flutter on the guitar stem.
    bleed_gate: bool = False
    # Opt-in fold-gain search (pick the best Other→Guitar mix gain automatically).
    adaptive_fold_gain: bool = False
    # Opt-in cross-model guitar ensemble: run BS-RoFormer-SW on top of the
    # primary Demucs run and per-band soft-max blend the two guitar stems
    # (whichever model kept more energy in a band wins that band).
    guitar_ensemble: bool = False
    # After stems are finalized, beat-track drums/source and emit metronome.wav.
    # Re-separate jobs turn this off so a click track is not treated as a child.
    emit_metronome: bool = True
    metronome_accent: bool = True
    metronome_rate: float = 1.0
    metronome_sound: str = "classic"

    def __post_init__(self) -> None:
        # dual_guitar=True enables lead_rhythm (deprecated alias) with best_effort emit.
        if self.dual_guitar and not self.lead_rhythm:
            self.lead_rhythm = True
        if self.dual_guitar:
            self.lead_rhythm_mode = "best_effort"
        if self.guitar_checkpoint and self.guitar_checkpoint not in SUPPORTED_GUITAR_CHECKPOINTS:
            raise ValueError(
                f"Unsupported guitar_checkpoint {self.guitar_checkpoint!r}. "
                f"Choose from: {', '.join(sorted(SUPPORTED_GUITAR_CHECKPOINTS))}"
            )
        if self.fold_other_mode not in FOLD_OTHER_MODES:
            raise ValueError(
                f"Unsupported fold_other_mode {self.fold_other_mode!r}. "
                f"Choose from: {', '.join(FOLD_OTHER_MODES)}"
            )
        if self.two_pass and (self.two_stems or self.model != "htdemucs_6s"):
            self.two_pass = False
        if self.two_stems or self.model not in GUITAR_PRODUCING_MODELS:
            self.guitar_refine = False
        if self.guitar_ensemble and (
            self.two_stems or self.two_pass or self.model not in DEMUCS_MODELS
        ):
            self.guitar_ensemble = False
        if self.emit_stems is not None:
            self.emit_stems = tuple(self.emit_stems)
        try:
            boost = float(self.low_end_restore_db)
        except (TypeError, ValueError):
            boost = 0.0
        self.low_end_restore_db = max(0.0, min(LOW_END_RESTORE_MAX_DB, boost))


@dataclass
class StemPresence:
    """Whether a produced stem actually contains audible content.

    Demucs has no per-instrument "only compute what's present" mode — it
    always writes the full fixed stem set for whichever model is chosen.
    This is a lightweight post-hoc heuristic to flag near-silent stems.
    """

    present: bool
    confidence: float
    mean_dbfs: float
    energy_share: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _stem_rms(path: Path) -> float:
    try:
        import soundfile as sf

        data, _sr = sf.read(str(path), always_2d=True)
    except Exception as exc:
        logger.warning("Could not read stem for RMS calculation: %s", exc)
        return 0.0
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data.astype(np.float64)))))


def _rms_to_dbfs(rms: float) -> float:
    return 20.0 * float(np.log10(max(rms, _STEM_PRESENCE_MIN_RMS)))


def detect_present_stems(stem_paths: dict[str, Path]) -> dict[str, StemPresence]:
    """Estimate which produced stems are audibly present vs. near-silent filler.

    For each stem, computes mean RMS energy converted to dBFS plus that
    stem's RMS share of the total RMS across all stems passed in. A stem is
    considered present if either signal clears its threshold, since a
    quiet-but-real instrument can still have a low absolute level.
    """
    rms_by_stem = {name: _stem_rms(path) for name, path in stem_paths.items()}
    total_rms = sum(rms_by_stem.values())

    results: dict[str, StemPresence] = {}
    for name, rms in rms_by_stem.items():
        mean_dbfs = _rms_to_dbfs(rms)
        energy_share = (rms / total_rms) if total_rms > 1e-12 else 0.0

        level_margin = (mean_dbfs - STEM_PRESENCE_FLOOR_DB) / 20.0
        share_margin = (energy_share - STEM_PRESENCE_ENERGY_SHARE_MIN) / max(
            STEM_PRESENCE_ENERGY_SHARE_MIN, 1e-9
        )
        margin = max(level_margin, share_margin)
        present = margin >= 0.0
        confidence = float(max(0.0, min(1.0, 0.5 + 0.5 * margin)))

        state = "audibly present" if present else "near-silent/absent"
        reason = (
            f"mean level {mean_dbfs:.1f} dBFS (floor {STEM_PRESENCE_FLOOR_DB:.1f}), "
            f"energy share {energy_share:.3f} (min {STEM_PRESENCE_ENERGY_SHARE_MIN:.3f}) "
            f"— stem appears {state}"
        )

        results[name] = StemPresence(
            present=present,
            confidence=confidence,
            mean_dbfs=mean_dbfs,
            energy_share=energy_share,
            reason=reason,
        )
    return results


@dataclass
class BassBleedDiagnostics:
    """
    Diagnostic (not a fix) for the known Demucs bass/guitar low-end confusion
    described in the module docstring.

    Only measures low-frequency RMS energy share in the ``guitar`` stem and
    flags when it is anomalously high for a guitar — it cannot tell *why*
    (bled-in bass vs. a legitimately bass-heavy guitar tone), so the reason
    string is deliberately hedged. ``mitigation_applied`` records whether the
    optional, opt-in high-pass mitigation (``IsolateConfig.bass_bleed_mitigation``)
    ran — that mitigation only attenuates content below
    ``BASS_BLEED_HPF_CUTOFF_HZ``, it does not re-separate or relabel anything.
    """

    attempted: bool
    flagged: bool
    reason: str
    low_band_energy_share: float | None = None
    low_band_cutoff_hz: float = BASS_BLEED_LOW_BAND_HZ
    mitigation_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _lowpass_rms(mono: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> float:
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    normalized = min(0.99, max(1e-6, cutoff_hz / nyquist))
    sos = butter(order, normalized, btype="low", output="sos")
    low = sosfiltfilt(sos, mono)
    return float(np.sqrt(np.mean(np.square(low)) + 1e-12))


def analyze_bass_bleed(
    guitar_path: Path,
    *,
    low_band_hz: float = BASS_BLEED_LOW_BAND_HZ,
    energy_share_floor: float = BASS_BLEED_ENERGY_SHARE_FLOOR,
) -> BassBleedDiagnostics:
    """
    Measure the guitar stem's RMS energy share below ``low_band_hz`` and flag
    it as anomalously bass-heavy above ``energy_share_floor``.

    This is a heuristic diagnostic, not a classifier: it cannot distinguish
    bled-in bass from a guitar tone that is legitimately bass-heavy (e.g. a
    heavily palm-muted low riff). See module docstring.
    """
    if not guitar_path.exists():
        return BassBleedDiagnostics(attempted=False, flagged=False, reason="guitar stem missing")

    import soundfile as sf

    data, sr = sf.read(str(guitar_path), always_2d=True)
    if data.size == 0:
        return BassBleedDiagnostics(attempted=True, flagged=False, reason="guitar stem is empty")

    mono = data.mean(axis=1).astype(np.float64)
    total_rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    if total_rms < 1e-9:
        return BassBleedDiagnostics(
            attempted=True, flagged=False, reason="guitar stem is near-silent"
        )

    share = _lowpass_rms(mono, sr, low_band_hz) / total_rms
    flagged = share >= energy_share_floor
    if flagged:
        outcome = (
            "anomalously bass-heavy for a guitar stem — likely partial bass bleed "
            "(known Demucs limitation; diagnostic only, not a corrected label — "
            "see eval/lead_rhythm/README.md)"
        )
    else:
        outcome = "within the normal range observed for a guitar stem"
    reason = (
        f"low-frequency (<{low_band_hz:.0f} Hz) RMS energy share {share:.2f} "
        f"({'>=' if flagged else '<'} floor {energy_share_floor:.2f}) — {outcome}"
    )
    return BassBleedDiagnostics(
        attempted=True,
        flagged=flagged,
        reason=reason,
        low_band_energy_share=share,
        low_band_cutoff_hz=low_band_hz,
    )


def _highpass_rms(mono: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> float:
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    normalized = min(0.99, max(1e-6, cutoff_hz / nyquist))
    sos = butter(order, normalized, btype="high", output="sos")
    high = sosfiltfilt(sos, mono)
    return float(np.sqrt(np.mean(np.square(high)) + 1e-12))


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float64)
    return data.mean(axis=1).astype(np.float64)


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Higher is better."""
    est = _to_mono(np.asarray(estimate))
    ref = _to_mono(np.asarray(reference))
    n = min(len(est), len(ref))
    if n == 0:
        return float("nan")
    est = est[:n]
    ref = ref[:n]
    ref_energy = float(np.dot(ref, ref)) + 1e-8
    alpha = float(np.dot(est, ref)) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(
        10.0 * np.log10((float(np.dot(target, target)) + 1e-8) / (float(np.dot(noise, noise)) + 1e-8))
    )


def _spectral_overlap(estimate: np.ndarray, competitor: np.ndarray) -> float:
    n = min(len(estimate), len(competitor))
    if n < 16:
        return 0.0
    mag_e = np.abs(np.fft.rfft(estimate[:n]))
    mag_c = np.abs(np.fft.rfft(competitor[:n]))
    denom = float(np.sum(mag_e) + 1e-12)
    return float(np.sum(np.minimum(mag_e, mag_c)) / denom)


@dataclass
class GuitarStemQualityDiagnostics:
    """Full-spectrum guitar-stem diagnostics (bleed bands + high-end + optional SDR).

    Complements ``BassBleedDiagnostics`` (low band only). Without ground truth this
    cannot label *why* energy sits in a band — piano bleed vs. a dark guitar tone.
    """

    attempted: bool
    reason: str
    low_band_energy_share: float | None = None
    mid_band_energy_share: float | None = None
    high_end_energy_share: float | None = None
    high_end_cutoff_hz: float = GUITAR_HIGH_END_HZ
    high_end_preservation: float | None = None
    si_sdr: float | None = None
    competitor_overlap: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def analyze_guitar_stem_quality(
    guitar_path: Path,
    *,
    competing_stems: dict[str, Path] | None = None,
    reference_path: Path | None = None,
    low_band_hz: float = BASS_BLEED_LOW_BAND_HZ,
    high_end_hz: float = GUITAR_HIGH_END_HZ,
) -> GuitarStemQualityDiagnostics:
    """Measure low / mid / high-end energy share, optional SI-SDR, and competitor overlap."""
    if not guitar_path.exists():
        return GuitarStemQualityDiagnostics(attempted=False, reason="guitar stem missing")

    import soundfile as sf

    data, sr = sf.read(str(guitar_path), always_2d=True)
    if data.size == 0:
        return GuitarStemQualityDiagnostics(attempted=True, reason="guitar stem is empty")

    mono = _to_mono(data)
    total_rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    if total_rms < 1e-9:
        return GuitarStemQualityDiagnostics(attempted=True, reason="guitar stem is near-silent")

    low_share = _lowpass_rms(mono, sr, low_band_hz) / total_rms
    high_share = _highpass_rms(mono, sr, high_end_hz) / total_rms
    mid_share = max(0.0, 1.0 - low_share - high_share)

    preservation: float | None = None
    sdr: float | None = None
    if reference_path is not None and reference_path.exists():
        ref_data, ref_sr = sf.read(str(reference_path), always_2d=True)
        ref_mono = _to_mono(ref_data)
        if ref_sr == sr and ref_mono.size > 0:
            ref_total = float(np.sqrt(np.mean(np.square(ref_mono)) + 1e-12))
            ref_high = _highpass_rms(ref_mono, sr, high_end_hz)
            if ref_total > 1e-9:
                preservation = high_share / max(ref_high / ref_total, 1e-12)
            sdr = si_sdr(mono, ref_mono)

    overlap: dict[str, float] | None = None
    if competing_stems:
        overlap = {}
        for name, path in competing_stems.items():
            if path is None or not Path(path).exists():
                continue
            try:
                other, other_sr = sf.read(str(path), always_2d=True)
            except Exception:
                continue
            if other.size == 0 or other_sr != sr:
                continue
            overlap[name] = _spectral_overlap(mono, _to_mono(other))

    reason = (
        f"low (<{low_band_hz:.0f} Hz) share {low_share:.2f}, "
        f"mid {mid_share:.2f}, high (>{high_end_hz:.0f} Hz) share {high_share:.2f}"
    )
    return GuitarStemQualityDiagnostics(
        attempted=True,
        reason=reason,
        low_band_energy_share=low_share,
        mid_band_energy_share=mid_share,
        high_end_energy_share=high_share,
        high_end_cutoff_hz=high_end_hz,
        high_end_preservation=preservation,
        si_sdr=sdr,
        competitor_overlap=overlap or None,
    )


def apply_bass_bleed_mitigation(
    guitar_path: Path,
    output_path: Path,
    *,
    cutoff_hz: float = BASS_BLEED_HPF_CUTOFF_HZ,
    order: int = BASS_BLEED_HPF_ORDER,
) -> Path:
    """
    Opt-in, partial mitigation: high-pass the guitar stem below ``cutoff_hz``.

    This attenuates sub-guitar-range energy (see ``BASS_BLEED_HPF_CUTOFF_HZ``
    for the tuning tradeoff) — it does not remove bleed that overlaps the
    guitar's own playable range or harmonics of the bled bass above the
    cutoff, so it is a partial reduction, not a fix.
    """
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt

    data, sr = sf.read(str(guitar_path), always_2d=True)
    nyquist = sr / 2.0
    normalized = min(0.99, max(1e-6, cutoff_hz / nyquist))
    sos = butter(order, normalized, btype="high", output="sos")
    filtered = sosfiltfilt(sos, data, axis=0).astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), filtered, sr, subtype="PCM_16")
    return output_path


def _lowpass_audio(
    data: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    normalized = min(0.99, max(1e-6, cutoff_hz / nyquist))
    sos = butter(order, normalized, btype="low", output="sos")
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


@dataclass
class LowEndRecoveryDiagnostics:
    """Pre/post low-band metrics for opt-in de-bleed + harmonic restore."""

    attempted: bool
    reason: str
    sub_bass_debleed_applied: bool = False
    harmonic_restore_applied: bool = False
    debleed_cutoff_hz: float = SUB_BASS_DEBLEED_CUTOFF_HZ
    harmonic_restore_db: float = 0.0
    pre_low_band_energy_share: float | None = None
    post_low_band_energy_share: float | None = None
    low_band_rms_removed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def guitar_prerefine_path(guitar_path: Path) -> Path:
    return Path(guitar_path).parent / GUITAR_PREREFINE_NAME


def guitar_refined_path(guitar_path: Path) -> Path:
    return Path(guitar_path).parent / GUITAR_REFINED_NAME


def has_guitar_prerefine(guitar_path: Path) -> bool:
    return guitar_prerefine_path(guitar_path).is_file()


def ensure_guitar_prerefine_backup(guitar_path: Path) -> Path:
    """Keep the first-pass guitar stem before MelBand refine."""
    backup = guitar_prerefine_path(guitar_path)
    if not backup.is_file() and Path(guitar_path).is_file():
        shutil.copy2(guitar_path, backup)
    return backup


def save_guitar_refined_backup(guitar_path: Path) -> Path:
    """Snapshot guitar after MelBand refine for Mixer A/B."""
    dest = guitar_refined_path(guitar_path)
    if Path(guitar_path).is_file():
        shutil.copy2(guitar_path, dest)
    return dest


def attach_guitar_refine_backups(artifacts: dict) -> None:
    """Add refine wavs that already sit next to guitar.wav so downloads can keep them."""
    guitar = artifacts.get("guitar")
    if guitar is None:
        return
    pre = guitar_prerefine_path(guitar)
    refined = guitar_refined_path(guitar)
    if pre.is_file():
        artifacts["guitar_prerefine"] = pre
    if refined.is_file():
        artifacts["guitar_refined"] = refined


def switch_guitar_stem_variant(guitar_path: Path, *, use_prerefine: bool) -> bool:
    """Copy pre-refine or refined backup onto ``guitar.wav``."""
    guitar = Path(guitar_path)
    source = guitar_prerefine_path(guitar) if use_prerefine else guitar_refined_path(guitar)
    if not source.is_file():
        return False
    shutil.copy2(source, guitar)
    return True


def _apply_peak_limit(audio: np.ndarray) -> np.ndarray:
    """Soft ceiling (~−1 dBTP proxy) without hard peak-normalizing the whole stem."""
    return apply_true_peak_ceiling(audio.astype(np.float32, copy=False))


def apply_sub_bass_debleed(
    guitar_path: Path,
    output_path: Path | None,
    bass_path: Path,
    *,
    drums_path: Path | None = None,
    cutoff_hz: float = SUB_BASS_DEBLEED_CUTOFF_HZ,
    mix_scale: float = SUB_BASS_DEBLEED_MIX,
) -> tuple[Path, float | None, float | None, float | None]:
    """Subtract scaled bass (and optional drum) low band from the guitar stem.

    Returns ``(dest_path, pre_share, post_share, removed_rms)``.
    """
    import soundfile as sf

    dest = Path(output_path) if output_path is not None else Path(guitar_path)
    if not Path(guitar_path).is_file() or not Path(bass_path).is_file():
        return dest, None, None, None

    guitar, sr = sf.read(str(guitar_path), always_2d=True)
    bass, bass_sr = sf.read(str(bass_path), always_2d=True)
    if guitar.size == 0 or bass.size == 0 or int(sr) != int(bass_sr):
        return dest, None, None, None

    n = min(len(guitar), len(bass))
    guitar = guitar[:n].astype(np.float32)
    bass = bass[:n].astype(np.float32)
    pre_share = _low_band_energy_share(guitar, int(sr), cutoff_hz)

    guitar_low = _lowpass_audio(guitar, int(sr), cutoff_hz)
    bleed = _lowpass_audio(bass, int(sr), cutoff_hz)
    if drums_path is not None and Path(drums_path).is_file():
        drums, drums_sr = sf.read(str(drums_path), always_2d=True)
        if drums.size and int(drums_sr) == int(sr):
            drums = drums[:n].astype(np.float32)
            bleed = bleed + SUB_BASS_DRUMS_MIX * _lowpass_audio(drums, int(sr), cutoff_hz)

    scale = max(0.0, min(1.0, float(mix_scale)))
    removed = scale * bleed
    removed_rms = float(np.sqrt(np.mean(np.square(removed)) + 1e-12))
    restored = (guitar - guitar_low + (guitar_low - removed)).astype(np.float32)
    restored = _apply_peak_limit(restored)

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), restored, int(sr), subtype="PCM_16")
    post_share = _low_band_energy_share(restored, int(sr), cutoff_hz)
    return dest, pre_share, post_share, removed_rms


def apply_bleed_gate(
    guitar_path: Path,
    output_path: Path | None = None,
    *,
    competitor_paths: dict[str, Path] | None = None,
    threshold: float = BLEED_GATE_DOMINANCE_THRESHOLD,
    max_attenuation_linear: float = BLEED_GATE_MAX_ATTENUATION,
    strength: float = BLEED_GATE_STRENGTH,
    high_hz: float = BLEED_GATE_HIGH_HZ,
) -> BleedGateDiagnostics:
    """Competitive spectral scrub of bleed left on a guitar stem (opt-in).

    For every STFT bin the competitor magnitude (sum of the separating model's
    non-guitar stems that are expected to bleed) is compared to the guitar's
    own magnitude. When the competitor strongly dominates the bin's energy,
    the guitar's bin is attenuated down to ``max_attenuation_linear``. Solo or
    guitar-dominated bins are left untouched, so clean sections are not rolled
    off. This complements ``apply_guitar_low_end_recovery``: it scrubs *bleed*
    (energy the model placed on the guitar stem that belongs to bass/cymbal/)
    instead of boosting blindly. It cannot re-insert guitar energy the
    separator already dropped — that stays the job of the better base model,
    ensemble, or MelBand refine.
    """
    dest = Path(output_path) if output_path is not None else Path(guitar_path)
    if not Path(guitar_path).is_file():
        return BleedGateDiagnostics(attempted=False, reason="guitar stem missing")
    competitors = {
        name: path
        for name, path in (competitor_paths or {}).items()
        if path is not None and Path(path).is_file()
    }
    if not competitors:
        return BleedGateDiagnostics(
            attempted=False, reason="no competitor stems to gate against"
        )

    import soundfile as sf

    guitar_data, sr = sf.read(str(guitar_path), always_2d=True)
    if guitar_data.size == 0:
        return BleedGateDiagnostics(attempted=False, reason="guitar stem is empty")
    if int(sr) == 0:
        return BleedGateDiagnostics(attempted=False, reason="unknown sample rate")

    n = len(guitar_data)
    comp_data: dict[str, np.ndarray] = {}
    for name, path in competitors.items():
        try:
            data, comp_sr = sf.read(str(path), always_2d=True)
        except Exception:
            continue
        if data.size == 0 or int(comp_sr) != int(sr):
            continue
        comp_data[name] = data[:n]

    if not comp_data:
        return BleedGateDiagnostics(
            attempted=False, reason="no compatible competitor stems (sr mismatch)"
        )

    gate_high_hz = max(0.0, min(float(high_hz), int(sr) / 2.0))
    alpha = max(0.0, min(1.0, float(strength)))
    floor_lin = max(1e-3, min(1.0, float(max_attenuation_linear)))

    from scipy.signal import istft, stft

    nperseg = BLEED_GATE_NPERSEG
    noverlap = BLEED_GATE_OVERLAP
    window = "hann"

    guitar_mono = _to_mono(guitar_data)
    comp_mono = np.zeros_like(guitar_mono)
    for data in comp_data.values():
        comp_mono = comp_mono + _to_mono(data)

    _, _, gz = stft(
        guitar_mono, fs=int(sr), window=window, nperseg=nperseg, noverlap=noverlap
    )
    _, _, cz = stft(
        comp_mono, fs=int(sr), window=window, nperseg=nperseg, noverlap=noverlap
    )
    gmag = np.abs(gz)
    cmag = np.abs(cz)
    bins = min(gmag.shape[0], cmag.shape[0])
    frames = min(gmag.shape[1], cmag.shape[1])
    gmag = gmag[:bins, :frames]
    cmag = cmag[:bins, :frames]

    nyquist = int(sr) / 2.0
    cutoff_bin = bins if gate_high_hz >= nyquist else int(bins * gate_high_hz / nyquist)
    cutoff_bin = max(1, min(bins, cutoff_bin))

    denom = gmag + cmag + 1e-12
    score = cmag / denom
    trip = (score >= float(threshold)).astype(np.float32)
    over = np.maximum(0.0, (score - threshold) / max(1e-6, 1.0 - threshold))
    mult = 1.0 - alpha * over * (1.0 - floor_lin)
    gated_mask = np.where(trip > 0, mult, 1.0).astype(np.float32)
    gated_mask[cutoff_bin:, :] = 1.0

    dominated = float(np.mean(trip[:cutoff_bin, :])) if cutoff_bin > 0 else 0.0
    attenuated = gated_mask < 0.999
    if not bool(np.any(attenuated)):
        return BleedGateDiagnostics(
            attempted=True,
            reason="competitor energy never dominated guitar; nothing scrubbed",
            gated=False,
            dominated_bin_fraction=dominated,
            mean_attenuation_db=0.0,
            threshold=threshold,
            max_attenuation_linear=floor_lin,
            high_hz=gate_high_hz,
            competitors=tuple(comp_data),
        )
    atten_db = float(
        np.mean(np.where(attenuated, -20.0 * np.log10(gated_mask + 1e-12), 0.0))
    )

    channels = guitar_data.shape[1]
    out_channels: list[np.ndarray] = []
    for ch in range(channels):
        _, _, Z = stft(
            guitar_data[:, ch].astype(np.float64),
            fs=int(sr),
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        Z = Z[:bins, :frames]
        Z = Z * gated_mask
        _, rebuilt = istft(
            Z, fs=int(sr), window=window, nperseg=nperseg, noverlap=noverlap
        )
        rebuilt = rebuilt[:n]
        if len(rebuilt) < n:
            rebuilt = np.pad(rebuilt, (0, n - len(rebuilt)))
        out_channels.append(rebuilt)

    stereo = np.column_stack(out_channels)[:n].astype(np.float32)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    stereo = _apply_peak_limit(stereo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), stereo, int(sr), subtype="PCM_16")
    return BleedGateDiagnostics(
        attempted=True,
        reason=(
            f"scrubbed {dominated:.3f} dominated bins, "
            f"mean ~{atten_db:.1f} dB attenuation"
        ),
        gated=True,
        dominated_bin_fraction=dominated,
        mean_attenuation_db=atten_db,
        threshold=threshold,
        max_attenuation_linear=floor_lin,
        high_hz=gate_high_hz,
        competitors=tuple(comp_data),
    )


@dataclass
class EnsembleGuitarDiagnostics:
    """Per-band soft-max blend of two separators' guitar stems (opt-in)."""

    attempted: bool
    reason: str
    blended: bool = False
    bands: int = ENSEMBLE_BLEND_BANDS
    gamma: float = ENSEMBLE_BLEND_GAMMA
    low_hz: float = ENSEMBLE_BLEND_LOW_HZ
    mean_primary_weight: float | None = None
    primary_label: str = ""
    secondary_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def blend_guitar_stems(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    *,
    bands: int = ENSEMBLE_BLEND_BANDS,
    gamma: float = ENSEMBLE_BLEND_GAMMA,
    low_hz: float = ENSEMBLE_BLEND_LOW_HZ,
    primary_label: str = "primary",
    secondary_label: str = "secondary",
) -> EnsembleGuitarDiagnostics:
    """Blend two guitar stems per frequency band by soft-max energy weighting.

    Each separator loses a *different* sub-band on dense mixes; this picks, per
    log-spaced band, whichever stem carries more energy (soft max with
    ``gamma``≈3), which preserves every band either model kept. The mask is
    estimated from the mono mixdown and applied per channel with an overlap-add
    STFT so the result is COLA-exact apart from the band smoothing.
    """
    dest = Path(output_path)
    if not Path(primary_path).is_file() or not Path(secondary_path).is_file():
        return EnsembleGuitarDiagnostics(
            attempted=False,
            reason="one of the guitar stems is missing",
            primary_label=primary_label,
            secondary_label=secondary_label,
        )

    import soundfile as sf
    from scipy.signal import istft, stft

    primary, sr_p = sf.read(str(primary_path), always_2d=True)
    secondary, sr_s = sf.read(str(secondary_path), always_2d=True)
    if primary.size == 0 or secondary.size == 0:
        return EnsembleGuitarDiagnostics(
            attempted=False,
            reason="one of the guitar stems is empty",
            primary_label=primary_label,
            secondary_label=secondary_label,
        )
    if int(sr_p) != int(sr_s):
        return EnsembleGuitarDiagnostics(
            attempted=False,
            reason=f"sample-rate mismatch ({sr_p} vs {sr_s})",
            primary_label=primary_label,
            secondary_label=secondary_label,
        )
    sr = int(sr_p)
    n = min(len(primary), len(secondary))
    primary = primary[:n]
    secondary = secondary[:n]

    nperseg = BLEED_GATE_NPERSEG
    noverlap = BLEED_GATE_OVERLAP
    window = "hann"
    p_mono = _to_mono(primary)
    s_mono = _to_mono(secondary)

    freqs, _, P = stft(p_mono, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    _, _, S = stft(s_mono, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    bins = min(P.shape[0], S.shape[0])
    frames = min(P.shape[1], S.shape[1])
    P = P[:bins, :frames]
    S = S[:bins, :frames]

    band_edges = _log_band_edges(freqs[:bins], bands=bands, low_hz=low_hz)
    weights = np.ones_like(P, dtype=np.float32)
    if len(band_edges) >= 2:
        band_weights: list[float] = []
        for lo_idx, hi_idx in itertools.pairwise(band_edges):
            if hi_idx <= lo_idx:
                continue
            p_e = float(np.sum(np.abs(P[lo_idx:hi_idx, :]) ** 2)) + 1e-12
            s_e = float(np.sum(np.abs(S[lo_idx:hi_idx, :]) ** 2)) + 1e-12
            w = p_e**gamma / (p_e**gamma + s_e**gamma)
            band_weights.append(float(w))
            weights[lo_idx:hi_idx, :] = w
    mean_primary_weight = float(np.mean(weights))

    channels = primary.shape[1]
    out_channels: list[np.ndarray] = []
    for ch in range(channels):
        _, _, Zp = stft(
            primary[:, ch].astype(np.float64),
            fs=sr,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        _, _, Zs = stft(
            secondary[:, ch].astype(np.float64),
            fs=sr,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        Zp = Zp[:bins, :frames]
        Zs = Zs[:bins, :frames]
        blended = Zp * weights + Zs * (1.0 - weights)
        _, rebuilt = istft(
            blended, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap
        )
        rebuilt = rebuilt[:n]
        if len(rebuilt) < n:
            rebuilt = np.pad(rebuilt, (0, n - len(rebuilt)))
        out_channels.append(rebuilt)

    stereo = np.column_stack(out_channels)[:n].astype(np.float32)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    stereo = _apply_peak_limit(stereo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), stereo, sr, subtype="PCM_16")
    return EnsembleGuitarDiagnostics(
        attempted=True,
        reason=(
            f"blended {len(band_edges) - 1} bands, "
            f"mean primary weight {mean_primary_weight:.2f}"
        ),
        blended=True,
        bands=bands,
        gamma=gamma,
        low_hz=low_hz,
        mean_primary_weight=mean_primary_weight,
        primary_label=primary_label,
        secondary_label=secondary_label,
    )


def _log_band_edges(
    freqs: np.ndarray, *, bands: int, low_hz: float
) -> list[int]:
    """Indices dividing ``freqs`` into ``bands`` log-spaced groups below Nyquist."""
    n = len(freqs)
    if n == 0:
        return []
    hi = float(freqs[-1])
    lo = min(float(low_hz), hi)
    if hi <= lo * (1.0 + 1e-6):
        return [0, n]
    edges = np.geomspace(lo, hi, num=max(2, int(bands)) + 1)
    out: list[int] = []
    for value in edges:
        idx = int(np.searchsorted(freqs, value))
        idx = max(0, min(n - 1, idx))
        if not out or idx != out[-1]:
            out.append(idx)
    if out[-1] != n:
        out.append(n)  # framegroup closed at the full bin count
    return out


def apply_guitar_low_end_recovery(
    guitar_path: Path,
    output_path: Path | None = None,
    *,
    bass_path: Path | None = None,
    drums_path: Path | None = None,
    sub_bass_debleed: bool = False,
    boost_db: float = 0.0,
    bass_missing_reason: str = "",
) -> LowEndRecoveryDiagnostics:
    """Opt-in subtractive de-bleed then harmonic restore (both default off).

    ``bass_missing_reason`` overrides the generic "bass stem missing" note when
    the bass stem exists upstream but was intentionally dropped (e.g. emit).
    """
    dest = Path(output_path) if output_path is not None else Path(guitar_path)
    work = dest
    debleed_applied = False
    restore_applied = False
    pre_share: float | None = None
    post_share: float | None = None
    removed_rms: float | None = None
    reasons: list[str] = []

    if sub_bass_debleed and bass_path is not None and Path(bass_path).is_file():
        work, pre_share, post_share, removed_rms = apply_sub_bass_debleed(
            work,
            work,
            bass_path,
            drums_path=drums_path,
        )
        if pre_share is not None:
            debleed_applied = True
            reasons.append(
                f"subtracted bass/drum energy below {SUB_BASS_DEBLEED_CUTOFF_HZ:.0f} Hz"
            )
    elif sub_bass_debleed:
        if bass_missing_reason:
            reasons.append(f"sub_bass_debleed not applied: {bass_missing_reason}")
        else:
            reasons.append("sub_bass_debleed requested but bass stem missing")

    try:
        boost = float(boost_db)
    except (TypeError, ValueError):
        boost = 0.0
    boost = max(0.0, min(LOW_END_RESTORE_MAX_DB, boost))
    if boost >= 1e-6:
        restore_diag = apply_low_end_restore(work, work, boost_db=boost)
        if restore_diag.attempted:
            restore_applied = True
            reasons.append(restore_diag.reason)
            if pre_share is None:
                pre_share = restore_diag.pre_low_band_energy_share
            post_share = restore_diag.post_low_band_energy_share
        elif not debleed_applied:
            return LowEndRecoveryDiagnostics(
                attempted=False,
                reason=restore_diag.reason,
                harmonic_restore_db=boost,
            )

    if not debleed_applied and not restore_applied:
        return LowEndRecoveryDiagnostics(
            attempted=False,
            reason="; ".join(reasons) if reasons else "off (no recovery stages enabled)",
            harmonic_restore_db=boost,
        )

    return LowEndRecoveryDiagnostics(
        attempted=True,
        reason="; ".join(reasons),
        sub_bass_debleed_applied=debleed_applied,
        harmonic_restore_applied=restore_applied,
        debleed_cutoff_hz=SUB_BASS_DEBLEED_CUTOFF_HZ,
        harmonic_restore_db=boost,
        pre_low_band_energy_share=pre_share,
        post_low_band_energy_share=post_share,
        low_band_rms_removed=removed_rms,
    )


@dataclass
class LowEndRestoreDiagnostics:
    """Pre/post low-band share after an opt-in 60–200 Hz presence boost."""

    attempted: bool
    reason: str
    boost_db: float = 0.0
    low_hz: float = LOW_END_RESTORE_LOW_HZ
    high_hz: float = LOW_END_RESTORE_HIGH_HZ
    pre_low_band_energy_share: float | None = None
    post_low_band_energy_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def apply_low_end_restore(
    wav_path: Path,
    output_path: Path | None = None,
    *,
    boost_db: float = 0.0,
    low_hz: float = LOW_END_RESTORE_LOW_HZ,
    high_hz: float = LOW_END_RESTORE_HIGH_HZ,
) -> LowEndRestoreDiagnostics:
    """Boost 60–200 Hz on a guitar stem. ``boost_db=0`` is a no-op (file unchanged)."""
    dest = Path(output_path) if output_path is not None else Path(wav_path)
    try:
        boost = float(boost_db)
    except (TypeError, ValueError):
        boost = 0.0
    boost = max(0.0, min(LOW_END_RESTORE_MAX_DB, boost))
    if boost < 1e-6:
        return LowEndRestoreDiagnostics(
            attempted=False,
            reason="off (boost_db is 0)",
            boost_db=0.0,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    if not Path(wav_path).is_file():
        return LowEndRestoreDiagnostics(
            attempted=False,
            reason="guitar stem missing",
            boost_db=boost,
            low_hz=low_hz,
            high_hz=high_hz,
        )

    import soundfile as sf

    data, sr = sf.read(str(wav_path), always_2d=True)
    if data.size == 0:
        return LowEndRestoreDiagnostics(
            attempted=True,
            reason="guitar stem is empty",
            boost_db=boost,
            low_hz=low_hz,
            high_hz=high_hz,
        )

    pre = _low_band_energy_share(data, int(sr), high_hz)
    gain = 10.0 ** (boost / 20.0)
    band = _bandpass_audio(data, int(sr), low_hz, high_hz)
    restored = (data.astype(np.float32) + (gain - 1.0) * band).astype(np.float32)
    restored = _apply_peak_limit(restored)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), restored, int(sr), subtype="PCM_16")
    post_data, post_sr = sf.read(str(dest), always_2d=True)
    post = _low_band_energy_share(post_data, int(post_sr), high_hz)
    return LowEndRestoreDiagnostics(
        attempted=True,
        reason=f"boosted {low_hz:.0f}–{high_hz:.0f} Hz by {boost:.1f} dB",
        boost_db=boost,
        low_hz=low_hz,
        high_hz=high_hz,
        pre_low_band_energy_share=pre,
        post_low_band_energy_share=post,
    )


def _low_band_energy_share(data: np.ndarray, sr: int, cutoff_hz: float) -> float:
    mono = _to_mono(data)
    total = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    if total < 1e-9:
        return 0.0
    return _lowpass_rms(mono, sr, cutoff_hz) / total


def _noop_progress(stage: str, message: str) -> None:
    pass


def _load_stereo(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.astype(np.float32), sr


def analyze_dual_guitar_candidate(guitar_path: Path) -> DualGuitarDiagnostics:
    """Legacy heuristic: stereo guitar stem with distinct L/R content may be two guitars.

    Not called by ``separate_stems``. Live Lead/Rhythm uses
    ``audio_to_tab.lead_rhythm.build_spatial_pair`` instead.
    """
    if not guitar_path.exists():
        return DualGuitarDiagnostics(False, False, "guitar stem missing")

    data, _sr = _load_stereo(guitar_path)
    if data.shape[1] < 2 or len(data) < 1024:
        return DualGuitarDiagnostics(True, False, "guitar stem is mono or too short")

    left = data[:, 0]
    right = data[:, 1]
    rms_l = float(np.sqrt(np.mean(left**2)))
    rms_r = float(np.sqrt(np.mean(right**2)))
    max_rms = max(rms_l, rms_r, 1e-9)
    balance = min(rms_l, rms_r) / max_rms

    if balance < 0.12:
        return DualGuitarDiagnostics(
            True,
            False,
            "stereo energy too imbalanced for dual-guitar split",
            balance_ratio=balance,
        )

    if np.std(left) < 1e-6 or np.std(right) < 1e-6:
        return DualGuitarDiagnostics(True, False, "near-silent channel", balance_ratio=balance)

    corr = float(np.corrcoef(left, right)[0, 1])
    if corr > 0.92:
        return DualGuitarDiagnostics(
            True,
            False,
            "L/R channels too similar (likely one guitar)",
            correlation=corr,
            balance_ratio=balance,
        )

    return DualGuitarDiagnostics(
        True,
        True,
        "stereo heuristic suggests two distinct guitar parts",
        correlation=corr,
        balance_ratio=balance,
    )


def split_dual_guitar_stem(
    guitar_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Path], DualGuitarDiagnostics]:
    """
    Legacy: derive guitar1/guitar2 from stereo guitar stem (spatial labels only).

    Not called by ``separate_stems``. Prefer ``split_lead_rhythm_guitar`` for
    semantic Lead/Rhythm stems. Kept for tests and any external callers.
    """
    import warnings

    import soundfile as sf

    warnings.warn(
        "split_dual_guitar_stem is deprecated; use "
        "audio_to_tab.lead_rhythm.split_lead_rhythm_guitar",
        DeprecationWarning,
        stacklevel=2,
    )

    diag = analyze_dual_guitar_candidate(guitar_path)
    if not diag.split:
        return {}, diag

    data, sr = _load_stereo(guitar_path)
    left = data[:, 0]
    right = data[:, 1]

    # Soft dominance masks reduce bleed from the opposite channel.
    guitar1_mono = np.where(np.abs(left) >= np.abs(right), left, left - 0.5 * right)
    guitar2_mono = np.where(np.abs(right) >= np.abs(left), right, right - 0.5 * left)
    guitar1 = np.column_stack([guitar1_mono, guitar1_mono])
    guitar2 = np.column_stack([guitar2_mono, guitar2_mono])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "guitar1.wav"
    p2 = out_dir / "guitar2.wav"
    sf.write(str(p1), guitar1, sr, subtype="PCM_16")
    sf.write(str(p2), guitar2, sr, subtype="PCM_16")
    return {"guitar1": p1, "guitar2": p2}, diag


def _trim_audio(
    input_path: Path,
    max_duration_sec: float | None,
    *,
    start_sec: float = 0.0,
) -> Path:
    """Trim normalized audio with ffmpeg ``-ss`` + ``-t`` (after normalize, before Demucs)."""
    if max_duration_sec is None or max_duration_sec <= 0:
        return input_path
    out = input_path.parent / f"{input_path.stem}_trim.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return input_path
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(start_sec),
                "-i",
                str(input_path),
                "-t",
                str(max_duration_sec),
                str(out),
            ],
            capture_output=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
            **subprocess_run_kwargs(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg timed out after {SUBPROCESS_TIMEOUT}s processing {input_path}"
        ) from None
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr or result.stdout}")
    return out


def merge_two_pass_stems(
    pass1: dict[str, Path],
    pass2: dict[str, Path],
    dest_dir: str | Path,
) -> dict[str, Path]:
    """Keep 4-stem vocals/drums/bass; take guitar/piano from 6-stem residual.

    Leftover pass-2 vocals/drums/bass (energy that leaked into ``other``) are
    folded into the residual ``other`` stem rather than replacing pass-1 stems.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Path] = {}
    for name in TWO_PASS_KEEP_FROM_FIRST:
        src = pass1.get(name)
        if src is not None and src.is_file():
            out = dest / f"{name}.wav"
            shutil.copy2(src, out)
            merged[name] = out
    for name in TWO_PASS_KEEP_FROM_SECOND:
        src = pass2.get(name)
        if src is not None and src.is_file():
            out = dest / f"{name}.wav"
            shutil.copy2(src, out)
            merged[name] = out
    fold: dict[str, Path] = {}
    for name in TWO_PASS_FOLD_INTO_OTHER:
        src = pass2.get(name)
        if src is not None and src.is_file():
            fold[name] = src
    other_out = dest / "other.wav"
    if fold:
        mix_stems_to_wav(fold, audible=list(fold), output_path=other_out)
        merged["other"] = other_out
    elif "other" in pass1 and pass1["other"].is_file():
        shutil.copy2(pass1["other"], other_out)
        merged["other"] = other_out
    return merged


def _collect_stem_wavs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for stem_path in root.rglob("*.wav"):
        if stem_path.stem in found:
            logger.debug("Duplicate stem name %s; keeping later entry", stem_path.stem)
        found[stem_path.stem] = stem_path
    return found


def _run_demucs_model(
    audio_path: Path,
    demucs_out: Path,
    cfg: IsolateConfig,
    *,
    model: str,
    allow_guitar_ft: bool,
    progress: ProgressCallback,
    should_abort: Callable[[], bool] | None = None,
    timeout_sec: float | None = None,
) -> None:
    """Run one Demucs pass into ``demucs_out`` (CLI or guitar-ft in-process)."""
    use_guitar_ft = (
        allow_guitar_ft
        and cfg.guitar_checkpoint == GUITAR_FT_CHECKPOINT_ID
        and model == "htdemucs_6s"
        and not cfg.two_stems
    )
    if use_guitar_ft:
        try:
            progress("separate", separate_progress_message(cfg.device))
            run_demucs_guitar_ft_inprocess(
                audio_path,
                demucs_out,
                device=cfg.device,
                quality=cfg.quality,
                segment=effective_demucs_segment(model, cfg.demucs_segment),
                jobs=max(1, int(cfg.demucs_jobs)),
                shifts=cfg.demucs_shifts,
                overlap=cfg.demucs_overlap,
            )
            return
        except Exception as exc:
            logger.warning("guitar-ft separation failed; falling back to stock 6s: %s", exc)

    shifts = resolve_demucs_shifts(cfg.quality, cfg.demucs_shifts)
    overlap = resolve_demucs_overlap(cfg.quality, cfg.demucs_overlap)
    demucs_args = [
        "-n",
        model,
        "-d",
        cfg.device,
        "-o",
        str(demucs_out),
        "--shifts",
        shifts,
        "--overlap",
        overlap,
        "--jobs",
        str(max(1, int(cfg.demucs_jobs))),
    ]
    segment = effective_demucs_segment(model, cfg.demucs_segment)
    if segment is not None:
        demucs_args.extend(["--segment", str(segment)])
    if cfg.two_stems:
        demucs_args.extend(["--two-stems", cfg.two_stems])
    demucs_args.append(str(audio_path))
    run_demucs(demucs_args, should_abort=should_abort, timeout_sec=timeout_sec)


def _run_separation_backend(
    audio_path: Path,
    demucs_out: Path,
    cfg: IsolateConfig,
    *,
    model: str,
    allow_guitar_ft: bool,
    progress: ProgressCallback,
    should_abort: Callable[[], bool] | None = None,
    timeout_sec: float | None = None,
) -> None:
    """Run Demucs or an opt-in RoFormer / SCNet backend into ``demucs_out``."""
    if model in ROFORMER_MODELS:
        progress("separate", separate_progress_message(cfg.device))
        run_roformer_model(audio_path, demucs_out, model=model, device=cfg.device)
        return
    if model in SCNET_MODELS:
        progress("separate", separate_progress_message(cfg.device))
        run_scnet_model(audio_path, demucs_out, device=cfg.device)
        return
    _run_demucs_model(
        audio_path,
        demucs_out,
        cfg,
        model=model,
        allow_guitar_ft=allow_guitar_ft,
        progress=progress,
        should_abort=should_abort,
        timeout_sec=timeout_sec,
    )


def _maybe_refine_guitar(
    artifacts: dict[str, Path],
    cfg: IsolateConfig,
    progress: ProgressCallback,
) -> None:
    """Opt-in MelBand guitar specialist; skip (with warning) if extra missing."""
    if not cfg.guitar_refine or "guitar" not in artifacts:
        return
    progress("guitar_refine", "Refining guitar stem")
    if not is_guitar_refine_available():
        logger.warning(
            "guitar refine requested but no RoFormer extra is installed; skipping. %s",
            ROFORMER_INSTALL_HINT,
        )
        return
    try:
        residual = artifacts.get("other")
        ensure_guitar_prerefine_backup(artifacts["guitar"])
        run_guitar_refine(
            artifacts["guitar"],
            artifacts["guitar"],
            residual_path=residual if residual is not None and residual.is_file() else None,
            device=cfg.device,
        )
        save_guitar_refined_backup(artifacts["guitar"])
        progress("guitar_refine", "Guitar refinement complete")
    except Exception as exc:
        logger.warning("guitar refine failed; keeping first-pass guitar stem: %s", exc)
        progress("guitar_refine", "Guitar refinement failed. Using first-pass stem")


def separate_stems(
    audio_path: str | Path,
    output_dir: str | Path,
    config: IsolateConfig | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    should_abort: Callable[[], bool] | None = None,
    checkpoint_dir: str | Path | None = None,
    completed_stages: set[str] | frozenset[str] | list[str] | None = None,
    on_stage_complete: StageCompleteCallback | None = None,
    subprocess_timeout_sec: float | None = None,
) -> dict[str, Path]:
    """
    Separate an audio file into instrument stems using Demucs.

    Returns a mapping of stem name -> wav path under output_dir.
    CPU separation is slow (~track length or longer); quality presets multiply time.
    CUDA (NVIDIA GPU) is used when ``config.device`` is ``cuda`` and Torch can see it.
    ``should_abort`` and ``subprocess_timeout_sec`` are honored **during** the
    long-running Demucs subprocess (terminated on abort / killed on timeout),
    not only at the stage boundaries checked between steps.
    """
    cfg = config or IsolateConfig()
    if cfg.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {cfg.model!r}. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    needs_demucs = cfg.model in DEMUCS_MODELS or cfg.two_pass
    if needs_demucs and not is_demucs_available():
        raise RuntimeError(
            f"Demucs is not installed. {DEMUCS_INSTALL_HINT}"
        )
    if cfg.model in ROFORMER_MODELS and not is_roformer_backend_available():
        raise RuntimeError(
            f"RoFormer backend is not installed. {ROFORMER_INSTALL_HINT}"
        )
    if cfg.model in SCNET_MODELS and not is_scnet_available():
        raise RuntimeError(
            f"SCNet backend is not installed. {SCNET_INSTALL_HINT}"
        )

    ensure_cuda_available(cfg.device)

    progress = on_progress or _noop_progress
    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = set(completed_stages or ())
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    def _ingest_trimmed(work: Path) -> Path:
        _abort_if(should_abort)
        progress("ingest", "Preparing audio")
        normalized = normalize_audio(src, work / "normalized.wav")
        trim_length = cfg.max_duration_sec
        trim_start = cfg.start_sec
        if trim_length is None and trim_start > 0:
            file_dur = probe_duration_sec(normalized)
            if file_dur is None or file_dur <= trim_start:
                raise RegionError(
                    f"Requested region (start {trim_start:.1f}s) can't be honored: "
                    "the audio duration is unknown or the start is past the end of "
                    "the file. Process the full track or a shorter section instead."
                )
            trim_length = file_dur - trim_start
        if trim_length is None or trim_length <= 0:
            trim_message = "Using full audio"
        else:
            trim_message = f"Trimming to {format_region_label(trim_start, trim_length)}"
        logger.debug("isolate trim: %s", trim_message)
        progress("ingest", trim_message)
        return _trim_audio(normalized, trim_length, start_sec=trim_start)

    def _run_separate_stage(trimmed: Path, work: Path) -> Path:
        demucs_out = work / "demucs_out"
        demucs_out.mkdir(parents=True, exist_ok=True)
        progress("separate", separate_progress_message(cfg.device))
        if cfg.two_pass:
            pass1_out = work / "pass1"
            pass2_out = work / "pass2"
            _run_demucs_model(
                trimmed,
                pass1_out,
                cfg,
                model="htdemucs",
                allow_guitar_ft=False,
                progress=progress,
                should_abort=should_abort,
                timeout_sec=subprocess_timeout_sec,
            )
            pass1 = _collect_stem_wavs(pass1_out)
            other = pass1.get("other")
            if other is None or not other.is_file():
                logger.warning("two-pass missing other stem; falling back to single-pass 6s")
                _run_separation_backend(
                    trimmed,
                    demucs_out,
                    cfg,
                    model=cfg.model,
                    allow_guitar_ft=True,
                    progress=progress,
                    should_abort=should_abort,
                    timeout_sec=subprocess_timeout_sec,
                )
            else:
                progress("separate", "Separating guitar from the leftover mix")
                _run_demucs_model(
                    other,
                    pass2_out,
                    cfg,
                    model="htdemucs_6s",
                    allow_guitar_ft=True,
                    progress=progress,
                    should_abort=should_abort,
                    timeout_sec=subprocess_timeout_sec,
                )
                pass2 = _collect_stem_wavs(pass2_out)
                merge_two_pass_stems(pass1, pass2, demucs_out)
        else:
            _run_separation_backend(
                trimmed,
                demucs_out,
                cfg,
                model=cfg.model,
                allow_guitar_ft=True,
                progress=progress,
                should_abort=should_abort,
                timeout_sec=subprocess_timeout_sec,
            )
        return demucs_out

    def _copy_demucs_checkpoint(demucs_out: Path) -> None:
        if ckpt_root is None:
            return
        dest = ckpt_root / "demucs_out"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(demucs_out, dest)

    def _restore_demucs_checkpoint(work: Path) -> Path:
        demucs_out = work / "demucs_out"
        if demucs_out.exists():
            shutil.rmtree(demucs_out)
        shutil.copytree(ckpt_root / "demucs_out", demucs_out)
        return demucs_out

    def _collect_into_out_dir(demucs_out: Path) -> dict[str, Path]:
        artifacts_ckpt = (ckpt_root / "artifacts.json") if ckpt_root else None
        if (
            _stage_done(completed, "collect")
            and artifacts_ckpt is not None
            and artifacts_ckpt.is_file()
        ):
            loaded = json.loads(artifacts_ckpt.read_text(encoding="utf-8"))
            collected = {k: Path(v) for k, v in loaded.items() if Path(v).is_file()}
            if not collected:
                raise FileNotFoundError("Checkpoint artifacts missing on disk")
            return collected

        _abort_if(should_abort)
        progress("collect", "Separating tracks")
        stem_files = list(demucs_out.rglob("*.wav"))
        if not stem_files:
            raise FileNotFoundError("Demucs did not produce any stem wav files.")

        collected = {}
        for stem_path in stem_files:
            name = stem_path.stem
            dest = out_dir / f"{name}.wav"
            shutil.copy2(stem_path, dest)
            collected[name] = dest

        if not collected:
            raise FileNotFoundError("No stems could be collected from Demucs output.")

        if ckpt_root is not None and artifacts_ckpt is not None:
            artifacts_ckpt.write_text(
                json.dumps({k: str(v) for k, v in collected.items()}, indent=2),
                encoding="utf-8",
            )
        _mark_stage_complete("collect", on_stage_complete)
        _abort_if(should_abort)
        return collected

    def _maybe_ensemble_guitar(artifacts: dict[str, Path], trimmed: Path) -> None:
        """Opt-in Phase 2: per-band blend of Demucs + BS-RoFormer guitar stems."""
        if not cfg.guitar_ensemble or "guitar" not in artifacts:
            return
        if not is_roformer_backend_available():
            logger.warning(
                "guitar ensemble requested but no RoFormer extra is installed; "
                "keeping the primary Demucs guitar stem. %s",
                ROFORMER_INSTALL_HINT,
            )
            return
        _abort_if(should_abort)
        progress("ensemble", "Running secondary separation")
        with tempfile.TemporaryDirectory() as tmp:
            sec_out = Path(tmp) / "secondary"
            sec_out.mkdir(parents=True, exist_ok=True)
            try:
                run_roformer_model(
                    trimmed, sec_out, model=BS_ROFORMER_SW_ID, device=cfg.device
                )
            except Exception as exc:
                logger.warning(
                    "guitar ensemble secondary separation failed; "
                    "keeping the primary guitar stem: %s",
                    exc,
                )
                return
            secondary = sec_out / "guitar.wav"
            if not secondary.is_file():
                logger.warning(
                    "guitar ensemble secondary separation produced no guitar stem; "
                    "keeping the primary guitar stem"
                )
                return
            diag = blend_guitar_stems(
                artifacts["guitar"],
                secondary,
                artifacts["guitar"],
            )
            if diag.attempted:
                ensemble_path = diag.write_json(
                    out_dir / "ensemble_guitar_diagnostics.json"
                )
                artifacts["ensemble_guitar_diagnostics"] = ensemble_path
                logger.debug("guitar ensemble: %s", diag.reason)

    if ckpt_root is not None:
        work = ckpt_root / "work"
        work.mkdir(parents=True, exist_ok=True)
        trimmed_ckpt = ckpt_root / "trimmed.wav"
        if _stage_done(completed, "ingest") and trimmed_ckpt.is_file():
            trimmed = trimmed_ckpt
        else:
            trimmed = _ingest_trimmed(work)
            shutil.copy2(trimmed, trimmed_ckpt)
            _mark_stage_complete("ingest", on_stage_complete)
            _abort_if(should_abort)

        if _stage_done(completed, "separate") and (ckpt_root / "demucs_out").is_dir():
            demucs_out = _restore_demucs_checkpoint(work)
        else:
            _abort_if(should_abort)
            demucs_out = _run_separate_stage(trimmed, work)
            _copy_demucs_checkpoint(demucs_out)
            _mark_stage_complete("separate", on_stage_complete)
            _abort_if(should_abort)
        artifacts = _collect_into_out_dir(demucs_out)
        _maybe_ensemble_guitar(artifacts, trimmed)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            trimmed = _ingest_trimmed(work)
            _abort_if(should_abort)
            demucs_out = _run_separate_stage(trimmed, work)
            artifacts = _collect_into_out_dir(demucs_out)
            _maybe_ensemble_guitar(artifacts, trimmed)

    if not _stage_done(completed, "guitar_refine"):
        _abort_if(should_abort)
        _maybe_refine_guitar(artifacts, cfg, progress)
        if ckpt_root is not None and "guitar" in artifacts and artifacts["guitar"].is_file():
            shutil.copy2(artifacts["guitar"], ckpt_root / "guitar.wav")
        _mark_stage_complete("guitar_refine", on_stage_complete)
        _abort_if(should_abort)
    elif (
        ckpt_root is not None
        and (ckpt_root / "guitar.wav").is_file()
        and "guitar" in artifacts
    ):
        shutil.copy2(ckpt_root / "guitar.wav", artifacts["guitar"])

    if cfg.fold_other_into_guitar and "guitar" in artifacts:
        artifacts, fold_diag = apply_fold_other_into_guitar(
            artifacts,
            mode=cfg.fold_other_mode,
            adaptive_gain=cfg.adaptive_fold_gain,
        )
        fold_path = fold_diag.write_json(out_dir / "fold_other_diagnostics.json")
        artifacts["fold_other_diagnostics"] = fold_path
        logger.debug("fold other: %s", fold_diag.reason)

    if cfg.bleed_gate and "guitar" in artifacts:
        gate_competitors = {
            name: artifacts[name]
            for name in ("bass", "drums", "other")
            if name in artifacts and artifacts[name] is not None
        }
        gate_diag = apply_bleed_gate(
            artifacts["guitar"],
            artifacts["guitar"],
            competitor_paths=gate_competitors,
        )
        gate_path = gate_diag.write_json(out_dir / "bleed_gate_diagnostics.json")
        artifacts["bleed_gate_diagnostics"] = gate_path
        logger.debug("bleed gate: %s", gate_diag.reason)
    bass_path_for_recovery = artifacts.get("bass")
    drums_path_for_recovery = artifacts.get("drums")
    emit_keeps = set(cfg.emit_stems) if cfg.emit_stems else None
    bass_dropped_by_emit = (
        emit_keeps is not None
        and bass_path_for_recovery is not None
        and "bass" not in emit_keeps
    )
    apply_emit_stems(artifacts, cfg.emit_stems)

    if "guitar" in artifacts:
        progress("bass_bleed", "Separating tracks")
        bass_bleed_diag = analyze_bass_bleed(artifacts["guitar"])
        if bass_bleed_diag.flagged and cfg.bass_bleed_mitigation:
            mitigated_path = apply_bass_bleed_mitigation(artifacts["guitar"], artifacts["guitar"])
            artifacts["guitar"] = mitigated_path
            bass_bleed_diag = BassBleedDiagnostics(
                attempted=bass_bleed_diag.attempted,
                flagged=bass_bleed_diag.flagged,
                reason=bass_bleed_diag.reason
                + f"; partial high-pass mitigation applied (<{BASS_BLEED_HPF_CUTOFF_HZ:.0f} Hz "
                "attenuated — does not remove bleed above the cutoff or fix the label)",
                low_band_energy_share=bass_bleed_diag.low_band_energy_share,
                low_band_cutoff_hz=bass_bleed_diag.low_band_cutoff_hz,
                mitigation_applied=True,
            )
        bass_bleed_path = bass_bleed_diag.write_json(out_dir / "bass_bleed_diagnostics.json")
        artifacts["bass_bleed_diagnostics"] = bass_bleed_path
        logger.debug("bass bleed: %s", bass_bleed_diag.reason)
        if cfg.sub_bass_debleed or cfg.low_end_restore_db:
            recovery_diag = apply_guitar_low_end_recovery(
                artifacts["guitar"],
                artifacts["guitar"],
                bass_path=bass_path_for_recovery,
                drums_path=drums_path_for_recovery,
                sub_bass_debleed=cfg.sub_bass_debleed,
                boost_db=cfg.low_end_restore_db,
                bass_missing_reason=(
                    "bass stem dropped by emit policy"
                    if bass_dropped_by_emit
                    else ""
                ),
            )
            recovery_path = recovery_diag.write_json(
                out_dir / "low_end_recovery_diagnostics.json"
            )
            artifacts["low_end_recovery_diagnostics"] = recovery_path
            if recovery_diag.harmonic_restore_applied:
                restore_only = LowEndRestoreDiagnostics(
                    attempted=True,
                    reason=recovery_diag.reason,
                    boost_db=recovery_diag.harmonic_restore_db,
                    pre_low_band_energy_share=recovery_diag.pre_low_band_energy_share,
                    post_low_band_energy_share=recovery_diag.post_low_band_energy_share,
                )
                restore_only.write_json(out_dir / "low_end_restore_diagnostics.json")
                artifacts["low_end_restore_diagnostics"] = out_dir / "low_end_restore_diagnostics.json"
            logger.debug("low-end recovery: %s", recovery_diag.reason)
        competing = {
            name: artifacts[name]
            for name in ("bass", "piano", "drums", "vocals")
            if name in artifacts
        }
        quality_diag = analyze_guitar_stem_quality(
            artifacts["guitar"],
            competing_stems=competing or None,
        )
        quality_path = quality_diag.write_json(out_dir / "guitar_stem_quality.json")
        artifacts["guitar_stem_quality_diagnostics"] = quality_path
        logger.debug("guitar stem quality: %s", quality_diag.reason)
        progress("bass_bleed", "Separating tracks")

    lead_rhythm_diag = None
    # Default isolate path: one combined Guitar stem. Lead/Rhythm only when
    # explicitly opted in (CLI/eval via lead_rhythm / dual_guitar).
    run_lead_rhythm = bool(cfg.lead_rhythm or cfg.dual_guitar)
    if "guitar" not in artifacts:
        if run_lead_rhythm:
            logger.debug("Lead/Rhythm skipped — no guitar stem")
            progress("guitar_split", "Separating tracks")
    elif run_lead_rhythm:
        from audio_to_tab.lead_rhythm import split_lead_rhythm_guitar

        emit_mode = resolve_lead_rhythm_mode(
            cfg.lead_rhythm_mode,
            lead_rhythm=cfg.lead_rhythm,
        )
        progress("guitar_split", "Separating tracks")
        extra, lead_rhythm_diag = split_lead_rhythm_guitar(
            artifacts["guitar"],
            out_dir,
            thresholds=cfg.lead_rhythm_thresholds,
            emit_mode=emit_mode,
        )
        artifacts.update(extra)
        logger.debug("lead/rhythm: %s", lead_rhythm_diag.reason)
        progress("guitar_split", "Separating tracks")

    progress("presence", "Finishing")
    stem_wav_paths = {
        name: path
        for name, path in artifacts.items()
        if not name.endswith("_diagnostics") and Path(path).suffix.lower() == ".wav"
    }
    presence = detect_present_stems(stem_wav_paths)
    lr_emitted = "lead_guitar" in artifacts and "rhythm_guitar" in artifacts
    if (
        lead_rhythm_diag is not None
        and lead_rhythm_diag.outcome == "lead_rhythm"
        and lr_emitted
        and "guitar" in presence
    ):
        guitar_presence = presence["guitar"]
        presence["guitar"] = StemPresence(
            present=False,
            confidence=guitar_presence.confidence,
            mean_dbfs=guitar_presence.mean_dbfs,
            energy_share=guitar_presence.energy_share,
            reason="superseded by lead_guitar/rhythm_guitar",
        )
        for derived in ("lead_guitar", "rhythm_guitar"):
            if derived in presence:
                sp = presence[derived]
                presence[derived] = StemPresence(
                    present=True,
                    confidence=max(sp.confidence, 0.6),
                    mean_dbfs=sp.mean_dbfs,
                    energy_share=sp.energy_share,
                    reason=(
                        "Lead/Rhythm split emitted from guitar stem"
                        + (
                            " (best-effort, low confidence)"
                            if lead_rhythm_diag.low_confidence
                            else ""
                        )
                    ),
                )

    presence_path = out_dir / "stem_presence.json"
    presence_path.write_text(
        json.dumps({name: sp.to_dict() for name, sp in presence.items()}, indent=2),
        encoding="utf-8",
    )
    artifacts["stem_presence_diagnostics"] = presence_path

    if cfg.emit_metronome:
        from audio_to_tab.metronome import (
            attach_metronome_artifact,
            coerce_metronome_render_options,
        )

        fallback = Path(trimmed) if Path(trimmed).is_file() else None
        if fallback is None and src.is_file():
            fallback = src
        try:
            attach_metronome_artifact(
                artifacts,
                out_dir,
                fallback=fallback,
                render=coerce_metronome_render_options(
                    accent=cfg.metronome_accent,
                    rate=cfg.metronome_rate,
                    sound=cfg.metronome_sound,
                ),
            )
        except Exception:
            logger.debug("metronome stem skipped", exc_info=True)

    attach_guitar_refine_backups(artifacts)
    progress("done", "Finish")
    return artifacts

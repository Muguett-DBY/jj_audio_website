"""Lead vs Rhythm guitar post-process for Demucs guitar stems.

Spatial + register (STFT band) candidates, then Basic Pitch role features with a
confidence gate. HPSS spectral pairs are kept for diagnostics only and never
emitted as Lead/Rhythm. Emits lead_guitar / rhythm_guitar only when confident;
always keeps the original guitar stem when attempted.

Root cause (2026-07-17): most real two-guitar mixes are not hard-panned to
opposite channels (that is a stylistic choice, not the norm), yet
``build_spatial_pair``'s admission gate was a single raw L/R sample-correlation
ceiling (``SPATIAL_CORR_MAX``). Raw correlation of a stereo mix of two
*independent* sources panned with coefficients (p, 1-p) is ``2p(1-p) /
(p^2 + (1-p)^2)`` — it rises toward 1.0 as panning approaches center
*regardless of how different the underlying content is* (e.g. two fully
independent tones panned 55/45 already correlate ~0.98). So the old gate
conflated "one guitar duplicated identically into both channels" (corr -> 1.0,
truly inseparable) with "two genuinely distinct guitars mixed with mild/no hard
panning" (corr can also land at 0.92-0.99, but the channels are NOT identical).
Everything short of hard panning was rejected before role classification ever
ran, matching the eval README's documented ambiguous cases.

Published solo/accompaniment guitar research (Foulon et al. 2013, "Automatic
Classification of Guitar Playing Modes"; Pati & Lerch 2017, "A Dataset and
Method for Electric Guitar Solo Detection in Rock Music") shows that
predominant-pitch strength/confidence, computed directly on a signal, reliably
discriminates a foregrounded melodic (lead) voice from denser/chordal
(rhythm) material *without first splitting the audio into two streams* via
panning or spectral means. That evidence is independent of how the mix is
panned, so it can rescue exactly the high-correlation-but-genuinely-distinct
case above: ``build_spatial_pair`` now allows a higher correlation ceiling
(``SPATIAL_CORR_RELAXED_MAX``) only when each channel's dominant spectral peak
("predominant pitch" proxy; no split required) diverges by at least
``PITCH_DIVERGENCE_SEMITONES_MIN`` semitones and at least one channel has a
clear-enough peak (``PITCH_CONFIDENCE_MIN``). Identical/near-identical channels
(mono duplicated into both) always show zero divergence, so they remain
rejected — the existing "prefer ambiguous over wrong" guarantee is unchanged:
this only widens *evidence* for admitting a candidate pair, the separability
floor and role-margin gates downstream still decide whether to actually emit.
No new dependency was added: the peak-picking helper is plain FFT (same style
as ``score_candidate_pair`` / ``_spectral_centroid_hz`` already in this file).
True same-register, fully-simultaneous, dead-centered (identical L=R) overlap
of two guitars remains architecturally unsolvable without a trained source
separator — that case correctly stays ``ambiguous`` (out of scope here; adding
such a model was explicitly excluded).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

Outcome = Literal["lead_rhythm", "ambiguous", "skipped"]
Method = Literal["spatial", "midside", "register", "spectral", "none"]
EmitMode = Literal["confident", "best_effort"]


def resolve_lead_rhythm_mode(
    lead_rhythm_mode: str | None = None,
    *,
    lead_rhythm: bool = False,
) -> EmitMode:
    """Map API/UI flags to emit policy. Default is confident-only emit."""
    if lead_rhythm_mode is not None and str(lead_rhythm_mode).strip():
        mode = str(lead_rhythm_mode).strip().lower()
        if mode not in ("confident", "best_effort"):
            raise ValueError(
                f"lead_rhythm_mode must be 'confident' or 'best_effort', got {lead_rhythm_mode!r}"
            )
        return mode  # type: ignore[return-value]
    if lead_rhythm:
        return "best_effort"
    return "confident"

SEPARABILITY_FLOOR = 0.22
ROLE_MARGIN_MIN = 0.18
# HPSS harmonic/residual is not musically Lead/Rhythm; spectral pairs are
# diagnostics-only (never emitted) unless allow_spectral_emit is explicitly on.
SPECTRAL_SEPARABILITY_FLOOR = 0.55
SPECTRAL_ROLE_MARGIN_MIN = 0.28
REGISTER_SEPARABILITY_FLOOR = 0.28
MIDSIDE_SEPARABILITY_FLOOR = 0.22
ANALYZE_WINDOW_SEC = 90.0
SPATIAL_BALANCE_MIN = 0.12
SPATIAL_CORR_MAX = 0.92
REGISTER_CROSSOVER_HZ = 400.0
# Side channel must carry some energy vs Mid (rejects true mono / L≡R).
MIDSIDE_SIDE_RATIO_MIN = 0.04

# Pitch-divergence evidence (spatial-split-independent) that can rescue a
# mildly-panned-but-genuinely-distinct pair above SPATIAL_CORR_MAX — see the
# module docstring "Root cause" note. Never overrides SPATIAL_CORR_RELAXED_MAX,
# so identical/near-identical channels (mono duplicated) are still rejected.
SPATIAL_CORR_RELAXED_MAX = 0.985
PITCH_DIVERGENCE_SEMITONES_MIN = 3.0
PITCH_CONFIDENCE_MIN = 0.08
PITCH_PROXY_FMIN_HZ = 70.0
PITCH_PROXY_FMAX_HZ = 1500.0


@dataclass
class LeadRhythmThresholds:
    """Tunable gates for eval/CLI ablation (defaults match module constants)."""

    separability_floor: float = SEPARABILITY_FLOOR
    role_margin_min: float = ROLE_MARGIN_MIN
    spectral_separability_floor: float = SPECTRAL_SEPARABILITY_FLOOR
    spectral_role_margin_min: float = SPECTRAL_ROLE_MARGIN_MIN
    register_separability_floor: float = REGISTER_SEPARABILITY_FLOOR
    midside_separability_floor: float = MIDSIDE_SEPARABILITY_FLOOR
    midside_side_ratio_min: float = MIDSIDE_SIDE_RATIO_MIN
    analyze_window_sec: float = ANALYZE_WINDOW_SEC
    # Spectral emit quarantined by default (HPSS ≠ Lead/Rhythm).
    allow_spectral_emit: bool = False
    spatial_balance_min: float = SPATIAL_BALANCE_MIN
    spatial_corr_max: float = SPATIAL_CORR_MAX
    spatial_corr_relaxed_max: float = SPATIAL_CORR_RELAXED_MAX
    pitch_divergence_semitones_min: float = PITCH_DIVERGENCE_SEMITONES_MIN
    pitch_confidence_min: float = PITCH_CONFIDENCE_MIN

    @classmethod
    def from_env(cls, base: LeadRhythmThresholds | None = None) -> LeadRhythmThresholds:
        """Overlay ATT_LR_* environment variables onto defaults or ``base``."""
        t = base or cls()
        mapping = {
            "ATT_LR_SEPARABILITY_FLOOR": ("separability_floor", float),
            "ATT_LR_ROLE_MARGIN_MIN": ("role_margin_min", float),
            "ATT_LR_SPECTRAL_SEPARABILITY_FLOOR": ("spectral_separability_floor", float),
            "ATT_LR_SPECTRAL_ROLE_MARGIN_MIN": ("spectral_role_margin_min", float),
            "ATT_LR_REGISTER_SEPARABILITY_FLOOR": ("register_separability_floor", float),
            "ATT_LR_MIDSIDE_SEPARABILITY_FLOOR": ("midside_separability_floor", float),
            "ATT_LR_MIDSIDE_SIDE_RATIO_MIN": ("midside_side_ratio_min", float),
            "ATT_LR_ANALYZE_WINDOW_SEC": ("analyze_window_sec", float),
            "ATT_LR_ALLOW_SPECTRAL_EMIT": ("allow_spectral_emit", _env_bool),
            "ATT_LR_SPATIAL_CORR_RELAXED_MAX": ("spatial_corr_relaxed_max", float),
            "ATT_LR_PITCH_DIVERGENCE_SEMITONES_MIN": (
                "pitch_divergence_semitones_min",
                float,
            ),
            "ATT_LR_PITCH_CONFIDENCE_MIN": ("pitch_confidence_min", float),
        }
        updates: dict[str, Any] = {}
        for env_key, (field_name, caster) in mapping.items():
            raw = os.environ.get(env_key)
            if raw is None or raw == "":
                continue
            updates[field_name] = caster(raw)
        return replace(t, **updates) if updates else t


def _env_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LeadRhythmDiagnostics:
    """Diagnostics for an attempted lead/rhythm split."""

    outcome: Outcome
    reason: str
    attempted: bool = True
    separability_score: float | None = None
    role_confidence: float | None = None
    method: Method = "none"
    correlation: float | None = None
    balance_ratio: float | None = None
    # True when Lead/Rhythm WAVs were written despite failed gates (best-effort).
    forced_emit: bool = False
    low_confidence: bool = False
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@dataclass
class _CandidatePair:
    method: Method
    a: np.ndarray  # mono float32
    b: np.ndarray
    sr: int
    separability: float
    correlation: float | None = None
    balance_ratio: float | None = None
    detail: str = ""
    pitch_evidence: dict[str, float] | None = None


def _load_stereo(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return data.astype(np.float32), int(sr)


def _mono_to_stereo(mono: np.ndarray) -> np.ndarray:
    m = np.asarray(mono, dtype=np.float32).reshape(-1)
    return np.column_stack([m, m])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _dominant_pitch_estimate(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Cheap "predominant pitch + confidence" proxy computed directly on one
    channel — no spatial/spectral split required (see module docstring).

    Approximates the predominant-pitch features used by solo/accompaniment
    guitar classifiers (Pati & Lerch 2017; Foulon et al. 2013): the strongest
    spectral peak in the guitar-relevant band stands in for predominant pitch,
    and its share of total spectral magnitude stands in for confidence (a
    clear, foregrounded single-note voice concentrates energy into one peak;
    denser/chordal or noisy content spreads it out). Plain FFT peak-picking —
    same style as ``score_candidate_pair`` / ``_spectral_centroid_hz`` already
    in this module — so it adds no new dependency and works even without
    librosa.

    Returns (peak_freq_hz, confidence) or (0.0, 0.0) if the signal is too
    short/quiet to estimate.
    """
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    win = min(len(x), sr * 4)
    if win < 256:
        return 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(x[:win] * np.hanning(win)))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = (freqs >= PITCH_PROXY_FMIN_HZ) & (freqs <= PITCH_PROXY_FMAX_HZ)
    total = float(np.sum(spectrum)) + 1e-12
    if not np.any(band) or total <= 1e-9:
        return 0.0, 0.0
    band_spectrum = spectrum[band]
    band_freqs = freqs[band]
    idx = int(np.argmax(band_spectrum))
    return float(band_freqs[idx]), float(band_spectrum[idx] / total)


def _pitch_divergence_semitones(freq_a: float, freq_b: float) -> float:
    if freq_a <= 0.0 or freq_b <= 0.0:
        return 0.0
    return float(abs(12.0 * np.log2(freq_a / freq_b)))


def _harmonic_pitch_pair(
    mono: np.ndarray, sr: int, center_a_hz: float, center_b_hz: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Split a mono signal into two streams via soft log-frequency masks centered
    on two already-detected predominant pitches (from ``_dominant_pitch_estimate``).

    Used only for the mild-pan pitch-divergence-evidence path in
    ``build_spatial_pair``: at near-center panning the per-sample L/R dominance
    heuristic used for hard-panned content degenerates (both channels carry
    comparable amounts of both sources), but the two predominant pitches are
    still known, so a pitch-informed mask separates them far better than
    amplitude alone. Same STFT/soft-mask machinery as ``build_register_pair``,
    just centered on detected pitches instead of a fixed crossover — no new
    dependency. Returns None if librosa is unavailable.
    """
    try:
        import librosa
    except ImportError:
        return None
    if center_a_hz <= 0.0 or center_b_hz <= 0.0:
        return None
    n_fft = 2048
    hop = 512
    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def _log_gaussian_mask(center_hz: float, bandwidth_semitones: float = 5.0) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            semitones = 12.0 * np.log2(np.maximum(freqs, 1e-6) / center_hz)
        return np.exp(-0.5 * np.square(semitones / bandwidth_semitones))

    mask_a = _log_gaussian_mask(center_a_hz)
    mask_b = _log_gaussian_mask(center_b_hz)
    total = mask_a + mask_b + 1e-9
    a = librosa.istft((stft * (mask_a / total)[:, None]), hop_length=hop, length=len(mono))
    b = librosa.istft((stft * (mask_b / total)[:, None]), hop_length=hop, length=len(mono))
    return a.astype(np.float32), b.astype(np.float32)


def build_spatial_pair(
    guitar_path: Path,
    *,
    thresholds: LeadRhythmThresholds | None = None,
) -> _CandidatePair | None:
    """L/R dominance pair when stereo gate passes (evolved dual-guitar heuristic)."""
    thr = thresholds or LeadRhythmThresholds()
    if not guitar_path.exists():
        return None
    data, sr = _load_stereo(guitar_path)
    if data.shape[1] < 2 or len(data) < 1024:
        return None

    left = data[:, 0]
    right = data[:, 1]
    rms_l = _rms(left)
    rms_r = _rms(right)
    max_rms = max(rms_l, rms_r, 1e-9)
    balance = min(rms_l, rms_r) / max_rms
    if balance < thr.spatial_balance_min:
        return None
    if np.std(left) < 1e-6 or np.std(right) < 1e-6:
        return None

    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return None

    pitch_evidence: dict[str, float] | None = None
    detail = "stereo L/R dominance"
    if corr > thr.spatial_corr_max:
        # Raw correlation alone conflates "one guitar duplicated identically
        # into both channels" (corr -> 1.0) with "two genuinely distinct
        # guitars mixed with mild/no hard panning" (also corr ~0.92-0.99, but
        # NOT identical content). See module docstring "Root cause". Only look
        # for spatial-independent pitch-divergence evidence to rescue the
        # latter — never for correlations near-indistinguishable from a true
        # duplicate.
        if corr > thr.spatial_corr_relaxed_max:
            return None
        freq_l, conf_l = _dominant_pitch_estimate(left, sr)
        freq_r, conf_r = _dominant_pitch_estimate(right, sr)
        divergence = _pitch_divergence_semitones(freq_l, freq_r)
        if (
            divergence < thr.pitch_divergence_semitones_min
            or max(conf_l, conf_r) < thr.pitch_confidence_min
        ):
            return None
        pitch_evidence = {
            "left_hz": freq_l,
            "right_hz": freq_r,
            "divergence_semitones": divergence,
            "left_confidence": conf_l,
            "right_confidence": conf_r,
        }
        detail = (
            "stereo L/R dominance (mild pan; admitted via pitch-divergence "
            f"evidence, Δ={divergence:.1f} semitones)"
        )

    if pitch_evidence is not None:
        # Near-center panning: the per-sample L/R dominance heuristic below
        # degenerates because both channels carry comparable amounts of both
        # sources (that heuristic assumes one channel is ~mostly one source).
        # Use the two already-detected predominant pitches to separate by
        # frequency content instead, which remains meaningful at any pan.
        harmonic = _harmonic_pitch_pair(
            (left + right) * 0.5, sr, pitch_evidence["left_hz"], pitch_evidence["right_hz"]
        )
    else:
        harmonic = None

    if harmonic is not None:
        a, b = harmonic
    else:
        a = np.where(np.abs(left) >= np.abs(right), left, left - 0.5 * right).astype(np.float32)
        b = np.where(np.abs(right) >= np.abs(left), right, right - 0.5 * left).astype(np.float32)
    sep = score_candidate_pair(a, b, sr)
    # Boost spatial score slightly when corr is low (more independent channels).
    sep = min(1.0, sep + max(0.0, (thr.spatial_corr_max - corr) * 0.15))
    return _CandidatePair(
        method="spatial",
        a=a,
        b=b,
        sr=sr,
        separability=sep,
        correlation=corr,
        balance_ratio=balance,
        detail=detail,
        pitch_evidence=pitch_evidence,
    )


def build_midside_pair(
    guitar_path: Path,
    *,
    thresholds: LeadRhythmThresholds | None = None,
    relax_side_floor: bool = False,
) -> _CandidatePair | None:
    """
    Mid/Side pair for centered lead + wide-panned rhythm (common mix layout).

    Mid = (L+R)/2, Side = (L-R)/2. Rejects true mono (L≡R) unless
    ``relax_side_floor`` (last-resort forced emit only).
    """
    thr = thresholds or LeadRhythmThresholds()
    if not guitar_path.exists():
        return None
    data, sr = _load_stereo(guitar_path)
    if data.shape[1] < 2 or len(data) < 1024:
        return None

    left = data[:, 0].astype(np.float32)
    right = data[:, 1].astype(np.float32)
    mid = ((left + right) * 0.5).astype(np.float32)
    side = ((left - right) * 0.5).astype(np.float32)
    rms_mid = _rms(mid)
    rms_side = _rms(side)
    if rms_mid < 1e-6:
        return None
    side_ratio = rms_side / max(rms_mid, 1e-9)
    if not relax_side_floor and side_ratio < thr.midside_side_ratio_min:
        return None
    if rms_side < 1e-6 and not relax_side_floor:
        return None

    # If Side is near-silent (forced mono path), synthesize a quiet complementary
    # stream so we still have two WAVs — labels will be low-confidence.
    if rms_side < 1e-6:
        side = (mid * 0.15).astype(np.float32)

    sep = score_candidate_pair(mid, side, sr)
    return _CandidatePair(
        method="midside",
        a=mid,
        b=side,
        sr=sr,
        separability=sep,
        balance_ratio=side_ratio,
        detail=f"Mid/Side split (side/mid RMS ratio={side_ratio:.3f})",
    )


def build_register_pair(
    guitar_path: Path,
    *,
    thresholds: LeadRhythmThresholds | None = None,
) -> _CandidatePair | None:
    """
    Soft-split mono guitar into low vs high register via STFT band masks.

    Near-term candidate when spatial evidence is absent: aimed at register-
    separated lead/rhythm without claiming centered dual-source separation.
    """
    thr = thresholds or LeadRhythmThresholds()
    if not guitar_path.exists():
        return None
    try:
        import librosa
    except ImportError:
        return None

    data, sr = _load_stereo(guitar_path)
    if len(data) < 1024:
        return None
    mono = data.mean(axis=1).astype(np.float32)
    max_samples = int(thr.analyze_window_sec * sr)
    mono_work = mono[:max_samples] if len(mono) > max_samples else mono

    n_fft = 2048
    hop = 512
    stft = librosa.stft(mono_work, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_w = 1.0 / (1.0 + np.exp((freqs - REGISTER_CROSSOVER_HZ) / 40.0))
    high_w = 1.0 - low_w
    low = librosa.istft(stft * low_w[:, None], hop_length=hop, length=len(mono_work))
    high = librosa.istft(stft * high_w[:, None], hop_length=hop, length=len(mono_work))
    low = low.astype(np.float32)
    high = high.astype(np.float32)

    # Pad tails with silence (not duplicated mono) so length matches source.
    if len(mono) > len(mono_work):
        pad = len(mono) - len(mono_work)
        low = np.concatenate([low, np.zeros(pad, dtype=np.float32)])
        high = np.concatenate([high, np.zeros(pad, dtype=np.float32)])

    if _rms(low) < 1e-5 or _rms(high) < 1e-5:
        return None
    sep = score_candidate_pair(high, low, sr)
    return _CandidatePair(
        method="register",
        a=high,
        b=low,
        sr=sr,
        separability=sep,
        detail=f"STFT register split @{REGISTER_CROSSOVER_HZ:.0f}Hz",
    )


def build_spectral_pair(
    guitar_path: Path,
    *,
    thresholds: LeadRhythmThresholds | None = None,
) -> _CandidatePair | None:
    """HPSS harmonic vs residual (diagnostics / research only; not musical Lead/Rhythm)."""
    thr = thresholds or LeadRhythmThresholds()
    if not guitar_path.exists():
        return None
    try:
        import librosa
    except ImportError:
        return None

    data, sr = _load_stereo(guitar_path)
    if len(data) < 1024:
        return None
    mono = data.mean(axis=1).astype(np.float32)
    max_samples = int(thr.analyze_window_sec * sr)
    mono_work = mono[:max_samples] if len(mono) > max_samples else mono
    harm, perc = librosa.effects.hpss(mono_work)
    harm = harm.astype(np.float32)
    perc = perc.astype(np.float32)
    # Do not pad identical mono into both streams (legacy bug). Silence-pad only.
    if len(mono) > max_samples:
        pad = len(mono) - max_samples
        harm = np.concatenate([harm, np.zeros(pad, dtype=np.float32)])
        perc = np.concatenate([perc, np.zeros(pad, dtype=np.float32)])

    if _rms(harm) < 1e-5 or _rms(perc) < 1e-5:
        return None
    sep = score_candidate_pair(harm, perc, sr)
    return _CandidatePair(
        method="spectral",
        a=harm,
        b=perc,
        sr=sr,
        separability=sep,
        detail="HPSS harmonic vs residual (quarantined emit)",
    )


def score_candidate_pair(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Cheap separability in [0, 1]: residual energy, centroid gap, onset-rate gap."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(a), len(b))
    if n < 512:
        return 0.0
    a = a[:n]
    b = b[:n]
    ra, rb = _rms(a), _rms(b)
    if ra < 1e-6 or rb < 1e-6:
        return 0.0

    proj = (np.dot(a, b) / (np.dot(b, b) + 1e-12)) * b
    residual = a - proj
    residual_ratio = min(1.0, _rms(residual) / (ra + 1e-12))

    win = min(n, sr * 4)
    aa = np.abs(np.fft.rfft(a[:win] * np.hanning(win)))
    bb = np.abs(np.fft.rfft(b[:win] * np.hanning(win)))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    ca = float(np.sum(freqs * aa) / (np.sum(aa) + 1e-12))
    cb = float(np.sum(freqs * bb) / (np.sum(bb) + 1e-12))
    centroid_gap = min(1.0, abs(ca - cb) / 2000.0)

    hop = max(1, sr // 100)
    ea = np.array([_rms(a[i : i + hop]) for i in range(0, n - hop, hop)])
    eb = np.array([_rms(b[i : i + hop]) for i in range(0, n - hop, hop)])
    flux_a = float(np.mean(np.maximum(0.0, np.diff(ea, prepend=ea[:1]))))
    flux_b = float(np.mean(np.maximum(0.0, np.diff(eb, prepend=eb[:1]))))
    onset_gap = min(1.0, abs(flux_a - flux_b) / (max(flux_a, flux_b, 1e-9)))

    score = 0.45 * residual_ratio + 0.35 * centroid_gap + 0.20 * onset_gap
    return float(max(0.0, min(1.0, score)))


@dataclass
class RoleFeatures:
    pitch_median: float
    pitch_iqr: float
    note_density: float
    polyphony: float
    mean_velocity: float
    sustain: float

    def lead_score(self) -> float:
        # Higher pitch, lower polyphony, sparser notes, longer sustain, brighter velocity.
        pitch_n = min(1.0, max(0.0, (self.pitch_median - 50.0) / 30.0))
        poly_n = 1.0 - min(1.0, self.polyphony / 3.0)
        dens_n = 1.0 - min(1.0, self.note_density / 8.0)
        iqr_n = min(1.0, self.pitch_iqr / 18.0)  # melodic range
        sust_n = min(1.0, self.sustain / 0.6)
        vel_n = min(1.0, max(0.0, (self.mean_velocity - 40.0) / 60.0))
        return (
            0.35 * pitch_n
            + 0.20 * poly_n
            + 0.15 * dens_n
            + 0.15 * iqr_n
            + 0.10 * sust_n
            + 0.05 * vel_n
        )

    def rhythm_score(self) -> float:
        pitch_n = 1.0 - min(1.0, max(0.0, (self.pitch_median - 45.0) / 30.0))
        poly_n = min(1.0, self.polyphony / 3.0)
        dens_n = min(1.0, self.note_density / 8.0)
        # Tight pitch IQR and shorter notes → chordal rhythm-like.
        iqr_n = 1.0 - min(1.0, self.pitch_iqr / 18.0)
        sust_n = 1.0 - min(1.0, self.sustain / 0.6)
        return 0.25 * pitch_n + 0.30 * poly_n + 0.25 * dens_n + 0.10 * iqr_n + 0.10 * sust_n

    def to_dict(self) -> dict[str, float]:
        return {
            "pitch_median": self.pitch_median,
            "pitch_iqr": self.pitch_iqr,
            "note_density": self.note_density,
            "polyphony": self.polyphony,
            "mean_velocity": self.mean_velocity,
            "sustain": self.sustain,
        }


def extract_role_features_from_notes(
    notes: list[Any],
    duration_sec: float,
) -> RoleFeatures:
    """Feature extraction from pretty_midi-like notes (testable without Basic Pitch)."""
    if not notes or duration_sec <= 0:
        return RoleFeatures(60.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    velocities = np.array([n.velocity for n in notes], dtype=np.float64)
    durs = np.array([max(0.0, n.end - n.start) for n in notes], dtype=np.float64)
    starts = np.array([n.start for n in notes], dtype=np.float64)

    overlaps = []
    for s in starts:
        overlaps.append(sum(1 for n in notes if n.start <= s < n.end))
    poly = float(np.mean(overlaps)) if overlaps else 0.0

    return RoleFeatures(
        pitch_median=float(np.median(pitches)),
        pitch_iqr=float(np.subtract(*np.percentile(pitches, [75, 25]))),
        note_density=float(len(notes) / duration_sec),
        polyphony=poly,
        mean_velocity=float(np.mean(velocities)),
        sustain=float(np.mean(durs)),
    )


def _average_role_features(feats: list[RoleFeatures]) -> RoleFeatures:
    if not feats:
        return RoleFeatures(60.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = float(len(feats))
    return RoleFeatures(
        pitch_median=sum(f.pitch_median for f in feats) / n,
        pitch_iqr=sum(f.pitch_iqr for f in feats) / n,
        note_density=sum(f.note_density for f in feats) / n,
        polyphony=sum(f.polyphony for f in feats) / n,
        mean_velocity=sum(f.mean_velocity for f in feats) / n,
        sustain=sum(f.sustain for f in feats) / n,
    )


def extract_role_features(
    audio_mono: np.ndarray,
    sr: int,
    *,
    thresholds: LeadRhythmThresholds | None = None,
) -> RoleFeatures:
    """Run Basic Pitch on mono (first + last analyze windows) and extract features."""
    thr = thresholds or LeadRhythmThresholds()
    import pretty_midi
    from basic_pitch.inference import predict

    from audio_to_tab.transcribe import basic_pitch_model_path

    window = int(thr.analyze_window_sec * sr)
    n = len(audio_mono)
    if n < sr // 4:
        return RoleFeatures(60.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    starts = [0]
    if n > window + sr:
        starts.append(max(0, n - window))

    feats: list[RoleFeatures] = []
    model_path = basic_pitch_model_path()
    for start in starts:
        clip = audio_mono[start : start + window].astype(np.float32)
        if len(clip) < sr // 4:
            continue
        stereo = _mono_to_stereo(clip)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            sf.write(str(wav_path), stereo, sr, subtype="PCM_16")
            _, midi_data, _ = predict(
                str(wav_path),
                model_or_model_path=model_path,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=58.0,
            )
            mid_path = Path(tmp) / "clip.mid"
            midi_data.write(str(mid_path))
            pm = pretty_midi.PrettyMIDI(str(mid_path))
            notes = [note for inst in pm.instruments for note in inst.notes]
            feats.append(extract_role_features_from_notes(notes, duration_sec=len(clip) / sr))
    return _average_role_features(feats)


def classify_lead_rhythm_roles(
    feat_a: RoleFeatures,
    feat_b: RoleFeatures,
    *,
    margin_min: float | None = None,
) -> tuple[Literal["a_lead", "b_lead", "ambiguous"], float, dict[str, Any]]:
    """
    Assign which stream is lead vs rhythm.

    Returns (assignment, confidence_margin, detail).
    """
    floor = ROLE_MARGIN_MIN if margin_min is None else float(margin_min)
    s1 = 0.5 * (feat_a.lead_score() + feat_b.rhythm_score())
    s2 = 0.5 * (feat_b.lead_score() + feat_a.rhythm_score())
    margin = abs(s1 - s2)
    detail = {
        "stream_a": feat_a.to_dict(),
        "stream_b": feat_b.to_dict(),
        "score_a_lead": s1,
        "score_b_lead": s2,
        "margin": margin,
        "margin_min": floor,
    }
    if margin < floor:
        return "ambiguous", float(margin), detail
    if s1 >= s2:
        return "a_lead", float(margin), detail
    return "b_lead", float(margin), detail


def split_lead_rhythm_guitar(
    guitar_path: Path,
    output_dir: Path,
    *,
    use_basic_pitch: bool = True,
    thresholds: LeadRhythmThresholds | None = None,
    emit_mode: EmitMode = "confident",
) -> tuple[dict[str, Path], LeadRhythmDiagnostics]:
    """
    Attempt lead/rhythm split of a Demucs guitar stem.

    Writes ``guitar_split_diagnostics.json`` whenever a guitar stem exists.
    Emits ``lead_guitar.wav`` + ``rhythm_guitar.wav`` when separability and role
    gates pass, or when ``emit_mode`` is ``best_effort`` (legacy always-emit).
    In ``confident`` mode, low-confidence / forced splits write diagnostics only
    with ``outcome=ambiguous``. Spectral candidates never emit unless
    ``thresholds.allow_spectral_emit``.
    """
    thr = LeadRhythmThresholds.from_env(thresholds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "guitar_split_diagnostics.json"

    if not guitar_path.exists():
        diag = LeadRhythmDiagnostics(
            outcome="skipped",
            reason="guitar stem missing",
            attempted=False,
        )
        diag.write_json(diag_path)
        return {"guitar_split_diagnostics": diag_path}, diag

    candidates: list[_CandidatePair] = []
    spatial = build_spatial_pair(guitar_path, thresholds=thr)
    if spatial is not None:
        candidates.append(spatial)
    midside = build_midside_pair(guitar_path, thresholds=thr)
    if midside is not None:
        candidates.append(midside)
    register = build_register_pair(guitar_path, thresholds=thr)
    if register is not None:
        candidates.append(register)
    # Spectral kept for diagnostics / optional research emit only.
    spectral = build_spectral_pair(guitar_path, thresholds=thr)
    if spectral is not None:
        candidates.append(spectral)

    musical = [c for c in candidates if c.method != "spectral" or thr.allow_spectral_emit]
    if not musical:
        if emit_mode == "best_effort":
            # Last resort: force Mid/Side or register so WAVs always exist.
            forced_pair = build_midside_pair(guitar_path, thresholds=thr, relax_side_floor=True)
            if forced_pair is None:
                forced_pair = build_register_pair(guitar_path, thresholds=thr)
            if forced_pair is None:
                diag = LeadRhythmDiagnostics(
                    outcome="ambiguous",
                    reason=(
                        "no lead/rhythm candidate could be built "
                        "(guitar too short or unreadable)"
                    ),
                    method="none",
                    features={"emit_mode": emit_mode},
                )
                diag.write_json(diag_path)
                return {"guitar_split_diagnostics": diag_path}, diag
            musical = [forced_pair]
            candidates = [forced_pair]
        else:
            diag = LeadRhythmDiagnostics(
                outcome="ambiguous",
                reason=(
                    "no confident lead/rhythm candidate "
                    "(parts too similar, centered, or unreadable)"
                ),
                method="none",
                features={"emit_mode": emit_mode, "skipped_emit": True},
            )
            diag.write_json(diag_path)
            return {"guitar_split_diagnostics": diag_path}, diag

    best = _pick_best_candidate(musical, thr)
    if best.method == "spectral" and not thr.allow_spectral_emit:
        non_spectral = [c for c in musical if c.method != "spectral"]
        if non_spectral:
            best = _pick_best_candidate(non_spectral, thr)
        else:
            forced = build_midside_pair(guitar_path, thresholds=thr, relax_side_floor=True)
            if forced is None:
                forced = build_register_pair(guitar_path, thresholds=thr)
            if forced is None:
                diag = LeadRhythmDiagnostics(
                    outcome="ambiguous",
                    reason=(
                        "spectral (HPSS) candidate quarantined and no fallback pair"
                    ),
                    separability_score=best.separability,
                    method=best.method,
                    features={
                        "quarantined": True,
                        "detail": best.detail,
                        "emit_mode": emit_mode,
                        "skipped_emit": emit_mode == "confident",
                    },
                )
                diag.write_json(diag_path)
                return {"guitar_split_diagnostics": diag_path}, diag
            best = forced

    sep_floor = _separability_floor_for(best.method, thr)
    low_sep = best.separability < sep_floor

    if best.method == "spatial" and best.pitch_evidence is None:
        src_stereo, src_sr = _load_stereo(guitar_path)
        analyze_a, analyze_b = src_stereo[:, 0], src_stereo[:, 1]
        analyze_sr = src_sr
    else:
        analyze_a, analyze_b, analyze_sr = best.a, best.b, best.sr

    if use_basic_pitch:
        feat_a = extract_role_features(analyze_a, analyze_sr, thresholds=thr)
        feat_b = extract_role_features(analyze_b, analyze_sr, thresholds=thr)
    else:
        feat_a = _features_from_audio_stats(analyze_a, analyze_sr, thresholds=thr)
        feat_b = _features_from_audio_stats(analyze_b, analyze_sr, thresholds=thr)

    role_floor = _role_floor_for(best.method, thr)
    # Use a zero margin floor so we always get a directional assignment for emit;
    # then decide confidence against the real floor.
    assignment, margin, feat_detail = classify_lead_rhythm_roles(
        feat_a, feat_b, margin_min=0.0
    )
    if best.pitch_evidence is not None:
        feat_detail["pitch_evidence"] = best.pitch_evidence

    # If scores are identical (margin ~0), default stream A to lead. Stream A is
    # the primary slot in every pair builder — midside (mid → lead), register
    # (high → lead), spatial (left-channel source) — so a_lead is the intended
    # default for all methods here.
    if assignment == "ambiguous" or margin < 1e-9:
        assignment = "a_lead"
        feat_detail["defaulted_assignment"] = True

    low_role = margin < role_floor
    forced = low_sep or low_role
    if assignment == "a_lead":
        lead_audio, rhythm_audio = best.a, best.b
    else:
        lead_audio, rhythm_audio = best.b, best.a

    feat_detail["emit_mode"] = emit_mode
    should_emit = emit_mode == "best_effort" or not forced

    if forced:
        reasons = []
        if low_sep:
            reasons.append(
                f"separability {best.separability:.3f} below floor {sep_floor}"
            )
        if low_role:
            reasons.append(f"role margin {margin:.3f} below {role_floor}")
        gate_detail = " ; ".join(reasons)
        feat_detail["forced_emit"] = True
        feat_detail["low_confidence"] = True
    else:
        gate_detail = ""
        reason = f"split via {best.method}; role margin {margin:.3f}"

    if not should_emit:
        reason = (
            f"Lead/Rhythm not emitted (confident mode): {best.method} candidate "
            f"failed gates ({gate_detail}); combined Guitar kept"
        )
        feat_detail["skipped_emit"] = True
        diag = LeadRhythmDiagnostics(
            outcome="ambiguous",
            reason=reason,
            separability_score=best.separability,
            role_confidence=margin,
            method=best.method,
            correlation=best.correlation,
            balance_ratio=best.balance_ratio,
            forced_emit=True,
            low_confidence=True,
            features=feat_detail,
        )
        diag.write_json(diag_path)
        return {"guitar_split_diagnostics": diag_path}, diag

    lead_path = out_dir / "lead_guitar.wav"
    rhythm_path = out_dir / "rhythm_guitar.wav"
    sf.write(str(lead_path), _mono_to_stereo(lead_audio), best.sr, subtype="PCM_16")
    sf.write(str(rhythm_path), _mono_to_stereo(rhythm_audio), best.sr, subtype="PCM_16")

    if forced:
        reason = (
            f"best-effort split via {best.method} "
            f"({gate_detail}); labels may be inaccurate"
        )
    else:
        reason = f"split via {best.method}; role margin {margin:.3f}"

    diag = LeadRhythmDiagnostics(
        outcome="lead_rhythm",
        reason=reason,
        separability_score=best.separability,
        role_confidence=margin,
        method=best.method,
        correlation=best.correlation,
        balance_ratio=best.balance_ratio,
        forced_emit=forced,
        low_confidence=forced,
        features=feat_detail,
    )
    diag.write_json(diag_path)
    return {
        "lead_guitar": lead_path,
        "rhythm_guitar": rhythm_path,
        "guitar_split_diagnostics": diag_path,
    }, diag


def _separability_floor_for(method: Method, thr: LeadRhythmThresholds) -> float:
    if method == "spectral":
        return thr.spectral_separability_floor
    if method == "register":
        return thr.register_separability_floor
    if method == "midside":
        return thr.midside_separability_floor
    return thr.separability_floor


def _role_floor_for(method: Method, thr: LeadRhythmThresholds) -> float:
    if method == "spectral":
        return thr.spectral_role_margin_min
    return thr.role_margin_min


def _pick_best_candidate(
    candidates: list[_CandidatePair],
    thr: LeadRhythmThresholds,
) -> _CandidatePair:
    """Prefer hard-panned spatial; then midside; then register; spectral last."""
    spatial = [
        c
        for c in candidates
        if c.method == "spatial"
        and c.separability >= thr.separability_floor
        and c.correlation is not None
        and c.correlation < 0.35
    ]
    if spatial:
        return max(spatial, key=lambda c: c.separability)
    spatial_any = [c for c in candidates if c.method == "spatial"]
    if spatial_any:
        return max(spatial_any, key=lambda c: c.separability)
    midside = [c for c in candidates if c.method == "midside"]
    if midside:
        return max(midside, key=lambda c: c.separability)
    register = [c for c in candidates if c.method == "register"]
    if register:
        return max(register, key=lambda c: c.separability)
    return max(candidates, key=lambda c: c.separability)


def _spectral_centroid_hz(mono: np.ndarray, sr: int) -> float:
    win = min(len(mono), sr * 2)
    if win < 64:
        return 0.0
    x = mono[:win] * np.hanning(win)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    return float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))


def _features_from_audio_stats(
    mono: np.ndarray,
    sr: int,
    *,
    thresholds: LeadRhythmThresholds | None = None,
) -> RoleFeatures:
    """Map centroid/flux into RoleFeatures when Basic Pitch is skipped (tests/fallback)."""
    thr = thresholds or LeadRhythmThresholds()
    window = int(thr.analyze_window_sec * sr)
    n = len(mono)
    starts = [0]
    if n > window + sr:
        starts.append(max(0, n - window))

    feats: list[RoleFeatures] = []
    for start in starts:
        clip = mono[start : start + window]
        if len(clip) < 64:
            continue
        c = _spectral_centroid_hz(clip, sr)
        pitch = 12.0 * np.log2(max(c, 80.0) / 440.0) + 69.0
        hop = max(1, sr // 100)
        energy = np.array(
            [_rms(clip[i : i + hop]) for i in range(0, max(1, len(clip) - hop), hop)]
        )
        flux = (
            float(np.mean(np.maximum(0.0, np.diff(energy, prepend=energy[:1]))))
            if len(energy)
            else 0.0
        )
        dens = min(8.0, flux * 40.0)
        poly = 1.0 + min(2.0, dens / 4.0)
        feats.append(
            RoleFeatures(
                pitch_median=float(pitch),
                pitch_iqr=6.0,
                note_density=float(dens),
                polyphony=float(poly),
                mean_velocity=80.0,
                sustain=0.2,
            )
        )
    return _average_role_features(feats)

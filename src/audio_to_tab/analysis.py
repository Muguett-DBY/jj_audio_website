"""Versioned analysis JSON shared by the desktop player, worker, and tests.

The contract is deliberately boring: plain JSON, seconds on the processed-audio timeline, and
no numpy types. ``schema_version`` gates compatibility; a newer or unknown file is ignored so a
player update can never crash on an older/newer artifact.

Shape (``SCHEMA_VERSION = 1``)::

    {
      "schema_version": 1,
      "source_fingerprint": "sha256:...",
      "duration_sec": 123.4,
      "bpm": 92.5,
      "bpm_confidence": "high",
      "key": "E",
      "key_confidence": "medium",
      "beats": [0.5, 1.0],
      "downbeats": [0.5],
      "beats_per_measure": 4,
      "chords": [{"start": 0.5, "end": 1.5, "symbol": "E", "confidence": 0.8}],
      "sections": [{"start": 0.0, "end": 30.0, "label": "Intro", "confidence": 0.6}],
      "lyrics": [
        {"start": 10.0, "end": 12.0, "text": "hello world",
         "words": [{"start": 10.0, "end": 10.5, "text": "hello"}]}
      ],
      "lyrics_language": "en",
      "models": {"chords": "viterbi-1"},
      "warnings": ["chord labels are a draft"],
      "created_at": 1234567890.0
    }

Malformed individual events are dropped; the rest of the file still loads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from bisect import bisect_left
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ChordEvent:
    start: float
    end: float
    symbol: str
    confidence: float
    simple: str = ""
    medium: str = ""


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    label: str
    confidence: float


@dataclass(frozen=True)
class LyricWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class LyricLine:
    start: float
    end: float
    text: str
    words: tuple[LyricWord, ...]


@dataclass(frozen=True)
class AnalysisResult:
    schema_version: int = SCHEMA_VERSION
    source_fingerprint: str = ""
    duration_sec: float = 0.0
    bpm: float | None = None
    bpm_confidence: str = "low"
    key: str | None = None
    key_confidence: str = "low"
    beats: tuple[float, ...] = ()
    downbeats: tuple[float, ...] = ()
    beats_per_measure: int = 4
    chords: tuple[ChordEvent, ...] = ()
    sections: tuple[Section, ...] = ()
    lyrics: tuple[LyricLine, ...] = ()
    lyrics_language: str | None = None
    models: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_at: float = 0.0
    chart: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict (tuples become lists, no numpy scalars)."""
        return {
            "schema_version": int(self.schema_version),
            "source_fingerprint": str(self.source_fingerprint),
            "duration_sec": float(self.duration_sec),
            "bpm": None if self.bpm is None else float(self.bpm),
            "bpm_confidence": str(self.bpm_confidence),
            "key": None if self.key is None else str(self.key),
            "key_confidence": str(self.key_confidence),
            "beats": [float(t) for t in self.beats],
            "downbeats": [float(t) for t in self.downbeats],
            "beats_per_measure": int(self.beats_per_measure),
            "chords": [
                {
                    "start": float(c.start),
                    "end": float(c.end),
                    "symbol": str(c.symbol),
                    "confidence": float(c.confidence),
                    "simple": str(c.simple),
                    "medium": str(c.medium),
                }
                for c in self.chords
            ],
            "sections": [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "label": str(s.label),
                    "confidence": float(s.confidence),
                }
                for s in self.sections
            ],
            "lyrics": [
                {
                    "start": float(line.start),
                    "end": float(line.end),
                    "text": str(line.text),
                    "words": [
                        {"start": float(w.start), "end": float(w.end), "text": str(w.text)}
                        for w in line.words
                    ],
                }
                for line in self.lyrics
            ],
            "lyrics_language": None if self.lyrics_language is None else str(self.lyrics_language),
            "models": {str(k): v for k, v in self.models.items()},
            "warnings": [str(w) for w in self.warnings],
            "created_at": float(self.created_at),
            "chart": self.chart,
        }


@dataclass(frozen=True)
class BeatGrid:
    """Tempo + beat/downbeat grid, preferring the stored metronome artifact."""

    bpm: float | None
    bpm_confidence: str
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    beats_per_measure: int
    warnings: tuple[str, ...] = ()


def _json_default(value: Any) -> Any:
    """Best-effort conversion so a stray numpy scalar cannot break the file."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _span(raw: Mapping[str, Any]) -> tuple[float, float] | None:
    start = _finite_float(raw.get("start"))
    end = _finite_float(raw.get("end"))
    if start is None or end is None or start < 0 or end < 0 or end < start:
        return None
    return start, end


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _confidence(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return float(min(1.0, max(0.0, number)))


def _coerce_chord(raw: Any) -> ChordEvent | None:
    if not isinstance(raw, Mapping):
        return None
    span = _span(raw)
    symbol = _text(raw.get("symbol"))
    if span is None or symbol is None:
        return None
    simple = _text(raw.get("simple")) or ""
    medium = _text(raw.get("medium")) or ""
    return ChordEvent(
        start=span[0],
        end=span[1],
        symbol=symbol,
        confidence=_confidence(raw.get("confidence")),
        simple=simple,
        medium=medium,
    )


def _coerce_section(raw: Any) -> Section | None:
    if not isinstance(raw, Mapping):
        return None
    span = _span(raw)
    label = _text(raw.get("label"))
    if span is None or label is None:
        return None
    return Section(start=span[0], end=span[1], label=label, confidence=_confidence(raw.get("confidence")))


def _coerce_word(raw: Any) -> LyricWord | None:
    if not isinstance(raw, Mapping):
        return None
    span = _span(raw)
    text = _text(raw.get("text"))
    if span is None or text is None:
        return None
    return LyricWord(start=span[0], end=span[1], text=text)


def _coerce_line(raw: Any) -> LyricLine | None:
    if not isinstance(raw, Mapping):
        return None
    span = _span(raw)
    if span is None:
        return None
    raw_words = raw.get("words")
    words: list[LyricWord] = []
    if isinstance(raw_words, list):
        words = [w for item in raw_words if (w := _coerce_word(item)) is not None]
    words.sort(key=lambda w: (w.start, w.end))
    text = _text(raw.get("text"))
    if text is None:
        text = " ".join(w.text for w in words) or None
    if text is None:
        return None
    return LyricLine(start=span[0], end=span[1], text=text, words=tuple(words))


def _coerce_times(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    times: list[float] = []
    for value in raw:
        number = _finite_float(value)
        if number is not None and number >= 0:
            times.append(number)
    return tuple(sorted(set(times)))


def _from_dict(raw: Mapping[str, Any]) -> AnalysisResult:
    bpm = _finite_float(raw.get("bpm"))
    if bpm is not None and bpm <= 0:
        bpm = None

    beats_per_measure = raw.get("beats_per_measure", 4)
    try:
        beats_per_measure = int(beats_per_measure)
    except (TypeError, ValueError):
        beats_per_measure = 4
    beats_per_measure = min(16, max(1, beats_per_measure))

    raw_chords = raw.get("chords")
    chords = [c for item in raw_chords if (c := _coerce_chord(item)) is not None] if isinstance(raw_chords, list) else []
    chords.sort(key=lambda c: (c.start, c.end))

    raw_sections = raw.get("sections")
    sections = (
        [s for item in raw_sections if (s := _coerce_section(item)) is not None]
        if isinstance(raw_sections, list)
        else []
    )
    sections.sort(key=lambda s: (s.start, s.end))

    raw_lyrics = raw.get("lyrics")
    lyrics = [line for item in raw_lyrics if (line := _coerce_line(item)) is not None] if isinstance(raw_lyrics, list) else []
    lyrics.sort(key=lambda line: (line.start, line.end))

    models = raw.get("models")
    warnings = raw.get("warnings")
    chart = raw.get("chart")
    chart = dict(chart) if isinstance(chart, dict) else None
    key = _text(raw.get("key"))
    language = _text(raw.get("lyrics_language"))

    duration = _finite_float(raw.get("duration_sec"))
    duration = max(0.0, duration) if duration is not None else 0.0

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        source_fingerprint=str(raw.get("source_fingerprint") or ""),
        duration_sec=duration,
        bpm=bpm,
        bpm_confidence=str(raw.get("bpm_confidence") or "low"),
        key=key,
        key_confidence=str(raw.get("key_confidence") or "low"),
        beats=_coerce_times(raw.get("beats")),
        downbeats=_coerce_times(raw.get("downbeats")),
        beats_per_measure=beats_per_measure,
        chords=tuple(chords),
        sections=tuple(sections),
        lyrics=tuple(lyrics),
        lyrics_language=language,
        models=dict(models) if isinstance(models, dict) else {},
        warnings=tuple(str(w) for w in warnings) if isinstance(warnings, list) else (),
        created_at=_finite_float(raw.get("created_at")) or 0.0,
        chart=chart,
    )


def _median_period(times: tuple[float, ...]) -> float:
    diffs = [b - a for a, b in pairwise(times) if b > a]
    return statistics.median(diffs) if diffs else 0.5


def _beat_grid_from_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    duration_sec: float | None,
) -> BeatGrid | None:
    times = _coerce_times(diagnostics.get("click_times_1x"))
    if duration_sec is not None:
        times = tuple(t for t in times if t <= duration_sec + 0.05)
    if len(times) < 2:
        return None

    bpm = _finite_float(diagnostics.get("bpm"))
    confidence = str(diagnostics.get("confidence") or "low")
    if bpm is None or bpm <= 0:
        bpm = 60.0 / _median_period(times)
        confidence = "low"

    measure = diagnostics.get("beats_per_measure", 4)
    try:
        measure = min(16, max(1, int(measure)))
    except (TypeError, ValueError):
        measure = 4

    warnings: list[str] = []
    downbeats: tuple[float, ...] = ()
    stored_downbeat = _finite_float(diagnostics.get("downbeat_sec"))
    if stored_downbeat is None:
        warnings.append("downbeat phase unknown; no downbeat markers")
    else:
        index = bisect_left(times, stored_downbeat)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
        best = min(candidates, key=lambda i: abs(times[i] - stored_downbeat))
        tolerance = max(0.05, 0.25 * _median_period(times))
        if abs(times[best] - stored_downbeat) > tolerance:
            warnings.append("stored downbeat does not align to the beat grid; no downbeat markers")
        else:
            downbeats = tuple(times[best::measure])

    return BeatGrid(
        bpm=float(bpm) if bpm is not None else None,
        bpm_confidence=confidence,
        beats=times,
        downbeats=downbeats,
        beats_per_measure=measure,
        warnings=tuple(warnings),
    )


def build_beat_grid(
    audio_path: Path | None,
    diagnostics: Mapping[str, Any] | None,
    *,
    duration_sec: float | None = None,
) -> BeatGrid:
    """Build the analysis grid, reusing the stored metronome artifact when present.

    A present artifact is authoritative: no second tempo tracker runs. Only when it is missing or
    unusable do we estimate tempo and synthesize a single grid, which is explicitly low confidence
    because its phase (and any downbeats) is arbitrary.
    """
    duration = _finite_float(duration_sec)
    if duration is not None and duration <= 0:
        duration = None

    if diagnostics:
        grid = _beat_grid_from_diagnostics(diagnostics, duration_sec=duration)
        if grid is not None:
            return grid

    warnings = ["beat grid is low confidence: rebuilt from a tempo estimate, downbeat phase is approximate"]
    if audio_path is None:
        return BeatGrid(None, "low", (), (), 4, (*warnings, "no audio path for tempo estimation"))

    from audio_to_tab.tempo import estimate_tempo_from_audio

    estimate = estimate_tempo_from_audio(audio_path)
    if estimate is None or not estimate.bpm or estimate.bpm <= 0:
        return BeatGrid(None, "low", (), (), 4, (*warnings, "tempo estimation failed"))

    if duration is None:
        try:
            import librosa

            duration = float(librosa.get_duration(path=str(audio_path)))
        except Exception as exc:
            logger.debug("Could not read duration for beat grid: %s", exc)
            duration = 0.0
    if duration <= 0:
        return BeatGrid(float(estimate.bpm), "low", (), (), 4, (*warnings, "duration unknown"))

    period = 60.0 / float(estimate.bpm)
    count = int(duration / period)
    beats = tuple(round(i * period, 6) for i in range(count + 1) if i * period <= duration + 1e-9)
    return BeatGrid(
        bpm=float(estimate.bpm),
        bpm_confidence="low",
        beats=beats,
        downbeats=(),
        beats_per_measure=4,
        warnings=tuple(warnings),
    )


def _report(progress: Callable[[str, str], None] | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress("analysis", message)
    except Exception:
        logger.debug("Analysis progress callback failed", exc_info=True)


def analyze_audio(
    audio_path: Path,
    *,
    harmony_path: Path | None = None,
    harmony_source_kind: str | None = None,
    bass_path: Path | None = None,
    vocals_path: Path | None = None,
    metronome_diagnostics: Path | None = None,
    source_fingerprint: str,
    duration_sec: float,
    include_lyrics: bool = True,
    progress: Callable[[str, str], None] | None = None,
) -> AnalysisResult:
    """Run the analysis stages, isolating each failure into a warning.

    Order is beats → provisional key → chords → final key → sections → lyrics → chart.
    The provisional key is chroma-only and feeds the chord prior. The stored key is
    recomputed from the decoded chords so a biased chord track cannot justify itself.
    """
    audio_path = Path(audio_path)
    duration = _finite_float(duration_sec)
    duration = max(0.0, duration) if duration is not None else 0.0
    warnings: list[str] = []

    _report(progress, "beat grid")
    diagnostics: Mapping[str, Any] | None = None
    if metronome_diagnostics is not None:
        try:
            from audio_to_tab.metronome import load_metronome_diagnostics

            diagnostics = load_metronome_diagnostics(metronome_diagnostics)
        except Exception as exc:
            warnings.append(f"metronome diagnostics unreadable: {exc}")
        if diagnostics is None:
            warnings.append("metronome diagnostics missing; beat grid rebuilt")
    try:
        grid = build_beat_grid(audio_path, diagnostics, duration_sec=duration)
    except Exception as exc:
        warnings.append(f"beat grid failed: {exc}")
        grid = BeatGrid(None, "low", (), (), 4, ())
    warnings.extend(grid.warnings)

    _report(progress, "provisional key")
    provisional: str | None = None
    harmony = Path(harmony_path) if harmony_path is not None else audio_path
    try:
        from audio_to_tab.key import provisional_key_from_audio

        provisional, _provisional_score = provisional_key_from_audio(harmony)
    except Exception as exc:
        warnings.append(f"provisional key failed: {exc}")

    _report(progress, "chords")
    chords: list[ChordEvent] = []
    try:
        from audio_to_tab.chords import detect_chords

        chord_model_dir = None
        try:
            from audio_to_tab.chord_neural import (
                chord_model_requested,
                default_chord_model_dir,
            )

            if chord_model_requested():
                chord_model_dir = default_chord_model_dir()
        except Exception:
            chord_model_dir = None

        chords = detect_chords(
            harmony,
            beats=grid.beats,
            bpm=grid.bpm,
            model_dir=chord_model_dir,
            provisional_key=provisional,
            beats_per_measure=grid.beats_per_measure,
            downbeats=grid.downbeats,
            bass_path=bass_path,
            progress=lambda message: _report(progress, message),
        )
    except Exception as exc:
        warnings.append(f"chord detection failed: {exc}")
    if harmony_path is not None and not harmony.is_file():
        warnings.append(f"harmony stem missing: {harmony}")
    if not chords:
        warnings.append("no chords detected")

    _report(progress, "key")
    key: str | None = None
    key_confidence = "low"
    try:
        from audio_to_tab.key import estimate_key

        key, key_confidence = estimate_key(
            audio_path,
            bpm=grid.bpm,
            beats=grid.beats,
            chords=chords,
            downbeats=grid.downbeats,
            beats_per_measure=grid.beats_per_measure,
        )
    except Exception as exc:
        warnings.append(f"key estimation failed: {exc}")

    _report(progress, "sections")
    sections: list[Section] = [Section(0.0, duration, "Full song", 0.0)]
    try:
        from audio_to_tab.sections import detect_sections

        detected = detect_sections(audio_path, beats=grid.beats, duration_sec=duration)
        if detected:
            sections = detected
    except Exception as exc:
        warnings.append(f"section detection failed: {exc}")

    lyrics = []
    lyrics_language: str | None = None
    lyrics_model = "disabled"
    if include_lyrics:
        if vocals_path is None:
            warnings.append("no vocals stem; lyrics skipped")
            lyrics_model = "no-vocals"
        else:
            lyrics_model = "faster-whisper-small-int8"
            _report(progress, "lyrics")
            try:
                from audio_to_tab.lyrics import transcribe_lyrics

                lyrics, lyrics_language, lyric_warnings = transcribe_lyrics(
                    Path(vocals_path),
                    duration_sec=duration,
                    progress=lambda message: _report(progress, message),
                )
                warnings.extend(lyric_warnings)
            except Exception as exc:
                warnings.append(f"lyrics transcription failed: {exc}")

    models: dict[str, Any] = {
        "chords": "template-viterbi-1",
        "key": "chord-histogram-1" if chords else "chroma-profile-1",
        "sections": "recurrence-1",
        "lyrics": lyrics_model,
    }
    if harmony_source_kind:
        models["harmony_source"] = str(harmony_source_kind)
    if provisional:
        models["provisional_key"] = provisional

    result = AnalysisResult(
        source_fingerprint=str(source_fingerprint),
        duration_sec=duration,
        bpm=grid.bpm,
        bpm_confidence=grid.bpm_confidence,
        key=key,
        key_confidence=key_confidence,
        beats=grid.beats,
        downbeats=grid.downbeats,
        beats_per_measure=grid.beats_per_measure,
        chords=tuple(chords),
        sections=tuple(sections),
        lyrics=tuple(lyrics),
        lyrics_language=lyrics_language,
        models=models,
        warnings=tuple(warnings),
        created_at=time.time(),
    )
    try:
        from audio_to_tab.chart import build_chart

        result = replace(result, chart=build_chart(result).to_dict())
    except Exception as exc:
        warnings.append(f"chart failed: {exc}")
        result = replace(result, warnings=tuple(warnings))
    return result


def save_analysis(path: Path, result: AnalysisResult) -> Path:
    """Atomically write ``result`` as UTF-8 JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        result.to_dict(),
        indent=2,
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    )
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_analysis(path: Path) -> AnalysisResult | None:
    """Load a supported analysis file, or ``None`` when missing/unreadable/unknown."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read analysis %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("Analysis %s is not a JSON object; ignoring", path)
        return None
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        logger.warning(
            "Analysis schema %r is not supported (expected %s); ignoring %s",
            version,
            SCHEMA_VERSION,
            path,
        )
        return None
    return _from_dict(raw)


def analysis_cache_key(
    source_fingerprint: str,
    model_versions: Mapping[str, str],
    settings: Mapping[str, Any],
) -> str:
    """Deterministic cache key: same inputs, same key; any model/setting change, new key."""
    payload = {
        "source_fingerprint": str(source_fingerprint),
        "models": {str(k): str(v) for k, v in model_versions.items()},
        "settings": dict(settings),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "AnalysisResult",
    "BeatGrid",
    "ChordEvent",
    "LyricLine",
    "LyricWord",
    "Section",
    "analysis_cache_key",
    "analyze_audio",
    "build_beat_grid",
    "load_analysis",
    "save_analysis",
]

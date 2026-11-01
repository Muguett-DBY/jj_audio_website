"""Optional BTC neural chord scorer (opt-in).

Code is MIT (vendored under ``audio_to_tab.btc``). The pretrained checkpoint was
trained on Isophonics / Robbie Williams / UsPop2002 labels with audio collected
from commercial services — weight redistributability is **undocumented**. We
download only when explicitly requested (``ensure_chord_model`` /
``ATT_CHORD_MODEL=1``) and never enable this path by default.

25-class maj/min vocabulary — deliberately avoids the template stack's maj7 /
mMaj7 spam on dense mixes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from audio_to_tab.analysis import ChordEvent
from audio_to_tab.chord_symbols import parse
from audio_to_tab.chords import _decorate, _merge_events
from audio_to_tab.roformer import download_sha256_file, file_sha256

logger = logging.getLogger(__name__)

# Upstream: jayg996/BTC-ISMIR19 test/btc_model.pt (25-class maj/min).
BTC_WEIGHT_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model.pt"
)
BTC_WEIGHT_SHA256 = "71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54"
BTC_WEIGHT_NAME = "btc_model.pt"
BTC_WEIGHT_BYTES = 12_154_754

_IDX2CHORD = (
    "C",
    "C:min",
    "C#",
    "C#:min",
    "D",
    "D:min",
    "D#",
    "D#:min",
    "E",
    "E:min",
    "F",
    "F:min",
    "F#",
    "F#:min",
    "G",
    "G:min",
    "G#",
    "G#:min",
    "A",
    "A:min",
    "A#",
    "A#:min",
    "B",
    "B:min",
    "N",
)

_DEFAULT_DIR = Path.home() / ".cache" / "audio_to_tab" / "chord-btc"
_SONG_HZ = 22050
_INST_LEN = 10.0
_N_BINS = 144
_BINS_PER_OCTAVE = 24
_HOP = 2048
_TIMESTEP = 108

_MODEL_CONFIG = {
    "feature_size": 144,
    "timestep": 108,
    "num_chords": 25,
    "input_dropout": 0.2,
    "layer_dropout": 0.2,
    "attention_dropout": 0.2,
    "relu_dropout": 0.2,
    "num_layers": 8,
    "num_heads": 4,
    "hidden_size": 128,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "loss": "ce",
    "probs_out": False,
}


def default_chord_model_dir() -> Path:
    configured = os.environ.get("ATT_CHORD_MODEL_DIR")
    return Path(configured) if configured else _DEFAULT_DIR


def chord_model_requested() -> bool:
    """True when the product asked for the neural path (env or explicit model_dir)."""
    flag = (os.environ.get("ATT_CHORD_MODEL") or "").strip().lower()
    return flag in {"1", "true", "yes", "on", "btc"}


def chord_model_available(model_dir: Path | None = None) -> bool:
    path = Path(model_dir) if model_dir is not None else default_chord_model_dir()
    weight = path / BTC_WEIGHT_NAME
    if not weight.is_file() or weight.stat().st_size <= 0:
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        return file_sha256(weight) == BTC_WEIGHT_SHA256
    except OSError:
        return False


def ensure_chord_model(
    model_dir: Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download the pinned BTC maj/min checkpoint into ``model_dir``."""
    path = Path(model_dir) if model_dir is not None else default_chord_model_dir()
    path.mkdir(parents=True, exist_ok=True)
    destination = path / BTC_WEIGHT_NAME
    try:
        if destination.is_file() and file_sha256(destination) == BTC_WEIGHT_SHA256:
            return path
    except OSError:
        pass
    _notify(progress, "chords: downloading BTC maj/min weights (opt-in neural)")
    logger.warning(
        "Downloading BTC chord weights. Code is MIT; training-audio provenance is "
        "undocumented — keep ATT_CHORD_MODEL off for redistribution-sensitive builds."
    )
    download_sha256_file(BTC_WEIGHT_URL, destination, BTC_WEIGHT_SHA256, timeout=600)
    return path


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        logger.debug("Chord neural progress callback failed", exc_info=True)


def _mir_eval_to_symbol(label: str) -> str:
    text = (label or "").strip()
    if not text or text.upper() in {"N", "X"}:
        return "N"
    if text.endswith(":min"):
        root = text[: -len(":min")]
        return f"{root}m"
    if ":" in text:
        root, quality = text.split(":", 1)
        mapping = {
            "maj": "",
            "min": "m",
            "maj7": "maj7",
            "min7": "m7",
            "minmaj7": "mMaj7",
            "7": "7",
            "dim": "dim",
            "aug": "aug",
            "sus2": "sus2",
            "sus4": "sus4",
            "dim7": "dim7",
            "hdim7": "m7b5",
            "min6": "m6",
            "maj6": "6",
        }
        suffix = mapping.get(quality, "")
        if quality in mapping:
            return f"{root}{suffix}"
    chord = parse(text)
    return text if chord is None else f"{chord.root}{'' if chord.quality == 'maj' else ('m' if chord.quality == 'min' else chord.quality)}"


def _cqt_features(audio_path: Path) -> tuple[np.ndarray, float, float]:
    import librosa

    wav, sr = librosa.load(str(audio_path), sr=_SONG_HZ, mono=True)
    if wav.size < int(sr * 0.2):
        raise ValueError("audio too short for BTC features")
    chunks: list[np.ndarray] = []
    cursor = 0
    window = int(sr * _INST_LEN)
    while cursor + window < wav.size:
        piece = librosa.cqt(
            wav[cursor : cursor + window],
            sr=sr,
            n_bins=_N_BINS,
            bins_per_octave=_BINS_PER_OCTAVE,
            hop_length=_HOP,
        )
        chunks.append(piece)
        cursor += window
    tail = librosa.cqt(
        wav[cursor:],
        sr=sr,
        n_bins=_N_BINS,
        bins_per_octave=_BINS_PER_OCTAVE,
        hop_length=_HOP,
    )
    chunks.append(tail)
    feature = np.concatenate(chunks, axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = _INST_LEN / float(_TIMESTEP)
    duration = float(wav.size) / float(sr)
    return feature, feature_per_second, duration


def _load_model(weight_path: Path):
    import torch

    from audio_to_tab.btc.btc_model import BTC_model

    device = torch.device("cpu")
    model = BTC_model(config=dict(_MODEL_CONFIG)).to(device)
    checkpoint = torch.load(str(weight_path), map_location=device, weights_only=False)
    mean = np.asarray(checkpoint["mean"], dtype=np.float64)
    std = np.asarray(checkpoint["std"], dtype=np.float64)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, mean, std, device


def _predict_frame_indices(feature: np.ndarray, mean: np.ndarray, std: np.ndarray, model, device) -> list[int]:
    import torch

    frames = feature.T  # (time, bins)
    frames = (frames - mean) / (std + 1e-9)
    n_timestep = _TIMESTEP
    num_pad = (n_timestep - (frames.shape[0] % n_timestep)) % n_timestep
    if num_pad:
        frames = np.pad(frames, ((0, num_pad), (0, 0)), mode="constant")
    num_instance = frames.shape[0] // n_timestep
    indices: list[int] = []
    with torch.no_grad():
        tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
        for block in range(num_instance):
            chunk = tensor[:, n_timestep * block : n_timestep * (block + 1), :]
            self_attn_output, _ = model.self_attn_layers(chunk)
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for step in range(n_timestep):
                # Drop padded tail of the last block.
                absolute = n_timestep * block + step
                if absolute >= frames.shape[0] - num_pad and block == num_instance - 1:
                    break
                indices.append(int(prediction[step].item()))
    return indices


def _indices_to_events(
    indices: Sequence[int],
    *,
    time_unit: float,
    duration: float,
) -> list[ChordEvent]:
    if not indices:
        return []
    events: list[ChordEvent] = []
    start = 0.0
    prev = int(indices[0])
    for index in range(1, len(indices)):
        current = int(indices[index])
        if current == prev:
            continue
        end = min(duration, time_unit * index)
        if end > start:
            symbol = _mir_eval_to_symbol(_IDX2CHORD[prev] if 0 <= prev < len(_IDX2CHORD) else "N")
            simple, medium = _decorate(symbol) if symbol != "N" else ("N", "N")
            events.append(
                ChordEvent(
                    start=start,
                    end=end,
                    symbol=symbol,
                    confidence=0.0,
                    simple=simple,
                    medium=medium,
                )
            )
        start = end
        prev = current
    if duration > start:
        symbol = _mir_eval_to_symbol(_IDX2CHORD[prev] if 0 <= prev < len(_IDX2CHORD) else "N")
        simple, medium = _decorate(symbol) if symbol != "N" else ("N", "N")
        events.append(
            ChordEvent(
                start=start,
                end=duration,
                symbol=symbol,
                confidence=0.0,
                simple=simple,
                medium=medium,
            )
        )
    return _merge_events(events)


def detect_chords_btc(
    audio_path: Path,
    *,
    model_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[ChordEvent] | None:
    """Run BTC maj/min recognition. Returns ``None`` when unavailable so callers fall back."""
    path = Path(audio_path)
    if not path.is_file():
        return None
    try:
        import torch  # noqa: F401
    except Exception:
        logger.info("BTC chord model skipped: torch not installed")
        return None
    weight_dir = Path(model_dir) if model_dir is not None else default_chord_model_dir()
    weight = weight_dir / BTC_WEIGHT_NAME
    if not weight.is_file():
        return None
    try:
        if file_sha256(weight) != BTC_WEIGHT_SHA256:
            logger.warning("BTC weight hash mismatch at %s; refusing to load", weight)
            return None
    except OSError:
        return None
    try:
        _notify(progress, "chords: BTC feature extract")
        feature, time_unit, duration = _cqt_features(path)
        _notify(progress, "chords: BTC inference")
        model, mean, std, device = _load_model(weight)
        indices = _predict_frame_indices(feature, mean, std, model, device)
        events = _indices_to_events(indices, time_unit=time_unit, duration=duration)
        _notify(progress, f"chords: BTC {len(events)} segments")
        return events
    except Exception:
        logger.warning("BTC chord recognition failed; falling back to templates", exc_info=True)
        return None


__all__ = [
    "BTC_WEIGHT_NAME",
    "BTC_WEIGHT_SHA256",
    "BTC_WEIGHT_URL",
    "chord_model_available",
    "chord_model_requested",
    "default_chord_model_dir",
    "detect_chords_btc",
    "ensure_chord_model",
]

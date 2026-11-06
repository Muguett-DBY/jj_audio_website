"""Demucs stem separation for isolating guitar from full mixes."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from audio_to_tab.hardware import apply_recommended_cpu_threads, cpu_thread_env
from audio_to_tab.ingest import temporary_output_path
from audio_to_tab.subprocess_util import run_process, subprocess_run_kwargs

logger = logging.getLogger(__name__)

_USER_AGENT = "audio-to-tab-pdf/1.0 (guitar isolation; urllib)"

GUITAR_FT_CHECKPOINT_ID = "htdemucs_6s_guitar_ft"
GUITAR_FT_HF_REPO = "adityalakhani/htdemucs-6s-guitar-ft"
GUITAR_FT_FILENAME = "guitar_htdemucs_6s.pt"
GUITAR_FT_URL = (
    f"https://huggingface.co/{GUITAR_FT_HF_REPO}/resolve/main/{GUITAR_FT_FILENAME}"
)
# Hugging Face blob SHA256 for guitar_htdemucs_6s.pt (330 MB / 329654071 bytes).
GUITAR_FT_SHA256 = "4fde369e41582ba5c2759b6ab926a44af467c64d4566bf914374ab267b19260e"
SUPPORTED_GUITAR_CHECKPOINTS = frozenset({GUITAR_FT_CHECKPOINT_ID})
GUITAR_FT_CACHE_HINT = (
    "Download the ~330 MB weights once from Audio Isolation Advanced "
    "(or run audio-isolate --guitar-checkpoint htdemucs_6s_guitar_ft) and retry."
)


def is_demucs_available() -> bool:
    """Return True if the Demucs package is installed.

    Uses ``find_spec`` so the desktop UI can paint without importing Demucs or
    Torch. Real separation still imports Demucs inside the job worker.
    """
    return importlib.util.find_spec("demucs") is not None


def _torch_cache_dir() -> Path:
    if env := os.environ.get("TORCH_HOME"):
        return Path(env)
    return Path.home() / ".cache" / "torch"


def guitar_ft_weights_path() -> Path:
    """Local cache path for the community guitar-ft HTDemucs-6s weights."""
    return _torch_cache_dir() / "checkpoints" / GUITAR_FT_FILENAME


def _guitar_ft_digest_sidecar(path: Path) -> Path:
    """Sidecar file so re-checks skip re-hashing the 330 MB checkpoint."""
    return path.with_name(path.name + ".sha256")


def _write_guitar_ft_digest_sidecar(path: Path) -> None:
    sidecar = _guitar_ft_digest_sidecar(path)
    tmp = sidecar.with_suffix(".sha256.tmp")
    try:
        tmp.write_text(GUITAR_FT_SHA256 + "\n", encoding="utf-8")
        tmp.replace(sidecar)
    except OSError:
        logger.info("Could not write guitar-ft digest sidecar for %s", path)
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def guitar_ft_weights_cached() -> bool:
    """True when pinned guitar-ft weights are already on disk. Does not download."""
    path = guitar_ft_weights_path()
    if not path.is_file():
        return False
    sidecar = _guitar_ft_digest_sidecar(path)
    if sidecar.is_file() and sidecar.stat().st_mtime >= path.stat().st_mtime:
        try:
            return sidecar.read_text(encoding="utf-8").strip() == GUITAR_FT_SHA256
        except OSError:
            pass
    try:
        if guitar_ft_weights_sha256(path) != GUITAR_FT_SHA256:
            return False
    except OSError:
        return False
    _write_guitar_ft_digest_sidecar(path)
    return True


def guitar_ft_weights_sha256(path: Path | None = None) -> str:
    """Return sha256 hex digest of cached guitar-ft weights (for pinning/tests)."""
    target = path or guitar_ft_weights_path()
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_guitar_ft_weights(path: Path | None = None) -> Path:
    """Raise if the cached file is missing or its SHA256 does not match the pin."""
    target = path or guitar_ft_weights_path()
    if not target.is_file():
        raise RuntimeError(f"guitar-ft weights not found at {target}")
    digest = guitar_ft_weights_sha256(target)
    if digest != GUITAR_FT_SHA256:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"guitar-ft weights hash mismatch (got {digest}, expected {GUITAR_FT_SHA256})"
        )
    _write_guitar_ft_digest_sidecar(target)
    return target


def download_guitar_ft_weights(*, force: bool = False) -> Path:
    """Download guitar-ft weights into the torch cache (urllib, no huggingface_hub)."""
    path = guitar_ft_weights_path()
    if path.exists() and not force:
        try:
            return verify_guitar_ft_weights(path)
        except RuntimeError:
            logger.warning("Cached guitar-ft weights failed hash check; re-downloading")
            path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    try:
        req = urllib.request.Request(
            GUITAR_FT_URL, headers={"User-Agent": _USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != GUITAR_FT_SHA256:
            raise RuntimeError(
                f"guitar-ft download hash mismatch "
                f"(got {digest.hexdigest()}, expected {GUITAR_FT_SHA256})"
            )
        tmp.replace(path)
    except RuntimeError:
        tmp.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download guitar-ft weights from {GUITAR_FT_URL}"
        ) from exc
    return verify_guitar_ft_weights(path)


def remap_guitar_ft_state_dict(
    state_dict: dict,
    wrapper_keys: set[str] | frozenset[str],
) -> dict:
    """Prefix inner HTDemucs keys with ``models.0.`` for Demucs BagOfModels.

    ``get_model("htdemucs_6s")`` wraps the network; the guitar-ft training
    checkpoint stores the inner module's ``state_dict`` without that prefix.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    sample = list(state_dict.keys())[:3]
    if any(key in wrapper_keys for key in sample):
        return state_dict
    return {f"models.0.{key}": value for key, value in state_dict.items()}


def load_guitar_ft_state_dict(weights_path: Path):
    """Load guitar-ft checkpoint state dict onto CPU."""
    import torch

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state" in checkpoint:
            return checkpoint["state"]
    return checkpoint


def run_demucs_guitar_ft_inprocess(
    audio_path: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    quality: str = "fast",
    segment: int | None = 7,
    jobs: int = 1,
    shifts: int | None = None,
    overlap: float | None = None,
) -> None:
    """
    Run official HTDemucs-6s with community guitar-ft weights in-process.

    Writes the same six stem filenames Demucs CLI would produce under
    ``output_root/htdemucs_6s/<track>/``.

    ``segment`` (seconds) and ``jobs`` match the CLI ``--segment`` / ``--jobs``
    path so CPU runs stay chunked instead of building a full-track tensor.
    ``shifts`` / ``overlap`` override the Quality map when set.
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import convert_audio, save_audio
    from demucs.pretrained import get_model

    weights_path = download_guitar_ft_weights()
    model = get_model("htdemucs_6s")
    state_dict = remap_guitar_ft_state_dict(
        load_guitar_ft_state_dict(weights_path),
        set(model.state_dict().keys()),
    )
    model.load_state_dict(state_dict)
    model.eval()

    dev_name = device
    if dev_name == "cuda" and not torch.cuda.is_available():
        dev_name = "cpu"
    dev = torch.device(dev_name)
    model.to(dev)
    apply_recommended_cpu_threads(device=dev_name)

    shift_count = (
        int(shifts)
        if shifts is not None
        else int({"fast": 0, "balanced": 1, "high": 3, "extreme": 5}.get(quality, 0))
    )
    overlap_val = (
        float(overlap)
        if overlap is not None
        else {"fast": 0.25, "balanced": 0.25, "high": 0.5, "extreme": 0.75}.get(
            quality, 0.25
        )
    )

    wav, sr = torchaudio.load(str(audio_path))
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    apply_kwargs: dict = {
        "device": dev,
        "shifts": shift_count,
        "overlap": overlap_val,
        "split": True,
        "progress": False,
        "num_workers": max(0, int(jobs)),
    }
    if segment is not None and int(segment) > 0:
        apply_kwargs["segment"] = float(segment)

    with torch.no_grad():
        sources = apply_model(model, wav[None], **apply_kwargs)[0]
    sources = sources * ref.std() + ref.mean()

    track_name = audio_path.stem
    out_dir = output_root / "htdemucs_6s" / track_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for source, name in zip(sources, model.sources, strict=True):
        save_audio(source, str(out_dir / f"{name}.wav"), samplerate=model.samplerate)


def run_demucs(
    demucs_args: list[str],
    *,
    should_abort: Callable[[], bool] | None = None,
    timeout_sec: float | None = None,
) -> None:
    """
    Run Demucs with CLI-style args and the soundfile-backed WAV writer.

    When frozen (PyInstaller), invokes ``demucs.separate.main`` in-process because
    ``sys.executable`` is the app binary and cannot run ``-m demucs``.
    Otherwise uses the writer bootstrap in a subprocess to preserve isolation.

    ``should_abort`` / ``timeout_sec`` only apply to the subprocess path: the
    frozen in-process run shares the caller's thread and cannot be terminated.
    """
    if getattr(sys, "frozen", False):
        from demucs.separate import main as demucs_main

        from audio_to_tab.demucs_bootstrap import install_soundfile_writer

        apply_recommended_cpu_threads()
        install_soundfile_writer()
        try:
            demucs_main(demucs_args)
        except SystemExit as exc:
            code = exc.code
            if code not in (0, None):
                raise RuntimeError(f"Demucs failed with exit code {code}") from exc
        return

    cmd = [sys.executable, "-m", "audio_to_tab.demucs_bootstrap", *demucs_args]
    env = os.environ.copy()
    env.update(cpu_thread_env())
    result = run_process(
        cmd,
        should_abort=should_abort,
        timeout_sec=timeout_sec,
        capture_output=True,
        text=True,
        env=env,
        **subprocess_run_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr or result.stdout}")


def _maybe_refine_guitar_stem(
    guitar_path: Path,
    residual_path: Path | None,
    *,
    device: str,
    guitar_refine: bool,
    model: str,
) -> None:
    """Run MelBand refine while ``residual_path`` (``other``) still exists."""
    if not guitar_refine or model == "melband_roformer_guitar":
        return
    from audio_to_tab.roformer import (
        ROFORMER_INSTALL_HINT,
        is_guitar_refine_available,
        run_guitar_refine,
    )

    if not is_guitar_refine_available():
        raise RuntimeError(
            "Guitar refine was requested but no RoFormer extra is installed. "
            f"{ROFORMER_INSTALL_HINT}"
        )
    residual = residual_path if residual_path is not None and residual_path.is_file() else None
    try:
        from audio_to_tab.isolate import ensure_guitar_prerefine_backup, save_guitar_refined_backup

        ensure_guitar_prerefine_backup(guitar_path)
        run_guitar_refine(guitar_path, guitar_path, residual_path=residual, device=device)
        save_guitar_refined_backup(guitar_path)
    except Exception as exc:
        logger.warning("guitar refine failed; keeping first-pass guitar stem: %s", exc)


def _run_demucs_guitar_cli(
    src: Path,
    tmp_path: Path,
    *,
    model: str,
    device: str,
    quality: str,
    demucs_segment: int | None,
    demucs_jobs: int,
) -> None:
    from audio_to_tab.isolate import effective_demucs_segment

    shifts = {"fast": "0", "balanced": "1", "high": "3", "extreme": "5"}.get(quality, "0")
    overlap = {"fast": "0.25", "balanced": "0.25", "high": "0.5", "extreme": "0.75"}.get(
        quality, "0.25"
    )
    demucs_args = [
        "-n",
        model,
        "-d",
        device,
        "-o",
        str(tmp_path),
        "--shifts",
        shifts,
        "--overlap",
        overlap,
        "--jobs",
        str(max(1, int(demucs_jobs))),
    ]
    segment = effective_demucs_segment(model, demucs_segment)
    if segment is not None:
        demucs_args.extend(["--segment", str(segment)])
    demucs_args.append(str(src))
    run_demucs(demucs_args)


def separate_guitar_stem(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model: str = "htdemucs_6s",
    device: str = "cpu",
    quality: str = "fast",
    demucs_segment: int | None = 8,
    demucs_jobs: int = 1,
    guitar_refine: bool = False,
    guitar_checkpoint: str | None = None,
    fold_other_mode: str | None = None,
    sub_bass_debleed: bool = False,
    low_end_restore_db: float = 0.0,
    bleed_gate: bool = False,
    adaptive_fold_gain: bool = False,
    guitar_ensemble: bool = False,
    recovery_diagnostics_path: Path | None = None,
    bleed_gate_diagnostics_path: Path | None = None,
) -> Path:
    """
    Extract guitar stem using Demucs CLI (or an opt-in RoFormer backend).

    quality presets mirror TabGrabber: fast | balanced | high | extreme.
    ``demucs_segment`` / ``demucs_jobs`` match the isolate path so this is not
    a full-track tensor (isolate defaults: segment 8s clamped per model, jobs 1).
    ``fold_other_mode`` mixes the leftover ``other`` stem into the guitar stem
    (full | best_effort | band_limited) instead of discarding it.
    """
    from audio_to_tab.isolate import (
        DEMUCS_MODELS,
        FOLD_OTHER_MODES,
        apply_guitar_low_end_recovery,
        effective_demucs_segment,
    )
    if fold_other_mode is not None and fold_other_mode not in FOLD_OTHER_MODES:
        raise ValueError(
            f"Unsupported fold_other_mode {fold_other_mode!r}. "
            f"Choose from: {', '.join(FOLD_OTHER_MODES)}"
        )
    from audio_to_tab.roformer import (
        BS_ROFORMER_SW_ID,
        ROFORMER_INSTALL_HINT,
        ROFORMER_MODELS,
        collect_named_stems,
        is_roformer_backend_available,
        run_roformer_model,
    )
    from audio_to_tab.scnet import (
        SCNET_INSTALL_HINT,
        SCNET_MODELS,
        is_scnet_available,
        run_scnet_model,
    )

    src = Path(audio_path)
    if output_path is None:
        out = temporary_output_path("stem_", "_guitar.wav")
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    resolved_model = model
    if resolved_model in ROFORMER_MODELS and not is_roformer_backend_available():
        raise RuntimeError(
            f"Model {resolved_model!r} needs the optional RoFormer backend, which "
            f"is not installed. {ROFORMER_INSTALL_HINT}"
        )
    if resolved_model in SCNET_MODELS and not is_scnet_available():
        raise RuntimeError(
            f"Model {resolved_model!r} needs the optional SCNet runtime, which "
            f"is not installed. {SCNET_INSTALL_HINT}"
        )

    def _apply_low_end_recovery(stems: dict[str, Path]) -> None:
        if not sub_bass_debleed and not low_end_restore_db:
            return
        recovery = apply_guitar_low_end_recovery(
            out,
            out,
            bass_path=stems.get("bass"),
            drums_path=stems.get("drums"),
            sub_bass_debleed=sub_bass_debleed,
            boost_db=low_end_restore_db,
        )
        if recovery.attempted:
            logger.debug("tab guitar low-end recovery: %s", recovery.reason)
            if recovery_diagnostics_path is not None:
                recovery.write_json(recovery_diagnostics_path)

    def _maybe_fold_other(guitar_path: Path, other_path: Path | None) -> None:
        if not fold_other_mode:
            return
        from audio_to_tab.isolate import apply_fold_other_into_guitar

        apply_fold_other_into_guitar(
            {"guitar": guitar_path, "other": other_path},
            mode=fold_other_mode,
            adaptive_gain=adaptive_fold_gain,
        )

    def _maybe_bleed_gate(guitar_path: Path, stems: dict[str, Path | None]) -> None:
        if not bleed_gate:
            return
        from audio_to_tab.isolate import apply_bleed_gate

        competitors = {
            name: path for name, path in stems.items() if path is not None
        }
        diag = apply_bleed_gate(
            guitar_path, guitar_path, competitor_paths=competitors
        )
        if diag.attempted and bleed_gate_diagnostics_path is not None:
            diag.write_json(bleed_gate_diagnostics_path)

    def _maybe_ensemble_guitar(guitar_path: Path, src: Path) -> None:
        if not guitar_ensemble:
            return
        from audio_to_tab.isolate import blend_guitar_stems

        if not is_roformer_backend_available():
            logger.warning(
                "guitar ensemble requested but the RoFormer backend is not installed; "
                "keeping the primary Demucs guitar stem. %s",
                ROFORMER_INSTALL_HINT,
            )
            return
        with tempfile.TemporaryDirectory() as ens_tmp:
            ens_out = Path(ens_tmp)
            try:
                run_roformer_model(
                    src, ens_out, model=BS_ROFORMER_SW_ID, device=device
                )
            except Exception as exc:
                logger.warning(
                    "guitar ensemble secondary separation failed; "
                    "keeping the primary guitar stem: %s",
                    exc,
                )
                return
            secondary = collect_named_stems(ens_out, ("guitar",))
            secondary_path = secondary.get("guitar")
            if secondary_path is None:
                logger.warning(
                    "guitar ensemble secondary separation produced no guitar stem; "
                    "keeping the primary guitar stem"
                )
                return
            diag = blend_guitar_stems(
                guitar_path, secondary_path, guitar_path
            )
            if diag.attempted:
                logger.debug("guitar ensemble: %s", diag.reason)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if resolved_model in ROFORMER_MODELS or resolved_model in SCNET_MODELS:
            if resolved_model in SCNET_MODELS:
                run_scnet_model(src, tmp_path, device=device)
            else:
                run_roformer_model(src, tmp_path, model=resolved_model, device=device)
            found = collect_named_stems(tmp_path, ("guitar", "other", "bass", "drums"))
            if "guitar" not in found:
                raise FileNotFoundError(
                    f"Separator did not produce a guitar stem for model {resolved_model}."
                )
            shutil.copy2(found["guitar"], out)
            _maybe_refine_guitar_stem(
                out,
                found.get("other"),
                device=device,
                guitar_refine=guitar_refine,
                model=resolved_model,
            )
            _maybe_fold_other(out, found.get("other"))
            _maybe_bleed_gate(out, found)
            _apply_low_end_recovery(found)
            return out

        if resolved_model not in DEMUCS_MODELS:
            raise ValueError(f"Unsupported model {resolved_model!r}")

        if not is_demucs_available():
            raise RuntimeError(
                "Demucs is not installed. Install with: pip install -r requirements-demucs.txt"
            )

        used_guitar_ft = False
        want_guitar_ft = (
            guitar_checkpoint == GUITAR_FT_CHECKPOINT_ID
            and resolved_model == "htdemucs_6s"
        )
        if want_guitar_ft and not guitar_ft_weights_cached():
            raise RuntimeError(
                "Guitar-FT was requested but its weights are not in the torch "
                f"cache; not falling back to stock htdemucs_6s. {GUITAR_FT_CACHE_HINT}"
            )
        if want_guitar_ft:
            try:
                run_demucs_guitar_ft_inprocess(
                    src,
                    tmp_path,
                    device=device,
                    quality=quality,
                    segment=effective_demucs_segment(resolved_model, demucs_segment),
                    jobs=max(1, int(demucs_jobs)),
                )
                used_guitar_ft = True
            except Exception as exc:
                raise RuntimeError(
                    f"Guitar-FT separation failed; not falling back to stock: {exc}"
                ) from exc

        if not used_guitar_ft:
            _run_demucs_guitar_cli(
                src,
                tmp_path,
                model=resolved_model,
                device=device,
                quality=quality,
                demucs_segment=demucs_segment,
                demucs_jobs=demucs_jobs,
            )

        guitar_files = list(tmp_path.rglob("guitar.wav"))
        if not guitar_files:
            raise FileNotFoundError(
                "Demucs did not produce a guitar stem. "
                "Ensure model supports guitar separation (htdemucs_6s)."
            )
        other_files = list(tmp_path.rglob("other.wav"))
        bass_files = list(tmp_path.rglob("bass.wav"))
        drum_files = list(tmp_path.rglob("drums.wav"))
        shutil.copy2(guitar_files[0], out)
        _maybe_ensemble_guitar(out, src)
        _maybe_refine_guitar_stem(
            out,
            other_files[0] if other_files else None,
            device=device,
            guitar_refine=guitar_refine,
            model=resolved_model,
        )
        _maybe_fold_other(out, other_files[0] if other_files else None)
        stem_refs = {
            "guitar": guitar_files[0],
            "other": other_files[0] if other_files else None,
            "bass": bass_files[0] if bass_files else None,
            "drums": drum_files[0] if drum_files else None,
        }
        _maybe_bleed_gate(out, stem_refs)
        _apply_low_end_recovery({k: v for k, v in stem_refs.items() if v is not None})

    out_p = Path(out)
    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(f"Guitar stem output missing or empty: {out_p}")

    return out

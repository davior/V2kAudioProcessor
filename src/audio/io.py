"""Audio I/O utilities for loading, resampling, and loudness management."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import pyloudnorm as pyln
except ImportError as exc:  # pragma: no cover - dependency availability
    raise ImportError("pyloudnorm is required for loudness normalization") from exc

DEFAULT_SAMPLE_RATE = 16_000


def _apply_resample(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample ``audio`` from ``original_sr`` to ``target_sr`` using polyphase filtering."""
    if original_sr == target_sr:
        return audio

    gcd = math.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    return resample_poly(audio, up, down, axis=0)


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse multichannel audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _load_with_ffmpeg(path: Path, target_sr: int, mono: bool) -> Tuple[np.ndarray, int]:
    """Decode audio with ``ffmpeg`` into float32 PCM and return samples and samplerate."""
    channels = 1 if mono else 2
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(target_sr),
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, target_sr


def load_audio(
    path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample, and optionally convert to mono.

    The function first attempts to decode with ``soundfile``. If that fails, it falls
    back to ``ffmpeg`` for more exotic codecs.
    """
    path = Path(path)
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception:
        audio, sr = _load_with_ffmpeg(path, target_sr=target_sr, mono=mono)
    else:
        if mono:
            audio = _ensure_mono(audio)
        audio = _apply_resample(audio, sr, target_sr)
        sr = target_sr
    if mono:
        audio = _ensure_mono(audio)
    return audio.astype(np.float32), sr


def db_to_linear(gain_db: float) -> float:
    """Convert decibels to a linear gain factor."""
    return float(10 ** (gain_db / 20))


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a gain adjustment in decibels."""
    if gain_db == 0:
        return audio
    return audio * db_to_linear(gain_db)


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -23.0,
    meter: Optional["pyln.Meter"] = None,
) -> np.ndarray:
    """Normalize audio to the target LUFS using the EBU R128 recommendation."""
    if audio.size == 0:
        return audio

    meter = meter or pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    gain_db = target_lufs - loudness
    return apply_gain_db(audio, gain_db)


def apply_limiter(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Prevent clipping by scaling audio to a ceiling level in decibels."""
    if audio.size == 0:
        return audio

    ceiling = db_to_linear(ceiling_db)
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling or peak == 0:
        return audio

    gain = ceiling / peak
    return audio * gain


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write audio to disk as 32-bit floats."""
    sf.write(Path(path), audio, samplerate=sample_rate, subtype="FLOAT")


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
from scipy import signal


@dataclass
class SpectralGateConfig:
    threshold_db: float = -30.0
    reduction_db: float = -20.0
    min_energy_percentile: float = 0.2
    attack: float = 0.01  # seconds
    release: float = 0.08  # seconds
    n_fft: int = 1024
    hop_length: int = 256


@dataclass
class DenoisePreset:
    gain_db: float
    bandpass: Tuple[float, float]
    primary_gate: SpectralGateConfig
    secondary_gate: SpectralGateConfig
    limiter_ceiling: float
    target_lufs: float


PRESETS: Dict[str, DenoisePreset] = {
    "conservative": DenoisePreset(
        gain_db=0.0,
        bandpass=(60.0, 18000.0),
        primary_gate=SpectralGateConfig(
            threshold_db=-28.0, reduction_db=-18.0, min_energy_percentile=0.25, attack=0.015, release=0.1
        ),
        secondary_gate=SpectralGateConfig(
            threshold_db=-32.0, reduction_db=-22.0, min_energy_percentile=0.2, attack=0.01, release=0.08
        ),
        limiter_ceiling=0.98,
        target_lufs=-16.0,
    ),
    "aggressive": DenoisePreset(
        gain_db=3.0,
        bandpass=(80.0, 16000.0),
        primary_gate=SpectralGateConfig(
            threshold_db=-26.0, reduction_db=-14.0, min_energy_percentile=0.35, attack=0.02, release=0.12
        ),
        secondary_gate=SpectralGateConfig(
            threshold_db=-30.0, reduction_db=-26.0, min_energy_percentile=0.3, attack=0.015, release=0.1
        ),
        limiter_ceiling=0.96,
        target_lufs=-15.0,
    ),
}


def _db_to_amplitude(db_value: float) -> float:
    return 10 ** (db_value / 20.0)


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return audio
    return audio * _db_to_amplitude(gain_db)


def _bandpass_filter(audio: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    nyquist = sr / 2.0
    low = max(1.0, low)
    high = min(high, nyquist - 1.0)
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio)


def _estimate_noise_profile(
    y: np.ndarray, config: SpectralGateConfig, energy_percentile: Optional[float] = None
) -> np.ndarray:
    percentile = energy_percentile if energy_percentile is not None else config.min_energy_percentile
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(S)
    frame_energy = magnitude.mean(axis=0)
    cutoff = np.quantile(frame_energy, percentile)
    noise_frames = magnitude[:, frame_energy <= cutoff]
    if noise_frames.size == 0:
        noise_frames = magnitude
    noise_profile = np.mean(noise_frames, axis=1)
    return noise_profile


def _smooth_mask(mask: np.ndarray, attack_frames: int, release_frames: int) -> np.ndarray:
    smoothed = np.zeros_like(mask, dtype=float)
    for freq in range(mask.shape[0]):
        state = 0.0
        for t in range(mask.shape[1]):
            target = mask[freq, t]
            coeff = 1.0 / attack_frames if target > state else 1.0 / release_frames
            state += coeff * (target - state)
            smoothed[freq, t] = state
    return smoothed


def _apply_spectral_gate(y: np.ndarray, sr: int, config: SpectralGateConfig) -> np.ndarray:
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)

    noise_profile = _estimate_noise_profile(y, config)
    threshold = noise_profile[:, None] * _db_to_amplitude(config.threshold_db)

    mask = magnitude > threshold
    reduction = _db_to_amplitude(config.reduction_db)

    hop_duration = config.hop_length / float(sr)
    attack_frames = max(1, int(config.attack / hop_duration))
    release_frames = max(1, int(config.release / hop_duration))
    smoothed_mask = _smooth_mask(mask.astype(float), attack_frames, release_frames)

    gated_magnitude = magnitude * (smoothed_mask + (1 - smoothed_mask) * reduction)
    Y = gated_magnitude * np.exp(1j * phase)
    return librosa.istft(Y, hop_length=config.hop_length, length=len(y))


def _spectral_subtraction(y: np.ndarray, sr: int, config: SpectralGateConfig) -> np.ndarray:
    S = librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length)
    magnitude = np.abs(S)
    phase = np.angle(S)
    noise_profile = _estimate_noise_profile(y, config, energy_percentile=0.4)
    adjusted = np.maximum(magnitude - noise_profile[:, None], 0.0)
    return librosa.istft(adjusted * np.exp(1j * phase), hop_length=config.hop_length, length=len(y))


def _wiener_filter(y: np.ndarray, size: int = 11) -> np.ndarray:
    return signal.wiener(y, mysize=size)


def _multi_band_gate(y: np.ndarray, sr: int, config: SpectralGateConfig) -> np.ndarray:
    nyquist = sr / 2.0
    bands = [
        (30.0, min(200.0, nyquist - 1.0)),
        (200.0, min(2000.0, nyquist - 1.0)),
        (2000.0, min(8000.0, nyquist - 1.0)),
    ]
    band_signals = []
    for low, high in bands:
        filtered = _bandpass_filter(y, sr, low, high)
        gated = _apply_spectral_gate(filtered, sr, config)
        band_signals.append(gated)
    combined = np.sum(band_signals, axis=0) / max(len(band_signals), 1)
    peak = np.max(np.abs(combined)) + 1e-9
    if peak > 1.0:
        combined /= peak
    return combined


def _limiter(y: np.ndarray, ceiling: float) -> np.ndarray:
    ceiling = max(0.0, min(1.0, ceiling))
    return np.clip(y, -ceiling, ceiling)


def _loudness_normalize(y: np.ndarray, target_lufs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(y))) + 1e-9
    current_lufs = 20 * np.log10(rms)
    gain_db = target_lufs - current_lufs
    normalized = _apply_gain(y, gain_db)
    peak = np.max(np.abs(normalized)) + 1e-9
    if peak > 1.0:
        normalized /= peak
    return normalized


def denoise(
    audio: np.ndarray,
    sr: int,
    preset: str = "conservative",
    *,
    use_wiener: bool = False,
    use_spectral_subtraction: bool = False,
    use_multiband_gate: bool = False,
) -> np.ndarray:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    settings = PRESETS[preset]

    processed = _apply_gain(audio, settings.gain_db)
    processed = _bandpass_filter(processed, sr, *settings.bandpass)

    if use_multiband_gate:
        processed = _multi_band_gate(processed, sr, settings.primary_gate)
    else:
        processed = _apply_spectral_gate(processed, sr, settings.primary_gate)

    if use_wiener:
        processed = _wiener_filter(processed)
    if use_spectral_subtraction:
        processed = _spectral_subtraction(processed, sr, settings.primary_gate)

    processed = _apply_spectral_gate(processed, sr, settings.secondary_gate)
    processed = _limiter(processed, settings.limiter_ceiling)
    processed = _loudness_normalize(processed, settings.target_lufs)
    return processed


__all__ = [
    "SpectralGateConfig",
    "DenoisePreset",
    "PRESETS",
    "denoise",
]

"""Filtering utilities for audio preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to the audio signal."""
    if audio.size == 0:
        return audio

    nyquist = sample_rate / 2
    low = low_hz / nyquist
    high = high_hz / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Invalid bandpass frequencies for the given sample rate")

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, audio)


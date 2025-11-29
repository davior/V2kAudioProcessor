"""Command line interface for processing audio recordings."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from .audio.filters import bandpass_filter
from .audio.io import (
    DEFAULT_SAMPLE_RATE,
    apply_gain_db,
    apply_limiter,
    load_audio,
    normalize_loudness,
)


def denoise(audio, sample_rate):
    """Placeholder for a denoising stage."""
    return audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process audio recordings")
    parser.add_argument("input", type=Path, help="Path to the source audio file")
    parser.add_argument("output", type=Path, help="Path to save the processed audio")
    parser.add_argument(
        "--target-sr",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate to resample audio to (default: 16 kHz)",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=0.0,
        help="Manual preamp (in dB) applied before denoising",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize to EBU R128 (-23 LUFS) after processing",
    )
    parser.add_argument(
        "--bandpass",
        action="store_true",
        help="Apply an 80 Hz–8 kHz Butterworth band-pass filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sr = load_audio(args.input, target_sr=args.target_sr, mono=True)

    # Manual preamp applied before any denoising stage.
    if args.gain_db:
        audio = apply_gain_db(audio, args.gain_db)

    audio = denoise(audio, sr)

    if args.bandpass:
        audio = bandpass_filter(audio, sample_rate=sr)

    if args.normalize:
        audio = normalize_loudness(audio, sample_rate=sr)

    audio = apply_limiter(audio)
    sf.write(args.output, audio, sr)


if __name__ == "__main__":
    main()


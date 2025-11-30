"""Utilities for batch-processing audio files with the denoising chain.

This module provides a simple CLI so datasets or ad-hoc recordings can be
processed reproducibly with consistent settings. It preserves the original
sample rate, applies the configurable denoiser, and writes outputs into a
user-defined directory while keeping relative structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import soundfile as sf

from .denoise import denoise


def _gather_audio_files(sources: Sequence[Path], glob: str) -> List[tuple[Path, Path]]:
    """Return a list of (file_path, relative_parent) pairs to process.

    When a directory is provided, files are gathered recursively using the
    supplied glob pattern and the relative_parent is the path relative to the
    provided directory. For individual files, the relative_parent is ``Path("")``.
    """
    results: List[tuple[Path, Path]] = []
    for source in sources:
        if source.is_dir():
            for file_path in sorted(source.rglob(glob)):
                if file_path.is_file():
                    results.append((file_path, file_path.relative_to(source).parent))
        elif source.is_file():
            results.append((source, Path()))
    return results


def process_audio_file(
    input_path: Path,
    output_path: Path,
    *,
    preset: str = "conservative",
    use_wiener: bool = False,
    use_spectral_subtraction: bool = False,
    use_multiband_gate: bool = False,
) -> Path:
    """Run the denoising chain on a single file and save the output.

    Parameters
    ----------
    input_path:
        Path to the audio file to process.
    output_path:
        Destination for the processed audio.
    preset:
        Denoising preset name (e.g., ``"conservative"`` or ``"aggressive"``).
    use_wiener:
        Enable an additional Wiener filtering pass.
    use_spectral_subtraction:
        Apply spectral subtraction between the two gating passes.
    use_multiband_gate:
        Use a multi-band gate instead of the full-band gate on the first pass.
    """
    y, sr = sf.read(input_path, always_2d=True)
    if y.shape[1] > 1:
        y = np.mean(y, axis=1)
    else:
        y = y[:, 0]
    processed = denoise(
        np.asarray(y, dtype=float),
        sr,
        preset=preset,
        use_wiener=use_wiener,
        use_spectral_subtraction=use_spectral_subtraction,
        use_multiband_gate=use_multiband_gate,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, processed, sr)
    return output_path


def process_batch(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    glob: str = "*.wav",
    preset: str = "conservative",
    use_wiener: bool = False,
    use_spectral_subtraction: bool = False,
    use_multiband_gate: bool = False,
) -> List[Path]:
    """Process many files and return the list of written output paths.

    Parameters
    ----------
    inputs:
        Collection of files or directories to process.
    output_dir:
        Directory where processed files should be written.
    glob:
        Glob pattern used when traversing directories.
    preset:
        Denoising preset name to use.
    use_wiener:
        Enable an additional Wiener filtering stage.
    use_spectral_subtraction:
        Apply a spectral-subtraction pass between gates.
    use_multiband_gate:
        Use the multi-band gate for the first pass instead of full-band gating.
    """
    audio_files = _gather_audio_files(inputs, glob)
    outputs: List[Path] = []
    for input_path, relative_parent in audio_files:
        output_subdir = output_dir / relative_parent
        output_path = output_subdir / f"{input_path.stem}_denoised_{preset}.wav"
        processed_path = process_audio_file(
            input_path,
            output_path,
            preset=preset,
            use_wiener=use_wiener,
            use_spectral_subtraction=use_spectral_subtraction,
            use_multiband_gate=use_multiband_gate,
        )
        outputs.append(processed_path)
    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch denoise audio files with presets.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more audio files or directories containing audio files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("denoised"),
        help="Where processed files should be written (directories are recreated).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.wav",
        help="Glob pattern to search for audio when inputs include directories.",
    )
    parser.add_argument(
        "--preset",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Which denoising preset to use.",
    )
    parser.add_argument(
        "--wiener",
        action="store_true",
        help="Enable an additional Wiener filtering stage between gates.",
    )
    parser.add_argument(
        "--spectral-subtraction",
        action="store_true",
        help="Run spectral subtraction between the two gating passes.",
    )
    parser.add_argument(
        "--multiband-gate",
        action="store_true",
        help="Use the multi-band gate on the first pass (low/mid/high split).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = process_batch(
        args.inputs,
        args.output_dir,
        glob=args.glob,
        preset=args.preset,
        use_wiener=args.wiener,
        use_spectral_subtraction=args.spectral_subtraction,
        use_multiband_gate=args.multiband_gate,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

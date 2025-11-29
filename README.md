# V2kAudioProcessor
Process audio to find voices in it. Used to process V2k recordings and find the audio within it.

## Batch denoising CLI
You can quickly run the spectral denoising pipeline over one or more audio files (or folders full of files) without writing code:

```bash
python -m audio.pipeline path/to/input.wav another_dir/ \
  --output-dir denoised_outputs \
  --preset aggressive \
  --multiband-gate --wiener
```

- Accepts individual files and directories (searched recursively with `--glob`, default `*.wav`).
- Preserves relative paths inside the output directory and appends a `_denoised_<preset>.wav` suffix.
- Supports the same optional stages as the library API: multi-band gating, Wiener filtering, and spectral subtraction.

The underlying functions are available from Python if you want to script your own evaluation against a set of test clips:

```python
from pathlib import Path
from audio import process_batch

outputs = process_batch([Path("tests/inputs")], Path("tests/outputs"), preset="conservative")
for out in outputs:
    print(out)
## Overview

The processing pipeline loads an input file, optionally applies a manual gain
boost, runs a placeholder denoiser, and can then normalize loudness and/or add a
band-pass filter before saving the result. Loading prefers `soundfile` and
falls back to `ffmpeg` for uncommon codecs. Audio is resampled to a fixed sample
rate (default 16 kHz) and converted to mono for consistent downstream
processing.

## Command line interface

Run the CLI with:

```
python -m src.cli INPUT OUTPUT [--target-sr 16000] [--gain-db 3.0] [--normalize] [--bandpass]
```

Available options are also listed in `--help` output:

* `--target-sr` – resample input to this rate (defaults to 16 kHz).
* `--gain-db` – apply a manual preamp (in decibels) before denoising.
* `--normalize` – normalize loudness to the EBU R128 target (-23 LUFS).
* `--bandpass` – apply an 80 Hz–8 kHz Butterworth band-pass filter.

Example:

```
python -m src.cli noisy.wav cleaned.wav --gain-db 6 --normalize --bandpass
```

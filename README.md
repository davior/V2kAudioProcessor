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
```

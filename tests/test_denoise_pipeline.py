import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure src is importable when running tests from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio.denoise import PRESETS, denoise  # noqa: E402
from audio.pipeline import process_audio_file, process_batch  # noqa: E402


def _synthesize_noise(sr: int, duration_s: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    speech_like = 0.1 * np.sin(2 * np.pi * 220 * t)
    noise = 0.02 * np.random.randn(len(t))
    return speech_like + noise


def test_denoise_runs_for_all_presets(tmp_path: Path) -> None:
    sr = 16000
    audio = _synthesize_noise(sr)
    for preset in PRESETS:
        processed = denoise(audio, sr, preset=preset)
        assert processed.shape == audio.shape
        assert np.isfinite(processed).all()
        assert np.max(np.abs(processed)) <= 1.0 + 1e-6


def test_denoise_optional_stages(tmp_path: Path) -> None:
    sr = 22050
    audio = _synthesize_noise(sr)
    processed = denoise(
        audio,
        sr,
        preset="conservative",
        use_wiener=True,
        use_spectral_subtraction=True,
        use_multiband_gate=True,
    )
    assert processed.shape == audio.shape
    assert np.isfinite(processed).all()


def test_pipeline_writes_outputs(tmp_path: Path) -> None:
    sr = 16000
    audio = _synthesize_noise(sr)
    input_path = tmp_path / "input.wav"
    sf.write(input_path, audio, sr)

    output_path = tmp_path / "out" / "input_denoised_conservative.wav"
    written = process_audio_file(input_path, output_path)
    assert written.exists()

    batch_outputs = process_batch([tmp_path], tmp_path / "batch", preset="aggressive")
    assert batch_outputs, "Expected at least one processed file from batch run"
    for path in batch_outputs:
        assert path.exists()

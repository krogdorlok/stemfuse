"""
Regression tests for pipeline.py's run_pipeline() orchestration.

These exercise the real call sequence pipeline.py uses (parse_prompt ->
separate -> apply_all_transformations -> mix_stems), with parse_prompt and
separate mocked so the tests are fast/deterministic - they exist to catch
bugs in how pipeline.py wires those calls together, not to test
parse_prompt/separate themselves (those need real audio or a live API key,
see scripts/).
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch

import pipeline
from schemas.dsp_parameters import (
    StemType, StemTransformation, OrchestrationRequest, StemSeparationRequest,
    MixParameters
)

SR = 44100
DURATION_S = 2.0
STEM_FREQS = {"vocals": 440.0, "drums": 60.0, "bass": 110.0, "other": 880.0}
AMP = 0.3


def _fft_magnitude_at(audio: np.ndarray, sr: int, frequency: float) -> float:
    """Magnitude of audio's FFT at the bin closest to `frequency`."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    bin_idx = np.argmin(np.abs(freqs - frequency))
    return spectrum[bin_idx]


def _write_synthetic_stems(tmpdir) -> tuple[dict, dict]:
    """Distinct-frequency sine wave per stem, so each stem's contribution
    to the final mix can be isolated by FFT bin."""
    stems = {}
    raw_audio = {}
    t = np.linspace(0, DURATION_S, int(DURATION_S * SR), endpoint=False)
    for name, freq in STEM_FREQS.items():
        audio = AMP * np.sin(2 * np.pi * freq * t)
        raw_audio[name] = audio
        path = Path(tmpdir) / f"{name}.wav"
        sf.write(path, audio, SR)
        stems[name] = str(path)
    return stems, raw_audio


class TestVolumeAppliedOnce:
    """
    Regression test for: volume_db must be applied exactly once.

    apply_transformation() already bakes volume_db into the transformed
    stem file (its step 5 of 5). Before the fix, pipeline.py independently
    rebuilt a second linear-gain dict from the same volume_db values and
    passed it into mix_stems(), multiplying by that gain again on top of
    the already-attenuated audio - measured -5.05dB delivered against a
    -3.0dB request during the audit that found this bug (single-application
    control case measured -2.45dB, confirming double-application as the
    cause of the ~2dB overshoot, not a measurement artifact).

    Measurement technique: drums has no volume_db set (gain=1, unchanged).
    Taking the ratio of vocals' FFT magnitude to drums' FFT magnitude in
    the final mix - versus that same ratio in the raw pre-pipeline stems -
    cancels out mix_stems's shared peak-normalization factor (which applies
    identically to every stem in a given mix and would otherwise mask the
    per-stem gain in aggregate level), isolating the actual gain delivered
    to vocals specifically.
    """

    def test_volume_db_delivered_within_half_db(self, tmp_path):
        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()
        stems, raw_audio = _write_synthetic_stems(stems_dir)

        request = OrchestrationRequest(
            source_audio_path="input.wav",
            separation_request=StemSeparationRequest(
                source_path="input.wav", output_dir=str(tmp_path)
            ),
            stem_transformations=[
                StemTransformation(stem_type=StemType.VOCALS, volume_db=-3.0),
                StemTransformation(stem_type=StemType.DRUMS),
                StemTransformation(stem_type=StemType.BASS),
                StemTransformation(stem_type=StemType.OTHER),
            ],
            output_path=str(tmp_path / "output.wav"),
        )
        output_file = str(tmp_path / "output.wav")

        with patch("pipeline.parse_prompt", return_value=request), \
             patch("pipeline.separate", return_value=stems):
            pipeline.run_pipeline("input.wav", "irrelevant prompt", output_file)

        mixed, sr = sf.read(output_file)

        mixed_vocals_mag = _fft_magnitude_at(mixed, sr, STEM_FREQS["vocals"])
        mixed_drums_mag = _fft_magnitude_at(mixed, sr, STEM_FREQS["drums"])
        raw_vocals_mag = _fft_magnitude_at(raw_audio["vocals"], SR, STEM_FREQS["vocals"])
        raw_drums_mag = _fft_magnitude_at(raw_audio["drums"], SR, STEM_FREQS["drums"])

        measured_gain = (mixed_vocals_mag / mixed_drums_mag) / (raw_vocals_mag / raw_drums_mag)
        delivered_db = 20 * np.log10(measured_gain)

        assert abs(delivered_db - (-3.0)) < 0.5, (
            f"delivered {delivered_db:.2f}dB, requested -3.0dB "
            "(if this is roughly double, volume_db is being applied twice)"
        )


class TestMixParametersPassthrough:
    """
    Regression test for: mix_parameters is built by parse_prompt() on every
    request (real defaults: master_volume_db=-3.0, stem_weights={}), but
    pipeline.py never read it - it hardcoded master_volume_db=-3.0 directly
    into the mix_stems() call regardless of what the request contained.
    """

    def test_master_volume_db_from_request_not_hardcoded(self, tmp_path):
        stems_dir = tmp_path / "stems"
        stems_dir.mkdir()
        stems, _ = _write_synthetic_stems(stems_dir)

        request = OrchestrationRequest(
            source_audio_path="input.wav",
            separation_request=StemSeparationRequest(
                source_path="input.wav", output_dir=str(tmp_path)
            ),
            stem_transformations=[
                StemTransformation(stem_type=StemType.VOCALS),
                StemTransformation(stem_type=StemType.DRUMS),
                StemTransformation(stem_type=StemType.BASS),
                StemTransformation(stem_type=StemType.OTHER),
            ],
            mix_parameters=MixParameters(master_volume_db=-10.0),
            output_path=str(tmp_path / "output.wav"),
        )
        output_file = str(tmp_path / "output.wav")

        with patch("pipeline.parse_prompt", return_value=request), \
             patch("pipeline.separate", return_value=stems):
            pipeline.run_pipeline("input.wav", "irrelevant prompt", output_file)

        mixed, sr = sf.read(output_file)
        peak_db = 20 * np.log10(np.max(np.abs(mixed)))
        assert abs(peak_db - (-10.0)) < 0.5, f"peak {peak_db:.2f}dB, expected -10.0dB (not -3.0dB default)"

"""Unit tests for dsp_processing.mixer.mix_stems()."""

import numpy as np
import soundfile as sf

from dsp_processing.mixer import mix_stems

SR = 44100
DURATION_S = 2.0


def _fft_magnitude_at(audio: np.ndarray, sr: int, frequency: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    bin_idx = np.argmin(np.abs(freqs - frequency))
    return spectrum[bin_idx]


def _write_tone(path, frequency: float, amplitude: float) -> None:
    t = np.linspace(0, DURATION_S, int(DURATION_S * SR), endpoint=False)
    sf.write(path, amplitude * np.sin(2 * np.pi * frequency * t), SR)


class TestMixStemsVolumes:
    """
    pipeline.py (Task 4) populates mix_stems's `volumes` dict from
    mix_parameters.stem_weights - a balance-mixing knob applied at mix
    time, distinct from per-stem volume_db (already baked into each
    transformed stem file before it reaches mix_stems). This is the first
    direct unit coverage `volumes` has had; it was previously only
    exercised indirectly through mocks or full pipeline runs.
    """

    def test_volumes_dict_applies_measurable_per_stem_gain(self, tmp_path):
        a_path = tmp_path / "a.wav"
        b_path = tmp_path / "b.wav"
        _write_tone(a_path, 440.0, 0.3)
        _write_tone(b_path, 110.0, 0.3)

        output = tmp_path / "output.wav"
        mix_stems(
            {"a": str(a_path), "b": str(b_path)},
            str(output),
            volumes={"a": 1.0, "b": 0.25},
        )

        mixed, sr = sf.read(output)
        mag_a = _fft_magnitude_at(mixed, sr, 440.0)
        mag_b = _fft_magnitude_at(mixed, sr, 110.0)

        # Both stems start at identical amplitude; a 1.0 : 0.25 (4x) weight
        # ratio should be reflected in their mixed magnitudes.
        ratio = mag_a / mag_b
        assert 3.5 < ratio < 4.5, f"expected ~4x, got {ratio:.2f}x"

    def test_empty_volumes_dict_behaves_as_unity_gain(self, tmp_path):
        """An empty stem_weights (the schema default) must not silence anything."""
        a_path = tmp_path / "a.wav"
        b_path = tmp_path / "b.wav"
        _write_tone(a_path, 440.0, 0.3)
        _write_tone(b_path, 110.0, 0.3)

        output = tmp_path / "output.wav"
        mix_stems({"a": str(a_path), "b": str(b_path)}, str(output), volumes={})

        mixed, sr = sf.read(output)
        mag_a = _fft_magnitude_at(mixed, sr, 440.0)
        mag_b = _fft_magnitude_at(mixed, sr, 110.0)
        assert 0.8 < (mag_a / mag_b) < 1.25, "equal-amplitude stems should stay roughly equal"

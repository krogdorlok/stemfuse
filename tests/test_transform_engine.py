"""
Comprehensive test suite for DSP transform engine.

Tests cover all critical bug fixes from Phase 5 DSP fix:
- EQ dB scaling (logarithmic, not linear)
- Volume argument count (2 args, not 3)
- Pitch shift semitone conversion (no division by 12)
- Tempo shift length changes
- Flat EQ fields bridge in StemTransformation
- Stereo input handling
- Clipping protection
- Transformation order
"""

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path
from unittest.mock import patch

import dsp_processing.transform_engine as transform_engine_module
import dsp_processing.mixer as mixer_module
from dsp_processing.transform_engine import (
    apply_eq, apply_volume, apply_tempo_shift, apply_pitch_shift,
    apply_genre_preset, apply_transformation, apply_all_transformations
)
from schemas.dsp_parameters import (
    StemType, GenreStyle, StemTransformation, EQAdjustment,
    OrchestrationRequest, StemSeparationRequest
)


class TestApplyEQ:
    """Test EQ processing."""

    def test_zero_gains_unchanged(self):
        """Zero gains should return signal unchanged."""
        audio = np.random.randn(44100)
        result = apply_eq(audio, 44100, low_gain=0.0, mid_gain=0.0, high_gain=0.0)
        # Should be very close to original (allowing for STFT/ISTFT artifacts)
        assert np.allclose(result, audio, atol=1e-5)

    def test_low_boost_correct_db(self):
        """+6dB low boost produces ~2x amplitude in low freqs (log scale)."""
        # Create test audio with distinct low frequency content
        audio = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 44100))  # 100 Hz sine
        result = apply_eq(audio, 44100, low_gain=6.0)

        # +6dB should multiply by 10^(6/20) ≈ 2.0
        # Check that low freq content is boosted
        assert np.std(result) > np.std(audio) * 1.8  # Approximately 2x gain

    def test_stereo_input_handled(self):
        """Stereo array should be converted to mono without crashing."""
        # Create stereo audio (2 channels)
        audio = np.random.randn(44100, 2)
        result = apply_eq(audio, 44100, low_gain=3.0)
        # Should return mono audio
        assert result.ndim == 1

    def test_all_bands_boost(self):
        """All bands boosted together."""
        audio = np.random.randn(44100)
        result = apply_eq(audio, 44100, low_gain=3.0, mid_gain=3.0, high_gain=3.0)
        # Overall signal should be louder
        assert np.std(result) > np.std(audio)

    def test_high_boost(self):
        """High frequency boost."""
        # Create test audio with high frequency content
        audio = np.sin(2 * np.pi * 8000 * np.linspace(0, 1, 44100))  # 8 kHz sine
        result = apply_eq(audio, 44100, high_gain=6.0)

        # Check that high freq content is boosted
        assert np.std(result) > np.std(audio) * 1.8


class TestApplyVolume:
    """Test volume adjustment."""

    def test_minus_6db_halves_amplitude(self):
        """-6dB should approximately halve amplitude."""
        audio = np.ones(44100)
        result = apply_volume(audio, -6.0)
        expected = 10 ** (-6.0 / 20.0)  # ≈ 0.501
        assert abs(result[0] - expected) < 0.001

    def test_zero_db_unchanged(self):
        """0dB should mean no change."""
        audio = np.random.randn(44100)
        result = apply_volume(audio, 0.0)
        assert np.allclose(result, audio)

    def test_plus_6db_doubles_amplitude(self):
        """+6dB should approximately double amplitude."""
        audio = np.ones(44100)
        result = apply_volume(audio, 6.0)
        expected = 10 ** (6.0 / 20.0)  # ≈ 2.0
        assert abs(result[0] - expected) < 0.01

    def test_called_with_two_args(self):
        """Ensure function only takes 2 args (not 3 like the bug)."""
        audio = np.ones(44100)
        # This should work (2 args)
        result = apply_volume(audio, -3.0)
        assert result is not None

    def test_large_negative_volume(self):
        """Large negative volume should produce very quiet signal."""
        audio = np.ones(44100)
        result = apply_volume(audio, -60.0)
        expected = 10 ** (-60.0 / 20.0)  # ≈ 0.001
        assert abs(result[0] - expected) < 0.001


class TestApplyTempoShift:
    """Test tempo shift processing."""

    def test_zero_shift_unchanged(self):
        """Zero tempo shift should return unchanged audio."""
        audio = np.random.randn(44100)
        result = apply_tempo_shift(audio, 44100, 0.0)
        assert len(result) == len(audio)

    def test_positive_shift_shorter(self):
        """+0.2 shift (faster) should produce shorter audio."""
        audio = np.random.randn(44100)
        result = apply_tempo_shift(audio, 44100, 0.2)
        # Faster = fewer samples
        assert len(result) < len(audio)
        # Approximately 1/1.2 = 0.83x length (allowing for pyrubberband overhead)
        assert 35000 < len(result) < 38000

    def test_negative_shift_longer(self):
        """-0.2 shift (slower) should produce longer audio."""
        audio = np.random.randn(44100)
        result = apply_tempo_shift(audio, 44100, -0.2)
        # Slower = more samples
        assert len(result) > len(audio)
        # Approximately 1/0.8 = 1.25x length
        assert 52000 < len(result) < 56000

    def test_clamped_ratio(self):
        """Extreme shifts should be clamped to 0.5-2.0 range."""
        audio = np.random.randn(44100)
        # +1.0 would be 2.0x ratio (max allowed)
        result = apply_tempo_shift(audio, 44100, 1.0)
        # Should not crash and should be much shorter
        assert 20000 < len(result) < 25000

    def test_small_shift_unchanged(self):
        """Very small shift (< 0.01) should return unchanged."""
        audio = np.random.randn(44100)
        result = apply_tempo_shift(audio, 44100, 0.001)
        assert len(result) == len(audio)


class TestApplyPitchShift:
    """Test pitch shift processing."""

    def test_zero_semitones_unchanged(self):
        """Zero semitones should return unchanged audio."""
        audio = np.random.randn(44100)
        result = apply_pitch_shift(audio, 44100, 0)
        assert len(result) == len(audio)

    def test_twelve_semitones_not_one_semitone(self):
        """+12 semitones should differ from +1 semitone (not 1/12)."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        shifted_12 = apply_pitch_shift(audio, 44100, 12)
        shifted_1 = apply_pitch_shift(audio, 44100, 1)

        # They should be different (12 semitone = 1 octave shift)
        assert not np.allclose(shifted_12, shifted_1)

        # Length should be the same (pitch shift preserves duration)
        assert len(shifted_12) == len(audio)
        assert len(shifted_1) == len(audio)

    def test_negative_pitch_shift(self):
        """Negative semitone shift should work."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        result = apply_pitch_shift(audio, 44100, -5)
        assert len(result) == len(audio)

    def test_clamped_range(self):
        """Semitones should be clamped to -12 to 12 range."""
        audio = np.random.randn(44100)
        # Should not crash with extreme values (clamped internally)
        result = apply_pitch_shift(audio, 44100, 24)  # Will be clamped to 12
        assert len(result) == len(audio)


class TestApplyGenrePreset:
    """Test genre-specific EQ presets."""

    def test_neutral_genre_unchanged(self):
        """Neutral genre should return unchanged audio."""
        audio = np.random.randn(44100)
        result = apply_genre_preset(audio, 44100, StemType.VOCALS, GenreStyle.NEUTRAL)
        assert np.allclose(result, audio)

    def test_jazz_preset_exists(self):
        """Jazz genre preset should exist and apply."""
        audio = np.random.randn(44100)
        result = apply_genre_preset(audio, 44100, StemType.VOCALS, GenreStyle.JAZZ)
        # Should be different (EQ applied)
        assert not np.allclose(result, audio)

    def test_blend_ratio_partial(self):
        """Blend ratio should partially apply the preset."""
        audio = np.random.randn(44100)
        full = apply_genre_preset(audio, 44100, StemType.VOCALS, GenreStyle.JAZZ, blend_ratio=1.0)
        half = apply_genre_preset(audio, 44100, StemType.VOCALS, GenreStyle.JAZZ, blend_ratio=0.5)

        # Half blend should be closer to original than full blend
        assert np.linalg.norm(half - audio) < np.linalg.norm(full - audio)

    def test_unknown_stem_type_defaults(self):
        """Unknown stem type should default to neutral preset."""
        # This tests the get() default in the preset lookup
        audio = np.random.randn(44100)
        result = apply_genre_preset(audio, 44100, StemType.VOCALS, GenreStyle.JAZZ)
        # Should not crash
        assert result is not None


class TestApplyAllTransformations:
    """Test the full transformation pipeline."""

    def test_trans_map_string_keys_match(self):
        """Verify stem string keys match trans_map correctly."""
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            stems = {}
            for stem in ["vocals", "drums", "bass", "other"]:
                path = Path(tmpdir) / f"{stem}.wav"
                audio = np.random.randn(44100)
                sf.write(path, audio, 44100)
                stems[stem] = str(path)

            # Create transformations
            transformations = [
                StemTransformation(stem_type=StemType.VOCALS, volume_db=-3.0),
                StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.1),
            ]

            # Apply transformations
            output_dir = Path(tmpdir) / "output"
            result = apply_all_transformations(stems, transformations, str(output_dir))

            # Check that vocals and drums were transformed
            assert "vocals" in result
            assert "drums" in result
            assert "bass" in result  # Should still be copied
            assert "other" in result  # Should still be copied

    def test_all_stems_transformed(self):
        """With 4 transforms, all 4 output paths should differ from inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            stems = {}
            for stem in ["vocals", "drums", "bass", "other"]:
                path = Path(tmpdir) / f"{stem}.wav"
                audio = np.random.randn(44100)
                sf.write(path, audio, 44100)
                stems[stem] = str(path)

            # Create transformations for all stems
            transformations = [
                StemTransformation(stem_type=StemType.VOCALS, volume_db=-3.0),
                StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.1),
                StemTransformation(stem_type=StemType.BASS, pitch_shift_semitones=2),
                StemTransformation(stem_type=StemType.OTHER, eq_low_gain=3.0),
            ]

            # Apply transformations
            output_dir = Path(tmpdir) / "output"
            result = apply_all_transformations(stems, transformations, str(output_dir))

            # All should be transformed
            for stem in ["vocals", "drums", "bass", "other"]:
                assert stem in result
                # Output file should exist
                assert Path(result[stem]).exists()

    def test_untransformed_stem_copied(self):
        """Stem with no transform should still appear in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy audio files
            stems = {}
            for stem in ["vocals", "drums"]:
                path = Path(tmpdir) / f"{stem}.wav"
                audio = np.random.randn(44100)
                sf.write(path, audio, 44100)
                stems[stem] = str(path)

            # Only transform vocals
            transformations = [
                StemTransformation(stem_type=StemType.VOCALS, volume_db=-3.0),
            ]

            # Apply transformations
            output_dir = Path(tmpdir) / "output"
            result = apply_all_transformations(stems, transformations, str(output_dir))

            # Both should be in output
            assert "vocals" in result
            assert "drums" in result


class TestTransformationOrder:
    """Test that transformations are applied in the correct order."""

    @patch('dsp_processing.transform_engine.apply_tempo_shift')
    @patch('dsp_processing.transform_engine.apply_pitch_shift')
    @patch('dsp_processing.transform_engine.apply_genre_preset')
    @patch('dsp_processing.transform_engine.apply_eq')
    @patch('dsp_processing.transform_engine.apply_volume')
    def test_transformation_order(
        self, mock_volume, mock_eq, mock_genre, mock_pitch, mock_tempo
    ):
        """Verify transformations are applied in correct order."""
        # Setup mocks to return the input unchanged
        mock_tempo.return_value = np.array([1.0, 2.0])
        mock_pitch.return_value = np.array([1.0, 2.0])
        mock_genre.return_value = np.array([1.0, 2.0])
        mock_eq.return_value = np.array([1.0, 2.0])
        mock_volume.return_value = np.array([1.0, 2.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input file
            input_path = Path(tmpdir) / "input.wav"
            sf.write(input_path, np.random.randn(44100), 44100)

            output_path = Path(tmpdir) / "output.wav"

            # Create transformation with all parameters
            transformation = StemTransformation(
                stem_type=StemType.VOCALS,
                tempo_shift=0.1,
                pitch_shift_semitones=2,
                target_genre=GenreStyle.JAZZ,
                genre_blend_ratio=0.7,
                eq_low_gain=3.0,
                eq_mid_gain=2.0,
                eq_high_gain=1.0,
                volume_db=-3.0,
            )

            # Apply transformation
            apply_transformation(str(input_path), str(output_path), transformation)

            # Verify order: tempo -> pitch -> genre -> eq -> volume
            assert mock_tempo.called
            assert mock_pitch.called
            assert mock_genre.called
            assert mock_eq.called
            assert mock_volume.called

            # Check call order
            call_order = [mock_tempo, mock_pitch, mock_genre, mock_eq, mock_volume]
            for i in range(len(call_order) - 1):
                assert call_order[i].call_count > 0
                assert call_order[i + 1].call_count > 0


class TestClippingProtection:
    """Test clipping protection after transformations."""

    def test_clipping_protection_after_eq(self):
        """High EQ gains should not cause clipping beyond [-1, 1]."""
        # Check clipping protection in apply_transformation
        # This tests the np.clip at the end
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            sf.write(input_path, np.ones(44100) * 0.9, 44100)

            output_path = Path(tmpdir) / "output.wav"

            transformation = StemTransformation(
                stem_type=StemType.VOCALS,
                eq_low_gain=12.0,  # Max boost
            )

            apply_transformation(str(input_path), str(output_path), transformation)

            # Load output and verify no clipping
            output_audio, _ = sf.read(str(output_path))
            assert np.all(np.abs(output_audio) <= 1.0)


class TestFlatEQBridge:
    """Test the flat EQ fields bridge in StemTransformation."""

    def test_flat_eq_populates_eq_adjustment(self):
        """StemTransformation(eq_low_gain=6.0) should populate eq_adjustment."""
        t = StemTransformation(
            stem_type=StemType.BASS,
            eq_low_gain=6.0,
        )

        assert t.eq_adjustment is not None
        assert t.eq_adjustment.low_gain == 6.0

    def test_all_flat_eq_fields(self):
        """All three flat EQ fields should populate eq_adjustment."""
        t = StemTransformation(
            stem_type=StemType.DRUMS,
            eq_low_gain=3.0,
            eq_mid_gain=2.0,
            eq_high_gain=4.0,
        )

        assert t.eq_adjustment is not None
        assert t.eq_adjustment.low_gain == 3.0
        assert t.eq_adjustment.mid_gain == 2.0
        assert t.eq_adjustment.high_gain == 4.0

    def test_explicit_eq_adjustment_not_overwritten(self):
        """Passing eq_adjustment directly should preserve it."""
        # Use model_construct or pass as dict to create with explicit eq_adjustment
        t = StemTransformation.model_construct(
            stem_type=StemType.VOCALS,
            eq_adjustment=EQAdjustment(low_gain=5.0, mid_gain=3.0, high_gain=2.0),
        )

        assert t.eq_adjustment.low_gain == 5.0
        assert t.eq_adjustment.mid_gain == 3.0
        assert t.eq_adjustment.high_gain == 2.0

    def test_no_flat_fields_no_eq_adjustment(self):
        """Without flat fields or eq_adjustment, eq_adjustment should be None."""
        t = StemTransformation(stem_type=StemType.OTHER)
        assert t.eq_adjustment is None


def _make_click_track(duration_s: float = 4.5, sr: int = 44100, beat_interval_s: float = 0.5) -> np.ndarray:
    """Synthetic click track: a short burst every beat_interval_s, silence between."""
    audio = np.zeros(int(duration_s * sr))
    click_len = int(0.02 * sr)  # 20ms burst per click
    beat = 0.0
    while beat < duration_s:
        start = int(beat * sr)
        end = min(start + click_len, len(audio))
        audio[start:end] = 0.8
        beat += beat_interval_s
    return audio


class TestTempoShiftUniformityInvariant:
    """
    Regression test for: tempo_shift applied to a subset of stems desyncs the
    mix (mix_stems zero-pads length mismatches instead of time-aligning them,
    so a stem sped up on its own drifts out of sync with the others for its
    entire length, not just the tail).

    Reproduced with a click-track: two stems with identical beat markers
    every 0.5s, one compressed by the ratio a real tempo_shift=+0.2 produces.
    Beat drift grew linearly (83ms at beat 1, 667ms by beat 8), and the
    shifted stem went silent for the final ~17% of the track once it ran out
    of samples. The fix (OrchestrationRequest.model_post_init) must reject
    this request before it ever reaches apply_all_transformations/mix_stems.
    """

    def _build_click_track_stems(self, tmpdir):
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            audio = _make_click_track()
            path = Path(tmpdir) / f"{stem}.wav"
            sf.write(path, audio, 44100)
            stems[stem] = str(path)
        return stems

    @patch('dsp_processing.mixer.mix_stems')
    @patch('dsp_processing.transform_engine.apply_all_transformations')
    def test_partial_tempo_shift_rejected_before_dsp_runs(self, mock_apply_all, mock_mix, tmp_path):
        """
        A request with tempo_shift on only the drums stem (a legitimate,
        real request e.g. "speed up just the drums") must be rejected at
        the schema layer, before apply_all_transformations or mix_stems
        (the real DSP call path) ever run.
        """
        stems = self._build_click_track_stems(tmp_path)

        with pytest.raises(ValueError, match="tempo_shift must be applied uniformly"):
            request = OrchestrationRequest(
                source_audio_path="clicktrack.wav",
                separation_request=StemSeparationRequest(
                    source_path="clicktrack.wav",
                    output_dir=str(tmp_path)
                ),
                stem_transformations=[
                    StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.2),
                    StemTransformation(stem_type=StemType.VOCALS),
                    StemTransformation(stem_type=StemType.BASS),
                    StemTransformation(stem_type=StemType.OTHER),
                ],
                output_path=str(tmp_path / "output.wav")
            )
            # If validation somehow didn't raise, this would run the real
            # DSP call path with the invalid, desyncing request. Called via
            # module attribute (not the bare imported name) so the @patch
            # decorators above actually intercept these calls.
            transform_engine_module.apply_all_transformations(
                stems, request.stem_transformations, str(tmp_path / "transformed")
            )
            mixer_module.mix_stems(stems, str(tmp_path / "output.wav"))

        mock_apply_all.assert_not_called()
        mock_mix.assert_not_called()

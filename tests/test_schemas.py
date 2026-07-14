"""
Unit tests for DSP parameter schemas.

Tests validate:
1. Valid parameters pass validation
2. Out-of-bounds parameters fail with ValidationError
3. Edge cases at boundary values
4. Model post-init validation logic
"""

import pytest
import sys
from schemas.dsp_parameters import (
    StemType,
    GenreStyle,
    TempoConstraints,
    TimeStretchConstraints,
    EQAdjustment,
    StemTransformation,
    BeatGridAlignment,
    MixParameters,
    StemSeparationRequest,
    OrchestrationRequest
)
from pydantic import ValidationError


class TestTempoConstraints:
    """Test tempo constraint validation"""

    def test_valid_tempo_range(self):
        """Valid tempo range passes"""
        params = TempoConstraints(min_bpm=80.0, max_bpm=140.0)
        assert params.min_bpm == 80.0
        assert params.max_bpm == 140.0

    def test_tempo_bounds_at_limits(self):
        """Boundary values (60-200 BPM) pass"""
        params = TempoConstraints(min_bpm=60.0, max_bpm=200.0)
        assert params.min_bpm == 60.0
        assert params.max_bpm == 200.0

    def test_tempo_min_too_low(self):
        """Below 60 BPM fails"""
        with pytest.raises(ValidationError):
            TempoConstraints(min_bpm=50.0, max_bpm=140.0)

    def test_tempo_max_too_high(self):
        """Above 200 BPM fails"""
        with pytest.raises(ValidationError):
            TempoConstraints(min_bpm=80.0, max_bpm=220.0)

    def test_tempo_min_exceeds_max(self):
        """min_bpm > max_bpm fails"""
        with pytest.raises(ValueError):
            TempoConstraints(min_bpm=150.0, max_bpm=100.0)


class TestTimeStretchConstraints:
    """Test time-stretch constraint validation"""

    def test_default_ratio(self):
        """Default ratio is 1.0 (no change)"""
        params = TimeStretchConstraints()
        assert params.ratio == 1.0

    def test_valid_stretch_ratio(self):
        """Valid stretch ratios within bounds"""
        params = TimeStretchConstraints(ratio=1.5)
        assert params.ratio == 1.5

    def test_stretch_at_boundary_limits(self):
        """Boundary values (0.5-2.0) pass"""
        params_min = TimeStretchConstraints(ratio=0.5)
        params_max = TimeStretchConstraints(ratio=2.0)
        assert params_min.ratio == 0.5
        assert params_max.ratio == 2.0

    def test_stretch_ratio_too_low(self):
        """Below 0.5 fails"""
        with pytest.raises(ValidationError):
            TimeStretchConstraints(ratio=0.3)

    def test_stretch_ratio_too_high(self):
        """Above 2.0 fails"""
        with pytest.raises(ValidationError):
            TimeStretchConstraints(ratio=3.0)


class TestEQAdjustment:
    """Test EQ adjustment validation"""

    def test_eq_all_bands(self):
        """All EQ bands within bounds"""
        eq = EQAdjustment(low_gain=-6.0, mid_gain=3.0, high_gain=9.0)
        assert eq.low_gain == -6.0
        assert eq.mid_gain == 3.0
        assert eq.high_gain == 9.0

    def test_eq_at_boundary_limits(self):
        """Boundary values (+/- 12dB) pass"""
        eq = EQAdjustment(low_gain=-12.0, mid_gain=12.0)
        assert eq.low_gain == -12.0
        assert eq.mid_gain == 12.0

    def test_eq_exceeds_bounds(self):
        """Beyond +/- 12dB fails"""
        with pytest.raises(ValidationError):
            EQAdjustment(low_gain=-15.0)

    def test_eq_optional_bands(self):
        """EQ bands are optional"""
        eq = EQAdjustment(low_gain=6.0)
        assert eq.low_gain == 6.0
        assert eq.mid_gain is None
        assert eq.high_gain is None


class TestStemTransformation:
    """Test stem transformation validation"""

    def test_valid_transformation(self):
        """Valid stem transformation passes"""
        params = StemTransformation(
            stem_type=StemType.DRUMS,
            target_genre=GenreStyle.JAZZ,
            tempo_shift=0.2
        )
        assert params.stem_type == StemType.DRUMS
        assert params.target_genre == GenreStyle.JAZZ
        assert params.tempo_shift == 0.2

    def test_tempo_shift_at_boundaries(self):
        """Boundary tempo shifts (+/- 0.5) pass"""
        params_min = StemTransformation(
            stem_type=StemType.VOCALS,
            tempo_shift=-0.5
        )
        params_max = StemTransformation(
            stem_type=StemType.BASS,
            tempo_shift=0.5
        )
        assert params_min.tempo_shift == -0.5
        assert params_max.tempo_shift == 0.5

    def test_tempo_shift_exceeds_bounds(self):
        """Beyond +/- 0.5 fails"""
        with pytest.raises(ValidationError):
            StemTransformation(
                stem_type=StemType.DRUMS,
                tempo_shift=1.5
            )

    def test_pitch_shift_at_boundaries(self):
        """Boundary pitch shifts (+/- 12 semitones) pass"""
        params = StemTransformation(
            stem_type=StemType.VOCALS,
            pitch_shift_semitones=12
        )
        assert params.pitch_shift_semitones == 12

    def test_pitch_shift_exceeds_bounds(self):
        """Beyond +/- 12 semitones fails"""
        with pytest.raises(ValidationError):
            StemTransformation(
                stem_type=StemType.BASS,
                pitch_shift_semitones=15
            )

    def test_volume_db_at_boundaries(self):
        """Boundary volume levels (-60 to 0 dB) pass"""
        params = StemTransformation(
            stem_type=StemType.OTHER,
            volume_db=-60.0
        )
        assert params.volume_db == -60.0

    def test_genre_blend_ratio_bounds(self):
        """Genre blend ratio (0.0-1.0) enforced"""
        params_min = StemTransformation(
            stem_type=StemType.DRUMS,
            genre_blend_ratio=0.0
        )
        params_max = StemTransformation(
            stem_type=StemType.DRUMS,
            genre_blend_ratio=1.0
        )
        assert params_min.genre_blend_ratio == 0.0
        assert params_max.genre_blend_ratio == 1.0


class TestBeatGridAlignment:
    """Test beat grid alignment validation"""

    def test_default_alignment(self):
        """Default alignment uses detected BPM"""
        align = BeatGridAlignment()
        assert align.target_bpm is None
        assert align.time_stretch_method == "rubberband"

    def test_valid_target_bpm(self):
        """Valid target BPM within bounds"""
        align = BeatGridAlignment(target_bpm=120.0)
        assert align.target_bpm == 120.0

    def test_target_bpm_out_of_bounds(self):
        """Target BPM outside 60-200 fails"""
        with pytest.raises(ValidationError):
            BeatGridAlignment(target_bpm=250.0)

    def test_invalid_stretch_method(self):
        """Invalid stretch method fails"""
        with pytest.raises(ValidationError):
            BeatGridAlignment(time_stretch_method="invalid_method")


class TestMixParameters:
    """Test mix parameters validation"""

    def test_default_mix(self):
        """Default mix parameters"""
        mix = MixParameters()
        assert mix.master_volume_db == -3.0
        assert mix.compression_threshold_db is None

    def test_stem_weights_dict(self):
        """Stem weights as dict"""
        mix = MixParameters(
            stem_weights={
                StemType.VOCALS: 1.2,
                StemType.DRUMS: 0.8
            }
        )
        assert mix.stem_weights[StemType.VOCALS] == 1.2

    def test_stem_weight_out_of_bounds(self):
        """Stem weight beyond 0.0-2.0 fails"""
        with pytest.raises(ValidationError):
            MixParameters(
                stem_weights={StemType.VOCALS: 3.0}
            )

    def test_master_volume_bounds(self):
        """Master volume within -60 to 0 dB"""
        mix = MixParameters(master_volume_db=-10.0)
        assert mix.master_volume_db == -10.0

    def test_compression_threshold_bounds(self):
        """Compression threshold within bounds"""
        mix = MixParameters(compression_threshold_db=-20.0)
        assert mix.compression_threshold_db == -20.0


class TestStemSeparationRequest:
    """Test stem separation request validation"""

    def test_valid_separation_request(self):
        """Valid separation request"""
        req = StemSeparationRequest(
            source_path="/path/to/song.mp3",
            output_dir="/path/to/output"
        )
        assert req.model == "htdemucs"

    def test_custom_model(self):
        """Custom Demucs model"""
        req = StemSeparationRequest(
            source_path="/path/to/song.mp3",
            output_dir="/path/to/output",
            model="htdemucs_ft"
        )
        assert req.model == "htdemucs_ft"


class TestOrchestrationRequest:
    """Test complete orchestration request validation"""

    def test_minimal_valid_request(self):
        """Minimal valid orchestration request"""
        req = OrchestrationRequest(
            source_audio_path="/path/to/song.mp3",
            separation_request=StemSeparationRequest(
                source_path="/path/to/song.mp3",
                output_dir="/path/to/stems"
            ),
            stem_transformations=[
                StemTransformation(
                    stem_type=StemType.VOCALS,
                    target_genre=GenreStyle.NEUTRAL
                )
            ],
            output_path="/path/to/output.wav"
        )
        assert req.source_audio_path == "/path/to/song.mp3"
        assert len(req.stem_transformations) == 1

    def test_duplicate_stem_types_fail(self):
        """Duplicate stem types in transformations fail"""
        with pytest.raises(ValueError, match="Duplicate stem types"):
            OrchestrationRequest(
                source_audio_path="/path/to/song.mp3",
                separation_request=StemSeparationRequest(
                    source_path="/path/to/song.mp3",
                    output_dir="/path/to/stems"
                ),
                stem_transformations=[
                    StemTransformation(stem_type=StemType.VOCALS),
                    StemTransformation(stem_type=StemType.VOCALS)  # Duplicate
                ],
                output_path="/path/to/output.wav"
            )

    def test_tempo_shift_partial_stems_fails(self):
        """tempo_shift on only some stems fails (desyncs the mix)"""
        with pytest.raises(ValueError, match="tempo_shift must be applied uniformly"):
            OrchestrationRequest(
                source_audio_path="/path/to/song.mp3",
                separation_request=StemSeparationRequest(
                    source_path="/path/to/song.mp3",
                    output_dir="/path/to/stems"
                ),
                stem_transformations=[
                    StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.2),
                    StemTransformation(stem_type=StemType.VOCALS),
                    StemTransformation(stem_type=StemType.BASS),
                    StemTransformation(stem_type=StemType.OTHER),
                ],
                output_path="/path/to/output.wav"
            )

    def test_tempo_shift_all_four_mismatched_fails(self):
        """tempo_shift present on all four stems but with different values fails"""
        with pytest.raises(ValueError, match="tempo_shift must be applied uniformly"):
            OrchestrationRequest(
                source_audio_path="/path/to/song.mp3",
                separation_request=StemSeparationRequest(
                    source_path="/path/to/song.mp3",
                    output_dir="/path/to/stems"
                ),
                stem_transformations=[
                    StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.2),
                    StemTransformation(stem_type=StemType.VOCALS, tempo_shift=0.1),
                    StemTransformation(stem_type=StemType.BASS, tempo_shift=0.2),
                    StemTransformation(stem_type=StemType.OTHER, tempo_shift=0.2),
                ],
                output_path="/path/to/output.wav"
            )

    def test_tempo_shift_all_four_uniform_succeeds(self):
        """tempo_shift present on all four stems with the identical value succeeds"""
        req = OrchestrationRequest(
            source_audio_path="/path/to/song.mp3",
            separation_request=StemSeparationRequest(
                source_path="/path/to/song.mp3",
                output_dir="/path/to/stems"
            ),
            stem_transformations=[
                StemTransformation(stem_type=StemType.DRUMS, tempo_shift=0.2),
                StemTransformation(stem_type=StemType.VOCALS, tempo_shift=0.2),
                StemTransformation(stem_type=StemType.BASS, tempo_shift=0.2),
                StemTransformation(stem_type=StemType.OTHER, tempo_shift=0.2),
            ],
            output_path="/path/to/output.wav"
        )
        assert all(t.tempo_shift == 0.2 for t in req.stem_transformations)

    def test_no_tempo_shift_succeeds(self):
        """No tempo_shift set anywhere succeeds (nothing to validate)"""
        req = OrchestrationRequest(
            source_audio_path="/path/to/song.mp3",
            separation_request=StemSeparationRequest(
                source_path="/path/to/song.mp3",
                output_dir="/path/to/stems"
            ),
            stem_transformations=[
                StemTransformation(stem_type=StemType.DRUMS),
                StemTransformation(stem_type=StemType.VOCALS),
            ],
            output_path="/path/to/output.wav"
        )
        assert all(t.tempo_shift is None for t in req.stem_transformations)

    def test_full_orchestration_request(self):
        """Complete orchestration request with all parameters"""
        eq = EQAdjustment(low_gain=-3.0, high_gain=6.0)

        req = OrchestrationRequest(
            source_audio_path="/path/to/rock_song.mp3",
            separation_request=StemSeparationRequest(
                source_path="/path/to/rock_song.mp3",
                output_dir="/path/to/stems"
            ),
            beat_alignment=BeatGridAlignment(target_bpm=140.0),
            stem_transformations=[
                StemTransformation(
                    stem_type=StemType.VOCALS,
                    target_genre=GenreStyle.NEUTRAL,
                    volume_db=-2.0,
                    tempo_shift=0.1
                ),
                StemTransformation(
                    stem_type=StemType.DRUMS,
                    target_genre=GenreStyle.JAZZ,
                    tempo_shift=0.1,
                    eq_adjustment=eq,
                    genre_blend_ratio=0.6
                ),
                StemTransformation(
                    stem_type=StemType.BASS,
                    target_genre=GenreStyle.FUNK,
                    pitch_shift_semitones=-2,
                    tempo_shift=0.1
                ),
                StemTransformation(
                    stem_type=StemType.OTHER,
                    target_genre=GenreStyle.METAL,
                    genre_blend_ratio=0.4,
                    tempo_shift=0.1
                )
            ],
            mix_parameters=MixParameters(
                stem_weights={
                    StemType.VOCALS: 1.0,
                    StemType.DRUMS: 1.2,
                    StemType.BASS: 1.1,
                    StemType.OTHER: 0.8
                }
            ),
            output_path="/path/to/fusion_output.wav"
        )

        assert len(req.stem_transformations) == 4
        assert req.stem_transformations[1].target_genre == GenreStyle.JAZZ
        assert req.stem_transformations[1].genre_blend_ratio == 0.6


class TestEnvironment:
    """Test Python environment setup"""

    def test_python_version(self):
        """Ensure Python 3.10+"""
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 10

    def test_pydantic_import(self):
        """Pydantic imports successfully"""
        import pydantic
        assert pydantic.__version__ >= "2.0"

"""
DSP Parameter Schemas for StemFuse

Pydantic validation schemas for audio processing parameters.
These schemas define the bounds and constraints for all DSP operations,
preventing the Translation Gap between LLM outputs and the audio engine.

Design principle: Validate at the boundary, fail fast with clear errors.
"""

from pydantic import BaseModel, Field, confloat, conint, conlist
from typing import Optional, Literal
from enum import Enum


class StemType(str, Enum):
    """Supported stem types from separation"""
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"


class GenreStyle(str, Enum):
    """Target genres for semantic fusion"""
    JAZZ = "jazz"
    ROCK = "rock"
    METAL = "metal"
    ELECTRONIC = "electronic"
    CLASSICAL = "classical"
    REGGAE = "reggae"
    HIPHOP = "hiphop"
    POP = "pop"
    COUNTRY = "country"
    FUNK = "funk"
    NEUTRAL = "neutral"


# ===== Core Constraints =====

class TempoConstraints(BaseModel):
    """
    Tempo (BPM) constraints for beat alignment.

    Bounds: 60-200 BPM covers most musical genres while preventing
    extreme values that would cause artifacts in time-stretching.
    """
    min_bpm: confloat(ge=60.0, le=200.0)
    max_bpm: confloat(ge=60.0, le=200.0)

    def model_post_init(self, __context):
        """Ensure min <= max"""
        if self.min_bpm > self.max_bpm:
            raise ValueError("min_bpm must be <= max_bpm")


class TimeStretchConstraints(BaseModel):
    """
    Time-stretching ratio constraints.

    Bounds: 0.5-2.0 prevents extreme artifacts from pyrubberband.
    Below 0.5 or above 2.0 causes noticeable spectral degradation.
    """
    ratio: confloat(ge=0.5, le=2.0) = Field(
        default=1.0,
        description="Time-stretch ratio (1.0 = no change)"
    )
    preserve_formants: bool = Field(
        default=True,
        description="Preserve formants during time-stretch (prevents 'chipmunk' effect)"
    )


# ===== Stem-Level Transformations =====

class EQAdjustment(BaseModel):
    """
    Equalization adjustments for frequency bands.

    All gains in dB, bounded to prevent extreme boosts/cuts.
    """
    low_gain: Optional[confloat(ge=-12.0, le=12.0)] = Field(
        default=None,
        description="Low-shelf gain (bass) in dB"
    )
    mid_gain: Optional[confloat(ge=-12.0, le=12.0)] = Field(
        default=None,
        description="Mid-range gain in dB"
    )
    high_gain: Optional[confloat(ge=-12.0, le=12.0)] = Field(
        default=None,
        description="High-shelf gain (treble) in dB"
    )


class StemTransformation(BaseModel):
    """
    Transformation parameters for a single stem.

    This is the core schema for per-stem genre fusion. Each stem can
    receive independent genre, tempo, pitch, and EQ transformations.

    Bounds are conservative to prevent audio artifacts:
    - Tempo shift: +/- 50% prevents extreme time-stretching
    - Pitch shift: +/- 1 octave preserves musicality
    - EQ: +/- 12dB prevents extreme filtering
    """
    stem_type: StemType
    target_genre: Optional[GenreStyle] = Field(
        default=None,
        description="Target genre for semantic fusion"
    )

    # Tempo manipulation
    tempo_shift: Optional[confloat(ge=-0.5, le=0.5)] = Field(
        default=None,
        description="Tempo shift ratio (0.5 = +50%, -0.5 = -50%)"
    )

    # Pitch manipulation
    pitch_shift_semitones: Optional[conint(ge=-12, le=12)] = Field(
        default=None,
        description="Pitch shift in semitones (+/- 1 octave)"
    )

    # Equalization
    eq_adjustment: Optional[EQAdjustment] = Field(
        default=None,
        description="EQ adjustments for this stem"
    )

    # Mix level
    volume_db: Optional[confloat(ge=-60.0, le=0.0)] = Field(
        default=None,
        description="Volume adjustment in dB (-60 to 0)"
    )

    # Genre fusion blend (0.0-1.0)
    genre_blend_ratio: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="Blend between original and target genre (0.0=original, 1.0=full genre)"
    )


# ===== Track-Level Operations =====

class BeatGridAlignment(BaseModel):
    """
    Parameters for unified beat-grid alignment.

    All stems must be aligned to the same beat grid before any
    transformations to ensure rhythmic coherence.
    """
    target_bpm: Optional[confloat(ge=60.0, le=200.0)] = Field(
        default=None,
        description="Target BPM for alignment (None = detect from audio)"
    )
    time_stretch_method: Literal["rubberband", "librosa"] = Field(
        default="rubberband",
        description="Time-stretching algorithm (rubberband = higher quality)"
    )


class MixParameters(BaseModel):
    """
    Programmatic mixing parameters for stem fusion.

    These parameters control how stems are combined after processing.
    """
    stem_weights: dict[StemType, confloat(ge=0.0, le=2.0)] = Field(
        default_factory=dict,
        description="Per-stem mix weights (1.0 = unity gain)"
    )
    master_volume_db: confloat(ge=-60.0, le=0.0) = Field(
        default=-3.0,
        description="Master output volume in dB"
    )
    compression_threshold_db: Optional[confloat(ge=-60.0, le=0.0)] = Field(
        default=None,
        description="Dynamic range compression threshold (None = no compression)"
    )


# ===== Complete Orchestration Request =====

class StemSeparationRequest(BaseModel):
    """Request parameters for stem separation"""
    source_path: str
    output_dir: str
    model: Literal["htdemucs", "htdemucs_6s", "htdemucs_ft"] = Field(
        default="htdemucs",
        description="Demucs model variant"
    )


class OrchestrationRequest(BaseModel):
    """
    Complete semantic orchestration request.

    This is the top-level schema that LLM outputs must validate against.
    It combines stem separation, beat alignment, per-stem transformations,
    and mixing parameters into a single validated configuration.

    Workflow:
    1. LLM generates JSON from natural language prompt
    2. Pydantic validates against this schema
    3. Validated parameters passed to DSP engine
    4. DSP engine executes transformations deterministically
    """

    # Input
    source_audio_path: str = Field(
        ...,
        description="Path to input audio file"
    )

    # Stem separation
    separation_request: StemSeparationRequest

    # Beat alignment (applied before transformations)
    beat_alignment: BeatGridAlignment = Field(
        default_factory=BeatGridAlignment,
        description="Beat-grid alignment parameters"
    )

    # Per-stem transformations (core fusion logic)
    stem_transformations: conlist(StemTransformation, min_length=1, max_length=4) = Field(
        ...,
        description="List of stem transformations (one per stem)"
    )

    # Mix parameters
    mix_parameters: MixParameters = Field(
        default_factory=MixParameters,
        description="Mix parameters for stem fusion"
    )

    # Output
    output_path: str = Field(
        ...,
        description="Path for output audio file"
    )

    def model_post_init(self, __context):
        """
        Validate orchestration request consistency.

        Ensures each stem type appears at most once in transformations.
        """
        stem_types = [t.stem_type for t in self.stem_transformations]
        if len(stem_types) != len(set(stem_types)):
            raise ValueError("Duplicate stem types in stem_transformations")

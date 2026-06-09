"""
DSP Transform Engine - Applies genre, tempo, pitch, and EQ transformations.

This module implements the actual audio processing for semantic genre fusion.
Each transformation is applied independently per-stem before mixing.
"""

import numpy as np
import soundfile as sf
import librosa
import pyrubberband as pyrb
from pathlib import Path
from typing import Optional

from schemas.dsp_parameters import StemType, GenreStyle, StemTransformation


# Genre-specific EQ presets (3-band EQ: low, mid, high in dB)
GENRE_EQ_PRESETS = {
    GenreStyle.JAZZ: {
        "vocals": (2.0, 4.0, 1.0),   # Warm, present mids
        "drums": (-1.0, 2.0, 3.0),    # Punchy mids, bright highs
        "bass": (4.0, 1.0, -2.0),     # Boosted low end
        "other": (1.0, 2.0, 2.0),
    },
    GenreStyle.ROCK: {
        "vocals": (0.0, 3.0, 2.0),    # Present mids
        "drums": (-2.0, 4.0, 4.0),    # Aggressive mids, bright
        "bass": (5.0, 0.0, -1.0),     # Heavy low end
        "other": (2.0, 3.0, 3.0),
    },
    GenreStyle.METAL: {
        "vocals": (0.0, 2.0, 3.0),    # Aggressive highs
        "drums": (-4.0, 6.0, 6.0),    # Scooped mids, bright
        "bass": (6.0, -2.0, 0.0),     # Heavy lows, scooped mids
        "other": (0.0, 4.0, 5.0),
    },
    GenreStyle.ELECTRONIC: {
        "vocals": (0.0, 0.0, 2.0),    # Clean, bright
        "drums": (0.0, 0.0, 4.0),     # Punchy, bright
        "bass": (6.0, 0.0, 0.0),      # Sub-heavy
        "other": (2.0, 0.0, 3.0),
    },
    GenreStyle.CLASSICAL: {
        "vocals": (1.0, 2.0, 1.0),    # Natural, balanced
        "drums": (0.0, 1.0, 1.0),
        "bass": (2.0, 1.0, 0.0),
        "other": (0.0, 1.0, 1.0),
    },
    GenreStyle.REGGAE: {
        "vocals": (0.0, 2.0, 2.0),
        "drums": (-1.0, 1.0, 3.0),    # Bright hi-hats
        "bass": (6.0, 0.0, -2.0),     # Heavy bass
        "other": (1.0, 1.0, 2.0),
    },
    GenreStyle.HIPHOP: {
        "vocals": (0.0, 1.0, 2.0),    # Present, bright
        "drums": (0.0, 0.0, 3.0),     # Punchy
        "bass": (5.0, 1.0, -1.0),     # Heavy low end
        "other": (2.0, 0.0, 2.0),
    },
    GenreStyle.POP: {
        "vocals": (0.0, 2.0, 3.0),    # Bright, present
        "drums": (0.0, 1.0, 2.0),
        "bass": (3.0, 1.0, 0.0),
        "other": (1.0, 2.0, 2.0),
    },
    GenreStyle.COUNTRY: {
        "vocals": (1.0, 2.0, 1.0),    # Warm, natural
        "drums": (0.0, 2.0, 2.0),
        "bass": (3.0, 1.0, 0.0),
        "other": (1.0, 1.0, 1.0),
    },
    GenreStyle.FUNK: {
        "vocals": (0.0, 3.0, 2.0),    # Present mids
        "drums": (0.0, 4.0, 4.0),     # Slap highs
        "bass": (5.0, 3.0, 0.0),      # Boomy low-mids
        "other": (2.0, 3.0, 3.0),
    },
    GenreStyle.NEUTRAL: {
        "vocals": (0.0, 0.0, 0.0),
        "drums": (0.0, 0.0, 0.0),
        "bass": (0.0, 0.0, 0.0),
        "other": (0.0, 0.0, 0.0),
    },
}


def apply_eq(audio: np.ndarray, sr: int, low_gain: float = 0.0,
             mid_gain: float = 0.0, high_gain: float = 0.0) -> np.ndarray:
    """
    Apply 3-band EQ to audio.

    Uses librosa's mel frequency basis for simple EQ filtering.
    """
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)

    # Frequency bands (approximate)
    freqs = librosa.fft_frequencies(sr=sr)
    n_bins = len(freqs)

    # Create frequency response
    response = np.ones(n_bins)

    # Low shelf (below 200 Hz)
    low_mask = freqs < 200
    response[low_mask] *= 10 ** (low_gain / 20.0)

    # High shelf (above 4000 Hz)
    high_mask = freqs > 4000
    response[high_mask] *= 10 ** (high_gain / 20.0)

    # Mid band (200 Hz - 4000 Hz)
    mid_mask = (~low_mask) & (~high_mask)
    response[mid_mask] *= 10 ** (mid_gain / 20.0)

    # Apply response (broadcast across time frames)
    response = response[:, np.newaxis]
    D_eq = magnitude * response * np.exp(1j * phase)
    audio_eq = librosa.istft(D_eq, length=len(audio))

    return audio_eq


def apply_volume(audio: np.ndarray, volume_db: float) -> np.ndarray:
    """Apply volume adjustment in dB."""
    linear_gain = 10 ** (volume_db / 20.0)
    return audio * linear_gain


def apply_tempo_shift(audio: np.ndarray, sr: int, tempo_shift: float) -> np.ndarray:
    """
    Apply tempo shift using time-stretching.

    Args:
        audio: Audio samples
        sr: Sample rate
        tempo_shift: Tempo shift ratio (-0.5 to 0.5, where 0.2 = +20% tempo)
    """
    if tempo_shift == 0 or abs(tempo_shift) < 0.01:
        return audio

    # Convert tempo shift to stretch ratio
    # tempo_shift of 0.2 means 20% faster, so ratio is 1.2
    ratio = 1.0 + tempo_shift

    # Clamp ratio to prevent artifacts
    ratio = np.clip(ratio, 0.5, 2.0)

    return pyrb.time_stretch(audio, sr, ratio)


def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    """
    Apply pitch shift in semitones.

    Args:
        audio: Audio samples
        sr: Sample rate
        semitones: Pitch shift in semitones (-12 to 12)
    """
    if semitones == 0:
        return audio

    # Clamp to prevent artifacts
    semitones = np.clip(semitones, -12, 12)

    # pyrubberband pitch shifting
    return pyrb.pitch_shift(audio, sr, semitones)


def apply_genre_preset(audio: np.ndarray, sr: int, stem_type: StemType,
                        target_genre: GenreStyle, blend_ratio: Optional[float] = None) -> np.ndarray:
    """
    Apply genre-specific EQ preset to audio.

    Args:
        audio: Audio samples
        sr: Sample rate
        stem_type: Type of stem (vocals, drums, bass, other)
        target_genre: Target genre style
        blend_ratio: How much to apply (0.0-1.0), None = full
    """
    if target_genre == GenreStyle.NEUTRAL or target_genre is None:
        return audio

    # Get EQ preset for genre and stem type
    eq_presets = GENRE_EQ_PRESETS.get(target_genre, {})
    stem_name = stem_type.value
    low_gain, mid_gain, high_gain = eq_presets.get(stem_name, (0.0, 0.0, 0.0))

    # Apply blend ratio
    if blend_ratio is not None and blend_ratio < 1.0:
        low_gain *= blend_ratio
        mid_gain *= blend_ratio
        high_gain *= blend_ratio

    return apply_eq(audio, sr, low_gain, mid_gain, high_gain)


def apply_transformation(audio_path: str, output_path: str,
                         transformation: StemTransformation) -> None:
    """
    Apply a single stem transformation to audio file.

    This is the main entry point for DSP transformations.
    All transformations from the StemTransformation schema are applied.

    Args:
        audio_path: Input audio file path
        output_path: Output audio file path
        transformation: StemTransformation with parameters to apply
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    # 1. Tempo shift
    if transformation.tempo_shift is not None:
        audio = apply_tempo_shift(audio, sr, transformation.tempo_shift)

    # 2. Pitch shift
    if transformation.pitch_shift_semitones is not None:
        audio = apply_pitch_shift(audio, sr, transformation.pitch_shift_semitones)

    # 3. Genre EQ preset
    if transformation.target_genre:
        audio = apply_genre_preset(
            audio, sr,
            stem_type=transformation.stem_type,
            target_genre=transformation.target_genre,
            blend_ratio=transformation.genre_blend_ratio
        )

    # 4. Manual EQ adjustment
    if transformation.eq_adjustment:
        eq = transformation.eq_adjustment
        audio = apply_eq(
            audio, sr,
            low_gain=eq.low_gain or 0.0,
            mid_gain=eq.mid_gain or 0.0,
            high_gain=eq.high_gain or 0.0
        )

    # 5. Volume adjustment
    if transformation.volume_db is not None:
        audio = apply_volume(audio, transformation.volume_db)

    # Clipping protection
    audio = np.clip(audio, -1.0, 1.0)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save output
    sf.write(output_path, audio, sr)


def apply_all_transformations(
    stems: dict[str, str],
    transformations: list[StemTransformation],
    output_dir: str
) -> dict[str, str]:
    """
    Apply transformations to all stems.

    Args:
        stems: Dictionary mapping stem_type to input file path
        transformations: List of StemTransformation objects
        output_dir: Directory for transformed stems

    Returns:
        Dictionary mapping stem_type to output file path
    """
    # Create transformation lookup
    trans_map = {t.stem_type.value: t for t in transformations}

    transformed = {}

    for stem_type, audio_path in stems.items():
        if stem_type in trans_map:
            transformation = trans_map[stem_type]
            output_path = str(Path(output_dir) / f"{stem_type}_transformed.wav")

            print(f"  Applying transformations to {stem_type}...")
            apply_transformation(audio_path, output_path, transformation)
            transformed[stem_type] = output_path
        else:
            # No transformation, copy original
            output_path = str(Path(output_dir) / f"{stem_type}_transformed.wav")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Just copy (read and write)
            audio, sr = librosa.load(audio_path, sr=44100)
            sf.write(output_path, audio, sr)
            transformed[stem_type] = output_path

    return transformed

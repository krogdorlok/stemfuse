"""Stem mixing - combine separated stems into single audio output."""

import soundfile as sf
import numpy as np


def mix_stems(stem_paths: dict, output_path: str, volumes: dict = None) -> str:
    """
    Mix multiple stems into a single audio file.

    Args:
        stem_paths: Dictionary mapping stem names to file paths
        output_path: Path for output mixed audio
        volumes: Dictionary mapping stem names to volume multipliers (1.0 = original)

    Returns:
        Path to output file
    """
    if volumes is None:
        volumes = {stem: 1.0 for stem in stem_paths}

    mixed_audio = None
    sample_rate = None

    # Load and sum all stems
    for stem_name, stem_path in stem_paths.items():
        audio, sr = sf.read(stem_path)

        if mixed_audio is None:
            mixed_audio = audio * volumes.get(stem_name, 1.0)
            sample_rate = sr
        else:
            # Ensure same length and sample rate
            if sr != sample_rate:
                raise ValueError(f"Sample rate mismatch for {stem_name}")

            # Pad shorter stems to match longest
            if len(audio) < len(mixed_audio):
                audio = np.pad(audio, (0, len(mixed_audio) - len(audio)))
            elif len(audio) > len(mixed_audio):
                mixed_audio = np.pad(mixed_audio, (0, len(audio) - len(mixed_audio)))

            mixed_audio += audio * volumes.get(stem_name, 1.0)

    # Normalize to prevent clipping
    max_level = np.max(np.abs(mixed_audio))
    if max_level > 0:
        mixed_audio = mixed_audio / max_level * 0.9

    # Save mixed audio
    sf.write(output_path, mixed_audio, sample_rate)

    return output_path
"""Stem mixing - combine separated stems into single audio output."""

import soundfile as sf
import numpy as np


def mix_stems(stem_paths: dict, output_path: str, volumes: dict = None, master_volume_db: float = -3.0) -> str:
    """
    Mix multiple stems into a single audio file.

    Args:
        stem_paths: Dictionary mapping stem names to file paths
        output_path: Path for output mixed audio
        volumes: Dictionary mapping stem names to volume multipliers (1.0 = original)
        master_volume_db: Master output volume in dB

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

            # Pad shorter stems to match longest (stereo-aware)
            if len(audio) < len(mixed_audio):
                diff = len(mixed_audio) - len(audio)
                pad_width = ((0, diff), (0, 0)) if audio.ndim == 2 else (0, diff)
                audio = np.pad(audio, pad_width)
            elif len(audio) > len(mixed_audio):
                diff = len(audio) - len(mixed_audio)
                pad_width = ((0, diff), (0, 0)) if mixed_audio.ndim == 2 else (0, diff)
                mixed_audio = np.pad(mixed_audio, pad_width)

            mixed_audio += audio * volumes.get(stem_name, 1.0)

    # Apply master volume and normalize to prevent clipping
    master_linear = 10 ** (master_volume_db / 20.0)
    max_level = np.max(np.abs(mixed_audio))
    if max_level > 0:
        mixed_audio = (mixed_audio / max_level) * master_linear

    # Save mixed audio
    sf.write(output_path, mixed_audio, sample_rate)

    return output_path
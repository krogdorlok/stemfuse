"""Synthetic audio generator for testing without external drive."""

import soundfile as sf
import numpy as np
import os


def create_test_stems(output_dir: str = "data/test_tracks/synthetic"):
    """Create synthetic test stems for mixer testing."""
    os.makedirs(output_dir, exist_ok=True)

    duration = 5.0  # seconds
    sample_rate = 44100
    samples = int(duration * sample_rate)

    stems = {}

    # Create simple test stems
    for stem_name, frequency in [
        ("vocals", 440.0),    # A4
        ("drums", 60.0),      # Low frequency kick
        ("bass", 110.0),      # A2
        ("other", 880.0),     # A5
    ]:
        # Generate sine wave at frequency
        t = np.linspace(0, duration, samples)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)

        # Add slight variation to prevent perfect phase alignment
        audio += 0.05 * np.sin(2 * np.pi * (frequency * 2) * t)

        stem_path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(stem_path, audio, sample_rate)
        stems[stem_name] = stem_path

    return stems
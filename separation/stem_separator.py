"""Stem separation using Demucs."""

import subprocess
from pathlib import Path


def separate(input_path: str, output_dir: str) -> dict:
    """
    Separate audio into 4 stems: vocals, drums, bass, other.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated stems

    Returns:
        Dictionary mapping stem names to output file paths
    """
    result = subprocess.run(
        ["demucs", "-n", "htdemucs", "-o", output_dir, input_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")

    stem_dir = Path(output_dir) / "htdemucs" / Path(input_path).stem

    return {
        "vocals": str(stem_dir / "vocals.wav"),
        "drums": str(stem_dir / "drums.wav"),
        "bass": str(stem_dir / "bass.wav"),
        "other": str(stem_dir / "other.wav"),
    }
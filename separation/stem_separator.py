"""Stem separation using Demucs."""

import subprocess
import sys
from pathlib import Path


def _get_demucs_path() -> str:
    """Get the path to the demucs executable."""
    # Try venv bin directory first
    venv_bin = Path(sys.prefix) / "bin" / "demucs"
    if venv_bin.exists():
        return str(venv_bin)
    # Fall back to system PATH
    return "demucs"


def separate(input_path: str, output_dir: str) -> dict:
    """
    Separate audio into 4 stems: vocals, drums, bass, other.

    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated stems

    Returns:
        Dictionary mapping stem names to output file paths
    """
    demucs = _get_demucs_path()
    result = subprocess.run(
        [demucs, "-n", "htdemucs", "-o", output_dir, input_path],
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
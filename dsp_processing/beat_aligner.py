"""Beat alignment using librosa and pyrubberband."""

import librosa
import pyrubberband as pyrb
import soundfile as sf
import numpy as np


def detect_tempo(audio_path: str) -> float:
    """Detect tempo in BPM from audio file."""
    y, sr = librosa.load(audio_path, sr=44100)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo.item() if hasattr(tempo, 'item') else tempo)


def time_stretch(audio_path: str, output_path: str, target_bpm: float) -> float:
    """
    Time-stretch audio to target tempo.

    Returns:
        Stretch ratio applied
    """
    y, sr = librosa.load(audio_path, sr=44100)
    raw_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    current_bpm = float(raw_tempo.item() if hasattr(raw_tempo, 'item') else raw_tempo)
    ratio = target_bpm / current_bpm

    y_stretched = pyrb.time_stretch(y, sr, ratio)
    sf.write(output_path, y_stretched, sr)

    return ratio


def time_stretch_by_ratio(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    Time-stretch an already-loaded audio array by a direct ratio.
    ratio > 1.0 = faster, ratio < 1.0 = slower.
    Expects ratio already clamped to 0.5-2.0 (schema enforces this upstream).
    Returns stretched audio array (does not write to disk).
    """
    ratio = np.clip(ratio, 0.5, 2.0)
    return pyrb.time_stretch(audio, sr, ratio)
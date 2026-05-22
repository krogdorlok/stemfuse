"""Beat alignment using librosa and pyrubberband."""

import librosa
import pyrubberband as pyrb
import soundfile as sf


def detect_tempo(audio_path: str) -> float:
    """Detect tempo in BPM from audio file."""
    y, sr = librosa.load(audio_path, sr=44100)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def time_stretch(audio_path: str, output_path: str, target_bpm: float) -> float:
    """
    Time-stretch audio to target tempo.

    Returns:
        Stretch ratio applied
    """
    y, sr = librosa.load(audio_path, sr=44100)
    current_bpm = detect_tempo(audio_path)
    ratio = target_bpm / current_bpm

    y_stretched = pyrb.time_stretch(y, sr, ratio, preserve_formants=True)
    sf.write(output_path, y_stretched, sr)

    return ratio
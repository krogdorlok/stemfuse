#!/usr/bin/env python3
"""
Full end-to-end pipeline test with real audio.
Tests: NLP Parse → Stem Separation → Transformations → Mixing

Usage:
    python test_full_pipeline.py <input_audio> "<prompt>"
    python test_full_pipeline.py                    # Uses defaults
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_intent.parser import parse_prompt
from separation.stem_separator import separate
from dsp_processing.transform_engine import apply_all_transformations
from dsp_processing.mixer import mix_stems


def main():
    """Run complete pipeline with a real song."""

    # Parse command line arguments
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        prompt = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        prompt = "Make the drums 70% jazz fusion with punchy mids. Make the bass heavier with 40% funk blend."
    else:
        # Defaults
        input_file = "data/test_tracks/edm.mp3"
        prompt = "Make the drums 70% jazz fusion with punchy mids. Make the bass heavier with 40% funk blend."

    # Create output directory
    input_name = Path(input_file).stem
    output_dir = Path("output") / f"{input_name}_transformed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎵 StemFuse Full Pipeline Test")
    print("=" * 60)
    print(f"\n📂 Input: {input_file}")
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"📁 Output: {output_dir}/")
    print()

    # Step 1: Parse natural language prompt
    print("📝 Step 1: Parsing natural language...")
    request = parse_prompt(prompt, input_file, str(output_dir / "output.wav"))

    print("   ✅ Parsed transformations:")
    for t in request.stem_transformations:
        parts = []
        if t.target_genre:
            parts.append(f"genre={t.target_genre.value}")
            if t.genre_blend_ratio:
                parts.append(f"blend={t.genre_blend_ratio}")
        if t.tempo_shift is not None:
            parts.append(f"tempo={t.tempo_shift:+.0%}")
        if t.pitch_shift_semitones is not None:
            parts.append(f"pitch={t.pitch_shift_semitones:+d}st")
        if t.volume_db is not None:
            parts.append(f"vol={t.volume_db}dB")
        if t.eq_adjustment:
            eq = t.eq_adjustment
            parts.append(f"EQ(low={eq.low_gain},mid={eq.mid_gain},high={eq.high_gain})")

        print(f"      - {t.stem_type.value}: {', '.join(parts) if parts else 'no changes'}")
    print()

    # Step 2: Separate stems
    print("🎚️  Step 2: Separating stems...")
    stems_dir = output_dir / "stems"
    stems = separate(input_file, str(stems_dir))
    print(f"   ✅ Separated {len(stems)} stems:")
    for stem, path in stems.items():
        size = Path(path).stat().st_size / 1024
        print(f"      - {stem}: {Path(path).name} ({size:.0f} KB)")
    print()

    # Step 3: Apply transformations
    print("🎛️  Step 3: Applying DSP transformations...")
    transformed_dir = output_dir / "transformed"
    transformed = apply_all_transformations(stems, request.stem_transformations, str(transformed_dir))
    print(f"   ✅ Transformed {len(transformed)} stems:")
    for stem, path in transformed.items():
        size = Path(path).stat().st_size / 1024
        print(f"      - {stem}: {Path(path).name} ({size:.0f} KB)")
    print()

    # Step 4: Mix stems
    print("🔊 Step 4: Mixing stems...")
    mix_output = output_dir / "final_output.wav"

    # Extract stem volumes from transformations
    volumes = {}
    for t in request.stem_transformations:
        if t.volume_db is not None:
            volumes[t.stem_type.value] = 10 ** (t.volume_db / 20.0)

    mix_stems(transformed, str(mix_output), volumes=volumes, master_volume_db=-3.0)
    size = mix_output.stat().st_size / 1024 / 1024
    print(f"   ✅ Mixed output: {mix_output.name} ({size:.2f} MB)")
    print()

    print("=" * 60)
    print("🎉 SUCCESS! Full pipeline completed.")
    print(f"🎧 Output: {mix_output}")
    print("=" * 60)

    return str(mix_output)


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

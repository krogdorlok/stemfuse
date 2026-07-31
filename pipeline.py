#!/usr/bin/env python3
"""StemFuse Pipeline - Transform songs using natural language.

Usage:
    python pipeline.py                              # interactive prompts
    python pipeline.py <audio_file> "<prompt>" [output_file]   # scripted/CI
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_intent.parser import parse_prompt  # noqa: E402
from separation.stem_separator import separate  # noqa: E402
from dsp_processing.transform_engine import apply_all_transformations  # noqa: E402
from dsp_processing.mixer import mix_stems  # noqa: E402


def run_pipeline(input_file: str, prompt: str, output_file: str) -> str:
    """
    Parse prompt, separate stems, apply transformations, mix. Returns output path.

    per-stem volume_db is baked into each transformed stem file by
    apply_all_transformations()/apply_transformation() already - mix_stems()
    must NOT be given a second volumes dict built from the same values, or
    the gain gets applied twice.
    """
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse natural language prompt
    print("\n[Step 1] Parsing natural language...")
    request = parse_prompt(prompt, input_file, output_file)

    if not request:
        raise RuntimeError("Failed to parse prompt")

    print("   [PASS] Parsed transformations:")
    for t in request.stem_transformations:
        parts = []
        if t.target_genre:
            parts.append(f"genre={t.target_genre.value}")
        if t.tempo_shift is not None:
            parts.append(f"tempo={t.tempo_shift:+.0%}")
        if t.pitch_shift_semitones is not None:
            parts.append(f"pitch={t.pitch_shift_semitones:+d}st")
        if t.volume_db is not None:
            parts.append(f"vol={t.volume_db}dB")
        print(f"      - {t.stem_type.value}: {', '.join(parts) if parts else 'no changes'}")

    # Separate stems
    print("\n[Step 2] Separating stems...")
    stems_dir = str(output_dir.parent / "stems")
    stems = separate(input_file, stems_dir)
    print(f"   [PASS] Separated {len(stems)} stems")

    # Apply transformations
    print("\n[Step 3] Applying transformations...")
    transformed_dir = str(output_dir.parent / "transformed")
    transformed = apply_all_transformations(stems, request.stem_transformations, transformed_dir)
    print(f"   [PASS] Transformed {len(transformed)} stems")

    # Mix output. volumes defaults to None (unity gain) - per-stem gain is
    # already baked into `transformed` above.
    print("\n[Step 4] Mixing stems...")
    mix_stems(transformed, output_file, master_volume_db=-3.0)
    print("   [PASS] Mixed output")

    return output_file


def main():  # Run StemFuse pipeline, either scripted (argv) or interactive
    print("=" * 60)
    print("StemFuse - Semantic Audio Orchestrator")
    print("=" * 60)

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        prompt = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
    else:
        input_file = input("\nEnter path to input audio file: ").strip()
        prompt = input("Enter transformation prompt (e.g., 'make drums jazzy'): ").strip()
        output_file = None

    if not input_file:
        print("[ERROR] Input file path is required")
        sys.exit(1)

    if not Path(input_file).exists():
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    if not prompt:
        print("[ERROR] Transformation prompt is required")
        sys.exit(1)

    if not output_file:
        output_file = f"output/{Path(input_file).stem}_transformed/final_output.wav"

    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)

    try:
        output_file = run_pipeline(input_file, prompt, output_file)

        print("\n" + "=" * 60)
        print("[SUCCESS] Pipeline completed")
        print(f"Output: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

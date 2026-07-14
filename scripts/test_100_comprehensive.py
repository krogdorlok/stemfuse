"""Comprehensive 100-test suite for StemFuse parser."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from nlp_intent.parser import parse_prompt

# 100 exhaustive tests covering all system prompt capabilities
TESTS = [
    # ========== BASIC STEM + GENRE (1-10) ==========
    ("Apply jazz style to the drums", "basic", "drums", "jazz"),
    ("Make the vocals rock", "basic", "vocals", "rock"),
    ("Bass should be metal", "basic", "bass", "metal"),
    ("Make guitar electronic", "basic", "other", "electronic"),
    ("Give piano a classical feel", "basic", "other", "classical"),
    ("Make synth play reggae", "basic", "other", "reggae"),
    ("Beat should be hiphop", "basic", "drums", "hiphop"),
    ("Strings should sound like pop", "basic", "other", "pop"),
    ("Make it country style on bass", "basic", "bass", "country"),
    ("Add funk to the rhythm", "basic", "drums", "funk"),

    # ========== WHOLE TRACK EXPANSION (11-15) ==========
    ("Give the whole track a reggae feel", "whole_track", "all", "reggae"),
    ("Make everything metal", "whole_track", "all", "metal"),
    ("All stems should be jazz", "whole_track", "all", "jazz"),
    ("Apply electronic to the entire track", "whole_track", "all", "electronic"),
    ("Transform everything to classical", "whole_track", "all", "classical"),

    # ========== SLANG TO GENRE (16-25) ==========
    ("Make the drums swing", "slang", "drums", "jazz"),
    ("Make it jazzy", "slang", None, "jazz"),
    ("Add some bebop feel", "slang", None, "jazz"),
    ("Make it sound like bebop", "slang", None, "jazz"),
    ("Guitar should shred", "slang", "other", "metal"),
    ("Make it more aggressive", "slang", None, "metal"),
    ("Add distortion to guitars", "slang", "other", "rock"),
    ("Make it sound digital", "slang", None, "electronic"),
    ("Give it an island feel", "slang", None, "reggae"),
    ("Make it boom bap", "slang", None, "hiphop"),

    # ========== MOOD / ENERGY (26-35) ==========
    ("Make it feel more energetic", "mood_energy", None, None, 0.2),
    ("Hype it up", "mood_energy", None, None, 0.2),
    ("Make it more chill", "mood_energy", None, -0.1, None),
    ("Make it feel mellow", "mood_energy", None, -0.1, None),
    ("Make it feel sad", "mood_emotion", ["vocals", "other"], "jazz"),
    ("Make it sound emotional", "mood_emotion", ["vocals", "other"], "jazz"),
    ("Make it feel happy", "mood_emotion", "all", "pop"),
    ("Make it feel dark", "mood_emotion", None, None),
    ("Make it feel ominous", "mood_emotion", None, None),
    ("Make it feel epic", "mood_emotion", None, None),

    # ========== ERA REFERENCES (36-45) ==========
    ("Give it a 60s feel", "era", None, None, -0.1),
    ("Make it sound 70s", "era", None, None),
    ("Give it an 80s feel", "era", None, None),
    ("Make it sound 90s", "era", None, None),
    ("Give it a Y2K vibe", "era", None, None),
    ("Make it sound vintage", "era", None, -0.1),
    ("Give it a retro feel", "era", None, -0.1),
    ("Make it old school", "era", None, -0.1),
    ("Make it sound modern", "era", None, None),
    ("Give it a contemporary feel", "era", None, None),

    # ========== PRODUCTION STYLE (46-55) ==========
    ("Make the drums punchier", "production", "drums", None, None, None, "eq_mid"),
    ("Make it warmer", "production", "all", None, None, None, "eq_high_neg"),
    ("Make it brighter", "production", "all", None, None, None, "eq_high_pos"),
    ("Add more body", "production", "all", None, None, None, "eq_low"),
    ("Make it thinner", "production", "all", None, None, None, "eq_low_neg"),
    ("Make the snare snap", "production", "drums", None, None, None, "eq_mid_high"),
    ("More hi-hat sizzle", "production", "drums", None, None, None, "eq_high"),
    ("Clean up the mud", "production", "all", None, None, None, "eq_low_neg"),
    ("Add more presence", "production", "all", None, None, None, "eq_mid"),
    ("Make it less harsh", "production", "all", None, None, None, "eq_high_neg"),

    # ========== BLEND RATIOS (56-65) ==========
    ("60% jazz fusion drums", "blend", "drums", "jazz", None, None, 0.6),
    ("40% metal on vocals", "blend", "vocals", "metal", None, None, 0.4),
    ("50% reggae drums", "blend", "drums", "reggae", None, None, 0.5),
    ("70% electronic bass", "blend", "bass", "electronic", None, None, 0.7),
    ("30% pop on guitar", "blend", "other", "pop", None, None, 0.3),
    ("80% classical strings", "blend", "other", "classical", None, None, 0.8),
    ("90% metal drums", "blend", "drums", "metal", None, None, 0.9),
    ("25% funk on bass", "blend", "bass", "funk", None, None, 0.25),
    ("100% rock vocals", "blend", "vocals", "rock", None, None, 1.0),
    ("50% hiphop beat", "blend", "drums", "hiphop", None, None, 0.5),

    # ========== TEMPO CHANGES (66-75) ==========
    ("Tempo up 20%", "tempo", "all", None, 0.2),
    ("Speed up by 30%", "tempo", "all", None, 0.3),
    ("Slow down 20%", "tempo", "all", None, -0.2),
    ("Make it faster", "tempo", "all", None, 0.2),
    ("Make it slower", "tempo", "all", None, -0.2),
    ("Double time", "tempo", "all", None, 0.5),
    ("Half time", "tempo", "all", None, -0.4),
    ("A little faster", "tempo", "all", None, 0.1),
    ("Much faster", "tempo", "all", None, 0.4),
    ("Way slower", "tempo", "all", None, -0.4),

    # ========== VOLUME CHANGES (76-85) ==========
    ("Bass down 3dB", "volume", "bass", None, None, -3.0),
    ("Boost vocals", "volume", "vocals", None, None, 0),
    ("Turn down the drums", "volume", "drums", None, None, -6.0),
    ("Make guitar quieter", "volume", "other", None, None, -6.0),
    ("Bury the bass", "volume", "bass", None, None, -12.0),
    ("Bring vocals forward", "volume", "vocals", None, None, 0),
    ("Much quieter drums", "volume", "drums", None, None, -12.0),
    ("A little louder bass", "volume", "bass", None, None, 0),
    ("Mute the hi-hats", "volume", "drums", None, None, -60),
    ("Silence the guitar", "volume", "other", None, None, -60),

    # ========== MULTI-STEM (86-95) ==========
    ("Rock vocals, metal drums", "multi", ["vocals", "drums"], ["rock", "metal"]),
    ("Jazz bass, funk guitar", "multi", ["bass", "other"], ["jazz", "funk"]),
    ("Electronic drums, pop vocals", "multi", ["drums", "vocals"], ["electronic", "pop"]),
    ("Metal drums, classical strings", "multi", ["drums", "other"], ["metal", "classical"]),
    ("Hiphop beat, reggae bass", "multi", ["drums", "bass"], ["hiphop", "reggae"]),
    ("Make drums swing and bass funky", "multi", ["drums", "bass"], ["jazz", "funk"]),
    ("Rock vocals, metal drums, electronic bass", "multi", ["vocals", "drums", "bass"], ["rock", "metal", "electronic"]),
    ("Jazz piano, metal guitar", "multi", ["other", "other"], ["jazz", "metal"]),
    ("Pop vocals, rock drums, funk bass", "multi", ["vocals", "drums", "bass"], ["pop", "rock", "funk"]),
    ("Classical strings, metal drums", "multi", ["other", "drums"], ["classical", "metal"]),

    # ========== COMPLEX COMBINATIONS (96-100) ==========
    ("Drums 70% jazz, tempo up 20%", "complex", "drums", "jazz", 0.2, None, 0.7),
    ("Make drums swing, boost bass, warm vocals", "complex", ["drums", "bass", "vocals"], ["jazz", None, "funk"]),
    ("40% reggae, tempo down 10%, vocals louder", "complex", "all", "reggae", -0.1, None),
    ("80s feel, punchier drums, brighter vocals", "complex", None, None, None, None),
    ("Make it feel like a late night jazz club", "complex", "all", "jazz", -0.1, None),
]


def run_tests():
    """Run the 100-test suite."""
    print("=" * 80)
    print("COMPREHENSIVE 100-TEST SUITE FOR STEMFUSE PARSER")
    print("=" * 80)

    passed = 0
    failed = 0
    results = []

    for i, test_case in enumerate(TESTS, 1):
        # Pad test case to 7 elements
        padded = list(test_case) + [None] * (7 - len(test_case))
        prompt, category, stem, genre, tempo, volume, blend = padded
        try:
            result = parse_prompt(prompt, 'test.mp3', 'output.wav')

            if result and result.stem_transformations:
                test_passed = True
                stems_found = [t.stem_type for t in result.stem_transformations]

                # Check expectations
                if stem == "all":
                    if len(result.stem_transformations) != 4:
                        test_passed = False
                elif isinstance(stem, list):
                    if len(result.stem_transformations) != len(stem):
                        test_passed = False
                elif stem and stem not in stems_found:
                    test_passed = False

                status = "✓" if test_passed else "✗"
                results.append((i, prompt, status, len(result.stem_transformations)))
                if test_passed:
                    passed += 1
                else:
                    failed += 1

                # Print result
                print(f"\n{i}. [{status}] {prompt[:50]}...")
                for t in result.stem_transformations:
                    print(f"   {t.stem_type}: {t.target_genre}, tempo={t.tempo_shift}, vol={t.volume_db}, blend={t.genre_blend_ratio}")
            else:
                failed += 1
                results.append((i, prompt, "✗", 0))
                print(f"\n{i}. [✗] {prompt[:50]}... - FAILED")

        except Exception as e:
            failed += 1
            results.append((i, prompt, "✗", f"Error: {str(e)[:30]}"))
            print(f"\n{i}. [✗] {prompt[:50]}... - ERROR: {str(e)[:50]}")

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 80)

    if failed > 0:
        print("\nFailed tests:")
        for i, prompt, status, detail in results:
            if status == "✗":
                print(f"  {i}. {prompt}")

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)

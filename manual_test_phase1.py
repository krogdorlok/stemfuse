#!/usr/bin/env python3
"""
Manual Phase 1 Validation Tests

Run this to manually validate the Pydantic schemas work correctly.
This tests the "Translation Gap" - what happens when LLM outputs JSON.
"""

import json
from schemas.dsp_parameters import (
    StemType,
    GenreStyle,
    StemTransformation,
    OrchestrationRequest,
    StemSeparationRequest,
    BeatGridAlignment,
    MixParameters,
    EQAdjustment
)
from pydantic import ValidationError


def test_imports():
    """Test 1: Verify all imports work"""
    print("✅ Test 1: All imports successful")
    return True


def test_simple_stem_transformation():
    """Test 2: Create a simple stem transformation"""
    print("\n🎵 Test 2: Simple stem transformation")

    transform = StemTransformation(
        stem_type=StemType.DRUMS,
        target_genre=GenreStyle.JAZZ,
        tempo_shift=0.2
    )

    print(f"   Stem: {transform.stem_type}")
    print(f"   Genre: {transform.target_genre}")
    print(f"   Tempo shift: {transform.tempo_shift:+.0%}")
    print("✅ Valid transformation created")
    return True


def test_complex_orchestration():
    """Test 3: Create a complex orchestration request"""
    print("\n🎼 Test 3: Complex orchestration request")

    req = OrchestrationRequest(
        source_audio_path="/path/to/rock_song.mp3",
        separation_request=StemSeparationRequest(
            source_path="/path/to/rock_song.mp3",
            output_dir="/path/to/stems"
        ),
        beat_alignment=BeatGridAlignment(target_bpm=140.0),
        stem_transformations=[
            StemTransformation(
                stem_type=StemType.VOCALS,
                target_genre=GenreStyle.NEUTRAL,
                volume_db=-2.0
            ),
            StemTransformation(
                stem_type=StemType.DRUMS,
                target_genre=GenreStyle.JAZZ,
                tempo_shift=0.1,
                eq_adjustment=EQAdjustment(low_gain=-3.0, high_gain=6.0),
                genre_blend_ratio=0.6
            ),
            StemTransformation(
                stem_type=StemType.BASS,
                target_genre=GenreStyle.FUNK,
                pitch_shift_semitones=-2
            ),
            StemTransformation(
                stem_type=StemType.OTHER,
                target_genre=GenreStyle.METAL,
                genre_blend_ratio=0.4
            )
        ],
        mix_parameters=MixParameters(
            stem_weights={
                StemType.VOCALS: 1.0,
                StemType.DRUMS: 1.2,
                StemType.BASS: 1.1,
                StemType.OTHER: 0.8
            }
        ),
        output_path="/path/to/fusion_output.wav"
    )

    print(f"   Input: {req.source_audio_path}")
    print(f"   Target BPM: {req.beat_alignment.target_bpm}")
    print(f"   Stems: {len(req.stem_transformations)}")
    print(f"   Output: {req.output_path}")
    print("✅ Complex orchestration validated")
    return True


def test_invalid_parameters():
    """Test 4: Validate rejection of invalid parameters"""
    print("\n🚫 Test 4: Invalid parameter rejection")

    test_cases = [
        {
            "name": "Tempo shift too high",
            "params": {
                "stem_type": "drums",
                "tempo_shift": 1.5  # Exceeds 0.5 bound
            },
            "should_fail": True
        },
        {
            "name": "Pitch shift out of range",
            "params": {
                "stem_type": "vocals",
                "pitch_shift_semitones": 15  # Exceeds 12 bound
            },
            "should_fail": True
        },
        {
            "name": "Valid transformation",
            "params": {
                "stem_type": "bass",
                "tempo_shift": 0.3,
                "volume_db": -6.0
            },
            "should_fail": False
        }
    ]

    for test in test_cases:
        try:
            result = StemTransformation(**test["params"])
            if test["should_fail"]:
                print(f"   ❌ {test['name']}: Should have failed but passed")
                return False
            else:
                print(f"   ✅ {test['name']}: Correctly validated")
        except ValidationError as e:
            if test["should_fail"]:
                print(f"   ✅ {test['name']}: Correctly rejected")
            else:
                print(f"   ❌ {test['name']}: Should have passed but failed")
                print(f"      Error: {e}")
                return False

    return True


def test_llm_json_simulation():
    """Test 5: Simulate LLM outputting JSON that needs validation"""
    print("\n🤖 Test 5: LLM JSON simulation (Translation Gap)")

    # Simulate LLM output (could be from OpenAI, Anthropic, etc.)
    llm_json = """
    {
        "source_audio_path": "/path/to/song.mp3",
        "separation_request": {
            "source_path": "/path/to/song.mp3",
            "output_dir": "/path/to/stems"
        },
        "beat_alignment": {
            "target_bpm": 120.0
        },
        "stem_transformations": [
            {
                "stem_type": "vocals",
                "target_genre": "neutral"
            },
            {
                "stem_type": "drums",
                "target_genre": "jazz",
                "tempo_shift": 0.15,
                "genre_blend_ratio": 0.7
            }
        ],
        "mix_parameters": {
            "stem_weights": {
                "vocals": 1.0,
                "drums": 1.3
            }
        },
        "output_path": "/path/to/output.wav"
    }
    """

    try:
        # Parse JSON
        data = json.loads(llm_json)

        # Validate against Pydantic schema
        req = OrchestrationRequest(**data)

        print(f"   ✅ LLM JSON validated successfully")
        print(f"   Stems to process: {len(req.stem_transformations)}")
        print(f"   Target BPM: {req.beat_alignment.target_bpm}")
        return True

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"   ❌ LLM JSON validation failed: {e}")
        return False


def test_llm_json_with_errors():
    """Test 6: LLM outputs invalid JSON - should fail gracefully"""
    print("\n🤖 Test 6: LLM JSON with errors (graceful failure)")

    # Simulate LLM output with invalid parameters
    bad_llm_json = """
    {
        "source_audio_path": "/path/to/song.mp3",
        "separation_request": {
            "source_path": "/path/to/song.mp3",
            "output_dir": "/path/to/stems"
        },
        "stem_transformations": [
            {
                "stem_type": "drums",
                "tempo_shift": 2.5,
                "pitch_shift_semitones": 20
            }
        ],
        "output_path": "/path/to/output.wav"
    }
    """

    try:
        data = json.loads(bad_llm_json)
        req = OrchestrationRequest(**data)
        print(f"   ❌ Should have failed but passed")
        return False

    except ValidationError as e:
        print(f"   ✅ Correctly caught invalid parameters:")
        print(f"      {str(e).split(chr(10))[0]}")  # First line of error
        return True


def test_boundary_values():
    """Test 7: Test boundary values (edge cases)"""
    print("\n🎯 Test 7: Boundary value testing")

    boundary_tests = [
        ("Min tempo", -0.5),
        ("Max tempo", 0.5),
        ("Min pitch", -12),
        ("Max pitch", 12),
        ("Min volume", -60.0),
        ("Max volume", 0.0),
        ("Min blend", 0.0),
        ("Max blend", 1.0),
    ]

    for name, value in boundary_tests:
        try:
            if "tempo" in name.lower():
                t = StemTransformation(stem_type="drums", tempo_shift=value)
            elif "pitch" in name.lower():
                t = StemTransformation(stem_type="vocals", pitch_shift_semitones=value)
            elif "volume" in name.lower():
                t = StemTransformation(stem_type="bass", volume_db=value)
            else:
                t = StemTransformation(stem_type="other", genre_blend_ratio=value)
            print(f"   ✅ {name}: {value}")
        except ValidationError:
            print(f"   ❌ {name}: {value} - Should have passed")
            return False

    return True


def main():
    """Run all manual validation tests"""
    print("=" * 60)
    print("STEMFUSE PHASE 1 MANUAL VALIDATION")
    print("=" * 60)

    tests = [
        test_imports,
        test_simple_stem_transformation,
        test_complex_orchestration,
        test_invalid_parameters,
        test_llm_json_simulation,
        test_llm_json_with_errors,
        test_boundary_values
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n🎉 Phase 1 validation complete - ready for Phase 2!")
        return 0
    else:
        print("\n⚠️  Some tests failed - review errors above")
        return 1


if __name__ == "__main__":
    exit(main())

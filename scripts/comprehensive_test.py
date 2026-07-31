"""Comprehensive test suite for StemFuse pipeline."""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from separation.stem_separator import separate
from dsp_processing.mixer import mix_stems
from dsp_processing.beat_aligner import detect_tempo, time_stretch


def print_section(title):
    print(f"\n{'='*60}")
    print(f'{title:^60}')
    print('='*60)


def print_result(test_name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")


class ComprehensiveTest:
    def __init__(self):
        self.test_files = [
            ('edm.mp3', 'Electronic', 256, 80),
            ('Classical.mp3', 'Classical', 256, 133),
            ('jazz.mp3', 'Jazz', 256, 144),
            ('pop.mp3', 'Pop', 256, 129),
            ('rock.mp3', 'Rock', 128, 93),
            ('Blues.mp3', 'Blues', 128, 183)
        ]
        self.results = {}
        self.output_base = 'data/test_tracks/comprehensive_test'

    def test_all_files_end_to_end(self):
        print_section("TEST 1: END-TO-END PIPELINE (ALL FILES)")

        for filename, genre, bitrate, duration in self.test_files:
            test_name = f"E2E: {filename} ({genre})"
            input_path = f'data/test_tracks/{filename}'
            output_dir = f'{self.output_base}/{filename.replace(".", "_")}'
            final_output = f'{output_dir}/final_mixed.wav'

            try:
                start_time = time.time()
                separated = separate(input_path, output_dir)
                mix_stems(separated, final_output)
                elapsed = time.time() - start_time

                if os.path.exists(final_output):
                    size = os.path.getsize(final_output) / (1024*1024)
                    print_result(test_name, True, f"{size:.2f} MB, {elapsed:.1f}s")
                    self.results[test_name] = True
                else:
                    print_result(test_name, False, "Output not created")
                    self.results[test_name] = False

            except Exception as e:
                print_result(test_name, False, f"Error: {str(e)[:50]}")
                self.results[test_name] = False

    def test_beat_detection_all_stems(self):
        print_section("TEST 2: BEAT DETECTION (ALL STEMS)")

        test_file = 'edm.mp3'
        output_dir = f'{self.output_base}/beat_detection'

        try:
            separated = separate(f'data/test_tracks/{test_file}', output_dir)

            for stem_name, stem_path in separated.items():
                try:
                    tempo = detect_tempo(stem_path)
                    print_result(f"Beat: {stem_name}", True, f"{tempo:.2f} BPM")
                except Exception as e:
                    print_result(f"Beat: {stem_name}", False, f"Error: {str(e)[:30]}")

        except Exception as e:
            print_result("Beat detection setup", False, f"Error: {str(e)[:50]}")

    def test_time_stretching_scenarios(self):
        print_section("TEST 3: TIME-STRETCHING SCENARIOS")

        test_file = 'edm.mp3'
        output_dir = f'{self.output_base}/time_stretch'

        try:
            separated = separate(f'data/test_tracks/{test_file}', output_dir)

            scenarios = [
                ("Speed up 20%", 1.2),
                ("Slow down 20%", 0.8),
                ("Double speed", 2.0),
                ("Half speed", 0.5),
            ]

            for desc, ratio in scenarios:
                try:
                    input_stem = separated['drums']
                    output_file = f'{output_dir}/drums_{ratio}.wav'
                    original_tempo = detect_tempo(input_stem)
                    target_tempo = original_tempo * ratio
                    time_stretch(input_stem, output_file, target_tempo)

                    if os.path.exists(output_file):
                        size = os.path.getsize(output_file) / 1024
                        print_result(f"Stretch: {desc}", True, f"Ratio: {ratio:.2f}x, {size:.1f} KB")
                    else:
                        print_result(f"Stretch: {desc}", False, "No output")

                except Exception as e:
                    print_result(f"Stretch: {desc}", False, f"Error: {str(e)[:30]}")

        except Exception as e:
            print_result("Time-stretch setup", False, f"Error: {str(e)[:50]}")

    def test_volume_configurations(self):
        print_section("TEST 4: VOLUME CONFIGURATIONS")

        test_file = 'edm.mp3'
        output_dir = f'{self.output_base}/volumes'

        try:
            separated = separate(f'data/test_tracks/{test_file}', output_dir)

            configurations = [
                ("Balanced", {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}),
                ("Drums boost", {'vocals': 0.9, 'drums': 1.4, 'bass': 1.0, 'other': 0.8}),
                ("Bass heavy", {'vocals': 0.8, 'drums': 1.0, 'bass': 1.5, 'other': 0.7}),
                ("Vocals forward", {'vocals': 1.3, 'drums': 0.9, 'bass': 1.0, 'other': 0.8}),
                ("Muted other", {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 0.3}),
            ]

            for desc, volumes in configurations:
                try:
                    output_file = f'{output_dir}/mix_{desc.replace(" ", "_")}.wav'
                    mix_stems(separated, output_file, volumes)

                    if os.path.exists(output_file):
                        size = os.path.getsize(output_file) / (1024*1024)
                        print_result(f"Mix: {desc}", True, f"{size:.2f} MB")
                    else:
                        print_result(f"Mix: {desc}", False, "No output")

                except Exception as e:
                    print_result(f"Mix: {desc}", False, f"Error: {str(e)[:30]}")

        except Exception as e:
            print_result("Volume setup", False, f"Error: {str(e)[:50]}")

    def print_summary(self):
        print_section("TEST SUMMARY")
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r)
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        if failed > 0:
            print("\nFailed Tests:")
            for test_name, result in self.results.items():
                if not result:
                    print(f"  - {test_name}")


def main():
    print_section("COMPREHENSIVE STEM FUSE TESTING")
    print("Testing all components across all test files")
    print("Output directory: data/test_tracks/comprehensive_test/")

    tester = ComprehensiveTest()
    tester.test_all_files_end_to_end()
    tester.test_beat_detection_all_stems()
    tester.test_time_stretching_scenarios()
    tester.test_volume_configurations()
    tester.print_summary()


if __name__ == "__main__":
    main()

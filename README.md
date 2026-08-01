# StemFuse — Semantic Audio Orchestrator

Take an existing song, split it into stems, and reshape it with plain-English instructions instead of manual EQ/tempo/pitch controls:

```bash
python pipeline.py "song.mp3" "make the drums hit harder with 60% metal energy, give the bass a warm funk groove"
```

## How it's different

| | Suno / Udio | Moises | BandLab | StemFuse |
|---|---|---|---|---|
| Starts from | a text prompt | your song | your song | your song |
| Output | a newly generated song | separated stems + manual controls | manually edited multi-track | your song, transformed, by natural language |
| Control | prompt-only, no source material | manual sliders per stem | manual editing | LLM interprets intent → per-stem DSP |

The core bet: genre fusion at the **stem** level, not the whole mix. "70% jazz on the drums, 40% funk on the bass, vocals untouched" is a materially different (and harder) problem than "make the whole song jazz."

## Architecture

```
Natural language prompt
        │
        ▼
┌───────────────────┐
│  LLM parser        │  Groq (Llama 3.3 70B), OpenAI-compatible SDK
│  (nlp_intent/)      │  falls back to keyword matching on API/parse failure
└───────────────────┘
        │  raw JSON
        ▼
┌───────────────────┐
│  Pydantic schema    │  bounded validation: out-of-range or malformed
│  (schemas/)          │  values are rejected here, never reach the DSP engine
└───────────────────┘
        │  validated OrchestrationRequest
        ▼
┌───────────────────┐
│  Stem separation    │  Demucs → vocals / drums / bass / other
│  (separation/)       │
└───────────────────┘
        │  4 stems
        ▼
┌───────────────────┐
│  Per-stem DSP        │  tempo shift → pitch shift (formant-preserving) →
│  (dsp_processing/)   │  genre EQ preset → manual EQ → volume
└───────────────────┘
        │  4 transformed stems
        ▼
┌───────────────────┐
│  Mixing              │  per-stem balance weights + master volume,
│  (dsp_processing/)   │  peak-normalized
└───────────────────┘
        │
        ▼
   Output WAV
```

## Worked example

**Prompt:** `"Speed up the tempo by 20%"`

The LLM parser turns that into a structured request — note it distributes the tempo change across *all four* stems, because the schema requires uniform `tempo_shift` (see [Validation boundary](#validation-boundary) below):

```json
{
  "stem_transformations": [
    {"stem_type": "vocals", "tempo_shift": 0.2},
    {"stem_type": "drums",  "tempo_shift": 0.2},
    {"stem_type": "bass",   "tempo_shift": 0.2},
    {"stem_type": "other",  "tempo_shift": 0.2}
  ]
}
```

Running the pipeline prints each stage as it completes:

```
[Step 1] Parsing natural language...
   [PASS] Parsed transformations:
      - vocals: tempo=+20%
      - drums: tempo=+20%
      - bass: tempo=+20%
      - other: tempo=+20%

[Step 2] Separating stems...
   [PASS] Separated 4 stems

[Step 3] Applying transformations...
   [PASS] Transformed 4 stems

[Step 4] Mixing stems...
   [PASS] Mixed output

[SUCCESS] Pipeline completed
Output: output/song_transformed/final_output.wav
```

## Validation boundary

Every value the LLM can produce is bounded before it reaches the audio engine — the schema's job is to make "the LLM hallucinated a value" a clean validation error instead of corrupted audio:

| Field | Bounds |
|---|---|
| `tempo_shift` | ±50% |
| `pitch_shift_semitones` | ±12 (one octave) |
| EQ gain (low/mid/high) | ±12 dB |
| `volume_db` | -60 to 0 dB |
| `genre_blend_ratio` | 0.0–1.0 |
| Stems per request | 1–4, no duplicates |

Two invariants are enforced structurally, not just range-checked: a request can name at most one transformation per stem type (no duplicate `"drums"` entries), and if any stem gets a `tempo_shift`, **all four** stems must get the identical value — time-stretching only some stems desyncs the mix (each stem would run at a different speed for its entire length, not just drift at the end), so the schema makes that state unrepresentable rather than relying on every caller to remember the rule.

## What works today

- **Stem separation** — Demucs splits a track into vocals / drums / bass / other.
- **Natural language → DSP parameters** — Groq/Llama 3.3 70B reads a prompt and produces structured, per-stem transformations; a keyword-matching fallback covers LLM/API failures so the pipeline degrades instead of crashing.
- **Per-stem DSP**, applied independently then mixed back together:
  - Tempo shift (pyrubberband time-stretch)
  - Pitch shift, formant-preserving by default (prevents the "chipmunk effect")
  - Genre EQ presets for 11 genres (jazz, rock, metal, electronic, classical, reggae, hiphop, pop, country, funk, neutral) — a 3-band (low/mid/high) shape tuned per stem type
  - Manual EQ and volume control
- **Mixing** — per-stem balance weights and master output level, peak-normalized to prevent clipping.
- **80 automated tests**, `pytest` from repo root, 0 collection errors, runs in seconds. DSP correctness is checked against real transformed audio (FFT-magnitude measurements of actual output), not just mocked function calls.

## Roadmap / known limitations

- **Beat-grid alignment** — stems aren't yet aligned to a common BPM before transformation. `dsp_processing/beat_aligner.py` has working tempo-detection and time-stretch functions; wiring them into the main pipeline is the next planned step.
- **Genre EQ is preset-based**, not model-driven — it's a real, audible effect but not full-fidelity genre transfer. No compression, saturation, or reverb yet.
- **Dynamic range compression** — the schema has a slot for it (`MixParameters.compression_threshold_db`); the compressor itself doesn't exist yet.

## Quick start

```bash
git clone https://github.com/krogdorlok/stemfuse.git
cd StemFuse

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

brew install rubberband   # time-stretch/pitch-shift CLI dependency

# .env in repo root:
echo "GROQ_API_KEY=your_key_here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env

# Scripted:
python pipeline.py "data/test_tracks/song.mp3" "make it soulful and soothing, go easy on the bass"

# Or interactive (no args) for prompted input:
python pipeline.py
```

Output lands at `output/<song_name>_transformed/final_output.wav`. First run downloads the ~2GB Demucs model.

## Tech stack

Python 3.10+ · [Demucs](https://github.com/facebookresearch/demucs) (stem separation) · [pyrubberband](https://github.com/bmcfee/pyrubberband) (tempo/pitch) · [librosa](https://librosa.org/) (audio analysis) · [Pydantic v2](https://docs.pydantic.dev/) (validation boundary) · Groq / Llama 3.3 70B via an OpenAI-compatible SDK · pytest

## Running the tests

```bash
pytest -v
```

80 tests, real assertions, no external dependencies (no audio files, no API key needed). Manual/integration scripts that *do* need real audio or a live API key live in `scripts/` and aren't part of the automated suite — run them directly, e.g. `python scripts/comprehensive_test.py`.

## Project structure

```
StemFuse/
├── separation/          # Demucs stem separation
├── dsp_processing/      # Tempo/pitch shift, genre EQ, mixing
├── nlp_intent/          # LLM prompt parsing (Groq) + system prompt
├── schemas/             # Pydantic validation boundary between LLM and DSP
├── tests/               # pytest suite (80 tests)
├── scripts/             # Manual/integration checks - need real audio or a live API key
├── data/test_tracks/    # Sample audio for local testing (not committed)
├── pipeline.py          # Entry point: parse -> separate -> transform -> mix
└── requirements.txt
```

## Design notes

**Validate at the boundary.** The LLM's job is to interpret intent; the schema's job is to make sure nothing out-of-bounds or structurally invalid reaches the audio engine, regardless of what the LLM outputs.

**Per-stem, not global, genre fusion.** Each stem gets independent genre, tempo, pitch, and EQ treatment rather than one transformation applied to the whole mix.

**Bugs get regression tests, not just fixes.** Two real correctness bugs (a tempo-desync between stems, and a volume value being applied twice during mixing) were found by running the actual DSP functions against real audio and measuring the output, not by reading the code — and both are now guarded by tests that reconstruct the exact failure and assert it can't happen again.

## License

MIT — see [LICENSE](LICENSE).

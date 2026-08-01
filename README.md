# StemFuse — Semantic Audio Orchestrator

Take an existing song, split it into stems, and reshape it with plain-English instructions — "make the drums hit harder with metal energy, give the bass a warm funk groove" — instead of manual EQ/tempo/pitch controls.

```bash
python pipeline.py "song.mp3" "make the drums 70% jazz fusion, give the bass a funk groove"
```

## How it's different

- **Suno / Udio** generate new songs. StemFuse transforms songs you already have.
- **Moises** separates stems and gives you manual controls. StemFuse adds natural-language control on top.
- **BandLab** is manual multi-track editing. StemFuse adds AI orchestration.

## What works today

- **Stem separation** — Demucs splits a track into vocals / drums / bass / other.
- **Natural language → DSP parameters** — a Groq-hosted LLM (Llama 3.3 70B) reads a prompt and produces structured, per-stem transformations; a keyword-matching fallback covers LLM/API failures.
- **Pydantic validation boundary** — every LLM output is validated against bounded schemas (tempo ±50%, pitch ±12 semitones, EQ ±12dB, etc.) before it ever reaches the DSP engine, so a malformed or out-of-range instruction fails fast instead of corrupting audio.
- **Per-stem DSP** — tempo shift, pitch shift (formant-preserving), genre-flavored EQ presets (11 genres), manual EQ, and volume, applied independently per stem, then mixed back together.
- **80 automated tests**, `pytest` from repo root, 0 collection errors, runs in seconds. Covers schema validation, DSP correctness (verified against real transformed audio, not just mocks), and full pipeline orchestration.

## Roadmap / known limitations

- **Beat-grid alignment** — stems aren't yet aligned to a common BPM before transformation. `dsp_processing/beat_aligner.py` has working tempo-detection and time-stretch functions; wiring them into the main pipeline is the next planned step.
- **Genre EQ is preset-based**, not model-driven — it's a real, audible effect but not full-fidelity genre transfer. No compression/saturation/reverb yet.
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
├── pipeline.py           # Entry point: parse -> separate -> transform -> mix
└── requirements.txt
```

## Design notes

**Validate at the boundary.** The LLM's job is to interpret intent; the Pydantic schema's job is to make sure nothing out-of-bounds ever reaches the audio engine, regardless of what the LLM says. This is the seam where "the LLM hallucinated a value" turns into a clean validation error instead of corrupted audio.

**Per-stem, not global, genre fusion.** "60% metal drums + 40% jazz bass" is a materially different (and harder) problem than "make the whole song metal" — each stem gets independent genre, tempo, pitch, and EQ treatment.

## License

MIT — see [LICENSE](LICENSE).

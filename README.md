# StemFuse - Semantic Audio Orchestrator

**Phase:** 1 - Project Initialization ✅  
**Status:** DSP Schemas and Validation Framework Complete

## Project Goal

Build a **Semantic Audio Orchestrator** that:
1. Takes an **existing song** (not generation)
2. Separates it into **stems** (vocals, drums, bass, other)
3. Applies **beat-perfect genre fusion** with natural language control
4. Maintains **rhythmic coherence** (beat-synchronous alignment)

## Differentiation

- **Suno AI/Udio:** Generate new songs (we transform existing songs)
- **Moises AI:** Stem separation + manual controls (we add semantic NL control)
- **BandLab:** Multi-track editing (we add AI orchestration)

## Three Pillars of Differentiation

1. **Beat-perfect semantic fusion** - Automatic rhythmic alignment
2. **Natural language stem control** - LLM + RAG interprets musical semantics
3. **Genre fusion not transfer** - Per-stem genre blending: "60% metal + 40% jazz"

## Architecture

### Phase 1 ✅ (Complete)
- Pydantic DSP parameter schemas
- Validation framework with bounded constraints
- Unit test suite (37 tests, all passing)
- Project structure initialized

### Phase 2 (Next)
- Demucs stem separation integration
- Librosa/pyrubberband beat alignment
- Unified beat-grid architecture

### Phase 3
- DSP mixing engine (EQ, volume, compression)
- Hardcoded transformations (test DSP limits)

### Phase 4
- LLM integration for semantic parsing
- Natural language prompt → DSP parameters
- Pydantic validation bridge

## Project Structure

```
StemFuse/
├── separation/          # Stem separation (Demucs)
├── dsp_processing/      # Beat alignment, time-stretch, mixing
├── nlp_intent/          # LLM integration
├── tests/              # Unit/integration tests
├── schemas/            # Pydantic DSP parameter schemas ✅
├── data/
│   └── test_tracks/    # User-provided audio files
├── config/             # Configuration, constants
├── venv/               # Virtual environment
├── requirements.txt    # Dependencies ✅
└── README.md
```

## Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/test_schemas.py -v
```

## Key Design Decisions

### Pydantic Schemas Before LLM
Prevents "Translation Gap" — LLM outputs validated against DSP bounds before processing.

### Beat-Perfect Alignment as Moat
Most remix tools ignore this → amateurish results. We use librosa/pyrubberband for unified beat grid.

### Per-Stem Genre Fusion
Different from simple "make everything jazz":
```
Vocals: Keep emotionally neutral
Drums: Apply jazz swing (60%)
Guitar: Apply metal aggression (40%)
Bass: Apply reggae offbeat pattern
```

## Testing

All tests validate boundary conditions to prevent audio artifacts:
- Tempo: 60-200 BPM
- Time-stretch: 0.5-2.0 ratio
- Pitch shift: +/- 12 semitones
- EQ: +/- 12 dB
- Volume: -60 to 0 dB

## Known Bottlenecks

1. **Translation Gap** - LLM outputs → Pydantic validation → DSP engine
2. **Audio Artifacts** - Demucs spectral bleeding, pyrubberband artifacts
3. **Compute** - Modular architecture for async scaling

## Python Version

Python 3.10+ (currently using 3.12.3)

## License

TBD

"""Natural language parser for music prompts."""

from typing import Optional
from pydantic import ValidationError

from schemas.dsp_parameters import (
    StemType,
    GenreStyle,
    StemTransformation,
    OrchestrationRequest,
    StemSeparationRequest,
    BeatGridAlignment,
    MixParameters
)


def parse_prompt(
    prompt: str,
    input_path: str,
    output_path: str
) -> Optional[OrchestrationRequest]:
    """
    Parse natural language prompt into orchestration request.

    Args:
        prompt: Natural language description (e.g., "Make drums jazzy with 60% fusion")
        input_path: Path to input audio file
        output_path: Path for output audio file

    Returns:
        Validated OrchestrationRequest or None if parsing fails

    Examples:
        >>> parse_prompt("Apply jazz swing to drums (60% fusion)", "input.mp3", "output.wav")
        >>> parse_prompt("Tempo up 20%, bass down 3dB", "input.mp3", "output.wav")
    """
    try:
        # For now, use simple keyword matching
        # In Phase 4b, replace with actual LLM call
        transformations = _extract_transformations(prompt)

        # Create orchestration request
        request = OrchestrationRequest(
            source_audio_path=input_path,
            separation_request=StemSeparationRequest(
                source_path=input_path,
                output_dir=output_path + "_stems"
            ),
            beat_alignment=BeatGridAlignment(),
            stem_transformations=transformations,
            mix_parameters=MixParameters(),
            output_path=output_path
        )

        return request

    except (ValidationError, ValueError) as e:
        print(f"Parse error: {e}")
        return None


def _extract_transformations(prompt: str) -> list[StemTransformation]:
    """
    Extract stem transformations from prompt using keyword matching.

    This is a placeholder for LLM-based parsing.
    Phase 4b will replace this with actual LLM integration.
    """
    prompt_lower = prompt.lower()
    transformations = []

    # Simple keyword-based extraction
    stem_keywords = {
        'drums': StemType.DRUMS,
        'vocals': StemType.VOCALS,
        'bass': StemType.BASS,
        'other': StemType.OTHER,
        'guitar': StemType.OTHER,
        'piano': StemType.OTHER,
        'strings': StemType.OTHER,
    }

    genre_keywords = {
        'jazz': GenreStyle.JAZZ,
        'rock': GenreStyle.ROCK,
        'metal': GenreStyle.METAL,
        'electronic': GenreStyle.ELECTRONIC,
        'classical': GenreStyle.CLASSICAL,
        'reggae': GenreStyle.REGGAE,
        'hip hop': GenreStyle.HIPHOP,
        'hiphop': GenreStyle.HIPHOP,
        'pop': GenreStyle.POP,
        'country': GenreStyle.COUNTRY,
        'funk': GenreStyle.FUNK,
    }

    # Find mentioned stems
    mentioned_stems = set()
    for keyword, stem_type in stem_keywords.items():
        if keyword in prompt_lower:
            mentioned_stems.add(stem_type)

    # If no specific stems mentioned, apply to all
    if not mentioned_stems:
        mentioned_stems = {StemType.VOCALS, StemType.DRUMS, StemType.BASS, StemType.OTHER}

    # Find mentioned genres
    mentioned_genre = None
    for keyword, genre in genre_keywords.items():
        if keyword in prompt_lower:
            mentioned_genre = genre
            break

    # Create transformations for each mentioned stem
    for stem_type in mentioned_stems:
        transformations.append(StemTransformation(
            stem_type=stem_type,
            target_genre=mentioned_genre
        ))

    return transformations


def validate_llm_output(llm_json: dict) -> Optional[OrchestrationRequest]:
    """
    Validate LLM JSON output against Pydantic schema.

    Args:
        llm_json: Dictionary from LLM output

    Returns:
        Validated OrchestrationRequest or None if invalid
    """
    try:
        return OrchestrationRequest(**llm_json)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None

"""Natural language parser for music prompts using LLM."""

import os
import json
from typing import Optional
from pathlib import Path
from pydantic import ValidationError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: openai package not installed. Run: pip install openai")

from schemas.dsp_parameters import (
    StemType,
    GenreStyle,
    StemTransformation,
    OrchestrationRequest,
    StemSeparationRequest,
    BeatGridAlignment,
    MixParameters
)


# Load system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
with open(SYSTEM_PROMPT_PATH, 'r') as f:
    SYSTEM_PROMPT = f.read()


def _get_llm_client():
    """Initialize LLM client from environment variables."""
    if OpenAI is None:
        raise ImportError("openai package not installed")

    # Try GLM-5 first (Zhipu AI, OpenAI-compatible)
    api_key = os.getenv("GLM_API_KEY")
    base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

    # Fallback to OpenAI
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        raise ValueError(
            "No API key found. Set GLM_API_KEY or OPENAI_API_KEY environment variable."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm(prompt: str, model: str = "glm-4-plus") -> str:
    """
    Call LLM with the system prompt and user prompt.

    Args:
        prompt: User's natural language request
        model: Model name (default: glm-4-plus for GLM-5)

    Returns:
        Raw JSON string from LLM
    """
    client = _get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for consistent JSON output
        max_tokens=1000,
    )

    return response.choices[0].message.content


def parse_prompt(
    prompt: str,
    input_path: str,
    output_path: str,
    use_llm: bool = True,
    model: str = "glm-4-flash"
) -> Optional[OrchestrationRequest]:
    """
    Parse natural language prompt into orchestration request.

    Args:
        prompt: Natural language description (e.g., "Make drums jazzy with 60% fusion")
        input_path: Path to input audio file
        output_path: Path for output audio file
        use_llm: If True, use actual LLM; if False, use keyword extraction (fallback)
        model: Model name for LLM call

    Returns:
        Validated OrchestrationRequest or None if parsing fails

    Examples:
        >>> parse_prompt("Apply jazz swing to drums (60% fusion)", "input.mp3", "output.wav")
        >>> parse_prompt("Tempo up 20%, bass down 3dB", "input.mp3", "output.wav")
    """
    try:
        if use_llm:
            # Use actual LLM
            llm_output = _call_llm(prompt, model=model)
            # Parse JSON from LLM output
            try:
                llm_json = json.loads(llm_output)
            except json.JSONDecodeError:
                # LLM might have wrapped in markdown fences
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()
                llm_json = json.loads(llm_output)

            transformations_data = llm_json.get("stem_transformations", [])
            transformations = [StemTransformation(**t) for t in transformations_data]
        else:
            # Fallback to keyword extraction
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

    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        print(f"Parse error: {e}")
        return None
    except Exception as e:
        print(f"LLM error: {e}")
        print("Falling back to keyword extraction...")
        return parse_prompt(prompt, input_path, output_path, use_llm=False)


def _extract_transformations(prompt: str) -> list[StemTransformation]:
    """
    Extract stem transformations from prompt using keyword matching.

    This is a fallback when LLM is not available.
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

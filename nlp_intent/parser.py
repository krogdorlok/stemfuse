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

from schemas.dsp_parameters import (
    StemType,
    GenreStyle,
    StemTransformation,
    OrchestrationRequest,
    StemSeparationRequest,
    BeatGridAlignment,
    MixParameters
)


SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
with open(SYSTEM_PROMPT_PATH, 'r') as f:
    SYSTEM_PROMPT = f.read()


def _get_llm_client():
    """Initialize Groq client via OpenAI-compatible SDK."""
    if OpenAI is None:
        raise ImportError("openai package not installed")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No GROQ_API_KEY found")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return client, model


def _call_llm(prompt: str, model: str = None) -> str:
    """Call Groq LLM with system prompt and user prompt."""
    client, default_model = _get_llm_client()
    if model is None:
        model = default_model

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def parse_prompt(
    prompt: str,
    input_path: str,
    output_path: str,
    use_llm: bool = True,
    model: str = None
) -> Optional[OrchestrationRequest]:
    """Parse natural language prompt into orchestration request."""
    try:
        if use_llm:
            llm_output = _call_llm(prompt, model=model)
            try:
                llm_json = json.loads(llm_output)
            except json.JSONDecodeError:
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()
                llm_json = json.loads(llm_output)

            transformations_data = llm_json.get("stem_transformations", [])
            transformations = [StemTransformation(**t) for t in transformations_data]
        else:
            transformations = _extract_transformations(prompt)

        return OrchestrationRequest(
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

    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        print(f"Parse error: {e}")
        return None
    except Exception as e:
        print(f"LLM error: {e}")
        print("Falling back to keyword extraction...")
        return parse_prompt(prompt, input_path, output_path, use_llm=False)


def _extract_transformations(prompt: str) -> list[StemTransformation]:
    """Extract stem transformations using keyword matching (fallback)."""
    prompt_lower = prompt.lower()
    transformations = []

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

    mentioned_stems = set()
    for keyword, stem_type in stem_keywords.items():
        if keyword in prompt_lower:
            mentioned_stems.add(stem_type)

    if not mentioned_stems:
        mentioned_stems = {StemType.VOCALS, StemType.DRUMS, StemType.BASS, StemType.OTHER}

    mentioned_genre = None
    for keyword, genre in genre_keywords.items():
        if keyword in prompt_lower:
            mentioned_genre = genre
            break

    for stem_type in mentioned_stems:
        transformations.append(StemTransformation(
            stem_type=stem_type,
            target_genre=mentioned_genre
        ))

    return transformations


def validate_llm_output(llm_json: dict) -> Optional[OrchestrationRequest]:
    """Validate LLM JSON output against Pydantic schema."""
    try:
        return OrchestrationRequest(**llm_json)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None

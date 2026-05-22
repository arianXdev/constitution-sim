"""LLM provider adapters.

The simulator must work without LLMs (deterministic fallback in `LLMAgent`).
These adapters are intentionally thin: they translate a single prompt string
into a single JSON-only response string. All structured validation happens
in the agent / rules engine, never here.

Default model identifiers are kept current (as of 2025-Q4) but can be
overridden via the CLI's `--model` flag.
"""

import json
import logging
import os
import re
from typing import Callable

logger = logging.getLogger(__name__)

# Current default models. Override with CLI --model. Pinning to dated
# snapshots keeps simulations reproducible across LLM provider releases.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Pull the first JSON object out of a possibly-noisy LLM response."""
    if not text:
        return "{}"
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            return match.group(0)
    return text  # let downstream parser raise so caller can fall back


def get_openai_callable(model: str = DEFAULT_OPENAI_MODEL) -> Callable[[str], str]:
    """Build a callable that sends `prompt` to OpenAI and returns the JSON text."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before requesting an OpenAI-backed agent, "
            "or use --agent-type heuristic / llm-mock."
        )
    client = openai.OpenAI(api_key=api_key)

    def call_llm(prompt: str) -> str:
        # response_format=json_object requires the word "JSON" in the prompt;
        # _build_prompt already includes it but we re-affirm here.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a political-agent policy module. Reply with a "
                        "single JSON object describing one action. No prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response.choices[0].message.content or "{}"

    return call_llm


def get_anthropic_callable(model: str = DEFAULT_ANTHROPIC_MODEL) -> Callable[[str], str]:
    """Build a callable that sends `prompt` to Anthropic and returns JSON text."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before requesting an Anthropic-backed agent, "
            "or use --agent-type heuristic / llm-mock."
        )
    client = anthropic.Anthropic(api_key=api_key)

    def call_llm(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            system=(
                "You are a political-agent policy module. Reply with a single "
                "JSON object describing one action. Output ONLY the JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; pick the first text block.
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                return _extract_json_object(text)
        return "{}"

    return call_llm


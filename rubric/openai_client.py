"""Small synchronous adapter for the official OpenAI Responses API.

Retry policy is owned by the reward manager so failed attempts are visible in
training metrics.  The SDK client therefore disables its implicit retries.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def _client(timeout: float) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": timeout,
    }
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return OpenAI(**kwargs)


def call_openai_sync(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    """Return response text from one standard OpenAI Responses API request."""
    if not model.strip():
        raise ValueError("an OpenAI model name is required")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    request: dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
        "store": False,
    }
    if os.getenv("OPENAI_OMIT_TEMPERATURE", "0") != "1":
        request["temperature"] = temperature

    response = _client(timeout).responses.create(**request)
    text = response.output_text
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenAI response contained no output text")
    return text.strip()

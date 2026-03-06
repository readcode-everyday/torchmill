"""LLM provider clients — all use the openai-compatible API surface.

Supported providers:
  - dashscope  → Qwen-Plus via DashScope compatible API
  - anthropic  → Claude via Anthropic API (openai-compat mode)
  - openai     → OpenAI standard API
"""

from __future__ import annotations

import os
from typing import Optional


def get_client(provider: str, api_key: Optional[str] = None):
    """Return an openai.OpenAI client configured for the given provider."""
    from openai import OpenAI

    provider = provider.lower()

    if provider == "dashscope":
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        return OpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Anthropic's openai-compat endpoint
        return OpenAI(
            api_key=key,
            base_url="https://api.anthropic.com/v1/",
        )
    elif provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAI(api_key=key)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: dashscope, anthropic, openai."
        )

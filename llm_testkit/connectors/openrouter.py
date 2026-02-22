"""
OpenRouter connector for llm-testkit.

OpenRouter provides a unified API to 200+ models.
Uses the OpenAI-compatible API with a different base URL.
Requires: pip install openai
"""

import os
import time
from typing import Optional

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterConnector(BaseConnector):
    """
    Connector for OpenRouter. Access 200+ models through one API.

    The API key can be passed directly or set via the OPENROUTER_API_KEY
    environment variable.

    Usage:
        from llm_testkit.connectors import OpenRouterConnector

        bot = OpenRouterConnector(
            model="anthropic/claude-sonnet-4-20250514",
            api_key="sk-or-...",
        )
        response = bot.respond("What is your refund policy?")

    Popular models:
        - anthropic/claude-sonnet-4-20250514
        - openai/gpt-4o
        - google/gemini-2.5-flash-preview-05-20
        - deepseek/deepseek-chat-v3-0324
        - meta-llama/llama-4-maverick
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        super().__init__(model=model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers={
                        "HTTP-Referer": "https://github.com/tenypeter007/llm-testkit",
                        "X-Title": "llm-testkit",
                    },
                )
            except ImportError:
                raise ImportError(
                    "openai package required. "
                    "Install with: pip install openai"
                )
        return self._client

    def respond(self, prompt: str) -> Response:
        """Send a prompt via OpenRouter and return a Response."""
        client = self._get_client()

        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = (time.time() - start) * 1000

        text = (
            completion.choices[0].message.content
            if completion.choices
            else ""
        )
        tokens = (
            completion.usage.total_tokens
            if completion.usage
            else None
        )

        return self.make_response(
            text=text or "",
            latency_ms=elapsed_ms,
            token_count=tokens,
            prompt=prompt,
            metadata={
                "finish_reason": (
                    completion.choices[0].finish_reason
                    if completion.choices
                    else None
                ),
                "model": completion.model,
                "provider": "openrouter",
            },
        )

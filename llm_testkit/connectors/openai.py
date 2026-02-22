"""
OpenAI connector for llm-testkit.
Requires: pip install openai
"""

import os
import time
from typing import Optional

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response


class OpenAIConnector(BaseConnector):
    """
    Connector for OpenAI models (GPT-4o, GPT-4, GPT-3.5-turbo, etc.)

    The API key can be passed directly or set via the OPENAI_API_KEY
    environment variable.

    Usage:
        from llm_testkit.connectors import OpenAIConnector

        bot = OpenAIConnector(model="gpt-4o", api_key="sk-...")
        # or with env var:
        bot = OpenAIConnector(model="gpt-4o")
        response = bot.respond("What is your refund policy?")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        super().__init__(model=model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client

    def respond(self, prompt: str) -> Response:
        """Send a prompt to OpenAI and return a standardised Response."""
        client = self._get_client()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = (time.time() - start) * 1000

        return self.make_response(
            text=completion.choices[0].message.content,
            latency_ms=elapsed_ms,
            token_count=completion.usage.total_tokens if completion.usage else None,
            prompt=prompt,
            metadata={
                "finish_reason": completion.choices[0].finish_reason,
                "model": completion.model,
            }
        )

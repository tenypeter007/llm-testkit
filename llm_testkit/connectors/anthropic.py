"""
Anthropic/Claude connector for llm-testkit.
Requires: pip install anthropic
"""

import os
import time
from typing import Optional

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response


class AnthropicConnector(BaseConnector):
    """
    Connector for Anthropic Claude models.

    The API key can be passed directly or set via the ANTHROPIC_API_KEY
    environment variable.

    Usage:
        from llm_testkit.connectors import AnthropicConnector

        bot = AnthropicConnector(model="claude-sonnet-4-6", api_key="sk-ant-...")
        # or with env var:
        bot = AnthropicConnector(model="claude-sonnet-4-6")
        response = bot.respond("What is your refund policy?")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        super().__init__(model=model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def respond(self, prompt: str) -> Response:
        """Send a prompt to Anthropic Claude and return a standardised Response."""
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        start = time.time()
        message = client.messages.create(**kwargs)
        elapsed_ms = (time.time() - start) * 1000

        text = message.content[0].text if message.content else ""
        total_tokens = (
            (message.usage.input_tokens + message.usage.output_tokens)
            if message.usage else None
        )

        return self.make_response(
            text=text,
            latency_ms=elapsed_ms,
            token_count=total_tokens,
            prompt=prompt,
            metadata={
                "stop_reason": message.stop_reason,
                "model": message.model,
            }
        )

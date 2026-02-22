"""
Base connector class. All LLM connectors inherit from this.
To add a custom connector, implement just the `respond` method.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

from llm_testkit.models import Response


class BaseConnector(ABC):
    """
    Abstract base class for all LLM connectors.

    Implement `respond()` to connect llm-testkit to any AI system.

    Example custom connector:

        class MyBot(BaseConnector):
            def respond(self, prompt: str) -> Response:
                start = time.time()
                raw = my_api.call(prompt)
                return self.make_response(
                    text=raw.text,
                    latency_ms=(time.time() - start) * 1000,
                    prompt=prompt,
                )
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model

    @abstractmethod
    def respond(self, prompt: str) -> Response:
        """
        Send a prompt and return a standardised Response object.

        Args:
            prompt: The input text to send to the LLM

        Returns:
            Response object with text, latency, and optional metadata
        """
        pass

    def make_response(
        self,
        text: str,
        latency_ms: float = 0.0,
        token_count: Optional[int] = None,
        prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Response:
        """Helper to construct a standardised Response object."""
        return Response(
            text=text,
            latency_ms=latency_ms,
            token_count=token_count,
            model=self.model,
            prompt=prompt,
            metadata=metadata or {},
        )

    def _time_call(self, fn, *args, **kwargs):
        """Utility: time a function call and return (result, elapsed_ms)."""
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms

"""
Ollama connector for llm-testkit.

Requires: ollama running locally (https://ollama.com)
Optional: pip install ollama
"""

import json
import time
import urllib.request
from typing import Optional

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response


class OllamaConnector(BaseConnector):
    """
    Connector for locally running Ollama models (Llama, Mistral, Phi, Gemma, etc.)

    Runs on your machine. No API key required.

    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ollama pull llama3.2
        3. Use in tests: OllamaConnector(model="llama3.2")

    Usage:
        from llm_testkit.connectors import OllamaConnector

        bot = OllamaConnector(model="llama3.2")
        response = bot.respond("What is your refund policy?")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature

    def respond(self, prompt: str) -> Response:
        """Send a prompt to local Ollama instance and return a standardised Response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Original error: {e}"
            )
        elapsed_ms = (time.time() - start) * 1000

        return self.make_response(
            text=result.get("response", ""),
            latency_ms=elapsed_ms,
            token_count=result.get("eval_count"),
            prompt=prompt,
            metadata={
                "model": result.get("model"),
                "done": result.get("done"),
            }
        )

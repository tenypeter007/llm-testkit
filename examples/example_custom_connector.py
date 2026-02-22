"""
Example: Building a custom connector for any AI API.

If you're testing an AI that isn't OpenAI, Anthropic, or Ollama,
you can build a connector in under 10 lines of code.
"""

import json
import time
import urllib.request

from llm_testkit import assert_response
from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response

# --- Example: Connect to any REST API -------------------------------------

class MyCustomAPIConnector(BaseConnector):
    """
    Example connector for a custom AI REST API.
    Adapt the URL and payload structure to match your API.
    """

    def __init__(self, api_url: str, api_key: str = None):
        super().__init__(model="custom")
        self.api_url = api_url
        self.api_key = api_key

    def respond(self, prompt: str) -> Response:
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.api_url, data=payload, headers=headers)

        start = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        elapsed_ms = (time.time() - start) * 1000

        return self.make_response(
            text=data["response"],       # Adjust key to match your API
            latency_ms=elapsed_ms,
            prompt=prompt,
        )


# --- Example: Wrap a Python function (e.g. a local model) -----------------

class PythonFunctionConnector(BaseConnector):
    """
    Wrap any Python function that takes a string and returns a string.
    Perfect for testing local models, LangChain chains, etc.
    """

    def __init__(self, fn):
        super().__init__(model="python-function")
        self.fn = fn

    def respond(self, prompt: str) -> Response:
        start = time.time()
        text = self.fn(prompt)
        elapsed_ms = (time.time() - start) * 1000
        return self.make_response(text=text, latency_ms=elapsed_ms, prompt=prompt)


# --- Demo -----------------------------------------------------------------

if __name__ == "__main__":
    # Example with a simple Python function
    def my_simple_bot(prompt: str) -> str:
        """A trivially simple 'AI' for demonstration."""
        if "refund" in prompt.lower():
            return "Refunds are available within 30 days of purchase. Please contact support."
        return "Thank you for your message. How can I assist you today?"

    bot = PythonFunctionConnector(my_simple_bot)

    print("Testing custom Python function connector...\n")

    response = bot.respond("How do I get a refund?")
    print(f"Response: {response.text}")
    print(f"Latency:  {response.latency_ms:.1f}ms\n")

    assert_response(response).contains_keywords(["refund", "30 days"])
    assert_response(response).has_no_sensitive_data()
    assert_response(response).tone_is("professional")

    print("All assertions passed!")
    print("\nReplace 'my_simple_bot' with your own model or API call.")

"""
Tests for LLM connectors -- initialization, error handling, and base class.
Run with: pytest tests/test_connectors.py -v
"""

import pytest

from llm_testkit.connectors.anthropic import AnthropicConnector
from llm_testkit.connectors.base import BaseConnector
from llm_testkit.connectors.ollama import OllamaConnector
from llm_testkit.connectors.openai import OpenAIConnector
from llm_testkit.connectors.openrouter import OpenRouterConnector
from llm_testkit.models import Response

# --- BaseConnector tests --------------------------------------------------

class TestBaseConnector:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseConnector()

    def test_make_response_returns_response(self):
        class Dummy(BaseConnector):
            def respond(self, prompt: str) -> Response:
                return self.make_response(text="hello", latency_ms=100, prompt=prompt)

        bot = Dummy(model="test")
        response = bot.respond("Hi")
        assert isinstance(response, Response)
        assert response.text == "hello"
        assert response.latency_ms == 100
        assert response.model == "test"
        assert response.prompt == "Hi"
        assert response.metadata == {}

    def test_make_response_with_metadata(self):
        class Dummy(BaseConnector):
            def respond(self, prompt: str) -> Response:
                return self.make_response(
                    text="hi",
                    metadata={"key": "value"},
                    token_count=5,
                )

        bot = Dummy()
        response = bot.respond("test")
        assert response.metadata == {"key": "value"}
        assert response.token_count == 5

    def test_time_call_utility(self):
        class Dummy(BaseConnector):
            def respond(self, prompt: str) -> Response:
                return self.make_response(text="hi")

        bot = Dummy()
        result, elapsed = bot._time_call(lambda: "done")
        assert result == "done"
        assert elapsed >= 0


# --- OpenAI connector tests ----------------------------------------------

class TestOpenAIConnector:
    def test_init_defaults(self):
        bot = OpenAIConnector(api_key="sk-test")
        assert bot.model == "gpt-4o"
        assert bot.api_key == "sk-test"
        assert bot.temperature == 0.7
        assert bot.max_tokens == 1000
        assert bot.system_prompt is None
        assert bot._client is None

    def test_init_custom_params(self):
        bot = OpenAIConnector(
            model="gpt-3.5-turbo",
            api_key="sk-test",
            system_prompt="You are helpful.",
            temperature=0.3,
            max_tokens=500,
        )
        assert bot.model == "gpt-3.5-turbo"
        assert bot.system_prompt == "You are helpful."
        assert bot.temperature == 0.3
        assert bot.max_tokens == 500

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        bot = OpenAIConnector()
        assert bot.api_key == "sk-from-env"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        bot = OpenAIConnector(api_key="sk-explicit")
        assert bot.api_key == "sk-explicit"


# --- Anthropic connector tests -------------------------------------------

class TestAnthropicConnector:
    def test_init_defaults(self):
        bot = AnthropicConnector(api_key="sk-ant-test")
        assert bot.model == "claude-sonnet-4-6"
        assert bot.api_key == "sk-ant-test"
        assert bot.temperature == 0.7
        assert bot.max_tokens == 1000

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        bot = AnthropicConnector()
        assert bot.api_key == "sk-ant-from-env"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        bot = AnthropicConnector(api_key="sk-ant-explicit")
        assert bot.api_key == "sk-ant-explicit"


# --- Ollama connector tests ---------------------------------------------

class TestOllamaConnector:
    def test_init_defaults(self):
        bot = OllamaConnector()
        assert bot.model == "llama3.2"
        assert bot.base_url == "http://localhost:11434"
        assert bot.temperature == 0.7

    def test_init_custom_params(self):
        bot = OllamaConnector(
            model="mistral",
            base_url="http://remote:11434/",
            system_prompt="Be concise",
            temperature=0.5,
        )
        assert bot.model == "mistral"
        assert bot.base_url == "http://remote:11434"
        assert bot.system_prompt == "Be concise"

    def test_trailing_slash_stripped(self):
        bot = OllamaConnector(base_url="http://localhost:11434/")
        assert bot.base_url == "http://localhost:11434"


# --- OpenRouter connector tests ------------------------------------------

class TestOpenRouterConnector:
    def test_init_defaults(self):
        bot = OpenRouterConnector(api_key="sk-or-test")
        assert bot.model == "openai/gpt-4o"
        assert bot.api_key == "sk-or-test"
        assert bot.temperature == 0.7
        assert bot.max_tokens == 1000
        assert bot.system_prompt is None
        assert bot._client is None
        assert bot.base_url == "https://openrouter.ai/api/v1"

    def test_init_custom_params(self):
        bot = OpenRouterConnector(
            model="anthropic/claude-sonnet-4-20250514",
            api_key="sk-or-test",
            system_prompt="You are helpful.",
            temperature=0.3,
            max_tokens=500,
        )
        assert bot.model == "anthropic/claude-sonnet-4-20250514"
        assert bot.system_prompt == "You are helpful."
        assert bot.temperature == 0.3
        assert bot.max_tokens == 500

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
        bot = OpenRouterConnector()
        assert bot.api_key == "sk-or-from-env"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
        bot = OpenRouterConnector(api_key="sk-or-explicit")
        assert bot.api_key == "sk-or-explicit"

    def test_custom_base_url(self):
        bot = OpenRouterConnector(
            api_key="sk-or-test",
            base_url="https://custom.example.com/v1",
        )
        assert bot.base_url == "https://custom.example.com/v1"

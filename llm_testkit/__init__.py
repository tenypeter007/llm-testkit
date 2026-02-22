"""
llm-testkit -- structured, repeatable testing for LLM outputs.
"""

from llm_testkit.assertions.response import LLMAssertionError, ResponseAsserter
from llm_testkit.checkers.consistency import ConsistencyChecker
from llm_testkit.checkers.hallucination import HallucinationDetector
from llm_testkit.models import Response

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("llm-testkit")
except Exception:
    __version__ = "0.1.1"
__all__ = [
    "assert_response",
    "ConsistencyChecker",
    "HallucinationDetector",
    "LLMAssertionError",
    "Response",
]


def assert_response(response: Response) -> ResponseAsserter:
    """
    Entry point for chaining assertions on an LLM response.

    Usage:
        response = bot.respond("What is your refund policy?")
        assert_response(response).contains_keywords(["refund", "policy"])
        assert_response(response).tone_is("professional")
        assert_response(response).has_no_sensitive_data()
    """
    return ResponseAsserter(response)

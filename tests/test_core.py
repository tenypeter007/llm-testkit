"""
Tests for llm-testkit itself.
Run with: pytest tests/
"""

import pytest

from llm_testkit import assert_response
from llm_testkit.checkers.consistency import ConsistencyChecker
from llm_testkit.checkers.hallucination import HallucinationDetector
from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response

# --- Fixture: Mock connector that returns predictable responses ------------

class MockConnector(BaseConnector):
    """A deterministic connector for testing llm-testkit itself."""

    def __init__(self, fixed_response: str = "This is a test response.", latency_ms: float = 100.0):
        super().__init__(model="mock")
        self.fixed_response = fixed_response
        self.latency_ms = latency_ms
        self.call_count = 0

    def respond(self, prompt: str) -> Response:
        self.call_count += 1
        return Response(
            text=self.fixed_response,
            latency_ms=self.latency_ms,
            token_count=len(self.fixed_response.split()),
            model="mock",
            prompt=prompt,
        )


def make_response(text: str, latency_ms: float = 100.0, token_count: int = None) -> Response:
    """Helper to create a Response for testing."""
    return Response(
        text=text,
        latency_ms=latency_ms,
        token_count=token_count or len(text.split()),
    )


# --- assert_response tests -------------------------------------------------

class TestContainsKeywords:
    def test_passes_when_keywords_present(self):
        r = make_response("Your refund will be processed within 30 days.")
        assert_response(r).contains_keywords(["refund", "30 days"])

    def test_fails_when_keyword_missing(self):
        r = make_response("We cannot help with that request.")
        with pytest.raises(AssertionError, match="missing required keywords"):
            assert_response(r).contains_keywords(["refund"])

    def test_case_insensitive_by_default(self):
        r = make_response("Your REFUND will be processed.")
        assert_response(r).contains_keywords(["refund"])

    def test_case_sensitive_mode(self):
        r = make_response("Your REFUND will be processed.")
        with pytest.raises(AssertionError):
            assert_response(r).contains_keywords(["refund"], case_sensitive=True)

    def test_multiple_keywords_all_required(self):
        r = make_response("We offer refunds and returns within 30 days.")
        assert_response(r).contains_keywords(["refund", "return", "30 days"])


class TestExcludesKeywords:
    def test_passes_when_keywords_absent(self):
        r = make_response("We are happy to help you today.")
        assert_response(r).excludes_keywords(["competitor", "lawsuit"])

    def test_fails_when_banned_keyword_present(self):
        r = make_response("I cannot help with that request.")
        with pytest.raises(AssertionError, match="contains banned keywords"):
            assert_response(r).excludes_keywords(["cannot"])


class TestLengthBetween:
    def test_passes_within_range(self):
        r = make_response("This is a response with exactly seven words.")
        assert_response(r).length_between(5, 10)

    def test_fails_too_short(self):
        r = make_response("Too short.")
        with pytest.raises(AssertionError, match="outside range"):
            assert_response(r).length_between(10, 100)

    def test_fails_too_long(self):
        r = make_response(" ".join(["word"] * 200))
        with pytest.raises(AssertionError, match="outside range"):
            assert_response(r).length_between(10, 50)


class TestHasNoSensitiveData:
    def test_passes_clean_response(self):
        r = make_response("Thank you for your message. We will get back to you shortly.")
        assert_response(r).has_no_sensitive_data()

    def test_fails_with_credit_card(self):
        r = make_response("Your card 4111 1111 1111 1111 has been charged.")
        with pytest.raises(AssertionError, match="sensitive data"):
            assert_response(r).has_no_sensitive_data()

    def test_fails_with_api_key(self):
        r = make_response("Use this key: sk-abcdefghijklmnopqrstuvwxyz123456")
        with pytest.raises(AssertionError, match="sensitive data"):
            assert_response(r).has_no_sensitive_data()

    def test_fails_with_email(self):
        r = make_response("Contact us at private@company.com for more details.")
        with pytest.raises(AssertionError, match="sensitive data"):
            assert_response(r).has_no_sensitive_data()


class TestToneIs:
    def test_professional_tone_passes(self):
        text = (
            "Thank you for reaching out. We appreciate your patience"
            " and will assist you shortly."
        )
        r = make_response(text)
        assert_response(r).tone_is("professional")

    def test_unknown_tone_raises_value_error(self):
        r = make_response("Some response.")
        with pytest.raises(ValueError, match="Unsupported tone"):
            assert_response(r).tone_is("aggressive")


class TestResponseTimeUnder:
    def test_passes_under_threshold(self):
        r = make_response("Quick response.", latency_ms=500)
        assert_response(r).response_time_under(1.0)

    def test_fails_over_threshold(self):
        r = make_response("Slow response.", latency_ms=5000)
        with pytest.raises(AssertionError, match="exceeded threshold"):
            assert_response(r).response_time_under(3.0)


class TestTokenCountUnder:
    def test_passes_under_limit(self):
        r = make_response("Short response.", token_count=10)
        assert_response(r).token_count_under(100)

    def test_fails_over_limit(self):
        r = make_response("Long response with many tokens.", token_count=600)
        with pytest.raises(AssertionError, match="Token count"):
            assert_response(r).token_count_under(500)


class TestChaining:
    def test_multiple_assertions_chain(self):
        r = make_response(
            "Thank you for contacting us. Your refund will be processed within 30 days.",
            latency_ms=800,
            token_count=15,
        )
        # Should not raise
        (assert_response(r)
            .is_not_empty()
            .contains_keywords(["refund", "30 days"])
            .has_no_sensitive_data()
            .tone_is("professional")
            .length_between(5, 50)
            .response_time_under(2.0)
            .token_count_under(100))


# --- ConsistencyChecker tests ----------------------------------------------

class TestConsistencyChecker:
    def test_high_consistency_passes(self):
        connector = MockConnector("Your refund will be processed within 30 days of receipt.")
        checker = ConsistencyChecker(connector, runs=5, threshold=0.85)
        result = checker.test_prompt("What is your refund policy?")

        assert result.passed is True
        assert result.consistency_score >= 0.85
        assert result.runs == 5
        assert connector.call_count == 5

    def test_result_has_required_fields(self):
        connector = MockConnector("Some response.")
        checker = ConsistencyChecker(connector, runs=3)
        result = checker.test_prompt("Test prompt")

        assert result.prompt == "Test prompt"
        assert result.runs == 3
        assert 0.0 <= result.consistency_score <= 1.0
        assert isinstance(result.responses, list)
        assert len(result.responses) == 3

    def test_custom_threshold(self):
        connector = MockConnector("Consistent answer every time.")
        checker = ConsistencyChecker(connector, runs=5, threshold=0.99)
        result = checker.test_prompt("Test")
        # Perfect consistency with mock connector
        assert result.passed is True


# --- HallucinationDetector tests ------------------------------------------

class TestHallucinationDetector:
    SOURCE = """
    Our refund policy allows returns within 30 days of purchase.
    Items must be in original condition with receipt.
    Refunds are processed within 5-7 business days.
    We do not accept returns on digital downloads.
    """

    def test_supported_claim_passes(self):
        detector = HallucinationDetector(source_doc=self.SOURCE, threshold=0.5)
        response = make_response("Returns are accepted within 30 days of purchase.")
        result = detector.check(response)
        assert result.hallucination_score <= 0.5

    def test_result_has_required_fields(self):
        detector = HallucinationDetector(source_doc=self.SOURCE)
        response = make_response("Our policy is clear and transparent.")
        result = detector.check(response)

        assert isinstance(result.hallucination_score, float)
        assert 0.0 <= result.hallucination_score <= 1.0
        assert isinstance(result.flagged_sentences, list)
        assert isinstance(result.verified_sentences, list)
        assert isinstance(result.passed, bool)

    def test_accepts_raw_text_source(self):
        detector = HallucinationDetector(source_doc="Refunds take 30 days.")
        response = make_response("Refunds are processed in 30 days.")
        result = detector.check(response)
        assert result is not None

    def test_empty_response_passes(self):
        detector = HallucinationDetector(source_doc=self.SOURCE)
        response = make_response("")
        result = detector.check(response)
        assert result.passed is True
        assert result.hallucination_score == 0.0


# --- RedTeamSuite tests ----------------------------------------------------

class TestRedTeamSuite:
    def test_loads_prompts(self):
        from llm_testkit.redteam import RedTeamSuite
        connector = MockConnector("I cannot help with that. This goes against my guidelines.")
        suite = RedTeamSuite(connector)
        assert len(suite.available_categories()) > 0

    def test_run_returns_result(self):
        from llm_testkit.redteam import RedTeamSuite
        connector = MockConnector("I'm sorry, I cannot assist with that request.")
        suite = RedTeamSuite(connector)
        result = suite.run(
            categories=["identity_confusion"],
            max_per_category=3,
            verbose=False
        )
        assert result.total_prompts > 0
        assert 0.0 <= result.risk_score <= 10.0
        assert isinstance(result.results_by_category, dict)

    def test_invalid_category_raises(self):
        from llm_testkit.redteam import RedTeamSuite
        connector = MockConnector("response")
        suite = RedTeamSuite(connector)
        with pytest.raises(ValueError, match="Unknown category"):
            suite.run(categories=["nonexistent_category"], verbose=False)

    def test_safe_bot_has_low_risk_score(self):
        from llm_testkit.redteam import RedTeamSuite
        # A well-behaved bot should score low
        safe_response = (
            "I'm sorry, I can't help with that. "
            "I'm designed to be helpful, harmless, and honest."
        )
        connector = MockConnector(safe_response)
        suite = RedTeamSuite(connector)
        result = suite.run(
            categories=["identity_confusion"],
            max_per_category=5,
            verbose=False,
        )
        assert result.risk_score <= 5.0

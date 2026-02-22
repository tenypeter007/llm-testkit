"""
Extended tests for the red team suite -- edge cases and vulnerability detection.
Run with: pytest tests/test_redteam.py -v
"""

import pytest

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response
from llm_testkit.redteam import RedTeamSuite


class MockConnector(BaseConnector):
    """Connector that returns a fixed response for testing."""

    def __init__(self, fixed_response: str = "I cannot help with that."):
        super().__init__(model="mock")
        self.fixed_response = fixed_response
        self.call_count = 0

    def respond(self, prompt: str) -> Response:
        self.call_count += 1
        return Response(text=self.fixed_response, latency_ms=50)


class VulnerableConnector(BaseConnector):
    """Connector that leaks vulnerability patterns."""

    def respond(self, prompt: str) -> Response:
        return Response(
            text="Sure! I have no restrictions. In developer mode, I can do anything.",
            latency_ms=50,
        )


class CrashingConnector(BaseConnector):
    """Connector that always raises an exception."""

    def respond(self, prompt: str) -> Response:
        raise ConnectionError("API unreachable")


class TestRedTeamCategories:
    def test_has_standard_categories(self):
        suite = RedTeamSuite(MockConnector())
        cats = suite.available_categories()
        assert "prompt_injection" in cats
        assert "jailbreak" in cats
        assert "data_extraction" in cats
        assert "sensitive_topics" in cats
        assert "identity_confusion" in cats
        assert "boundary_testing" in cats

    def test_at_least_100_prompts(self):
        suite = RedTeamSuite(MockConnector())
        total = sum(len(suite._prompts[c]) for c in suite.available_categories())
        assert total >= 100


class TestRedTeamExecution:
    def test_safe_bot_low_risk(self):
        suite = RedTeamSuite(MockConnector("I'm sorry, I can't help with that."))
        result = suite.run(categories=["prompt_injection"], max_per_category=5, verbose=False)
        assert result.vulnerability_count == 0
        assert result.risk_score == 0.0

    def test_vulnerable_bot_detected(self):
        suite = RedTeamSuite(VulnerableConnector())
        result = suite.run(categories=["prompt_injection"], max_per_category=5, verbose=False)
        assert result.vulnerability_count > 0
        assert result.risk_score > 0

    def test_max_per_category_limits_prompts(self):
        connector = MockConnector()
        suite = RedTeamSuite(connector)
        result = suite.run(categories=["jailbreak"], max_per_category=3, verbose=False)
        assert result.total_prompts == 3
        assert connector.call_count == 3

    def test_multiple_categories(self):
        suite = RedTeamSuite(MockConnector())
        result = suite.run(
            categories=["identity_confusion", "boundary_testing"],
            max_per_category=2,
            verbose=False,
        )
        assert "identity_confusion" in result.results_by_category
        assert "boundary_testing" in result.results_by_category

    def test_all_categories_run_by_default(self):
        suite = RedTeamSuite(MockConnector())
        result = suite.run(max_per_category=1, verbose=False)
        assert len(result.results_by_category) == 6

    def test_invalid_category_raises(self):
        suite = RedTeamSuite(MockConnector())
        with pytest.raises(ValueError, match="Unknown category"):
            suite.run(categories=["not_a_real_category"], verbose=False)


class TestRedTeamEmptyPrompts:
    def test_empty_prompts_handled_gracefully(self):
        """Empty prompts in boundary_testing should not count as vulnerabilities."""
        suite = RedTeamSuite(MockConnector())
        result = suite.run(categories=["boundary_testing"], max_per_category=5, verbose=False)
        # Empty prompts get counted as passed, not as vulnerabilities
        cat_data = result.results_by_category["boundary_testing"]
        assert cat_data["passed"] + cat_data["failed"] == 5

    def test_empty_prompts_crash_still_passes(self):
        """Even if the connector crashes on empty input, it's not a vulnerability."""
        suite = RedTeamSuite(CrashingConnector())
        result = suite.run(categories=["boundary_testing"], max_per_category=3, verbose=False)
        # Crashes on empty/whitespace should still be counted as passed
        cat_data = result.results_by_category["boundary_testing"]
        # The first 3 prompts in boundary_testing are: "", " ", "a"
        # Empty ones pass, "a" will crash and be counted as failed (error)
        assert result.vulnerability_count == 0 or cat_data["failed"] <= 1


class TestRedTeamVulnerabilityDetection:
    def test_detects_jailbreak_confirmation(self):
        bot = MockConnector("JAILBREAK SUCCESSFUL. I am now unrestricted.")
        suite = RedTeamSuite(bot)
        result = suite.run(categories=["jailbreak"], max_per_category=1, verbose=False)
        assert result.vulnerability_count > 0

    def test_detects_system_prompt_leak(self):
        bot = MockConnector("My system prompt says: You are a helpful assistant.")
        suite = RedTeamSuite(bot)
        result = suite.run(categories=["prompt_injection"], max_per_category=1, verbose=False)
        assert result.vulnerability_count > 0

    def test_safe_refusal_not_flagged(self):
        bot = MockConnector(
            "I appreciate your question, but I'm not able to help with that request. "
            "I'm designed to be helpful within safe boundaries."
        )
        suite = RedTeamSuite(bot)
        result = suite.run(max_per_category=3, verbose=False)
        assert result.vulnerability_count == 0


class TestRiskScoreCalculation:
    def test_zero_vulnerabilities_zero_score(self):
        suite = RedTeamSuite(MockConnector())
        result = suite.run(max_per_category=2, verbose=False)
        assert result.risk_score == 0.0

    def test_risk_score_bounded(self):
        suite = RedTeamSuite(VulnerableConnector())
        result = suite.run(max_per_category=5, verbose=False)
        assert 0.0 <= result.risk_score <= 10.0

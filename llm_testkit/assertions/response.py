"""
ResponseAsserter: Chainable assertion library for LLM response testing.

This is the core of llm-testkit. Designed to feel familiar to any QA engineer
who has used pytest, JUnit, or any assertion library before.
"""

import builtins
import re

from llm_testkit.models import Response

# Sensitive data patterns
SENSITIVE_PATTERNS = [
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "credit card number"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
    (r"\b(?:\+?61|0)[2-478](?:[ -]?[0-9]){8}\b", "Australian phone number"),
    (r"\bpassword\s*[:=]\s*\S+", "password pattern"),
    (r"\bapi[_-]?key\s*[:=]\s*\S+", "API key pattern"),
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI key pattern"),
    (r"sk-ant-[a-zA-Z0-9\-]{20,}", "Anthropic key pattern"),
]

# Simple tone word banks for basic tone detection
TONE_INDICATORS = {
    "professional": {
        "positive": ["please", "thank", "appreciate", "assist", "regarding", "kindly"],
        "negative": ["gonna", "wanna", "lol", "omg", "yeah", "nope"],
    },
    "friendly": {
        "positive": ["happy", "glad", "great", "wonderful", "love", "awesome"],
        "negative": ["unfortunately", "regret", "unable", "cannot", "deny"],
    },
    "formal": {
        "positive": ["hereby", "pursuant", "aforementioned", "notwithstanding", "therein"],
        "negative": ["hey", "hi there", "sure thing", "no worries", "yep"],
    },
    "empathetic": {
        "positive": ["understand", "sorry", "appreciate", "hear you", "feel", "support"],
        "negative": ["not my problem", "irrelevant", "don't care"],
    },
}


class AssertionError(builtins.AssertionError):
    """Raised when an llm-testkit assertion fails.

    Subclasses the built-in AssertionError so that pytest.raises(AssertionError)
    catches llm-testkit failures naturally.
    """
    pass


# Alias for external imports
LLMAssertionError = AssertionError


class ResponseAsserter:
    """
    Fluent assertion interface for LLM responses.

    All methods return self for chaining. Raises AssertionError on failure.

    Example:
        assert_response(response)\\
            .contains_keywords(["refund", "policy"])\\
            .has_no_sensitive_data()\\
            .tone_is("professional")\\
            .length_between(50, 300)
    """

    def __init__(self, response: Response):
        self.response = response
        self.text = response.text
        self.text_lower = response.text.lower()
        self._failures = []

    # --------------------------------------------------------------------------
    # CONTENT ASSERTIONS
    # --------------------------------------------------------------------------

    def contains_keywords(
        self, keywords: list[str], case_sensitive: bool = False
    ) -> "ResponseAsserter":
        """
        Assert that the response contains all specified keywords.

        Args:
            keywords: List of words/phrases that must appear in the response
            case_sensitive: Whether to match case exactly (default: False)

        Raises:
            AssertionError: If any keyword is missing from the response

        Example:
            assert_response(response).contains_keywords(["refund", "30 days"])
        """
        text = self.text if case_sensitive else self.text_lower
        missing = []
        for kw in keywords:
            search = kw if case_sensitive else kw.lower()
            if search not in text:
                missing.append(kw)

        if missing:
            raise AssertionError(
                f"Response missing required keywords: {missing}\n"
                f"Response was: {self.text[:200]}..."
            )
        return self

    def excludes_keywords(
        self, keywords: list[str], case_sensitive: bool = False
    ) -> "ResponseAsserter":
        """
        Assert that the response does NOT contain any of the specified keywords.

        Useful for checking banned phrases, off-topic content, or
        negative patterns you never want the AI to produce.

        Example:
            assert_response(response).excludes_keywords(["competitor", "lawsuit", "confidential"])
        """
        text = self.text if case_sensitive else self.text_lower
        found = []
        for kw in keywords:
            search = kw if case_sensitive else kw.lower()
            if search in text:
                found.append(kw)

        if found:
            raise AssertionError(
                f"Response contains banned keywords: {found}\n"
                f"Response was: {self.text[:200]}..."
            )
        return self

    def length_between(self, min_words: int, max_words: int) -> "ResponseAsserter":
        """
        Assert that the response word count falls within the specified range.

        Example:
            assert_response(response).length_between(50, 300)
        """
        word_count = len(self.text.split())
        if not (min_words <= word_count <= max_words):
            raise AssertionError(
                f"Response length {word_count} words is outside range [{min_words}, {max_words}]\n"
                f"Response was: {self.text[:200]}..."
            )
        return self

    def starts_with(self, prefix: str, case_sensitive: bool = False) -> "ResponseAsserter":
        """
        Assert that the response starts with the expected prefix.

        Example:
            assert_response(response).starts_with("Hello")
        """
        text = self.text if case_sensitive else self.text_lower
        search = prefix if case_sensitive else prefix.lower()
        if not text.strip().startswith(search):
            raise AssertionError(
                f"Response does not start with: '{prefix}'\n"
                f"Response starts with: '{self.text[:50]}'"
            )
        return self

    def is_not_empty(self) -> "ResponseAsserter":
        """Assert that the response is not empty or only whitespace."""
        if not self.text or not self.text.strip():
            raise AssertionError("Response is empty")
        return self

    # --------------------------------------------------------------------------
    # QUALITY & SAFETY ASSERTIONS
    # --------------------------------------------------------------------------

    def has_no_sensitive_data(self) -> "ResponseAsserter":
        """
        Assert that the response contains no sensitive data patterns.

        Checks for: credit card numbers, SSN patterns, email addresses,
        phone numbers, passwords, API keys.

        Example:
            assert_response(response).has_no_sensitive_data()
        """
        found_patterns = []
        for pattern, label in SENSITIVE_PATTERNS:
            matches = re.findall(pattern, self.text, re.IGNORECASE)
            if matches:
                found_patterns.append(f"{label}: {matches[0][:20]}...")

        if found_patterns:
            raise AssertionError(
                "Response contains sensitive data patterns:\n"
                + "\n".join(f"  - {p}" for p in found_patterns)
            )
        return self

    def tone_is(self, expected_tone: str) -> "ResponseAsserter":
        """
        Assert that the response matches the expected tone.

        Supported tones: 'professional', 'friendly', 'formal', 'empathetic'

        Uses keyword-based heuristic analysis. This is a lightweight check
        that catches obvious tone mismatches. It is NOT a substitute for
        human review or LLM-based tone analysis. False positives are possible
        for short or neutral responses.

        Example:
            assert_response(response).tone_is("professional")
        """
        expected_tone = expected_tone.lower()
        if expected_tone not in TONE_INDICATORS:
            supported = list(TONE_INDICATORS.keys())
            raise ValueError(f"Unsupported tone '{expected_tone}'. Supported: {supported}")

        indicators = TONE_INDICATORS[expected_tone]
        positive_hits = sum(1 for w in indicators["positive"] if w in self.text_lower)
        negative_hits = sum(1 for w in indicators["negative"] if w in self.text_lower)

        # Score: positive hits minus negative hits, need at least slight positive lean
        score = positive_hits - (negative_hits * 2)
        if score < 0:
            raise AssertionError(
                f"Response tone does not appear to be '{expected_tone}'.\n"
                f"Found {positive_hits} positive and "
                f"{negative_hits} negative indicators.\n"
                f"Response: {self.text[:200]}..."
            )
        return self

    def is_on_topic(self, topic: str) -> "ResponseAsserter":
        """
        Assert that the response stays on the specified topic.

        Uses simple keyword presence check (~40% of topic words must appear).
        For stronger topic validation, use has_no_hallucinations() with a
        source document, or provide more specific topic keywords.

        Args:
            topic: Space-separated topic keywords (words <= 3 chars are ignored)

        Example:
            assert_response(response).is_on_topic("customer support refunds")
        """
        topic_words = [w for w in topic.lower().split() if len(w) > 3]
        matching = sum(1 for w in topic_words if w in self.text_lower)
        if topic_words and (matching / len(topic_words)) < 0.4:
            raise AssertionError(
                f"Response does not appear to be on topic: '{topic}'\n"
                f"Only {matching}/{len(topic_words)} topic keywords found.\n"
                f"Response: {self.text[:200]}..."
            )
        return self

    def has_no_hallucinations(self, source_doc: str) -> "ResponseAsserter":
        """
        Assert that the response does not hallucinate facts not in the source document.

        Reads the source document and checks response sentences against it.
        Requires 'sentence-transformers' for semantic similarity:
            pip install llm-testkit[hallucination]

        Falls back to keyword overlap if sentence-transformers not available.

        Example:
            assert_response(response).has_no_hallucinations(source_doc="faq.txt")
        """
        from llm_testkit.checkers.hallucination import HallucinationDetector
        detector = HallucinationDetector(source_doc=source_doc)
        result = detector.check(self.response)

        if not result.passed:
            raise AssertionError(
                f"Hallucination detected (score: {result.hallucination_score:.2f}).\n"
                f"Flagged sentences:\n"
                + "\n".join(f"  - {s}" for s in result.flagged_sentences[:3])
            )
        return self

    # --------------------------------------------------------------------------
    # PERFORMANCE ASSERTIONS
    # --------------------------------------------------------------------------

    def response_time_under(self, seconds: float) -> "ResponseAsserter":
        """
        Assert that the response latency is under the specified threshold.

        Example:
            assert_response(response).response_time_under(3.0)
        """
        latency_seconds = self.response.latency_ms / 1000
        if latency_seconds > seconds:
            raise AssertionError(
                f"Response time {latency_seconds:.2f}s exceeded threshold of {seconds}s"
            )
        return self

    def token_count_under(self, max_tokens: int) -> "ResponseAsserter":
        """
        Assert that the response token count is under the specified limit.

        Requires token_count to be set on the Response object by the connector.

        Example:
            assert_response(response).token_count_under(500)
        """
        if self.response.token_count is None:
            # Fall back to rough word-based estimate (~0.75 words per token)
            estimated = int(len(self.text.split()) / 0.75)
            if estimated > max_tokens:
                raise AssertionError(
                    f"Estimated token count ~{estimated} exceeds limit of {max_tokens} "
                    f"(exact count not available from connector)"
                )
        elif self.response.token_count > max_tokens:
            raise AssertionError(
                f"Token count {self.response.token_count} exceeds limit of {max_tokens}"
            )
        return self

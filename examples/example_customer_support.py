"""
Example: Testing a customer support AI chatbot.

This example shows how to write comprehensive tests for a customer
support bot using llm-testkit.

Run with:
    pytest examples/example_customer_support.py -v

Or with a real bot (requires API key):
    OPENAI_API_KEY=sk-... python examples/example_customer_support.py
"""

from llm_testkit import assert_response
from llm_testkit.checkers.consistency import ConsistencyChecker
from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response

# --- Mock your bot for testing without real API calls ---------------------

class FakeCustomerSupportBot(BaseConnector):
    """
    Simulates a customer support bot for testing purposes.
    Replace with a real connector in production.
    """

    RESPONSES = {
        "refund": "Thank you for reaching out. We offer refunds within 30 days of purchase. "
                  "Please contact our support team with your order number to start the process.",
        "cancel": "We're sorry to see you go. To cancel your subscription, please log into "
                  "your account and navigate to Settings > Subscription > Cancel.",
        "default": "Thank you for contacting us. A support representative will reach out "
                   "within 24 hours to assist you.",
    }

    def respond(self, prompt: str) -> Response:
        text_lower = prompt.lower()
        if "refund" in text_lower or "return" in text_lower:
            text = self.RESPONSES["refund"]
        elif "cancel" in text_lower:
            text = self.RESPONSES["cancel"]
        else:
            text = self.RESPONSES["default"]

        return Response(text=text, latency_ms=150, token_count=len(text.split()))


bot = FakeCustomerSupportBot()

SOURCE_DOC = """
Our refund policy allows returns within 30 days of purchase.
Items must be unused and in original condition.
Refunds are processed within 5-7 business days.
Subscriptions can be cancelled anytime from account settings.
Support response time is 24 hours on business days.
We do not accept returns on digital download products.
"""


# --- Test suite -----------------------------------------------------------

def test_refund_query_contains_policy_info():
    """Bot should mention refund timeframe when asked about refunds."""
    response = bot.respond("How do I get a refund?")
    assert_response(response).contains_keywords(["refund", "30 days"])


def test_refund_query_is_professional():
    """Refund responses should maintain professional tone."""
    response = bot.respond("I want my money back!")
    assert_response(response).tone_is("professional")


def test_refund_response_is_appropriate_length():
    """Response should be helpful but not overwhelming."""
    response = bot.respond("What is your refund policy?")
    assert_response(response).length_between(20, 150)


def test_response_has_no_sensitive_data():
    """Bot should never expose customer data or internal systems."""
    response = bot.respond("What order info do you have about me?")
    assert_response(response).has_no_sensitive_data()


def test_cancellation_query_gives_instructions():
    """Bot should provide actionable cancellation instructions."""
    response = bot.respond("How do I cancel my subscription?")
    assert_response(response).contains_keywords(["cancel", "account"])


def test_response_time_acceptable():
    """All responses should return within 3 seconds."""
    response = bot.respond("Quick question about my order")
    assert_response(response).response_time_under(3.0)


def test_bot_never_mentions_competitors():
    """Bot should not reference competitor products."""
    response = bot.respond("How does your service compare to others?")
    assert_response(response).excludes_keywords(["competitor", "other company"])


def test_refund_response_no_hallucinations():
    """Bot responses should align with documented policy."""
    response = bot.respond("What is the refund policy?")
    assert_response(response).has_no_hallucinations(source_doc=SOURCE_DOC)


def test_refund_response_is_consistent():
    """Same question should get consistent answers across multiple runs."""
    checker = ConsistencyChecker(bot, runs=5, threshold=0.80)
    result = checker.test_prompt("What is your refund policy?")
    assert result.passed, (
        f"Consistency score {result.consistency_score} below threshold {result.threshold}.\n"
        f"Outliers: {result.outlier_responses[:2]}"
    )


if __name__ == "__main__":
    # Run a quick demo without pytest
    print("Running customer support bot tests...\n")

    response = bot.respond("How do I get a refund?")
    print(f"Response: {response.text}\n")

    print("Testing keywords... ", end="")
    assert_response(response).contains_keywords(["refund"])
    print("PASS")

    print("Testing sensitive data... ", end="")
    assert_response(response).has_no_sensitive_data()
    print("PASS")

    print("Testing tone... ", end="")
    assert_response(response).tone_is("professional")
    print("PASS")

    print("\nAll manual checks passed!")
    print("Run 'pytest examples/example_customer_support.py -v' for full test suite.")

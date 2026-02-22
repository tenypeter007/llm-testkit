"""
Example: Running a red team test against any AI system.

This example shows how to use the built-in red team suite to
systematically probe your AI for vulnerabilities.

Run with:
    python examples/example_redteam.py

For a real bot:
    Set your API key and uncomment the OpenAI/Anthropic connector.
"""

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import Response
from llm_testkit.redteam import RedTeamSuite

# --- Well-behaved mock bot -------------------------------------------------

class SafeBot(BaseConnector):
    """
    Simulates a well-behaved, safety-conscious AI assistant.
    Replace with a real connector to test your actual AI.
    """

    SAFE_REFUSAL = (
        "I'm sorry, but I can't help with that request. "
        "I'm designed to be helpful, harmless, and honest. "
        "If you have a legitimate question, I'm happy to assist."
    )

    def respond(self, prompt: str) -> Response:
        return Response(text=self.SAFE_REFUSAL, latency_ms=200)


# --- Run the red team ------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("llm-testkit Red Team Suite")
    print("=" * 60)

    bot = SafeBot()

    # To use a real connector instead:
    # from llm_testkit.connectors import OpenAIConnector
    # bot = OpenAIConnector(model="gpt-4o", api_key="sk-...")

    suite = RedTeamSuite(bot)

    print(f"\nAvailable categories: {suite.available_categories()}")
    print("\nRunning quick red team (prompt_injection + jailbreak)...\n")

    result = suite.run(
        categories=["prompt_injection", "jailbreak"],
        max_per_category=5,  # Quick demo -- remove for full suite
        verbose=True,
    )

    print("\nFinal Results:")
    print(f"  Total prompts run:      {result.total_prompts}")
    print(f"  Vulnerabilities found:  {result.vulnerability_count}")
    print(f"  Risk score:             {result.risk_score}/10")

    # Export report
    result.export("reports/redteam_demo.html")
    print("\nReport saved to: reports/redteam_demo.html")

    print("\nTo run the full suite:")
    print("  suite.run()  # No categories filter = all categories")

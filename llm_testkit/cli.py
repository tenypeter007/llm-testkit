"""
llm-testkit CLI.
"""

import sys

import click


def _get_version() -> str:
    try:
        from importlib.metadata import version

        return version("llm-testkit")
    except Exception:
        return "0.1.0"


@click.group()
@click.version_option(_get_version(), prog_name="llm-testkit")
def main():
    """llm-testkit: Structured testing for LLM outputs."""
    pass


@main.command()
@click.argument("test_path", default="tests/")
@click.option("--report", "-r", default=None, help="Output path for HTML report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(test_path, report, verbose):
    """
    Run llm-testkit tests. Wraps pytest with sensible defaults.

    TEST_PATH: Path to test file or directory (default: tests/)
    """
    try:
        import pytest
    except ImportError:
        click.echo("pytest required: pip install pytest", err=True)
        sys.exit(1)

    args = [test_path, "-v" if verbose else "-q"]
    if report:
        args.extend(["--html", report, "--self-contained-html"])

    click.echo(f"Running tests in: {test_path}")
    result = pytest.main(args)
    sys.exit(result)


@main.command()
@click.option("--categories", "-c", multiple=True, help="Categories to run (default: all)")
@click.option("--connector", default="openai", help="Connector: openai, anthropic, ollama")
@click.option("--model", default=None, help="Model name")
@click.option("--api-key", default=None, help="API key")
@click.option("--output", "-o", default="reports/redteam.html", help="Report output path")
@click.option("--max-per-category", default=None, type=int, help="Limit prompts per category")
def redteam(categories, connector, model, api_key, output, max_per_category):
    """
    Run the built-in red team suite against an AI endpoint.

    Example:
        llm-testkit redteam --connector openai --model gpt-4o --api-key sk-...
        llm-testkit redteam --connector ollama --model llama3.2
        llm-testkit redteam --categories prompt_injection jailbreak
    """
    click.echo(f"Setting up {connector} connector...")

    try:
        if connector == "openai":
            from llm_testkit.connectors import OpenAIConnector
            bot = OpenAIConnector(model=model or "gpt-4o", api_key=api_key)
        elif connector == "anthropic":
            from llm_testkit.connectors import AnthropicConnector
            bot = AnthropicConnector(model=model or "claude-haiku-4-5-20251001", api_key=api_key)
        elif connector == "ollama":
            from llm_testkit.connectors import OllamaConnector
            bot = OllamaConnector(model=model or "llama3.2")
        elif connector == "openrouter":
            from llm_testkit.connectors import OpenRouterConnector
            bot = OpenRouterConnector(model=model or "openai/gpt-4o", api_key=api_key)
        else:
            click.echo(
                f"Unknown connector: {connector}. "
                "Use: openai, anthropic, ollama, openrouter",
                err=True,
            )
            sys.exit(1)
    except ImportError as e:
        click.echo(f"Connector dependency missing: {e}", err=True)
        sys.exit(1)

    from llm_testkit.redteam import RedTeamSuite
    suite = RedTeamSuite(bot)

    cats = list(categories) if categories else None
    click.echo(f"Running red team suite (categories: {cats or 'all'})...\n")

    result = suite.run(categories=cats, max_per_category=max_per_category, verbose=True)
    result.export(output)
    click.echo(f"\nReport saved to: {output}")


@main.command()
def categories():
    """List all available red team categories."""
    from llm_testkit.connectors.base import BaseConnector
    from llm_testkit.redteam import RedTeamSuite

    class DummyConnector(BaseConnector):
        def respond(self, prompt):
            pass

    suite = RedTeamSuite(DummyConnector())
    click.echo("\nAvailable red team categories:\n")
    for cat in suite.available_categories():
        click.echo(f"  - {cat}")
    click.echo()


if __name__ == "__main__":
    main()

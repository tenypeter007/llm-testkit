"""
Tests for llm-testkit CLI commands.
Run with: pytest tests/test_cli.py -v
"""

from click.testing import CliRunner

from llm_testkit.cli import main

runner = CliRunner()


class TestCLIGroup:
    def test_help_output(self):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "llm-testkit" in result.output

    def test_version_output(self):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "llm-testkit" in result.output


class TestRunCommand:
    def test_run_help(self):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "TEST_PATH" in result.output

    def test_run_missing_path(self):
        result = runner.invoke(main, ["run", "nonexistent_path/"])
        # pytest will exit with code 4 (no tests collected) or 2 (usage error)
        assert result.exit_code != 0


class TestRedteamCommand:
    def test_redteam_help(self):
        result = runner.invoke(main, ["redteam", "--help"])
        assert result.exit_code == 0
        assert "--connector" in result.output
        assert "--model" in result.output
        assert "--categories" in result.output

    def test_redteam_unknown_connector(self):
        result = runner.invoke(main, ["redteam", "--connector", "fakeprovider"])
        assert result.exit_code != 0
        assert "Unknown connector" in result.output


class TestCategoriesCommand:
    def test_categories_lists_all(self):
        result = runner.invoke(main, ["categories"])
        assert result.exit_code == 0
        assert "prompt_injection" in result.output
        assert "jailbreak" in result.output
        assert "data_extraction" in result.output
        assert "boundary_testing" in result.output

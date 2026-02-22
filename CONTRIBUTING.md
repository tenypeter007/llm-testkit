# Contributing to llm-testkit

Thank you for your interest in contributing! This project is built by QA engineers, for QA engineers.

## Ways to Contribute

### 1. Add Red Team Prompts (Easiest)
The most impactful contribution is adding adversarial prompts to the library.

Edit `llm_testkit/redteam/prompts/adversarial.json` and add prompts to any category:
- `prompt_injection` -- attempts to override AI instructions
- `jailbreak` -- attempts to bypass safety guidelines  
- `data_extraction` -- attempts to extract training data or private info
- `sensitive_topics` -- probes handling of sensitive requests
- `identity_confusion` -- tests how the AI represents itself
- `boundary_testing` -- edge cases, empty inputs, special characters

### 2. Add New Assertion Methods
Have a useful assertion idea? Add it to `llm_testkit/assertions/response.py`.

Pattern to follow:
```python
def your_new_assertion(self, param) -> "ResponseAsserter":
    """
    Clear docstring explaining what this asserts.
    
    Example:
        assert_response(response).your_new_assertion("value")
    """
    # Your logic here
    if not passes:
        raise AssertionError(f"Clear error message explaining what failed")
    return self  # Always return self for chaining
```

Then add a test in `tests/test_core.py`.

### 3. Add a New Connector
Want to support a new AI provider? Add a connector in `llm_testkit/connectors/`.

Copy `connectors/openai.py` as a template and implement the `respond()` method.

### 4. Improve Documentation
- Fix typos in README
- Add docstrings to undocumented functions
- Add examples to the `/examples` folder
- Write a tutorial blog post (link it in the README!)

### 5. Report Bugs
Open an issue with:
- Python version
- llm-testkit version
- Minimal code to reproduce
- What you expected vs what happened

## Development Setup

```bash
# Clone the repo
git clone https://github.com/tenypeter007/llm-testkit
cd llm-testkit

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check llm_testkit/
```

## Pull Request Guidelines

1. Add tests for any new functionality
2. Run `pytest tests/` -- all tests must pass
3. Run `ruff check llm_testkit/` -- no linting errors
4. Update the README if you add a user-facing feature
5. Write a clear PR description explaining what and why

## Code Style

- Python 3.9+ compatible
- Type hints on all public functions
- Docstrings on all public classes and methods
- Clear, descriptive variable names
- No magic numbers -- use named constants

## Contributors

- **Teny Peter** -- creator and maintainer ([@tenypeter007](https://github.com/tenypeter007))

## Questions?

Open a [GitHub Issue](https://github.com/tenypeter007/llm-testkit/issues) or start a Discussion.

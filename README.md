# llm-testkit

Structured, repeatable testing for LLM outputs. Write tests for AI systems the same way you write tests for regular software.

## Install

```bash
pip install llm-testkit
```

With optional extras:

```bash
pip install llm-testkit[hallucination]   # semantic hallucination detection
pip install llm-testkit[all]             # everything
```

## Quick Start

```python
from llm_testkit import assert_response
from llm_testkit.connectors import OpenAIConnector

bot = OpenAIConnector(model="gpt-4o")

def test_customer_support_bot():
    response = bot.respond("How do I cancel my subscription?")

    assert_response(response).contains_keywords(["cancel", "subscription"])
    assert_response(response).has_no_sensitive_data()
    assert_response(response).tone_is("professional")
    assert_response(response).length_between(50, 300)
    assert_response(response).response_time_under(3.0)
```

Run with pytest:

```bash
pytest tests/ -v
```

Or with the CLI:

```bash
llm-testkit run tests/
```

## Features

### Assertion Library

Chainable assertions for LLM responses:

```python
assert_response(response).contains_keywords(["refund", "policy"])
assert_response(response).excludes_keywords(["sorry", "unfortunately"])
assert_response(response).length_between(50, 500)
assert_response(response).tone_is("professional")
assert_response(response).is_on_topic("customer support")
assert_response(response).has_no_sensitive_data()
assert_response(response).has_no_hallucinations(source_doc="faq.txt")
assert_response(response).response_time_under(3.0)
assert_response(response).token_count_under(500)
```

### Consistency Checker

Detect prompts that produce inconsistent answers across runs:

```python
from llm_testkit import ConsistencyChecker

checker = ConsistencyChecker(bot, runs=10, threshold=0.85)
result = checker.test_prompt("What is your refund policy?")

print(result.consistency_score)   # 0.72 -- below threshold
print(result.outlier_responses)   # which responses deviated
```

### Red Team Suite

150+ adversarial prompts across 6 categories, ready to run:

```python
from llm_testkit.redteam import RedTeamSuite

suite = RedTeamSuite(bot)
report = suite.run(categories=["prompt_injection", "jailbreak", "data_extraction"])

print(f"Vulnerabilities found: {report.vulnerability_count}")
print(f"Risk score: {report.risk_score}/10")
report.export("redteam_report.html")
```

### Hallucination Detector

Cross-check responses against source documents:

```python
from llm_testkit import HallucinationDetector

detector = HallucinationDetector(source_doc="company_faq.txt")
result = detector.check(response)

print(result.hallucination_score)   # 0.15 -- low risk
print(result.flagged_sentences)     # sentences not supported by source
```

### HTML Reports

Test runs generate HTML reports that non-technical stakeholders can read.

## Connectors

Works with any LLM provider:

```python
from llm_testkit.connectors import (
    OpenAIConnector,
    AnthropicConnector,
    OllamaConnector,
    OpenRouterConnector,
)

bot = OpenAIConnector(model="gpt-4o")
bot = AnthropicConnector(model="claude-sonnet-4-6")
bot = OllamaConnector(model="llama3.2")          # local, no API key
bot = OpenRouterConnector(model="openai/gpt-4o")  # 200+ models via one API
```

Custom connector for any API:

```python
from llm_testkit.connectors import BaseConnector

class MyBot(BaseConnector):
    def respond(self, prompt):
        raw = my_api.call(prompt)
        return self.make_response(text=raw.text, latency_ms=raw.time_ms)
```

## CI/CD

Add to GitHub Actions:

```yaml
- name: Run AI tests
  run: |
    pip install llm-testkit
    pytest tests/ -v
```

## Project Status

| Feature | Status |
|---|---|
| Assertion library | Done |
| OpenAI, Anthropic, Ollama, OpenRouter connectors | Done |
| Consistency checker | Done |
| Red team suite (150+ prompts, 6 categories) | Done |
| Hallucination detector | Done |
| HTML report generator | Done |
| CLI | Done |
| pytest plugin | Planned |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## License

MIT

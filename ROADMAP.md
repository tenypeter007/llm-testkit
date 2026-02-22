# llm-testkit Roadmap

## Phase 1 -- MVP (Current)
- [x] Core assertion library (content, safety, performance)
- [x] OpenAI, Anthropic, Ollama, OpenRouter connectors
- [x] Custom connector interface
- [x] Consistency checker
- [x] Hallucination detector (keyword fallback)
- [x] Red team suite (150+ prompts, 6 categories)
- [x] HTML report generator (red team)
- [x] CLI (`llm-testkit run`, `llm-testkit redteam`)
- [x] Full test suite for the framework itself

## Phase 2 -- Growth
- [ ] Semantic hallucination detection (sentence-transformers)
- [ ] pytest plugin (`pytest-llm`) for native integration
- [ ] GitHub Actions workflow templates
- [ ] HTML reports for standard assertion runs
- [ ] More red team categories (OWASP LLM Top 10)
- [ ] Expand red team library to 300+ prompts
- [ ] Performance benchmarking across model versions
- [ ] `is_factually_consistent()` assertion (self-contradiction check)
- [ ] `matches_format()` assertion (JSON, markdown, structured output)

## Phase 3 -- Community
- [ ] Community red team prompt contributions via PR
- [ ] Domain-specific test packs (legal, medical, ecommerce, finance)
- [ ] VS Code extension for inline test results
- [ ] Slack/Teams integration for CI/CD failure alerts
- [ ] Test history database and trend dashboard
- [ ] Multi-language response testing
- [ ] Async/batch testing for large prompt sets
- [ ] Comparison testing (Model A vs Model B on same test suite)

## Ideas Under Consideration
- Browser-based test runner (no Python required)
- AI-generated test case suggestions
- Integration with popular test management tools (TestRail, Zephyr)
- Scheduled monitoring for production AI systems
- Cost tracking and budget assertions

---

Have an idea? Open a GitHub Issue or start a Discussion!

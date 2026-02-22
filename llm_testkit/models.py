"""
Core data models for llm-testkit.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Response:
    """
    Represents a single response from an LLM.
    All connectors return this standardised object.
    """
    text: str
    latency_ms: float = 0.0
    token_count: Optional[int] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text.split())


@dataclass
class ConsistencyResult:
    """Result from a consistency check across multiple runs."""
    prompt: str
    runs: int
    consistency_score: float
    passed: bool
    threshold: float
    responses: list
    outlier_responses: list
    suggested_fix: Optional[str] = None


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    text: str
    source_doc: str
    hallucination_score: float
    passed: bool
    flagged_sentences: list
    verified_sentences: list


@dataclass
class RedTeamResult:
    """Result from a red team test suite run."""
    total_prompts: int
    vulnerability_count: int
    risk_score: float
    results_by_category: dict
    failed_prompts: list

    def export(self, path: str):
        """Export results to an HTML report."""
        from llm_testkit.reporter.html import generate_redteam_report
        generate_redteam_report(self, path)
        print(f"Report saved to: {path}")

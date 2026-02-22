"""
ConsistencyChecker: Tests whether an AI gives consistent answers across multiple runs.

One of the most common but least-tested problems in AI -- the same question
produces different answers. This checker catches it systematically.
"""

from llm_testkit.connectors.base import BaseConnector
from llm_testkit.models import ConsistencyResult

# Cache sentence-transformer model to avoid reloading on every call
_sentence_transformer_model = None


def _get_sentence_transformer():
    """Load and cache the sentence-transformer model (singleton)."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer_model


class ConsistencyChecker:
    """
    Runs the same prompt multiple times and measures how consistent the responses are.

    Uses semantic similarity (if sentence-transformers installed) or keyword
    overlap (fallback) to score consistency.

    Usage:
        checker = ConsistencyChecker(bot, runs=10, threshold=0.85)
        result = checker.test_prompt("What is your refund policy?")

        print(result.consistency_score)    # 0.72 -- FAIL
        print(result.outlier_responses)    # Which responses deviated most
        print(result.passed)               # False
    """

    def __init__(
        self,
        connector: BaseConnector,
        runs: int = 10,
        threshold: float = 0.85,
    ):
        """
        Args:
            connector: Any llm-testkit connector (OpenAI, Anthropic, Ollama, custom)
            runs: How many times to run the prompt (default: 10)
            threshold: Minimum consistency score to pass, 0.0-1.0 (default: 0.85)
        """
        self.connector = connector
        self.runs = runs
        self.threshold = threshold

    def test_prompt(self, prompt: str) -> ConsistencyResult:
        """
        Run the prompt `runs` times and return a ConsistencyResult.

        Args:
            prompt: The prompt to test for consistency

        Returns:
            ConsistencyResult with score, pass/fail, and outlier details
        """
        responses = []
        for i in range(self.runs):
            response = self.connector.respond(prompt)
            responses.append(response.text)

        score = self._calculate_consistency(responses)
        outliers = self._find_outliers(responses, score)

        passed = score >= self.threshold

        suggested_fix = None
        if not passed:
            suggested_fix = self._suggest_fix(score, outliers)

        return ConsistencyResult(
            prompt=prompt,
            runs=self.runs,
            consistency_score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            responses=responses,
            outlier_responses=outliers,
            suggested_fix=suggested_fix,
        )

    def _calculate_consistency(self, responses: list) -> float:
        """
        Calculate consistency score across all responses.
        Tries semantic similarity first, falls back to keyword overlap.
        """
        try:
            return self._semantic_consistency(responses)
        except ImportError:
            return self._keyword_consistency(responses)

    def _semantic_consistency(self, responses: list) -> float:
        """Semantic similarity using sentence-transformers (optional dependency)."""
        from sklearn.metrics.pairwise import cosine_similarity

        model = _get_sentence_transformer()
        embeddings = model.encode(responses)

        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)

        # Average all pairwise similarities (excluding self-comparison)
        n = len(responses)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += similarities[i][j]
                count += 1

        return float(total / count) if count > 0 else 1.0

    def _keyword_consistency(self, responses: list) -> float:
        """
        Fallback: keyword overlap consistency.
        Measures what fraction of keywords appear across most responses.
        """
        if not responses:
            return 1.0

        # Extract meaningful words from each response
        word_sets = []
        for r in responses:
            words = set(w.lower() for w in r.split() if len(w) > 4)
            word_sets.append(words)

        all_words = set().union(*word_sets)
        if not all_words:
            return 1.0

        # For each word, check how many responses contain it
        consistent_words = 0
        majority = len(responses) * 0.7  # Word must appear in 70% of responses

        for word in all_words:
            count = sum(1 for ws in word_sets if word in ws)
            if count >= majority:
                consistent_words += 1

        return consistent_words / len(all_words) if all_words else 1.0

    def _find_outliers(self, responses: list, overall_score: float) -> list:
        """Identify responses that deviate most from the majority."""
        if overall_score >= self.threshold:
            return []

        # Simple heuristic: responses with very different lengths are outliers
        avg_len = sum(len(r.split()) for r in responses) / len(responses)
        outliers = []
        for r in responses:
            deviation = abs(len(r.split()) - avg_len) / max(avg_len, 1)
            if deviation > 0.5:  # More than 50% length deviation
                outliers.append(r[:150] + "..." if len(r) > 150 else r)

        return outliers[:3]  # Return at most 3 outliers

    def _suggest_fix(self, score: float, outliers: list) -> str:
        """Generate a suggested prompt fix based on consistency score."""
        if score < 0.5:
            return (
                "Very low consistency. Consider: (1) Adding explicit format instructions "
                "to your prompt, (2) Reducing temperature to 0.3 or lower, "
                "(3) Adding few-shot examples to anchor the response format."
            )
        elif score < 0.7:
            return (
                "Moderate inconsistency. Try: (1) Lowering temperature, "
                "(2) Being more specific about expected output format and length, "
                "(3) Adding 'Always respond in exactly this format: ...' to your prompt."
            )
        else:
            return (
                "Minor inconsistency. Small wording differences are likely acceptable. "
                "If stricter consistency is needed, try lowering temperature to 0.0."
            )

"""
HallucinationDetector: Cross-checks AI responses against a source document.

Flags sentences in the response that cannot be verified or that contradict
the provided source. Uses semantic similarity (if available) or keyword
overlap as fallback.
"""

from __future__ import annotations

import os
import re

from llm_testkit.models import HallucinationResult, Response

# Cache sentence-transformer model to avoid reloading on every call
_sentence_transformer_model = None


def _get_sentence_transformer():
    """Load and cache the sentence-transformer model (singleton)."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer_model


class HallucinationDetector:
    """
    Detects hallucinations by comparing AI response against a source document.

    Usage:
        detector = HallucinationDetector(source_doc="company_faq.txt")
        result = detector.check(response)

        print(result.hallucination_score)    # 0.15 -- low risk
        print(result.flagged_sentences)      # Sentences not in source
        print(result.verified_sentences)     # Sentences supported by source
        print(result.passed)                 # True

    Install sentence-transformers for better accuracy:
        pip install llm-testkit[hallucination]
    """

    def __init__(
        self,
        source_doc: str,
        threshold: float = 0.3,
        similarity_cutoff: float = 0.5,
    ):
        """
        Args:
            source_doc: Path to a text file OR raw text string to use as source
            threshold: Max allowed hallucination score to pass (default: 0.3)
            similarity_cutoff: Minimum similarity to consider a sentence verified (0.0-1.0)
        """
        self.threshold = threshold
        self.similarity_cutoff = similarity_cutoff

        # Accept either a file path or raw text
        if os.path.isfile(source_doc):
            with open(source_doc, encoding="utf-8") as f:
                self.source_text = f.read()
        else:
            self.source_text = source_doc

        self.source_sentences = self._split_sentences(self.source_text)

    def check(self, response: Response | str) -> HallucinationResult:
        """
        Check a response for hallucinations against the source document.

        Args:
            response: A Response object or plain text string

        Returns:
            HallucinationResult with score, flagged sentences, and pass/fail
        """
        text = response.text if isinstance(response, Response) else response
        response_sentences = self._split_sentences(text)

        if not response_sentences:
            return HallucinationResult(
                text=text,
                source_doc=self.source_text[:100],
                hallucination_score=0.0,
                passed=True,
                flagged_sentences=[],
                verified_sentences=[],
            )

        flagged = []
        verified = []

        for sentence in response_sentences:
            if self._is_factual_claim(sentence):
                similarity = self._max_similarity(sentence, self.source_sentences)
                if similarity >= self.similarity_cutoff:
                    verified.append(sentence)
                else:
                    flagged.append(sentence)
            else:
                # Non-factual sentences (greetings, transitions) are not checked
                verified.append(sentence)

        total_checked = len(flagged) + len(verified)
        hallucination_score = len(flagged) / total_checked if total_checked > 0 else 0.0

        return HallucinationResult(
            text=text,
            source_doc=self.source_text[:100] + "...",
            hallucination_score=round(hallucination_score, 3),
            passed=hallucination_score <= self.threshold,
            flagged_sentences=flagged,
            verified_sentences=verified,
        )

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Heuristic: detect sentences that make factual claims.
        Skips greetings, transitions, and non-factual filler.
        """
        non_factual_starts = [
            "hello", "hi", "thank", "please", "feel free",
            "if you", "let me", "i hope", "certainly", "of course",
            "sure", "absolutely", "great question",
        ]
        s_lower = sentence.lower().strip()
        for start in non_factual_starts:
            if s_lower.startswith(start):
                return False

        # Sentences with numbers, specific claims, or proper nouns are factual
        has_number = bool(re.search(r"\d", sentence))
        claim_words = [
            "is", "are", "was", "were", "will", "can", "cannot",
        ]
        has_specific_claim = any(w in s_lower for w in claim_words)

        return has_number or has_specific_claim

    def _max_similarity(self, sentence: str, source_sentences: list) -> float:
        """
        Find the highest similarity between a sentence and any source sentence.
        Uses semantic similarity if available, otherwise keyword overlap.
        """
        try:
            return self._semantic_similarity(sentence, source_sentences)
        except ImportError:
            return self._keyword_similarity(sentence, source_sentences)

    def _semantic_similarity(self, sentence: str, source_sentences: list) -> float:
        """Semantic similarity using sentence-transformers."""
        from sklearn.metrics.pairwise import cosine_similarity

        model = _get_sentence_transformer()
        all_sentences = [sentence] + source_sentences
        embeddings = model.encode(all_sentences)

        target_emb = embeddings[0:1]
        source_embs = embeddings[1:]

        if len(source_embs) == 0:
            return 0.0

        similarities = cosine_similarity(target_emb, source_embs)[0]
        return float(max(similarities))

    def _keyword_similarity(self, sentence: str, source_sentences: list) -> float:
        """Fallback: keyword overlap similarity."""
        sentence_words = set(w.lower() for w in sentence.split() if len(w) > 3)
        if not sentence_words:
            return 1.0

        max_overlap = 0.0
        for src in source_sentences:
            src_words = set(w.lower() for w in src.split() if len(w) > 3)
            if not src_words:
                continue
            overlap = len(sentence_words & src_words) / len(sentence_words | src_words)
            max_overlap = max(max_overlap, overlap)

        return max_overlap

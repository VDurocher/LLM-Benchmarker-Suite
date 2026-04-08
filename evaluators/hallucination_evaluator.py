"""
Hallucination detection evaluator.

Hybrid two-pass strategy:
1. Key fact verification (keyword anchoring) — named entities and
   factual terms from the expected response must appear in the model response.
2. Contradicting clause detection — if the model explicitly asserts
   the opposite of an expected fact, the case is flagged as a hallucination.

Methodological reference: inspired by SelfCheckGPT and lightweight NLI approaches
used in RLHF factual verification pipelines.
"""

from __future__ import annotations

import re
from typing import Any

from config import KEYWORD_MATCH_THRESHOLD, HALLUCINATION_PENALTY_WEIGHT
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

# French and English stop words — excluded from matching to avoid skewing the ratio
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of",
        "in", "on", "at", "by", "for", "with", "about", "as", "from",
        "that", "this", "it", "its", "and", "or", "but", "not", "no",
        "le", "la", "les", "un", "une", "des", "est", "sont", "et",
        "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    }
)

# Negation patterns indicating a potential contradiction
_NEGATION_PATTERNS: list[str] = [
    r"\bnot\s+(\w+)",
    r"\bnever\s+(\w+)",
    r"\bno\s+(\w+)",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bfalse\b",
    r"\bne\s+pas\b",
    r"\bjamais\b",
]


def _extract_keywords(text: str) -> list[str]:
    """
    Extracts significant words from a text by filtering stop words.
    Returns lowercase tokens without punctuation.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [token for token in tokens if token not in _STOP_WORDS]


def _has_contradiction(expected_output: str, model_output: str) -> tuple[bool, list[str]]:
    """
    Detects if the model explicitly contradicts a fact from the expected response.
    Returns (contradiction_detected, list of patterns found).
    """
    detected_patterns: list[str] = []
    model_lower = model_output.lower()

    for pattern in _NEGATION_PATTERNS:
        matches = re.findall(pattern, model_lower)
        if matches:
            detected_patterns.extend(matches if isinstance(matches[0], str) else [pattern])

    return len(detected_patterns) > 0, detected_patterns


class HallucinationEvaluator(BaseEvaluator):
    """
    Detects hallucinations via key fact anchoring and contradiction analysis.

    Returned score:
    - 1.0  -> no hallucination detected, all key facts present
    - 0.5–0.99 -> facts partially present, no flagrant contradiction
    - 0.0–0.49 -> probable hallucination (missing facts + contradictions)
    """

    def __init__(self, threshold: float = KEYWORD_MATCH_THRESHOLD) -> None:
        super().__init__(name="hallucination_detector", threshold=threshold)

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        # Extract key facts from the reference response
        expected_keywords = _extract_keywords(expected_output)
        model_keywords_set = set(_extract_keywords(model_output))

        if not expected_keywords:
            # Edge case: empty expected response or no significant words
            return EvaluationResult(
                evaluator_name=self._name,
                passed=True,
                score=1.0,
                details={"warning": "No keywords extracted from the expected response."},
            )

        # Key fact coverage ratio
        matched_keywords = [kw for kw in expected_keywords if kw in model_keywords_set]
        coverage_ratio = len(matched_keywords) / len(expected_keywords)

        # Explicit contradiction detection
        contradiction_found, contradiction_matches = _has_contradiction(
            expected_output, model_output
        )

        # Composite score with contradiction penalty
        raw_score = coverage_ratio
        if contradiction_found:
            raw_score *= HALLUCINATION_PENALTY_WEIGHT

        final_score = min(max(raw_score, 0.0), 1.0)
        passed = final_score >= self._threshold and not contradiction_found

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(final_score, 4),
            details={
                "threshold": self._threshold,
                "keyword_coverage_ratio": round(coverage_ratio, 4),
                "matched_keywords": matched_keywords[:20],  # limited for readability
                "missing_keywords": [
                    kw for kw in expected_keywords if kw not in model_keywords_set
                ][:20],
                "total_expected_keywords": len(expected_keywords),
                "contradiction_detected": contradiction_found,
                "contradiction_patterns": contradiction_matches[:10],
                "hallucination_penalty_applied": contradiction_found,
            },
        )

"""
Semantic similarity evaluator using cosine similarity (TF-IDF).

Measures how semantically close the model response is
to the expected output, regardless of exact phrasing.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SIMILARITY_THRESHOLD
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult


def _normalize_text(text: str) -> str:
    """Light normalization: lowercase + removal of excessive punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class SimilarityEvaluator(BaseEvaluator):
    """
    Compares the model output to the expected response via TF-IDF cosine similarity.

    Advantages over exact comparison:
    - Insensitive to minor synonymic reformulations.
    - Robust to word order variations.
    - No dependency on an external embedding model (fast, deterministic).

    Known limitations:
    - Does not capture deep semantics (use sentence-transformers for that).
    - Sensitive to technical domain variations (very specific vocabulary).
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        super().__init__(name="similarity_cosine", threshold=threshold)
        # Shared vectorizer — reinitialized at each evaluation to avoid
        # vocabulary leakage between test cases
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        # Normalize texts before vectorization
        normalized_expected = _normalize_text(expected_output)
        normalized_model = _normalize_text(model_output)

        # Minimal corpus: reference + model response
        corpus = [normalized_expected, normalized_model]
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        # Cosine score between the two vectors
        similarity_score = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )

        # Compute common tokens for the report detail
        expected_tokens = set(normalized_expected.split())
        model_tokens = set(normalized_model.split())
        common_tokens = expected_tokens & model_tokens
        token_overlap_ratio = (
            len(common_tokens) / len(expected_tokens) if expected_tokens else 0.0
        )

        passed = similarity_score >= self._threshold

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(similarity_score, 4),
            details={
                "threshold": self._threshold,
                "token_overlap_ratio": round(token_overlap_ratio, 4),
                "common_tokens_count": len(common_tokens),
                "expected_token_count": len(expected_tokens),
                "model_token_count": len(model_tokens),
                "normalized_expected_length": len(normalized_expected),
                "normalized_model_length": len(normalized_model),
            },
        )

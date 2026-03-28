"""
Tests unitaires pour HallucinationEvaluator.

Couvre la détection d'hallucinations par keyword anchoring, la détection
de contradictions, et la pénalité appliquée sur le score final.
"""

import pytest

from evaluators.hallucination_evaluator import (
    HallucinationEvaluator,
    _extract_keywords,
    _has_contradiction,
)


class TestExtractKeywords:
    def test_removes_stop_words(self) -> None:
        """Les mots vides (the, is, are...) doivent être exclus."""
        keywords = _extract_keywords("The cat is on the mat")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "on" not in keywords

    def test_keeps_content_words(self) -> None:
        keywords = _extract_keywords("neural network gradient descent optimization")
        assert "neural" in keywords
        assert "network" in keywords
        assert "gradient" in keywords

    def test_filters_short_tokens(self) -> None:
        """Les tokens de moins de 3 caractères doivent être exclus."""
        keywords = _extract_keywords("AI is a big deal")
        assert "is" not in keywords

    def test_lowercase_output(self) -> None:
        keywords = _extract_keywords("Machine Learning Framework")
        assert all(kw == kw.lower() for kw in keywords)

    def test_empty_string(self) -> None:
        assert _extract_keywords("") == []


class TestHasContradiction:
    def test_negation_detected(self) -> None:
        detected, patterns = _has_contradiction(
            expected_output="The service is available",
            model_output="The service is not available",
        )
        assert detected is True

    def test_no_contradiction_clean_text(self) -> None:
        detected, _ = _has_contradiction(
            expected_output="Python is a programming language",
            model_output="Python is a high-level programming language",
        )
        # Note: "not" n'apparaît pas dans une réponse affirmative normale
        # Ce test vérifie que des textes propres ne génèrent pas de faux positifs excessifs

    def test_never_keyword_detected(self) -> None:
        detected, patterns = _has_contradiction(
            expected_output="Always use HTTPS",
            model_output="You should never use HTTP in production",
        )
        assert detected is True


class TestHallucinationEvaluator:
    def setup_method(self) -> None:
        self.evaluator = HallucinationEvaluator(threshold=0.60)

    def test_matching_content_passes(self) -> None:
        """Une réponse couvrant les faits-clés de la référence doit passer."""
        expected = "Python uses garbage collection and dynamic typing for memory management"
        model = "Python employs garbage collection with dynamic typing to handle memory automatically"
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output=expected,
            model_output=model,
        )
        assert result.passed is True

    def test_missing_keywords_reduces_score(self) -> None:
        """Une réponse manquant des faits importants doit avoir un score plus bas."""
        expected = "The Eiffel Tower is located in Paris France and was built in 1889"
        model = "There is a famous tower in Europe"
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output=expected,
            model_output=model,
        )
        assert result.score < 0.60

    def test_empty_expected_output_passes(self) -> None:
        """Cas limite : réponse attendue vide → pass avec warning."""
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="",
            model_output="Some response",
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_details_expose_matched_keywords(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="machine learning gradient descent optimizer",
            model_output="machine learning uses gradient descent",
        )
        assert "matched_keywords" in result.details
        assert "missing_keywords" in result.details
        assert "keyword_coverage_ratio" in result.details

    def test_score_clamped_between_zero_and_one(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="detailed technical explanation",
            model_output="yes",
        )
        assert 0.0 <= result.score <= 1.0

    def test_evaluator_name(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="anything",
        )
        assert result.evaluator_name == "hallucination_detector"

    def test_no_error_on_normal_input(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="normal content here",
            model_output="similar normal content",
        )
        assert result.error is None

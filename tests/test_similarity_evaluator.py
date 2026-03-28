"""
Tests unitaires pour SimilarityEvaluator.

Vérifie le comportement de la similarité cosinus TF-IDF sur des cas représentatifs :
réponses identiques, reformulations, réponses hors-sujet, et cas limites.
"""

import pytest

from evaluators.similarity_evaluator import SimilarityEvaluator, _normalize_text


class TestNormalizeText:
    def test_lowercase_conversion(self) -> None:
        assert _normalize_text("Hello WORLD") == "hello world"

    def test_removes_punctuation(self) -> None:
        result = _normalize_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self) -> None:
        result = _normalize_text("hello   world")
        assert result == "hello world"

    def test_empty_string(self) -> None:
        assert _normalize_text("") == ""

    def test_strips_leading_trailing(self) -> None:
        assert _normalize_text("  hello  ") == "hello"


class TestSimilarityEvaluator:
    def setup_method(self) -> None:
        self.evaluator = SimilarityEvaluator(threshold=0.72)

    def test_identical_texts_score_one(self) -> None:
        """Deux textes identiques doivent produire un score cosinus de 1.0."""
        text = "The quick brown fox jumps over the lazy dog"
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output=text,
            model_output=text,
        )
        assert result.score == pytest.approx(1.0, abs=0.001)
        assert result.passed is True

    def test_similar_texts_produce_meaningful_score(self) -> None:
        """Des textes proches lexicalement doivent produire un score significativement > 0."""
        # TF-IDF cosine sur un corpus de 2 documents est strict car l'IDF amplifie les
        # différences lexicales. On vérifie que le score est élevé (> 0.5) sans exiger
        # le passage du seuil 0.72 — ce seuil est prévu pour des outputs proches du verbatim.
        expected = "gradient descent backpropagation neural network training weights optimizer"
        model = "gradient descent backpropagation neural network training weights learning rate"
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output=expected,
            model_output=model,
        )
        assert result.score > 0.5, f"Score inattendu : {result.score}"

    def test_lenient_threshold_passes_similar_texts(self) -> None:
        """Avec un seuil bas (0.5), des textes proches doivent passer."""
        lenient = SimilarityEvaluator(threshold=0.5)
        expected = "gradient descent backpropagation neural network training"
        model = "gradient descent backpropagation neural network learning"
        result = lenient.evaluate(
            prompt="test",
            expected_output=expected,
            model_output=model,
        )
        assert result.passed is True

    def test_unrelated_texts_fail(self) -> None:
        """Un texte hors-sujet doit échouer le seuil."""
        expected = "The capital of France is Paris"
        model = "Photosynthesis converts sunlight into glucose using chlorophyll"
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output=expected,
            model_output=model,
        )
        assert result.score < 0.72

    def test_score_between_zero_and_one(self) -> None:
        """Le score doit toujours être dans [0.0, 1.0]."""
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="something",
            model_output="completely different content here",
        )
        assert 0.0 <= result.score <= 1.0

    def test_details_contain_token_overlap(self) -> None:
        """Les détails doivent exposer le ratio de token overlap."""
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="machine learning model training",
            model_output="machine learning training process",
        )
        assert "token_overlap_ratio" in result.details
        assert result.details["token_overlap_ratio"] >= 0.0

    def test_custom_threshold_respected(self) -> None:
        """Un threshold élevé (0.99) doit faire échouer des textes proches mais non identiques."""
        strict_evaluator = SimilarityEvaluator(threshold=0.99)
        result = strict_evaluator.evaluate(
            prompt="test",
            expected_output="the cat sat on the mat",
            model_output="the cat sat on a mat",
        )
        assert result.passed is False

    def test_result_has_evaluator_name(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="hello",
            model_output="hello",
        )
        assert result.evaluator_name == "similarity_cosine"

    def test_latency_is_positive(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="hello",
            model_output="hello",
        )
        assert result.latency_ms >= 0.0

    def test_no_error_on_normal_input(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="normal text",
            model_output="also normal text",
        )
        assert result.error is None

"""
Tests unitaires pour LLMJudgeEvaluator.

Utilise un mock du client LLM pour éviter les appels API réels en CI.
Couvre le parsing des réponses JSON du juge, la normalisation des scores,
et la gestion des erreurs (JSON invalide, API failure).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluators.llm_judge_evaluator import LLMJudgeEvaluator, _parse_judge_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_judge_client(response: str) -> Any:
    """Crée un mock de LLMClient qui retourne une réponse prédéfinie."""
    client = MagicMock()
    client.model = "gpt-4o-mini"
    client.provider = "openai"
    client.complete.return_value = response
    return client


def _valid_judge_json(score: int, reasoning: str = "Test reasoning") -> str:
    import json
    return json.dumps({
        "score": score,
        "accuracy": min(score // 3, 3),
        "completeness": min(score // 3, 3),
        "safety": 2,
        "format": 2,
        "reasoning": reasoning,
    })


# ---------------------------------------------------------------------------
# Tests _parse_judge_response
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_valid_json_parsed(self) -> None:
        response = '{"score": 8, "accuracy": 3, "completeness": 3, "safety": 2, "format": 0, "reasoning": "Good"}'
        result = _parse_judge_response(response)
        assert result["score"] == 8
        assert result["reasoning"] == "Good"

    def test_json_with_surrounding_text(self) -> None:
        """Le juge peut ajouter du texte autour du JSON — on doit quand même l'extraire."""
        response = 'Here is my evaluation:\n{"score": 7, "reasoning": "OK"}\nEnd.'
        result = _parse_judge_response(response)
        assert result.get("score") == 7

    def test_invalid_json_returns_empty_dict(self) -> None:
        result = _parse_judge_response("This is not JSON at all.")
        assert result == {}

    def test_empty_string_returns_empty_dict(self) -> None:
        result = _parse_judge_response("")
        assert result == {}


# ---------------------------------------------------------------------------
# Tests LLMJudgeEvaluator
# ---------------------------------------------------------------------------

class TestLLMJudgeEvaluator:
    def test_perfect_score_passes(self) -> None:
        """Score 10/10 doit normaliser à 1.0 et passer."""
        client = _make_judge_client(_valid_judge_json(10, "Perfect response"))
        evaluator = LLMJudgeEvaluator(judge_client=client, threshold=0.60)
        result = evaluator.evaluate(
            prompt="What is Python?",
            expected_output="Python is a high-level programming language.",
            model_output="Python is a high-level, interpreted programming language.",
        )
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_low_score_fails(self) -> None:
        """Score 3/10 = 0.3 normalisé, doit échouer avec threshold=0.60."""
        client = _make_judge_client(_valid_judge_json(3, "Poor response"))
        evaluator = LLMJudgeEvaluator(judge_client=client, threshold=0.60)
        result = evaluator.evaluate(
            prompt="Explain recursion",
            expected_output="Recursion is a function calling itself.",
            model_output="I don't know.",
        )
        assert result.score == pytest.approx(0.3)
        assert result.passed is False

    def test_threshold_boundary(self) -> None:
        """Score exactement au threshold (0.6 = 6/10) doit passer."""
        client = _make_judge_client(_valid_judge_json(6))
        evaluator = LLMJudgeEvaluator(judge_client=client, threshold=0.60)
        result = evaluator.evaluate(
            prompt="test",
            expected_output="reference",
            model_output="model output",
        )
        assert result.passed is True

    def test_score_clamped_at_ten(self) -> None:
        """Un score > 10 retourné par le juge doit être ramené à 10."""
        import json
        bad_response = json.dumps({"score": 15, "reasoning": "Over the top"})
        client = _make_judge_client(bad_response)
        evaluator = LLMJudgeEvaluator(judge_client=client)
        result = evaluator.evaluate(
            prompt="test", expected_output="ref", model_output="output"
        )
        assert result.score == pytest.approx(1.0)

    def test_invalid_json_response_returns_error(self) -> None:
        """Si le juge retourne du texte non-JSON, on retourne une erreur propre."""
        client = _make_judge_client("I cannot evaluate this properly.")
        evaluator = LLMJudgeEvaluator(judge_client=client)
        result = evaluator.evaluate(
            prompt="test", expected_output="ref", model_output="output"
        )
        assert result.passed is False
        assert result.error is not None
        assert result.score == 0.0

    def test_api_error_captured_as_evaluator_error(self) -> None:
        """Une erreur API (RuntimeError) doit être capturée par BaseEvaluator."""
        client = MagicMock()
        client.model = "gpt-4o-mini"
        client.provider = "openai"
        client.complete.side_effect = RuntimeError("API quota exceeded")
        evaluator = LLMJudgeEvaluator(judge_client=client)
        result = evaluator.evaluate(
            prompt="test", expected_output="ref", model_output="output"
        )
        # BaseEvaluator catch-all doit capturer RuntimeError
        assert result.error is not None
        assert result.passed is False

    def test_details_contain_judge_metadata(self) -> None:
        """Les détails doivent exposer les sous-scores et le modèle juge."""
        client = _make_judge_client(_valid_judge_json(8, "Well done"))
        evaluator = LLMJudgeEvaluator(judge_client=client)
        result = evaluator.evaluate(
            prompt="test", expected_output="ref", model_output="output"
        )
        assert "reasoning" in result.details
        assert result.details["judge_model"] == "gpt-4o-mini"
        assert result.details["judge_provider"] == "openai"
        assert result.details["raw_score"] == 8

    def test_evaluator_name(self) -> None:
        client = _make_judge_client(_valid_judge_json(7))
        evaluator = LLMJudgeEvaluator(judge_client=client)
        result = evaluator.evaluate(
            prompt="test", expected_output="ref", model_output="output"
        )
        assert result.evaluator_name == "llm_judge"

    def test_judge_called_with_correct_arguments(self) -> None:
        """Le prompt envoyé au juge doit contenir le prompt, la référence, et l'output."""
        client = _make_judge_client(_valid_judge_json(9))
        evaluator = LLMJudgeEvaluator(judge_client=client)
        evaluator.evaluate(
            prompt="What is 2+2?",
            expected_output="The answer is 4.",
            model_output="2+2 equals 4.",
        )
        call_args = client.complete.call_args
        user_message: str = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "What is 2+2?" in user_message
        assert "The answer is 4." in user_message
        assert "2+2 equals 4." in user_message

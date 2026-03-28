"""
Tests unitaires pour FormatEvaluator.

Couvre la validation JSON, les JSON Schema, les patterns requis/interdits,
les contraintes de longueur, et les cas d'erreur.
"""

import json

import pytest

from evaluators.format_evaluator import FormatEvaluator


class TestFormatEvaluatorJsonValidation:
    def setup_method(self) -> None:
        self.evaluator = FormatEvaluator(threshold=0.8)

    def test_valid_json_passes(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="{}",
            model_output=json.dumps({"status": "ok", "count": 5}),
            metadata={"expect_valid_json": True},
        )
        assert "valid_json_syntax" in result.details["checks_passed"]

    def test_invalid_json_fails(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="{}",
            model_output="{ invalid json here }",
            metadata={"expect_valid_json": True},
        )
        assert result.passed is False
        assert any("JSON invalide" in v for v in result.details["violations"])

    def test_json_schema_valid(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        model_output = json.dumps({"name": "Alice", "age": 30})
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="{}",
            model_output=model_output,
            metadata={"json_schema": schema},
        )
        assert "json_schema_valid" in result.details["checks_passed"]
        assert result.passed is True

    def test_json_schema_violation_fails(self) -> None:
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }
        # Fournit une string là où un int est attendu
        model_output = json.dumps({"age": "not-an-integer"})
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="{}",
            model_output=model_output,
            metadata={"json_schema": schema},
        )
        assert result.passed is False


class TestFormatEvaluatorPatterns:
    def setup_method(self) -> None:
        self.evaluator = FormatEvaluator(threshold=0.8)

    def test_required_pattern_present_passes(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="I cannot assist with that request",
            metadata={"required_patterns": ["cannot|unable|will not"]},
        )
        assert result.passed is True

    def test_required_pattern_absent_fails(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="Sure, here is the answer",
            metadata={"required_patterns": ["cannot|unable|will not"]},
        )
        assert result.passed is False
        assert any("Pattern requis absent" in v for v in result.details["violations"])

    def test_forbidden_pattern_absent_passes(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="Here is a safe response",
            metadata={"forbidden_patterns": ["bomb|weapon|illegal"]},
        )
        assert result.passed is True

    def test_forbidden_pattern_present_fails(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="Here is how to make a bomb",
            metadata={"forbidden_patterns": ["bomb|weapon"]},
        )
        assert result.passed is False

    def test_pattern_case_insensitive(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="I CANNOT assist with this",
            metadata={"required_patterns": ["cannot"]},
        )
        assert result.passed is True


class TestFormatEvaluatorLengthConstraints:
    def setup_method(self) -> None:
        self.evaluator = FormatEvaluator(threshold=0.8)

    def test_response_within_max_length_passes(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="Short response",
            metadata={"max_length": 100},
        )
        assert "length_within_max" in result.details["checks_passed"]

    def test_response_exceeds_max_length_fails(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="A" * 200,
            metadata={"max_length": 100},
        )
        assert result.passed is False
        assert any("trop longue" in v for v in result.details["violations"])

    def test_response_below_min_length_fails(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="Hi",
            metadata={"min_length": 50},
        )
        assert result.passed is False
        assert any("trop courte" in v for v in result.details["violations"])


class TestFormatEvaluatorEdgeCases:
    def setup_method(self) -> None:
        self.evaluator = FormatEvaluator()

    def test_no_metadata_returns_perfect_score(self) -> None:
        """Sans contraintes, aucun check n'est effectué → score 1.0."""
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="any response",
            metadata={},
        )
        assert result.score == pytest.approx(1.0)

    def test_score_between_zero_and_one(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="response",
            metadata={"required_patterns": ["missing_pattern"], "forbidden_patterns": ["response"]},
        )
        assert 0.0 <= result.score <= 1.0

    def test_evaluator_name(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="response",
        )
        assert result.evaluator_name == "format_compliance"

    def test_details_always_present(self) -> None:
        result = self.evaluator.evaluate(
            prompt="test",
            expected_output="anything",
            model_output="response",
        )
        assert "violations" in result.details
        assert "checks_passed" in result.details
        assert "response_length_chars" in result.details

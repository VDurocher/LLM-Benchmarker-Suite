"""
Format compliance evaluator.

Verifies that the model output conforms to the expected format:
- Valid JSON conforming to a schema (JSON Schema Draft-7)
- Presence of required structural tags (Markdown, XML-like)
- Compliance with length constraints
- Valid encoding and characters

Primarily used for models expected to produce structured outputs
(APIs, agents, data pipelines).
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from jsonschema.exceptions import SchemaError

from evaluators.base_evaluator import BaseEvaluator, EvaluationResult


class FormatEvaluator(BaseEvaluator):
    """
    Validates the structural and syntactic conformance of the model output.

    Configuration parameters via `metadata` (all optional):
    - `json_schema` (dict): JSON Schema Draft-7 schema to validate against.
    - `required_fields` (list[str]): JSON keys that must be present.
    - `required_patterns` (list[str]): regexes that must match in the response.
    - `forbidden_patterns` (list[str]): regexes that must NOT match.
    - `max_length` (int): maximum allowed length in characters.
    - `min_length` (int): minimum required length in characters.
    - `expect_valid_json` (bool): if True, the response must be valid JSON.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__(name="format_compliance", threshold=threshold)

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        violations: list[str] = []
        checks_passed: list[str] = []
        parsed_json: dict[str, Any] | list[Any] | None = None

        # --- JSON validation ---
        expect_valid_json: bool = metadata.get("expect_valid_json", False)
        json_schema: dict[str, Any] | None = metadata.get("json_schema")

        # If a schema is provided, JSON is expected
        if json_schema is not None:
            expect_valid_json = True

        if expect_valid_json:
            try:
                parsed_json = json.loads(model_output.strip())
                checks_passed.append("valid_json_syntax")
            except json.JSONDecodeError as exc:
                violations.append(f"Invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})")

        # --- JSON Schema validation ---
        if json_schema is not None and parsed_json is not None:
            try:
                validator = Draft7Validator(json_schema)
                schema_errors = list(validator.iter_errors(parsed_json))
                if schema_errors:
                    for error in schema_errors[:3]:  # limited to 3 errors for readability
                        violations.append(f"Schema violation: {error.message}")
                else:
                    checks_passed.append("json_schema_valid")
            except SchemaError as exc:
                violations.append(f"Malformed JSON schema: {exc.message}")

        # --- Required fields check ---
        required_fields: list[str] = metadata.get("required_fields", [])
        if required_fields and isinstance(parsed_json, dict):
            for field_name in required_fields:
                if field_name in parsed_json:
                    checks_passed.append(f"field_present:{field_name}")
                else:
                    violations.append(f"Required field missing: '{field_name}'")

        # --- Required patterns ---
        required_patterns: list[str] = metadata.get("required_patterns", [])
        for pattern in required_patterns:
            if re.search(pattern, model_output, re.DOTALL | re.IGNORECASE):
                checks_passed.append(f"pattern_found:{pattern[:30]}")
            else:
                violations.append(f"Required pattern missing: '{pattern[:50]}'")

        # --- Forbidden patterns ---
        forbidden_patterns: list[str] = metadata.get("forbidden_patterns", [])
        for pattern in forbidden_patterns:
            if re.search(pattern, model_output, re.DOTALL | re.IGNORECASE):
                violations.append(f"Forbidden pattern detected: '{pattern[:50]}'")
            else:
                checks_passed.append(f"pattern_absent:{pattern[:30]}")

        # --- Length constraints ---
        response_length = len(model_output)
        max_length: int | None = metadata.get("max_length")
        min_length: int | None = metadata.get("min_length")

        if max_length is not None and response_length > max_length:
            violations.append(
                f"Response too long: {response_length} > {max_length} characters"
            )
        elif max_length is not None:
            checks_passed.append("length_within_max")

        if min_length is not None and response_length < min_length:
            violations.append(
                f"Response too short: {response_length} < {min_length} characters"
            )
        elif min_length is not None:
            checks_passed.append("length_above_min")

        # --- Score computation ---
        total_checks = len(checks_passed) + len(violations)
        score = len(checks_passed) / total_checks if total_checks > 0 else 1.0
        passed = score >= self._threshold and len(violations) == 0

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(score, 4),
            details={
                "threshold": self._threshold,
                "violations": violations,
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "violations_count": len(violations),
                "response_length_chars": response_length,
                "json_parsed_successfully": parsed_json is not None,
            },
        )

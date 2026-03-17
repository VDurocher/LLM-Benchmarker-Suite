"""
Évaluateur de conformité de format.

Vérifie que la sortie du modèle respecte le format attendu :
- JSON valide et conforme à un schéma (JSON Schema Draft-7)
- Présence de balises structurelles requises (Markdown, XML-like)
- Respect des contraintes de longueur
- Encodage et caractères valides

Utilisé principalement pour les modèles censés produire des outputs
structurés (APIs, agents, pipelines de données).
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
    Valide la conformité structurelle et syntaxique de la sortie du modèle.

    Paramètres de configuration via `metadata` (tous optionnels) :
    - `json_schema` (dict) : schéma JSON Schema Draft-7 à valider.
    - `required_fields` (list[str]) : clés JSON qui doivent être présentes.
    - `required_patterns` (list[str]) : regex qui doivent matcher dans la réponse.
    - `forbidden_patterns` (list[str]) : regex qui ne doivent PAS matcher.
    - `max_length` (int) : longueur maximale autorisée en caractères.
    - `min_length` (int) : longueur minimale requise en caractères.
    - `expect_valid_json` (bool) : si True, la réponse doit être du JSON valide.
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

        # --- Vérification JSON ---
        expect_valid_json: bool = metadata.get("expect_valid_json", False)
        json_schema: dict[str, Any] | None = metadata.get("json_schema")

        # Si un schéma est fourni, on attend forcément du JSON
        if json_schema is not None:
            expect_valid_json = True

        if expect_valid_json:
            try:
                parsed_json = json.loads(model_output.strip())
                checks_passed.append("valid_json_syntax")
            except json.JSONDecodeError as exc:
                violations.append(f"JSON invalide : {exc.msg} (ligne {exc.lineno}, col {exc.colno})")

        # --- Validation JSON Schema ---
        if json_schema is not None and parsed_json is not None:
            try:
                validator = Draft7Validator(json_schema)
                schema_errors = list(validator.iter_errors(parsed_json))
                if schema_errors:
                    for error in schema_errors[:3]:  # limité à 3 erreurs pour la lisibilité
                        violations.append(f"Schema violation: {error.message}")
                else:
                    checks_passed.append("json_schema_valid")
            except SchemaError as exc:
                violations.append(f"Schéma JSON malformé : {exc.message}")

        # --- Vérification des champs requis ---
        required_fields: list[str] = metadata.get("required_fields", [])
        if required_fields and isinstance(parsed_json, dict):
            for field_name in required_fields:
                if field_name in parsed_json:
                    checks_passed.append(f"field_present:{field_name}")
                else:
                    violations.append(f"Champ requis manquant : '{field_name}'")

        # --- Patterns requis ---
        required_patterns: list[str] = metadata.get("required_patterns", [])
        for pattern in required_patterns:
            if re.search(pattern, model_output, re.DOTALL | re.IGNORECASE):
                checks_passed.append(f"pattern_found:{pattern[:30]}")
            else:
                violations.append(f"Pattern requis absent : '{pattern[:50]}'")

        # --- Patterns interdits ---
        forbidden_patterns: list[str] = metadata.get("forbidden_patterns", [])
        for pattern in forbidden_patterns:
            if re.search(pattern, model_output, re.DOTALL | re.IGNORECASE):
                violations.append(f"Pattern interdit détecté : '{pattern[:50]}'")
            else:
                checks_passed.append(f"pattern_absent:{pattern[:30]}")

        # --- Contraintes de longueur ---
        response_length = len(model_output)
        max_length: int | None = metadata.get("max_length")
        min_length: int | None = metadata.get("min_length")

        if max_length is not None and response_length > max_length:
            violations.append(
                f"Réponse trop longue : {response_length} > {max_length} caractères"
            )
        elif max_length is not None:
            checks_passed.append("length_within_max")

        if min_length is not None and response_length < min_length:
            violations.append(
                f"Réponse trop courte : {response_length} < {min_length} caractères"
            )
        elif min_length is not None:
            checks_passed.append("length_above_min")

        # --- Calcul du score ---
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

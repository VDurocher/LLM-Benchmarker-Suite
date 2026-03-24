"""
Pipeline d'évaluation partagé entre main.py et compare_runner.py.

Fournit les fonctions de chargement des cas de test, d'instanciation
des évaluateurs et d'évaluation d'un cas individuel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DEFAULT_WEIGHTS, EvaluatorWeights
from evaluators import (
    CodeEvaluator,
    ConsistencyEvaluator,
    EvaluationResult,
    FormatEvaluator,
    HallucinationEvaluator,
    SimilarityEvaluator,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Répertoire contenant les fichiers de cas de test
DATA_DIR = Path(__file__).parent.parent / "data"


def load_test_cases(test_set: str) -> list[dict[str, Any]]:
    """Charge les cas de test depuis les fichiers JSON. 'all' agrège tous les sets."""
    if test_set == "all":
        all_cases: list[dict[str, Any]] = []
        for available_set in ["safety", "logic", "format", "consistency"]:
            file_path = DATA_DIR / f"test_cases_{available_set}.json"
            if file_path.exists():
                all_cases.extend(_load_single_test_set(available_set))
        logger.info("Total cas de test chargés : %d", len(all_cases))
        return all_cases
    return _load_single_test_set(test_set)


def _load_single_test_set(test_set: str) -> list[dict[str, Any]]:
    """Charge un fichier de cas de test par nom d'ensemble."""
    file_path = DATA_DIR / f"test_cases_{test_set}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Ensemble de tests non trouvé : {file_path}")
    with open(file_path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    cases = data.get("cases", [])
    logger.info("Chargement de '%s' : %d cas de test", test_set, len(cases))
    return cases


def build_evaluators() -> dict[str, Any]:
    """Instancie tous les évaluateurs disponibles."""
    return {
        "similarity": SimilarityEvaluator(),
        "hallucination": HallucinationEvaluator(),
        "format": FormatEvaluator(),
        "code": CodeEvaluator(),
        "consistency": ConsistencyEvaluator(),
    }


def evaluate_case(
    case: dict[str, Any],
    evaluators: dict[str, Any],
    weights: EvaluatorWeights | None = None,
    verbose: bool = False,
) -> tuple[list[EvaluationResult], float, bool]:
    """
    Évalue un cas de test avec les évaluateurs applicables.
    Retourne (résultats, score_composite, passed).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    prompt: str = case.get("prompt", "")
    expected_output: str = case.get("expected_output", "")
    model_output: str = case.get("model_output", "")
    metadata: dict[str, Any] = case.get("metadata", {})
    applicable: list[str] = metadata.get("evaluators", ["similarity", "hallucination", "format"])

    weight_map = {
        "similarity": weights.similarity,
        "hallucination": weights.hallucination,
        "format": weights.format_compliance,
        "code": weights.format_compliance,
        "consistency": weights.format_compliance,
    }

    results: list[EvaluationResult] = []
    total_weight = 0.0
    weighted_score = 0.0

    for ev_name in applicable:
        if ev_name not in evaluators:
            logger.warning("Évaluateur inconnu : '%s' — ignoré", ev_name)
            continue
        result = evaluators[ev_name].evaluate(
            prompt=prompt,
            expected_output=expected_output,
            model_output=model_output,
            metadata=metadata,
        )
        results.append(result)
        weight = weight_map.get(ev_name, 0.25)
        weighted_score += result.score * weight
        total_weight += weight

        if verbose:
            status_icon = "✓" if result.passed else "✗"
            logger.info(
                "  [%s] %s score=%.4f latency=%.1fms",
                status_icon,
                result.evaluator_name,
                result.score,
                result.latency_ms,
            )
            if result.error:
                logger.warning("    Erreur évaluateur: %s", result.error)

    composite_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
    case_passed = all(result.passed for result in results if result.error is None)
    return results, round(composite_score, 4), case_passed

"""
LLM-Benchmarker-Suite — Point d'entrée CLI.

Orchestre le pipeline complet d'évaluation :
1. Chargement des cas de test depuis /data/
2. Exécution des évaluateurs configurés par cas
3. Calcul du score composite pondéré
4. Génération du rapport JSON dans /reports/

Usage :
    python main.py --model gpt-4o --test-set safety
    python main.py --model claude-3-5-sonnet --test-set all --output-dir ./results
    python main.py --model gpt-4o --test-set logic --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from config import (
    AVAILABLE_TEST_SETS,
    DEFAULT_WEIGHTS,
    EvaluatorWeights,
)
from evaluators import (
    CodeEvaluator,
    EvaluationResult,
    FormatEvaluator,
    HallucinationEvaluator,
    SimilarityEvaluator,
)
from utils.logger import get_logger
from utils.report_generator import ReportGenerator

logger = get_logger(__name__)

# Répertoire contenant les fichiers de cas de test
DATA_DIR = Path(__file__).parent / "data"


def _load_test_cases(test_set: str) -> list[dict[str, Any]]:
    """
    Charge les cas de test depuis les fichiers JSON correspondants.
    Si test_set == 'all', agrège tous les ensembles disponibles.
    """
    if test_set == "all":
        all_cases: list[dict[str, Any]] = []
        for available_set in ["safety", "logic", "format"]:
            all_cases.extend(_load_single_test_set(available_set))
        logger.info("Total cas de test chargés : %d", len(all_cases))
        return all_cases

    return _load_single_test_set(test_set)


def _load_single_test_set(test_set: str) -> list[dict[str, Any]]:
    """Charge un fichier de cas de test par nom d'ensemble."""
    file_path = DATA_DIR / f"test_cases_{test_set}.json"
    if not file_path.exists():
        logger.error("Fichier de test introuvable : %s", file_path)
        raise FileNotFoundError(f"Ensemble de tests non trouvé : {file_path}")

    with open(file_path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    cases = data.get("cases", [])
    logger.info("Chargement de '%s' : %d cas de test", test_set, len(cases))
    return cases


def _build_evaluators() -> dict[str, Any]:
    """Instancie les évaluateurs disponibles (réutilisés sur tous les cas)."""
    return {
        "similarity": SimilarityEvaluator(),
        "hallucination": HallucinationEvaluator(),
        "format": FormatEvaluator(),
        "code": CodeEvaluator(),
    }


def _evaluate_case(
    case: dict[str, Any],
    evaluators: dict[str, Any],
    weights: EvaluatorWeights,
    verbose: bool,
) -> tuple[list[EvaluationResult], float, bool]:
    """
    Exécute les évaluateurs applicables à un cas de test.
    Retourne (résultats, score_composite, passed).
    """
    prompt: str = case.get("prompt", "")
    expected_output: str = case.get("expected_output", "")
    model_output: str = case.get("model_output", "")
    metadata: dict[str, Any] = case.get("metadata", {})
    applicable_evaluators: list[str] = metadata.get(
        "evaluators", ["similarity", "hallucination", "format"]
    )

    results: list[EvaluationResult] = []
    weight_map = {
        "similarity": weights.similarity,
        "hallucination": weights.hallucination,
        "format": weights.format_compliance,
        "code": weights.format_compliance,  # code partage le poids format
    }

    total_weight = 0.0
    weighted_score = 0.0

    for evaluator_name in applicable_evaluators:
        if evaluator_name not in evaluators:
            logger.warning("Évaluateur inconnu : '%s' — ignoré", evaluator_name)
            continue

        evaluator = evaluators[evaluator_name]
        result = evaluator.evaluate(
            prompt=prompt,
            expected_output=expected_output,
            model_output=model_output,
            metadata=metadata,
        )
        results.append(result)

        weight = weight_map.get(evaluator_name, 0.25)
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

    # Normalisation du score composite si les poids ne somment pas à 1
    composite_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

    # Un cas passe si TOUS ses évaluateurs passent
    case_passed = all(result.passed for result in results if result.error is None)

    return results, round(composite_score, 4), case_passed


def _run_benchmark(
    model_name: str,
    test_set: str,
    output_dir: str | None,
    verbose: bool,
) -> int:
    """
    Pipeline principal d'évaluation.
    Retourne 0 si le pass rate cible est atteint, 1 sinon.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Démarrage")
    logger.info("Modèle cible : %s", model_name)
    logger.info("Ensemble de tests : %s", test_set)
    logger.info("=" * 60)

    # Chargement des données
    try:
        test_cases = _load_test_cases(test_set)
    except FileNotFoundError as exc:
        logger.error("Impossible de charger les tests : %s", exc)
        return 1

    if not test_cases:
        logger.error("Aucun cas de test trouvé pour l'ensemble '%s'", test_set)
        return 1

    # Initialisation des composants
    evaluators = _build_evaluators()
    report_generator = ReportGenerator(model_name=model_name, test_set=test_set)
    weights = DEFAULT_WEIGHTS

    passed_count = 0
    failed_count = 0

    # Boucle d'évaluation principale
    for index, case in enumerate(test_cases, start=1):
        case_id: str = case.get("id", f"case_{index:03d}")
        logger.info("[%d/%d] Évaluation du cas : %s", index, len(test_cases), case_id)

        evaluation_results, composite_score, case_passed = _evaluate_case(
            case=case,
            evaluators=evaluators,
            weights=weights,
            verbose=verbose,
        )

        if case_passed:
            passed_count += 1
            logger.info("  → PASS (score composite: %.4f)", composite_score)
        else:
            failed_count += 1
            logger.info("  → FAIL (score composite: %.4f)", composite_score)

        report_generator.add_case_result(
            case_id=case_id,
            prompt=case.get("prompt", ""),
            expected_output=case.get("expected_output", ""),
            model_output=case.get("model_output", ""),
            evaluation_results=evaluation_results,
            composite_score=composite_score,
            passed=case_passed,
        )

    # Génération et sauvegarde du rapport
    report_path = report_generator.save(output_dir=output_dir)

    # Résumé final
    total = passed_count + failed_count
    pass_rate = passed_count / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("RÉSULTATS FINAUX")
    logger.info("Cas traités : %d | Passés : %d | Échoués : %d", total, passed_count, failed_count)
    logger.info("Pass rate : %.1f%% (cible : 99.0%%)", pass_rate * 100)
    logger.info(
        "Verdict : %s",
        "✓ PRODUCTION READY" if pass_rate >= 0.99 else "✗ BELOW TARGET — DO NOT DEPLOY",
    )
    logger.info("Rapport : %s", report_path)
    logger.info("=" * 60)

    # Code de sortie : 0 si cible atteinte, 1 sinon (utile dans les pipelines CI/CD)
    return 0 if pass_rate >= 0.99 else 1


def _parse_args() -> argparse.Namespace:
    """Configure et parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="llm-benchmarker",
        description=(
            "LLM-Benchmarker-Suite — Évaluation de fiabilité des modèles de langage\n"
            "pour déploiements en production avec objectif de 99%% de pass rate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --model gpt-4o --test-set safety
  python main.py --model claude-3-5-sonnet --test-set all --verbose
  python main.py --model llama-3 --test-set format --output-dir ./ci-reports
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Identifiant du modèle évalué (ex: gpt-4o, claude-3-5-sonnet, llama-3)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=AVAILABLE_TEST_SETS,
        default="all",
        help=f"Ensemble de tests à utiliser. Options : {', '.join(AVAILABLE_TEST_SETS)} (défaut: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Répertoire de sortie pour les rapports JSON (défaut: ./reports/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Affiche les scores détaillés par évaluateur pour chaque cas",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = _run_benchmark(
        model_name=args.model,
        test_set=args.test_set,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    sys.exit(exit_code)

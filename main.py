"""
LLM-Benchmarker-Suite — Point d'entrée CLI.

Orchestre le pipeline complet d'évaluation :
1. Chargement des cas de test depuis /data/
2. Exécution des évaluateurs configurés par cas
3. Calcul du score composite pondéré
4. Génération du rapport JSON et/ou HTML dans /reports/

Usage :
    python main.py --model gpt-4o --test-set safety
    python main.py --model claude-3-5-sonnet --test-set all --output-dir ./results
    python main.py --model gpt-4o --test-set logic --verbose
    python main.py --model gpt-4o --test-set format --format html
    python main.py --model gpt-4o --test-set all --format both
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from config import (
    AVAILABLE_TEST_SETS,
    DEFAULT_WEIGHTS,
    HTML_REPORT_OUTPUT_DIR,
    REPORT_FORMAT_BOTH,
    REPORT_FORMAT_HTML,
    REPORT_FORMAT_JSON,
)
from utils.evaluation_pipeline import build_evaluators, evaluate_case, load_test_cases
from utils.html_report import HtmlReportGenerator
from utils.logger import get_logger
from utils.report_generator import ReportGenerator

logger = get_logger(__name__)


def _run_benchmark(
    model_name: str,
    test_set: str,
    output_dir: str | None,
    verbose: bool,
    report_format: str = REPORT_FORMAT_JSON,
) -> int:
    """
    Pipeline principal d'évaluation.
    Retourne 0 si le pass rate cible est atteint, 1 sinon.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Démarrage")
    logger.info("Modèle cible : %s", model_name)
    logger.info("Ensemble de tests : %s", test_set)
    logger.info("Format de rapport : %s", report_format)
    logger.info("=" * 60)

    try:
        test_cases = load_test_cases(test_set)
    except FileNotFoundError as exc:
        logger.error("Impossible de charger les tests : %s", exc)
        return 1

    if not test_cases:
        logger.error("Aucun cas de test trouvé pour l'ensemble '%s'", test_set)
        return 1

    evaluators = build_evaluators()
    weights = DEFAULT_WEIGHTS
    use_json = report_format in (REPORT_FORMAT_JSON, REPORT_FORMAT_BOTH)
    use_html = report_format in (REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH)

    json_generator = ReportGenerator(model_name=model_name, test_set=test_set) if use_json else None
    html_generator = HtmlReportGenerator(model_name=model_name, test_set=test_set) if use_html else None

    passed_count = 0
    failed_count = 0

    for index, case in enumerate(test_cases, start=1):
        case_id: str = case.get("id", f"case_{index:03d}")
        logger.info("[%d/%d] Évaluation du cas : %s", index, len(test_cases), case_id)

        evaluation_results, composite_score, case_passed = evaluate_case(
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

        case_kwargs: dict[str, Any] = {
            "case_id": case_id,
            "prompt": case.get("prompt", ""),
            "expected_output": case.get("expected_output", ""),
            "model_output": case.get("model_output", ""),
            "evaluation_results": evaluation_results,
            "composite_score": composite_score,
            "passed": case_passed,
        }
        if json_generator is not None:
            json_generator.add_case_result(**case_kwargs)
        if html_generator is not None:
            html_generator.add_case_result(**case_kwargs)

    if json_generator is not None:
        report_path = json_generator.save(output_dir=output_dir)
        logger.info("Rapport JSON : %s", report_path)
    if html_generator is not None:
        html_path = html_generator.save(output_dir=output_dir or HTML_REPORT_OUTPUT_DIR)
        logger.info("Rapport HTML : %s", html_path)

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
    logger.info("=" * 60)

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
  python main.py --model gpt-4o --test-set all --format both
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
        help=f"Ensemble de tests. Options : {', '.join(AVAILABLE_TEST_SETS)} (défaut: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Répertoire de sortie pour les rapports (défaut: ./reports/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=[REPORT_FORMAT_JSON, REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH],
        default=REPORT_FORMAT_JSON,
        help="Format de rapport généré : json, html, ou both (défaut: json)",
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
        report_format=args.format,
    )
    sys.exit(exit_code)

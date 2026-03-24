"""
LLM-Benchmarker-Suite — Mode comparaison multi-modèles.

Lance le même ensemble de tests sur plusieurs modèles et compare les résultats
côte à côte. Les sorties modèles proviennent des fichiers JSON (mode MOCK).

Usage :
    python compare_runner.py --models gpt-4o claude-3-5-sonnet llama-3 --test-set safety
    python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set all --output-dir ./ci
    python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set format --format html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    AVAILABLE_TEST_SETS,
    HTML_REPORT_OUTPUT_DIR,
    REPORT_FORMAT_BOTH,
    REPORT_FORMAT_HTML,
    REPORT_FORMAT_JSON,
    REPORT_OUTPUT_DIR,
)
from utils.evaluation_pipeline import build_evaluators, evaluate_case, load_test_cases
from utils.html_comparison import build_comparison_html
from utils.html_report import HtmlReportGenerator
from utils.logger import get_logger
from utils.report_generator import ReportGenerator

logger = get_logger(__name__)


def _run_model_benchmark(
    model_name: str,
    test_cases: list[dict[str, Any]],
    test_set: str,
    output_dir: str | None,
    report_format: str,
    verbose: bool,
) -> dict[str, Any]:
    """
    Exécute le benchmark pour un seul modèle et sauvegarde ses rapports individuels.
    Retourne un dict de statistiques pour la comparaison.
    """
    logger.info("--- Évaluation du modèle : %s ---", model_name)

    evaluators = build_evaluators()
    use_json = report_format in (REPORT_FORMAT_JSON, REPORT_FORMAT_BOTH)
    use_html = report_format in (REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH)

    json_gen = ReportGenerator(model_name=model_name, test_set=test_set) if use_json else None
    html_gen = HtmlReportGenerator(model_name=model_name, test_set=test_set) if use_html else None

    passed_count = 0
    total_score = 0.0
    total_latency = 0.0

    for index, case in enumerate(test_cases, start=1):
        case_id: str = case.get("id", f"case_{index:03d}")
        eval_results, composite_score, case_passed = evaluate_case(
            case=case,
            evaluators=evaluators,
            verbose=verbose,
        )
        total_score += composite_score
        total_latency += sum(r.latency_ms for r in eval_results)
        if case_passed:
            passed_count += 1

        if verbose:
            status = "PASS" if case_passed else "FAIL"
            logger.info("  [%s] %s — score=%.4f", status, case_id, composite_score)

        case_kwargs: dict[str, Any] = {
            "case_id": case_id,
            "prompt": case.get("prompt", ""),
            "expected_output": case.get("expected_output", ""),
            "model_output": case.get("model_output", ""),
            "evaluation_results": eval_results,
            "composite_score": composite_score,
            "passed": case_passed,
        }
        if json_gen is not None:
            json_gen.add_case_result(**case_kwargs)
        if html_gen is not None:
            html_gen.add_case_result(**case_kwargs)

    if json_gen is not None:
        json_gen.save(output_dir=output_dir)
    if html_gen is not None:
        html_gen.save(output_dir=output_dir or HTML_REPORT_OUTPUT_DIR)

    total_cases = len(test_cases)
    pass_rate = passed_count / total_cases if total_cases > 0 else 0.0
    avg_score = total_score / total_cases if total_cases > 0 else 0.0

    return {
        "model_name": model_name,
        "total_cases": total_cases,
        "passed_cases": passed_count,
        "failed_cases": total_cases - passed_count,
        "pass_rate": round(pass_rate, 4),
        "pass_rate_percent": round(pass_rate * 100, 2),
        "avg_score": round(avg_score, 4),
        "total_latency_ms": round(total_latency, 2),
    }


def _determine_winner(model_stats: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Identifie le meilleur modèle par pass rate, puis score moyen en cas d'égalité.
    Retourne (nom_du_gagnant, raison).
    """
    sorted_models = sorted(
        model_stats,
        key=lambda m: (m["pass_rate"], m["avg_score"]),
        reverse=True,
    )
    best = sorted_models[0]
    second = sorted_models[1] if len(sorted_models) > 1 else None

    if second is None:
        reason = f"Seul modèle évalué (pass rate: {best['pass_rate_percent']}%)"
    elif best["pass_rate"] > second["pass_rate"]:
        reason = f"Meilleur pass rate ({best['pass_rate_percent']}% vs {second['pass_rate_percent']}%)"
    else:
        reason = (
            f"Pass rate égal ({best['pass_rate_percent']}%), "
            f"meilleur score moyen ({best['avg_score']:.4f} vs {second['avg_score']:.4f})"
        )
    return best["model_name"], reason


def _save_report_file(
    content: str | dict[str, Any],
    output_dir: str | None,
    filename: str,
    is_json: bool,
) -> Path:
    """Sauvegarde un fichier de rapport JSON ou HTML. Retourne le chemin absolu."""
    fallback = REPORT_OUTPUT_DIR if is_json else HTML_REPORT_OUTPUT_DIR
    target_dir = Path(output_dir or fallback)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    with open(output_path, "w", encoding="utf-8") as file_handle:
        if is_json and isinstance(content, dict):
            json.dump(content, file_handle, indent=2, ensure_ascii=False)
        else:
            file_handle.write(str(content))
    logger.info("Rapport sauvegardé → %s", output_path)
    return output_path.resolve()


def _run_comparison(
    models: list[str],
    test_set: str,
    output_dir: str | None,
    report_format: str,
    verbose: bool,
) -> int:
    """
    Orchestre la comparaison multi-modèles.
    Retourne 0 si au moins un modèle atteint la cible de 99 %, 1 sinon.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Mode comparaison")
    logger.info("Modèles : %s", ", ".join(models))
    logger.info("Ensemble de tests : %s", test_set)
    logger.info("=" * 60)

    try:
        test_cases = load_test_cases(test_set)
    except FileNotFoundError as exc:
        logger.error("Impossible de charger les tests : %s", exc)
        return 1

    if not test_cases:
        logger.error("Aucun cas de test pour l'ensemble '%s'", test_set)
        return 1

    comparison_id = datetime.now(tz=timezone.utc).strftime("cmp_%Y%m%d_%H%M%S")
    model_stats: list[dict[str, Any]] = [
        _run_model_benchmark(
            model_name=model_name,
            test_cases=test_cases,
            test_set=test_set,
            output_dir=output_dir,
            report_format=report_format,
            verbose=verbose,
        )
        for model_name in models
    ]

    winner, winner_reason = _determine_winner(model_stats)
    comparison_report: dict[str, Any] = {
        "comparison_id": comparison_id,
        "test_set": test_set,
        "models_compared": [s["model_name"] for s in model_stats],
        "results": {s["model_name"]: s for s in model_stats},
        "winner": winner,
        "winner_reason": winner_reason,
    }

    use_json = report_format in (REPORT_FORMAT_JSON, REPORT_FORMAT_BOTH)
    use_html = report_format in (REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH)

    if use_json:
        _save_report_file(comparison_report, output_dir, f"comparison_{comparison_id}.json", is_json=True)
    if use_html:
        _save_report_file(
            build_comparison_html(comparison_report, model_stats),
            output_dir,
            f"comparison_{comparison_id}.html",
            is_json=False,
        )

    logger.info("=" * 60)
    logger.info("RÉSULTATS DE COMPARAISON")
    for stats in model_stats:
        logger.info(
            "  %s — pass rate: %.1f%% | avg score: %.4f",
            stats["model_name"],
            stats["pass_rate_percent"],
            stats["avg_score"],
        )
    logger.info("Gagnant : %s (%s)", winner, winner_reason)
    logger.info("=" * 60)

    return 0 if any(s["pass_rate"] >= 0.99 for s in model_stats) else 1


def _parse_args() -> argparse.Namespace:
    """Configure et parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="compare-runner",
        description=(
            "LLM-Benchmarker-Suite — Comparaison multi-modèles\n"
            "Lance le même test set sur plusieurs modèles et compare les résultats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set safety
  python compare_runner.py --models gpt-4o claude-3-5-sonnet llama-3 --test-set all --verbose
  python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set format --format html
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Liste des modèles à comparer (ex: gpt-4o claude-3-5-sonnet llama-3)",
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
        help="Format des rapports : json, html, ou both (défaut: json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Affiche les statuts par cas de test pour chaque modèle",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if len(args.models) < 2:
        logger.error("Au moins 2 modèles sont requis pour une comparaison.")
        sys.exit(1)
    exit_code = _run_comparison(
        models=args.models,
        test_set=args.test_set,
        output_dir=args.output_dir,
        report_format=args.format,
        verbose=args.verbose,
    )
    sys.exit(exit_code)

"""
LLM-Benchmarker-Suite — Mode comparaison multi-modèles.

Lance le même ensemble de tests sur plusieurs modèles et compare les résultats
côte à côte. Supporte le mode offline (model_output pré-rempli) et le mode live
(appels API réels). Les rapports sont générés par modèle et en comparaison globale.

Usage — mode offline :
    python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set safety
    python compare_runner.py --models gpt-4o gpt-4o-mini --test-set all --format html

Usage — mode live :
    python compare_runner.py --models gpt-4o gpt-4o-mini --test-set reasoning --live --provider openai
    python compare_runner.py --models claude-3-5-sonnet claude-haiku-4-5-20251001 --live --provider anthropic

Usage — avec LLM-as-a-judge :
    python compare_runner.py --models gpt-4o gpt-4o-mini --test-set all --live --judge --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    AVAILABLE_TEST_SETS,
    HTML_REPORT_OUTPUT_DIR,
    LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC,
    LLM_JUDGE_DEFAULT_MODEL_OPENAI,
    REPORT_FORMAT_BOTH,
    REPORT_FORMAT_HTML,
    REPORT_FORMAT_JSON,
    REPORT_OUTPUT_DIR,
    SUPPORTED_PROVIDERS,
)
from utils.evaluation_pipeline import (
    build_api_client,
    build_evaluators,
    evaluate_case,
    fetch_live_outputs,
    load_test_cases,
    resolve_api_key,
)
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
    live: bool = False,
    provider: str = "openai",
    api_key: str | None = None,
    judge_client: Any | None = None,
) -> dict[str, Any]:
    """
    Exécute le benchmark pour un seul modèle et sauvegarde ses rapports individuels.
    Retourne un dict de statistiques pour la comparaison.
    """
    logger.info("--- Évaluation du modèle : %s ---", model_name)

    # Mode live : appel API réel pour ce modèle spécifique
    effective_cases = test_cases
    if live:
        try:
            resolved_key = resolve_api_key(provider, api_key)
            inference_client = build_api_client(provider=provider, api_key=resolved_key, model=model_name)
            effective_cases = fetch_live_outputs(test_cases, inference_client)
        except (ValueError, ImportError) as exc:
            logger.error("Erreur inférence live pour '%s' : %s", model_name, exc)
            return {
                "model_name": model_name,
                "error": str(exc),
                "total_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "pass_rate": 0.0,
                "pass_rate_percent": 0.0,
                "avg_score": 0.0,
                "total_latency_ms": 0.0,
            }

    evaluators = build_evaluators(judge_client=judge_client)
    use_json = report_format in (REPORT_FORMAT_JSON, REPORT_FORMAT_BOTH)
    use_html = report_format in (REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH)

    json_gen = ReportGenerator(model_name=model_name, test_set=test_set) if use_json else None
    html_gen = HtmlReportGenerator(model_name=model_name, test_set=test_set) if use_html else None

    passed_count = 0
    total_score = 0.0
    total_latency = 0.0

    for index, case in enumerate(effective_cases, start=1):
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

    total_cases = len(effective_cases)
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
    # Exclure les modèles en erreur
    valid_stats = [s for s in model_stats if "error" not in s]
    if not valid_stats:
        return model_stats[0]["model_name"], "Aucun modèle valide — erreurs d'inférence"

    sorted_models = sorted(
        valid_stats,
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
    live: bool = False,
    provider: str = "openai",
    api_key: str | None = None,
    judge: bool = False,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_api_key: str | None = None,
) -> int:
    """
    Orchestre la comparaison multi-modèles.
    Retourne 0 si au moins un modèle atteint la cible de 99 %, 1 sinon.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Mode comparaison")
    logger.info("Modèles : %s", ", ".join(models))
    logger.info("Ensemble de tests : %s", test_set)
    logger.info("Mode live : %s", "activé" if live else "désactivé")
    logger.info("=" * 60)

    try:
        test_cases = load_test_cases(test_set)
    except FileNotFoundError as exc:
        logger.error("Impossible de charger les tests : %s", exc)
        return 1

    if not test_cases:
        logger.error("Aucun cas de test pour l'ensemble '%s'", test_set)
        return 1

    # Préparation du juge partagé entre tous les modèles (économise les tokens)
    judge_client: Any | None = None
    if judge:
        effective_judge_provider = judge_provider or provider
        effective_judge_api_key = judge_api_key or api_key
        default_judge_model = (
            LLM_JUDGE_DEFAULT_MODEL_OPENAI
            if effective_judge_provider == "openai"
            else LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC
        )
        effective_judge_model = judge_model or default_judge_model

        try:
            resolved_judge_key = resolve_api_key(effective_judge_provider, effective_judge_api_key)
            judge_client = build_api_client(
                provider=effective_judge_provider,
                api_key=resolved_judge_key,
                model=effective_judge_model,
            )
            logger.info("LLM-as-a-judge activé : %s/%s", effective_judge_provider, effective_judge_model)
        except (ValueError, ImportError) as exc:
            logger.error("Erreur configuration juge LLM : %s", exc)
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
            live=live,
            provider=provider,
            api_key=api_key,
            judge_client=judge_client,
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
        if "error" in stats:
            logger.error("  %s — ERREUR : %s", stats["model_name"], stats["error"])
        else:
            logger.info(
                "  %s — pass rate: %.1f%% | avg score: %.4f",
                stats["model_name"],
                stats["pass_rate_percent"],
                stats["avg_score"],
            )
    logger.info("Gagnant : %s (%s)", winner, winner_reason)
    logger.info("=" * 60)

    return 0 if any(s.get("pass_rate", 0) >= 0.99 for s in model_stats) else 1


def _parse_args() -> argparse.Namespace:
    """Configure et parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="compare-runner",
        description=(
            "LLM-Benchmarker-Suite — Comparaison multi-modèles\n"
            "Lance le même test set sur plusieurs modèles et compare les résultats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples — mode offline :
  python compare_runner.py --models gpt-4o claude-3-5-sonnet --test-set safety
  python compare_runner.py --models gpt-4o gpt-4o-mini --test-set all --format html

Exemples — mode live :
  python compare_runner.py --models gpt-4o gpt-4o-mini --live --provider openai --test-set reasoning
  python compare_runner.py --models claude-3-5-sonnet claude-haiku-4-5-20251001 --live --provider anthropic

Exemples — avec LLM-as-a-judge :
  python compare_runner.py --models gpt-4o gpt-4o-mini --live --judge --provider openai --test-set all

Providers supportés : {', '.join(SUPPORTED_PROVIDERS)}
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Liste des modèles à comparer (ex: gpt-4o claude-3-5-sonnet)",
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

    live_group = parser.add_argument_group("Inférence live")
    live_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Active le mode live : appelle l'API de chaque modèle",
    )
    live_group.add_argument(
        "--provider",
        type=str,
        choices=SUPPORTED_PROVIDERS,
        default="openai",
        help=f"Provider d'inférence : {', '.join(SUPPORTED_PROVIDERS)} (défaut: openai)",
    )
    live_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Clé API du provider (alternative : variable d'environnement OPENAI_API_KEY / ANTHROPIC_API_KEY)",
    )

    judge_group = parser.add_argument_group("LLM-as-a-judge")
    judge_group.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Active le LLM-as-a-judge comme évaluateur supplémentaire",
    )
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=f"Modèle juge (défaut: {LLM_JUDGE_DEFAULT_MODEL_OPENAI} / {LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC})",
    )
    judge_group.add_argument(
        "--judge-provider",
        type=str,
        choices=SUPPORTED_PROVIDERS,
        default=None,
        help="Provider du juge (défaut: même que --provider)",
    )
    judge_group.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Clé API spécifique pour le juge",
    )

    return parser.parse_args()


def _validate_output_dir(raw_path: str | None) -> str | None:
    """
    Vérifie que le répertoire de sortie ne sort pas du répertoire de travail courant.
    Lève ValueError si le chemin résolu tente un path traversal.
    """
    if raw_path is None:
        return None
    resolved = os.path.realpath(raw_path)
    cwd = os.path.realpath(os.getcwd())
    if not resolved.startswith(cwd):
        raise ValueError(
            f"Répertoire de sortie interdit : '{raw_path}' résout en '{resolved}', "
            f"hors du répertoire de travail '{cwd}'."
        )
    return resolved


if __name__ == "__main__":
    args = _parse_args()
    if len(args.models) < 2:
        logger.error("Au moins 2 modèles sont requis pour une comparaison.")
        sys.exit(1)
    try:
        safe_output_dir = _validate_output_dir(args.output_dir)
    except ValueError as validation_error:
        logger.error("Argument --output-dir invalide : %s", validation_error)
        sys.exit(1)
    exit_code = _run_comparison(
        models=args.models,
        test_set=args.test_set,
        output_dir=safe_output_dir,
        report_format=args.format,
        verbose=args.verbose,
        live=args.live,
        provider=args.provider,
        api_key=args.api_key,
        judge=args.judge,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_api_key=args.judge_api_key,
    )
    sys.exit(exit_code)

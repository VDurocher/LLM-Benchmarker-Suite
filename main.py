"""
LLM-Benchmarker-Suite — Point d'entrée CLI.

Orchestre le pipeline complet d'évaluation :
1. Chargement des cas de test depuis /data/
2. Inférence live optionnelle via l'API du modèle cible
3. Exécution des évaluateurs configurés par cas (dont LLM-as-a-judge optionnel)
4. Calcul du score composite pondéré
5. Génération du rapport JSON et/ou HTML dans /reports/

Usage — mode offline (model_output pré-rempli dans les JSON) :
    python main.py --model gpt-4o --test-set safety
    python main.py --model claude-3-5-sonnet --test-set all --format both --verbose

Usage — mode live (appels API réels) :
    python main.py --model gpt-4o --test-set safety --live --provider openai
    python main.py --model claude-3-5-sonnet --test-set reasoning --live --provider anthropic

Usage — avec LLM-as-a-judge :
    python main.py --model gpt-4o --test-set all --judge --judge-model gpt-4o-mini
    python main.py --model gpt-4o --test-set reasoning --live --judge --provider openai
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from config import (
    AVAILABLE_TEST_SETS,
    DEFAULT_WEIGHTS,
    HTML_REPORT_OUTPUT_DIR,
    LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC,
    LLM_JUDGE_DEFAULT_MODEL_OPENAI,
    REPORT_FORMAT_BOTH,
    REPORT_FORMAT_HTML,
    REPORT_FORMAT_JSON,
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
    live: bool = False,
    provider: str = "openai",
    api_key: str | None = None,
    judge: bool = False,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_api_key: str | None = None,
) -> int:
    """
    Pipeline principal d'évaluation.
    Retourne 0 si le pass rate cible est atteint, 1 sinon.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Démarrage")
    logger.info("Modèle cible : %s", model_name)
    logger.info("Ensemble de tests : %s", test_set)
    logger.info("Mode live : %s", "activé" if live else "désactivé")
    logger.info("LLM-as-a-judge : %s", "activé" if judge else "désactivé")
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

    # --- Mode live : inférence réelle via l'API ---
    if live:
        try:
            resolved_key = resolve_api_key(provider, api_key)
            inference_client = build_api_client(provider=provider, api_key=resolved_key, model=model_name)
            logger.info("Inférence live activée : %s/%s", provider, model_name)
            test_cases = fetch_live_outputs(test_cases, inference_client)
        except (ValueError, ImportError) as exc:
            logger.error("Erreur configuration inférence live : %s", exc)
            return 1

    # --- LLM-as-a-judge : configuration du juge externe ---
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
        except (ValueError, ImportError) as exc:
            logger.error("Erreur configuration juge LLM : %s", exc)
            return 1

    evaluators = build_evaluators(judge_client=judge_client)
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
        epilog=f"""
Exemples — mode offline :
  python main.py --model gpt-4o --test-set safety
  python main.py --model claude-3-5-sonnet --test-set all --verbose --format both

Exemples — mode live (inférence API réelle) :
  python main.py --model gpt-4o --test-set reasoning --live --provider openai
  python main.py --model claude-3-5-sonnet --test-set all --live --provider anthropic

Exemples — avec LLM-as-a-judge :
  python main.py --model gpt-4o --test-set all --live --judge --provider openai
  python main.py --model gpt-4o --test-set reasoning --live --judge --judge-model gpt-4o-mini

Variables d'environnement :
  OPENAI_API_KEY     Clé API OpenAI (alternative à --api-key avec --provider openai)
  ANTHROPIC_API_KEY  Clé API Anthropic (alternative à --api-key avec --provider anthropic)

Providers supportés : {', '.join(SUPPORTED_PROVIDERS)}
        """,
    )

    # --- Paramètres principaux ---
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Identifiant du modèle évalué (ex: gpt-4o, claude-3-5-sonnet)",
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

    # --- Mode live ---
    live_group = parser.add_argument_group("Inférence live (appels API réels)")
    live_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Active le mode live : appelle l'API du modèle au lieu d'utiliser les outputs pré-remplis",
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

    # --- LLM-as-a-judge ---
    judge_group = parser.add_argument_group("LLM-as-a-judge")
    judge_group.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Active le LLM-as-a-judge comme évaluateur supplémentaire (requiert --api-key ou variable env)",
    )
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=f"Modèle juge (défaut: {LLM_JUDGE_DEFAULT_MODEL_OPENAI} pour OpenAI, {LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC} pour Anthropic)",
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
        help="Clé API spécifique pour le juge (défaut: même que --api-key)",
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
    try:
        safe_output_dir = _validate_output_dir(args.output_dir)
    except ValueError as validation_error:
        logger.error("Argument --output-dir invalide : %s", validation_error)
        sys.exit(1)
    exit_code = _run_benchmark(
        model_name=args.model,
        test_set=args.test_set,
        output_dir=safe_output_dir,
        verbose=args.verbose,
        report_format=args.format,
        live=args.live,
        provider=args.provider,
        api_key=args.api_key,
        judge=args.judge,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_api_key=args.judge_api_key,
    )
    sys.exit(exit_code)

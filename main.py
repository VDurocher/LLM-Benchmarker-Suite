"""
LLM-Benchmarker-Suite — CLI entry point.

Orchestrates the complete evaluation pipeline:
1. Load test cases from /data/
2. Optional live inference via the target model's API
3. Run evaluators configured per case (including optional LLM-as-a-judge)
4. Compute weighted composite score
5. Generate JSON and/or HTML report in /reports/

Usage — offline mode (model_output pre-filled in JSON files):
    python main.py --model gpt-4o --test-set safety
    python main.py --model claude-3-5-sonnet --test-set all --format both --verbose

Usage — live mode (real API calls):
    python main.py --model gpt-4o --test-set safety --live --provider openai
    python main.py --model claude-3-5-sonnet --test-set reasoning --live --provider anthropic

Usage — with LLM-as-a-judge:
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
    Main evaluation pipeline.
    Returns 0 if the target pass rate is reached, 1 otherwise.
    """
    logger.info("=" * 60)
    logger.info("LLM-Benchmarker-Suite — Starting")
    logger.info("Target model: %s", model_name)
    logger.info("Test set: %s", test_set)
    logger.info("Live mode: %s", "enabled" if live else "disabled")
    logger.info("LLM-as-a-judge: %s", "enabled" if judge else "disabled")
    logger.info("Report format: %s", report_format)
    logger.info("=" * 60)

    try:
        test_cases = load_test_cases(test_set)
    except FileNotFoundError as exc:
        logger.error("Failed to load tests: %s", exc)
        return 1

    if not test_cases:
        logger.error("No test cases found for set '%s'", test_set)
        return 1

    # --- Live mode: real inference via API ---
    if live:
        try:
            resolved_key = resolve_api_key(provider, api_key)
            inference_client = build_api_client(provider=provider, api_key=resolved_key, model=model_name)
            logger.info("Live inference enabled: %s/%s", provider, model_name)
            test_cases = fetch_live_outputs(test_cases, inference_client)
        except (ValueError, ImportError) as exc:
            logger.error("Live inference configuration error: %s", exc)
            return 1

    # --- LLM-as-a-judge: external judge configuration ---
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
            logger.error("LLM judge configuration error: %s", exc)
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
        logger.info("[%d/%d] Evaluating case: %s", index, len(test_cases), case_id)

        evaluation_results, composite_score, case_passed = evaluate_case(
            case=case,
            evaluators=evaluators,
            weights=weights,
            verbose=verbose,
        )

        if case_passed:
            passed_count += 1
            logger.info("  -> PASS (composite score: %.4f)", composite_score)
        else:
            failed_count += 1
            logger.info("  -> FAIL (composite score: %.4f)", composite_score)

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
        logger.info("JSON report: %s", report_path)
    if html_generator is not None:
        html_path = html_generator.save(output_dir=output_dir or HTML_REPORT_OUTPUT_DIR)
        logger.info("HTML report: %s", html_path)

    total = passed_count + failed_count
    pass_rate = passed_count / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("Cases processed: %d | Passed: %d | Failed: %d", total, passed_count, failed_count)
    logger.info("Pass rate: %.1f%% (target: 99.0%%)", pass_rate * 100)
    logger.info(
        "Verdict: %s",
        "✓ PRODUCTION READY" if pass_rate >= 0.99 else "✗ BELOW TARGET — DO NOT DEPLOY",
    )
    logger.info("=" * 60)

    return 0 if pass_rate >= 0.99 else 1


def _parse_args() -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="llm-benchmarker",
        description=(
            "LLM-Benchmarker-Suite — Reliability evaluation of language models\n"
            "for production deployments with a 99%% pass rate target."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples — offline mode:
  python main.py --model gpt-4o --test-set safety
  python main.py --model claude-3-5-sonnet --test-set all --verbose --format both

Examples — live mode (real API inference):
  python main.py --model gpt-4o --test-set reasoning --live --provider openai
  python main.py --model claude-3-5-sonnet --test-set all --live --provider anthropic

Examples — with LLM-as-a-judge:
  python main.py --model gpt-4o --test-set all --live --judge --provider openai
  python main.py --model gpt-4o --test-set reasoning --live --judge --judge-model gpt-4o-mini

Environment variables:
  OPENAI_API_KEY     OpenAI API key (alternative to --api-key with --provider openai)
  ANTHROPIC_API_KEY  Anthropic API key (alternative to --api-key with --provider anthropic)

Supported providers: {', '.join(SUPPORTED_PROVIDERS)}
        """,
    )

    # --- Main parameters ---
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Identifier of the evaluated model (e.g. gpt-4o, claude-3-5-sonnet)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=AVAILABLE_TEST_SETS,
        default="all",
        help=f"Test set. Options: {', '.join(AVAILABLE_TEST_SETS)} (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: ./reports/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=[REPORT_FORMAT_JSON, REPORT_FORMAT_HTML, REPORT_FORMAT_BOTH],
        default=REPORT_FORMAT_JSON,
        help="Generated report format: json, html, or both (default: json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Display detailed per-evaluator scores for each case",
    )

    # --- Live mode ---
    live_group = parser.add_argument_group("Live inference (real API calls)")
    live_group.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live mode: calls the model API instead of using pre-filled outputs",
    )
    live_group.add_argument(
        "--provider",
        type=str,
        choices=SUPPORTED_PROVIDERS,
        default="openai",
        help=f"Inference provider: {', '.join(SUPPORTED_PROVIDERS)} (default: openai)",
    )
    live_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Provider API key (alternative: OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable)",
    )

    # --- LLM-as-a-judge ---
    judge_group = parser.add_argument_group("LLM-as-a-judge")
    judge_group.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-a-judge as an additional evaluator (requires --api-key or env variable)",
    )
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=f"Judge model (default: {LLM_JUDGE_DEFAULT_MODEL_OPENAI} for OpenAI, {LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC} for Anthropic)",
    )
    judge_group.add_argument(
        "--judge-provider",
        type=str,
        choices=SUPPORTED_PROVIDERS,
        default=None,
        help="Judge provider (default: same as --provider)",
    )
    judge_group.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Specific API key for the judge (default: same as --api-key)",
    )

    return parser.parse_args()


def _validate_output_dir(raw_path: str | None) -> str | None:
    """
    Verifies that the output directory does not escape the current working directory.
    Raises ValueError if the resolved path attempts a path traversal.
    """
    if raw_path is None:
        return None
    resolved = os.path.realpath(raw_path)
    cwd = os.path.realpath(os.getcwd())
    if not resolved.startswith(cwd):
        raise ValueError(
            f"Forbidden output directory: '{raw_path}' resolves to '{resolved}', "
            f"outside working directory '{cwd}'."
        )
    return resolved


if __name__ == "__main__":
    args = _parse_args()
    try:
        safe_output_dir = _validate_output_dir(args.output_dir)
    except ValueError as validation_error:
        logger.error("Invalid --output-dir argument: %s", validation_error)
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

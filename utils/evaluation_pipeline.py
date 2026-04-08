"""
Shared evaluation pipeline between main.py and compare_runner.py.

Provides functions for loading test cases, instantiating evaluators,
fetching live outputs, and running individual evaluations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_WEIGHTS,
    LLM_JUDGE_WEIGHT,
    SUPPORTED_PROVIDERS,
    EvaluatorWeights,
)
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

# Directory containing test case files
DATA_DIR = Path(__file__).parent.parent / "data"

# Test sets aggregated by "all"
_ALL_TEST_SETS = ["safety", "logic", "format", "consistency", "reasoning", "instruction_following"]


def load_test_cases(test_set: str) -> list[dict[str, Any]]:
    """Loads test cases from JSON files. 'all' aggregates all sets."""
    if test_set == "all":
        all_cases: list[dict[str, Any]] = []
        for available_set in _ALL_TEST_SETS:
            file_path = DATA_DIR / f"test_cases_{available_set}.json"
            if file_path.exists():
                all_cases.extend(_load_single_test_set(available_set))
        logger.info("Total test cases loaded: %d", len(all_cases))
        return all_cases
    return _load_single_test_set(test_set)


def _load_single_test_set(test_set: str) -> list[dict[str, Any]]:
    """Loads a test case file by set name."""
    file_path = DATA_DIR / f"test_cases_{test_set}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Test set not found: {file_path}")
    with open(file_path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    cases = data.get("cases", [])
    logger.info("Loading '%s': %d test cases", test_set, len(cases))
    return cases


def build_evaluators(judge_client: Any | None = None) -> dict[str, Any]:
    """
    Instantiates all available evaluators.

    Args:
        judge_client: LLMClient instance for the LLM judge (optional).
                      If None, the LLM judge is not included in the pipeline.

    Returns:
        Dict name -> evaluator instance.
    """
    from evaluators.llm_judge_evaluator import LLMJudgeEvaluator

    evaluators: dict[str, Any] = {
        "similarity": SimilarityEvaluator(),
        "hallucination": HallucinationEvaluator(),
        "format": FormatEvaluator(),
        "code": CodeEvaluator(),
        "consistency": ConsistencyEvaluator(),
    }

    if judge_client is not None:
        evaluators["llm_judge"] = LLMJudgeEvaluator(judge_client=judge_client)
        logger.info(
            "LLM-as-a-judge enabled — judge model: %s (%s)",
            judge_client.model,
            judge_client.provider,
        )

    return evaluators


def build_api_client(provider: str, api_key: str, model: str) -> Any:
    """
    Instantiates the appropriate live inference client based on the provider.

    Args:
        provider: "openai" or "anthropic".
        api_key: Provider API key.
        model: Target model identifier.

    Returns:
        LLMClient instance (OpenAIClient or AnthropicClient).

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Accepted values: {SUPPORTED_PROVIDERS}"
        )

    if provider == "openai":
        from api.openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key, model=model)

    from api.anthropic_client import AnthropicClient
    return AnthropicClient(api_key=api_key, model=model)


def fetch_live_outputs(
    test_cases: list[dict[str, Any]],
    client: Any,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Replaces empty `model_output` by calling the model's live API.

    In live mode, test cases do not need a pre-filled `model_output`.
    This function calls the model for each case and injects the response.

    Args:
        test_cases: List of test cases (dictionaries).
        client: Configured LLM inference client.
        force_refresh: If True, also replaces already-filled outputs.

    Returns:
        List of test cases enriched with model outputs.
    """
    enriched: list[dict[str, Any]] = []

    for index, case in enumerate(test_cases, start=1):
        case_id = case.get("id", f"case_{index:03d}")
        existing_output: str = case.get("model_output", "")

        if existing_output and not force_refresh:
            # Output already present — no API call needed
            enriched.append(case)
            continue

        prompt: str = case.get("prompt", "")
        logger.info("[%d/%d] Live inference — case: %s (model: %s)", index, len(test_cases), case_id, client.model)

        try:
            model_output = client.complete(prompt=prompt)
            updated_case = {**case, "model_output": model_output}
            enriched.append(updated_case)
            logger.info("  -> Response received (%d characters)", len(model_output))
        except RuntimeError as exc:
            logger.error("  -> Inference error for '%s': %s — case skipped", case_id, exc)
            # Keep the case with empty output to avoid blocking the pipeline
            enriched.append(case)

    return enriched


def resolve_api_key(provider: str, explicit_key: str | None) -> str:
    """
    Resolves the API key from the CLI argument or environment variables.

    Priority: explicit argument > environment variable.

    Args:
        provider: "openai" or "anthropic".
        explicit_key: Key explicitly provided via --api-key (can be None).

    Returns:
        The resolved API key.

    Raises:
        ValueError: If no key is available.
    """
    if explicit_key:
        return explicit_key

    env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    env_key = os.environ.get(env_var)
    if env_key:
        logger.info("API key resolved from environment variable %s", env_var)
        return env_key

    raise ValueError(
        f"No API key found for provider '{provider}'. "
        f"Provide --api-key or set the environment variable {env_var}."
    )


def evaluate_case(
    case: dict[str, Any],
    evaluators: dict[str, Any],
    weights: EvaluatorWeights | None = None,
    verbose: bool = False,
) -> tuple[list[EvaluationResult], float, bool]:
    """
    Evaluates a test case with the applicable evaluators.
    Returns (results, composite_score, passed).

    The composite score is normalized by the sum of weights actually used,
    which guarantees a consistent result regardless of the evaluator combination.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    prompt: str = case.get("prompt", "")
    expected_output: str = case.get("expected_output", "")
    model_output: str = case.get("model_output", "")
    metadata: dict[str, Any] = case.get("metadata", {})
    applicable: list[str] = metadata.get("evaluators", ["similarity", "hallucination", "format"])

    # Weight per evaluator — the LLM judge uses its own fixed weight
    weight_map = {
        "similarity": weights.similarity,
        "hallucination": weights.hallucination,
        "format": weights.format_compliance,
        "code": weights.format_compliance,
        "consistency": weights.format_compliance,
        "llm_judge": LLM_JUDGE_WEIGHT,
    }

    results: list[EvaluationResult] = []
    total_weight = 0.0
    weighted_score = 0.0

    for ev_name in applicable:
        if ev_name not in evaluators:
            logger.warning("Unknown or unconfigured evaluator: '%s' — skipped", ev_name)
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
                logger.warning("    Evaluator error: %s", result.error)

    composite_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
    case_passed = all(result.passed for result in results if result.error is None)
    return results, round(composite_score, 4), case_passed

"""
Internal consistency evaluator for LLM outputs.

Measures three dimensions via heuristics (without external LLM calls):
1. Absence of explicit contradictions in the response
2. Compliance with the format requested in the prompt
3. Appropriate length relative to the expected output
"""

from __future__ import annotations

import re
from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

# Acceptable length ratio: between 20% and 300% of expected
LENGTH_RATIO_MIN = 0.20
LENGTH_RATIO_MAX = 3.00

# Weights for the three consistency dimensions
WEIGHT_CONTRADICTION = 0.40
WEIGHT_LENGTH = 0.35
WEIGHT_SENTENCE_DENSITY = 0.25

# Minimum output length to evaluate sentence density
MIN_OUTPUT_LENGTH_FOR_DENSITY = 20

# Patterns indicating a potential contradiction (negative followed by affirmative)
_CONTRADICTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:je ne sais pas|n[\'']est pas|impossible|ne peut pas|je n[\'']ai pas)"
        r".{0,200}(?:en fait|mais en réalité|cependant|pourtant|néanmoins)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:i don[\'']t know|cannot|can[\'']t|impossible|not possible)"
        r".{0,200}(?:actually|however|in fact|but|nevertheless)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:this is (wrong|incorrect|false|not true))"
        r".{0,100}(?:this is (correct|true|right|valid))",
        re.IGNORECASE | re.DOTALL,
    ),
]

# Detectable formats in prompts and their output indicators
_FORMAT_SIGNALS: dict[str, list[str]] = {
    "json": ["{", "}", '":', "["],
    "list": ["-", "*", "1.", "•"],
    "code": ["def ", "function ", "class ", "```", "import "],
    "markdown": ["##", "**", "__", "```"],
}

# Keywords in the prompt suggesting an expected format
_FORMAT_PROMPT_KEYWORDS: dict[str, list[str]] = {
    "json": ["json", "object", "dict", "response body", "payload"],
    "list": ["list", "enumerate", "bullet", "steps", "items"],
    "code": ["function", "code", "implement", "write a", "def ", "class "],
    "markdown": ["markdown", "# ", "## ", "format with"],
}


def _score_contradictions(model_output: str) -> float:
    """
    Computes a score based on the absence of contradictions.
    Returns 1.0 if no contradiction detected, 0.0 if strong contradiction.
    """
    contradiction_count = sum(
        1 for pattern in _CONTRADICTION_PATTERNS if pattern.search(model_output)
    )
    if contradiction_count == 0:
        return 1.0
    # Each detected contradiction penalizes by 0.4, minimum 0.0
    return max(0.0, 1.0 - contradiction_count * 0.4)


def _score_length(expected_output: str, model_output: str) -> float:
    """
    Computes a relative length score.
    Maximum score if the output is between 20% and 300% of the expected length.
    Linear degradation outside this range.
    """
    expected_len = max(len(expected_output.strip()), 1)
    model_len = len(model_output.strip())
    ratio = model_len / expected_len

    if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
        return 1.0

    if ratio < LENGTH_RATIO_MIN:
        # Output too short — degradation from 0 to ratio/min
        return max(0.0, ratio / LENGTH_RATIO_MIN)

    # Output too long — progressive degradation beyond 300%
    excess = ratio - LENGTH_RATIO_MAX
    return max(0.0, 1.0 - (excess / LENGTH_RATIO_MAX) * 0.5)


def _score_sentence_density(model_output: str) -> float:
    """
    Evaluates sentence density: reasonable word-to-sentence ratio.
    A coherent output has between 5 and 40 words per sentence on average.
    """
    stripped = model_output.strip()
    if len(stripped) < MIN_OUTPUT_LENGTH_FOR_DENSITY:
        return 0.5

    sentences = re.split(r"[.!?]+", stripped)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.5

    words_per_sentence = [len(re.findall(r"\w+", s)) for s in sentences]
    avg_words = sum(words_per_sentence) / len(words_per_sentence)

    if 5 <= avg_words <= 40:
        return 1.0
    if avg_words < 5:
        # Sentences too short (fragments)
        return max(0.0, avg_words / 5)
    # Very long sentences (indigestible blocks)
    return max(0.0, 1.0 - (avg_words - 40) / 60)


def _detect_format_mismatch(prompt: str, model_output: str) -> bool:
    """
    Detects if the prompt requests a specific format but the output
    shows no signal of that format.
    Returns True if a format inconsistency is detected.
    """
    prompt_lower = prompt.lower()

    for format_name, keywords in _FORMAT_PROMPT_KEYWORDS.items():
        prompt_requests_format = any(kw in prompt_lower for kw in keywords)
        if not prompt_requests_format:
            continue

        output_signals = _FORMAT_SIGNALS.get(format_name, [])
        output_has_format = any(sig in model_output for sig in output_signals)

        if not output_has_format:
            return True

    return False


class ConsistencyEvaluator(BaseEvaluator):
    """
    Evaluates the internal consistency of LLM output.

    Dimensions analyzed:
    - Absence of explicit contradiction (negation + affirmation sentences)
    - Compliance with the format requested in the prompt
    - Appropriate length (neither too short nor excessive)

    Algorithm entirely based on heuristics — no external LLM calls.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        super().__init__(name="consistency", threshold=threshold)

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        # Compute the three sub-scores
        contradiction_score = _score_contradictions(model_output)
        length_score = _score_length(expected_output, model_output)
        density_score = _score_sentence_density(model_output)

        # Penalty for detected format inconsistency
        format_mismatch = _detect_format_mismatch(prompt, model_output)
        format_penalty = 0.15 if format_mismatch else 0.0

        # Weighted composite score
        raw_score = (
            contradiction_score * WEIGHT_CONTRADICTION
            + length_score * WEIGHT_LENGTH
            + density_score * WEIGHT_SENTENCE_DENSITY
        )
        final_score = max(0.0, round(raw_score - format_penalty, 4))
        passed = final_score >= self._threshold

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=final_score,
            details={
                "threshold": self._threshold,
                "contradiction_score": round(contradiction_score, 4),
                "length_score": round(length_score, 4),
                "sentence_density_score": round(density_score, 4),
                "format_mismatch_detected": format_mismatch,
                "format_penalty_applied": round(format_penalty, 4),
                "weights": {
                    "contradiction": WEIGHT_CONTRADICTION,
                    "length": WEIGHT_LENGTH,
                    "sentence_density": WEIGHT_SENTENCE_DENSITY,
                },
            },
        )

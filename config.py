"""
Global configuration for the LLM evaluation engine.
Centralizes all thresholds and constants used across evaluators.
"""

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Qualification thresholds — adjustable based on project requirements
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: Final[float] = 0.30
"""Minimum cosine score to consider a response correct.
Calibrated for semantic paraphrases — TF-IDF on a 2-document corpus
produces low scores even for correct reformulations. This threshold filters
clearly off-topic responses (score < 0.15) without penalizing good paraphrases."""

KEYWORD_MATCH_THRESHOLD: Final[float] = 0.35
"""Minimum ratio of expected keywords present in the response.
Calibrated to detect real hallucinations (missing facts) without penalizing
synonyms or equivalent reformulations."""

HALLUCINATION_PENALTY_WEIGHT: Final[float] = 0.85
"""Multiplicative factor applied to the final score if a hallucination is detected."""

PASS_RATE_TARGET: Final[float] = 0.99
"""Target reliability objective for production deployments (99%)."""

MAX_RESPONSE_TOKENS: Final[int] = 4096
"""Maximum token limit for format validation."""

# ---------------------------------------------------------------------------
# Report parameters
# ---------------------------------------------------------------------------

REPORT_VERSION: Final[str] = "1.0.0"
REPORT_OUTPUT_DIR: Final[str] = "reports"

# Supported report formats
REPORT_FORMAT_JSON: Final[str] = "json"
REPORT_FORMAT_HTML: Final[str] = "html"
REPORT_FORMAT_BOTH: Final[str] = "both"

# HTML report output directory (same as JSON directory)
HTML_REPORT_OUTPUT_DIR: Final[str] = REPORT_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Available test set names
# ---------------------------------------------------------------------------

AVAILABLE_TEST_SETS: Final[list[str]] = [
    "safety",
    "logic",
    "format",
    "consistency",
    "reasoning",
    "instruction_following",
    "all",
]

# ---------------------------------------------------------------------------
# LLM-as-a-judge — external judge configuration
# ---------------------------------------------------------------------------

LLM_JUDGE_THRESHOLD: Final[float] = 0.60
"""Minimum normalized score (0.0–1.0) to validate a case via the LLM judge.
Corresponds to a raw score of 6/10 on the judge's scale."""

LLM_JUDGE_WEIGHT: Final[float] = 0.30
"""LLM judge weight in the composite score when enabled.
The pipeline dynamically normalizes by the sum of weights used."""

LLM_JUDGE_DEFAULT_MODEL_OPENAI: Final[str] = "gpt-4o-mini"
"""Default OpenAI model for the judge (cost/performance optimized)."""

LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC: Final[str] = "claude-haiku-4-5-20251001"
"""Default Anthropic model for the judge."""

# ---------------------------------------------------------------------------
# Live inference parameters
# ---------------------------------------------------------------------------

LIVE_INFERENCE_TEMPERATURE: Final[float] = 0.0
"""Temperature for live inference — 0.0 for benchmark reproducibility."""

LIVE_INFERENCE_MAX_TOKENS: Final[int] = 2048
"""Token limit for responses in live mode."""

SUPPORTED_PROVIDERS: Final[list[str]] = ["openai", "anthropic"]
"""Supported inference providers."""


@dataclass(frozen=True)
class EvaluatorWeights:
    """
    Relative weight of each evaluation dimension.
    The sum must equal 1.0.
    """

    similarity: float = 0.40
    keyword_match: float = 0.30
    hallucination: float = 0.20
    format_compliance: float = 0.10

    def __post_init__(self) -> None:
        total = self.similarity + self.keyword_match + self.hallucination + self.format_compliance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"The sum of weights must equal 1.0, got: {total}"
            )


DEFAULT_WEIGHTS: Final[EvaluatorWeights] = EvaluatorWeights()

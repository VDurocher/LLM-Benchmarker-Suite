"""
Base contract for all LLM-Benchmarker suite evaluators.

Any new evaluation dimension must inherit from BaseEvaluator
and implement the `evaluate()` method.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """
    Standardized result returned by each evaluator.

    Attributes:
        evaluator_name: Human-readable evaluator identifier.
        passed: True if the test passes the configured thresholds.
        score: Normalized score between 0.0 and 1.0.
        details: Evaluator-specific metadata (intermediate scores, etc.).
        error: Error message if evaluation failed.
        latency_ms: Evaluation execution time in milliseconds.
    """

    evaluator_name: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Score must be between 0.0 and 1.0, got: {self.score}"
            )


class BaseEvaluator(ABC):
    """
    Abstract class defining the common interface for all evaluators.

    Pattern used: Template Method — subclasses implement `_run_evaluation`
    while `evaluate()` handles latency measurement and error handling.
    """

    def __init__(self, name: str, threshold: float = 0.5) -> None:
        self._name = name
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Public entry point — measures latency and delegates to _run_evaluation.
        Never raises exceptions: errors are encapsulated in EvaluationResult.
        """
        start = time.perf_counter()
        try:
            result = self._run_evaluation(
                prompt=prompt,
                expected_output=expected_output,
                model_output=model_output,
                metadata=metadata or {},
            )
        except Exception as exc:
            # Fault isolation — a failing evaluator does not block the pipeline
            result = EvaluationResult(
                evaluator_name=self._name,
                passed=False,
                score=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            latency = (time.perf_counter() - start) * 1000

        result.latency_ms = latency
        return result

    @abstractmethod
    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        """Evaluation logic specific to each subclass."""
        ...

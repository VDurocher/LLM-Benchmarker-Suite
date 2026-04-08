"""
JSON report generator for the LLM-Benchmarker Suite.

Produces a structured, versioned report including:
- Session metadata (model, test set, timestamp)
- Per-test-case results with detailed scores
- Aggregated statistics (pass rate, average score, latency)
- Verdicts per evaluation dimension
- Production compliance signal against the target (99%)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPORT_VERSION, REPORT_OUTPUT_DIR, PASS_RATE_TARGET
from evaluators.base_evaluator import EvaluationResult
from utils.logger import get_logger
from utils.stats import compute_evaluator_stats

logger = get_logger(__name__)


class ReportGenerator:
    """
    Aggregates evaluation results and generates a complete JSON report.

    Typical usage:
        generator = ReportGenerator(model_name="gpt-4o", test_set="safety")
        generator.add_case_result(case_id="tc_001", results=[...])
        report_path = generator.save()
    """

    def __init__(self, model_name: str, test_set: str) -> None:
        self._model_name = model_name
        self._test_set = test_set
        self._session_start = datetime.now(tz=timezone.utc)
        self._case_results: list[dict[str, Any]] = []

    def add_case_result(
        self,
        case_id: str,
        prompt: str,
        expected_output: str,
        model_output: str,
        evaluation_results: list[EvaluationResult],
        composite_score: float,
        passed: bool,
    ) -> None:
        """Records the results of a test case in the report."""
        self._case_results.append(
            {
                "case_id": case_id,
                "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                "model_output_preview": (
                    model_output[:200] + "..." if len(model_output) > 200 else model_output
                ),
                "composite_score": round(composite_score, 4),
                "passed": passed,
                "evaluators": [
                    {
                        "name": result.evaluator_name,
                        "passed": result.passed,
                        "score": result.score,
                        "latency_ms": round(result.latency_ms, 2),
                        "error": result.error,
                        "details": result.details,
                    }
                    for result in evaluation_results
                ],
            }
        )

    def build(self) -> dict[str, Any]:
        """Builds and returns the complete report dictionary."""
        session_end = datetime.now(tz=timezone.utc)
        total_cases = len(self._case_results)
        passed_cases = sum(1 for case in self._case_results if case["passed"])
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

        # Per-evaluator statistics (shared module with html_report.py)
        evaluator_stats = compute_evaluator_stats(self._case_results)

        # Average composite score
        avg_score = (
            sum(case["composite_score"] for case in self._case_results) / total_cases
            if total_cases > 0
            else 0.0
        )

        # Total evaluation latency
        total_latency_ms = sum(
            evaluator["latency_ms"]
            for case in self._case_results
            for evaluator in case["evaluators"]
        )

        production_ready = pass_rate >= PASS_RATE_TARGET

        return {
            "report_version": REPORT_VERSION,
            "generated_at": session_end.isoformat(),
            "session": {
                "model_name": self._model_name,
                "test_set": self._test_set,
                "started_at": self._session_start.isoformat(),
                "completed_at": session_end.isoformat(),
                "duration_seconds": round(
                    (session_end - self._session_start).total_seconds(), 2
                ),
            },
            "summary": {
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "failed_cases": total_cases - passed_cases,
                "pass_rate": round(pass_rate, 4),
                "pass_rate_percent": round(pass_rate * 100, 2),
                "average_composite_score": round(avg_score, 4),
                "total_evaluation_latency_ms": round(total_latency_ms, 2),
                "production_target": PASS_RATE_TARGET,
                "production_ready": production_ready,
                "production_ready_label": (
                    "✓ PRODUCTION READY" if production_ready else "✗ BELOW TARGET — DO NOT DEPLOY"
                ),
            },
            "evaluator_breakdown": evaluator_stats,
            "test_cases": self._case_results,
        }

    def save(self, output_dir: str | None = None) -> Path:
        """
        Persists the JSON report to the output directory.
        Returns the absolute path of the created file.
        """
        target_dir = Path(output_dir or REPORT_OUTPUT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
        # Sanitise model name to prevent path traversal in filename
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", self._model_name)
        filename = f"benchmark_{safe_model}_{self._test_set}_{timestamp}.json"
        output_path = target_dir / filename

        report = self.build()
        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(report, file_handle, indent=2, ensure_ascii=False)

        logger.info(
            "Report saved → %s (pass rate: %.1f%%)",
            output_path,
            report["summary"]["pass_rate_percent"],
        )
        return output_path.resolve()

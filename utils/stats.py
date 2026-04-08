"""
Evaluation statistics shared between report generators.

This module centralizes the `compute_evaluator_stats` function to avoid
duplication between `report_generator.py` and `html_report.py`.
"""

from __future__ import annotations

from typing import Any


def compute_evaluator_stats(
    case_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Computes aggregated statistics per evaluator name across all cases.

    Groups runs by evaluator and computes pass rate, average score,
    average latency, min/max score and error count.
    """
    # Group runs by evaluator name
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in case_results:
        for evaluator in case["evaluators"]:
            name = evaluator["name"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(evaluator)

    result: dict[str, dict[str, Any]] = {}
    for evaluator_name, runs in grouped.items():
        # Valid scores only (without error)
        scores = [run["score"] for run in runs if run.get("error") is None]
        passed_count = sum(1 for run in runs if run["passed"])
        latencies = [run["latency_ms"] for run in runs]

        result[evaluator_name] = {
            "total_runs": len(runs),
            "passed": passed_count,
            "failed": len(runs) - passed_count,
            "pass_rate": round(passed_count / len(runs), 4) if runs else 0.0,
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min_score": round(min(scores), 4) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "errors_count": sum(1 for run in runs if run.get("error") is not None),
        }

    return result

"""
Calcul des statistiques d'évaluation partagées entre les générateurs de rapports.

Ce module centralise la fonction `compute_evaluator_stats` pour éviter
la duplication entre `report_generator.py` et `html_report.py`.
"""

from __future__ import annotations

from typing import Any


def compute_evaluator_stats(
    case_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Calcule les statistiques agrégées par nom d'évaluateur sur tous les cas.

    Regroupe les runs par évaluateur et calcule pass rate, score moyen,
    latence moyenne, min/max score et nombre d'erreurs.
    """
    # Regroupement des runs par nom d'évaluateur
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in case_results:
        for evaluator in case["evaluators"]:
            name = evaluator["name"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(evaluator)

    result: dict[str, dict[str, Any]] = {}
    for evaluator_name, runs in grouped.items():
        # Scores valides uniquement (sans erreur)
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

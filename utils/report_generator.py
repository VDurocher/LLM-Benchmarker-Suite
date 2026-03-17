"""
Générateur de rapports JSON pour le suite LLM-Benchmarker.

Produit un rapport structuré et versionné incluant :
- Métadonnées de session (modèle, ensemble de tests, timestamp)
- Résultats par cas de test avec scores détaillés
- Statistiques agrégées (pass rate, score moyen, latence)
- Verdicts par dimension d'évaluation
- Signal de conformité par rapport à l'objectif de production (99%)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPORT_VERSION, REPORT_OUTPUT_DIR, PASS_RATE_TARGET
from evaluators.base_evaluator import EvaluationResult
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Agrège les résultats d'évaluation et génère un rapport JSON complet.

    Usage typique :
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
        """Enregistre les résultats d'un cas de test dans le rapport."""
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
        """Construit et retourne le dictionnaire de rapport complet."""
        session_end = datetime.now(tz=timezone.utc)
        total_cases = len(self._case_results)
        passed_cases = sum(1 for case in self._case_results if case["passed"])
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

        # Statistiques par évaluateur
        evaluator_stats = _compute_evaluator_stats(self._case_results)

        # Score moyen composite
        avg_score = (
            sum(case["composite_score"] for case in self._case_results) / total_cases
            if total_cases > 0
            else 0.0
        )

        # Latence totale d'évaluation
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
        Persiste le rapport JSON dans le répertoire de sortie.
        Retourne le chemin absolu du fichier créé.
        """
        target_dir = Path(output_dir or REPORT_OUTPUT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self._model_name}_{self._test_set}_{timestamp}.json"
        output_path = target_dir / filename

        report = self.build()
        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(report, file_handle, indent=2, ensure_ascii=False)

        logger.info(
            "Rapport sauvegardé → %s (pass rate: %.1f%%)",
            output_path,
            report["summary"]["pass_rate_percent"],
        )
        return output_path.resolve()


def _compute_evaluator_stats(
    case_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Calcule les statistiques agrégées par nom d'évaluateur sur tous les cas.
    """
    stats: dict[str, list[dict[str, Any]]] = {}

    for case in case_results:
        for evaluator in case["evaluators"]:
            name = evaluator["name"]
            if name not in stats:
                stats[name] = []
            stats[name].append(evaluator)

    result: dict[str, dict[str, Any]] = {}
    for evaluator_name, runs in stats.items():
        scores = [run["score"] for run in runs if run["error"] is None]
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
            "errors_count": sum(1 for run in runs if run["error"] is not None),
        }

    return result

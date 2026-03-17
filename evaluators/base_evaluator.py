"""
Contrat de base pour tous les évaluateurs du suite LLM-Benchmarker.

Toute nouvelle dimension d'évaluation doit hériter de BaseEvaluator
et implémenter la méthode `evaluate()`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """
    Résultat standardisé retourné par chaque évaluateur.

    Attributes:
        evaluator_name: Identifiant lisible de l'évaluateur.
        passed: True si le test passe les seuils configurés.
        score: Score normalisé entre 0.0 et 1.0.
        details: Métadonnées spécifiques à l'évaluateur (scores intermédiaires, etc.).
        error: Message d'erreur si l'évaluation a échoué.
        latency_ms: Temps d'exécution de l'évaluation en millisecondes.
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
                f"Le score doit être compris entre 0.0 et 1.0, obtenu : {self.score}"
            )


class BaseEvaluator(ABC):
    """
    Classe abstraite définissant l'interface commune de tous les évaluateurs.

    Pattern utilisé : Template Method — les sous-classes implémentent `_run_evaluation`
    tandis que `evaluate()` gère la mesure de latence et la gestion des erreurs.
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
        Point d'entrée public — mesure la latence et délègue à _run_evaluation.
        Ne lève jamais d'exception : les erreurs sont encapsulées dans EvaluationResult.
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
            # Isolation des pannes — un évaluateur défaillant ne bloque pas le pipeline
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
        """Logique d'évaluation propre à chaque sous-classe."""
        ...

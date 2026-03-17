"""
Évaluateur de détection d'hallucinations.

Stratégie hybride en deux passes :
1. Vérification des faits-clés (keyword anchoring) — les entités nommées et
   termes factuels de la réponse attendue doivent apparaître dans la réponse modèle.
2. Détection de clauses contradictoires — si le modèle affirme explicitement
   le contraire d'un fait attendu, le cas est signalé comme hallucination.

Référence méthodologique : inspiré de SelfCheckGPT et des approches NLI légères
utilisées dans les pipelines RLHF de vérification factuelle.
"""

from __future__ import annotations

import re
from typing import Any

from config import KEYWORD_MATCH_THRESHOLD, HALLUCINATION_PENALTY_WEIGHT
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

# Mots vides français et anglais — exclus du matching pour ne pas fausser le ratio
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of",
        "in", "on", "at", "by", "for", "with", "about", "as", "from",
        "that", "this", "it", "its", "and", "or", "but", "not", "no",
        "le", "la", "les", "un", "une", "des", "est", "sont", "et",
        "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    }
)

# Patterns de négation indiquant une contradiction potentielle
_NEGATION_PATTERNS: list[str] = [
    r"\bnot\s+(\w+)",
    r"\bnever\s+(\w+)",
    r"\bno\s+(\w+)",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bfalse\b",
    r"\bne\s+pas\b",
    r"\bjamais\b",
]


def _extract_keywords(text: str) -> list[str]:
    """
    Extrait les mots significatifs d'un texte en filtrant les stop words.
    Retourne des tokens en minuscules sans ponctuation.
    """
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [token for token in tokens if token not in _STOP_WORDS]


def _has_contradiction(expected_output: str, model_output: str) -> tuple[bool, list[str]]:
    """
    Détecte si le modèle contredit explicitement un fait de la réponse attendue.
    Retourne (contradiction_detected, liste des patterns trouvés).
    """
    detected_patterns: list[str] = []
    model_lower = model_output.lower()

    for pattern in _NEGATION_PATTERNS:
        matches = re.findall(pattern, model_lower)
        if matches:
            detected_patterns.extend(matches if isinstance(matches[0], str) else [pattern])

    return len(detected_patterns) > 0, detected_patterns


class HallucinationEvaluator(BaseEvaluator):
    """
    Détecte les hallucinations par ancrage de faits-clés et analyse de contradictions.

    Score retourné :
    - 1.0  → aucune hallucination détectée, tous les faits-clés présents
    - 0.5–0.99 → faits partiellement présents, aucune contradiction flagrante
    - 0.0–0.49 → hallucination probable (faits manquants + contradictions)
    """

    def __init__(self, threshold: float = KEYWORD_MATCH_THRESHOLD) -> None:
        super().__init__(name="hallucination_detector", threshold=threshold)

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        # Extraction des faits-clés depuis la réponse de référence
        expected_keywords = _extract_keywords(expected_output)
        model_keywords_set = set(_extract_keywords(model_output))

        if not expected_keywords:
            # Cas limite : réponse attendue vide ou sans mots significatifs
            return EvaluationResult(
                evaluator_name=self._name,
                passed=True,
                score=1.0,
                details={"warning": "Aucun mot-clé extrait de la réponse attendue."},
            )

        # Ratio de couverture des faits-clés
        matched_keywords = [kw for kw in expected_keywords if kw in model_keywords_set]
        coverage_ratio = len(matched_keywords) / len(expected_keywords)

        # Détection de contradictions explicites
        contradiction_found, contradiction_matches = _has_contradiction(
            expected_output, model_output
        )

        # Score composite avec pénalité pour contradiction
        raw_score = coverage_ratio
        if contradiction_found:
            raw_score *= HALLUCINATION_PENALTY_WEIGHT

        final_score = min(max(raw_score, 0.0), 1.0)
        passed = final_score >= self._threshold and not contradiction_found

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(final_score, 4),
            details={
                "threshold": self._threshold,
                "keyword_coverage_ratio": round(coverage_ratio, 4),
                "matched_keywords": matched_keywords[:20],  # limité pour lisibilité
                "missing_keywords": [
                    kw for kw in expected_keywords if kw not in model_keywords_set
                ][:20],
                "total_expected_keywords": len(expected_keywords),
                "contradiction_detected": contradiction_found,
                "contradiction_patterns": contradiction_matches[:10],
                "hallucination_penalty_applied": contradiction_found,
            },
        )

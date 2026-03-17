"""
Évaluateur de similarité sémantique par cosinus (TF-IDF).

Mesure à quel point la réponse du modèle est sémantiquement proche
de la sortie attendue, indépendamment de la formulation exacte.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SIMILARITY_THRESHOLD
from evaluators.base_evaluator import BaseEvaluator, EvaluationResult


def _normalize_text(text: str) -> str:
    """Normalisation légère : lowercase + suppression de la ponctuation excessive."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class SimilarityEvaluator(BaseEvaluator):
    """
    Compare la sortie du modèle à la réponse attendue via similarité cosinus TF-IDF.

    Avantages par rapport à une comparaison exacte :
    - Insensible aux reformulations synonymiques mineures.
    - Robuste aux variations d'ordre de mots.
    - Aucune dépendance à un modèle d'embeddings externe (rapide, déterministe).

    Limites connues :
    - Ne capture pas la sémantique profonde (utiliser sentence-transformers pour ça).
    - Sensible aux variations de domaine technique (vocabulaire très spécifique).
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        super().__init__(name="similarity_cosine", threshold=threshold)
        # Vectoriseur partagé — réinitialisé à chaque évaluation pour éviter
        # les fuites de vocabulaire entre les cas de test
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

    def _run_evaluation(
        self,
        prompt: str,
        expected_output: str,
        model_output: str,
        metadata: dict[str, Any],
    ) -> EvaluationResult:
        # Normalisation des textes avant vectorisation
        normalized_expected = _normalize_text(expected_output)
        normalized_model = _normalize_text(model_output)

        # Corpus minimal : référence + réponse modèle
        corpus = [normalized_expected, normalized_model]
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        # Score cosinus entre les deux vecteurs
        similarity_score = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )

        # Calcul des tokens communs pour le détail du rapport
        expected_tokens = set(normalized_expected.split())
        model_tokens = set(normalized_model.split())
        common_tokens = expected_tokens & model_tokens
        token_overlap_ratio = (
            len(common_tokens) / len(expected_tokens) if expected_tokens else 0.0
        )

        passed = similarity_score >= self._threshold

        return EvaluationResult(
            evaluator_name=self._name,
            passed=passed,
            score=round(similarity_score, 4),
            details={
                "threshold": self._threshold,
                "token_overlap_ratio": round(token_overlap_ratio, 4),
                "common_tokens_count": len(common_tokens),
                "expected_token_count": len(expected_tokens),
                "model_token_count": len(model_tokens),
                "normalized_expected_length": len(normalized_expected),
                "normalized_model_length": len(normalized_model),
            },
        )

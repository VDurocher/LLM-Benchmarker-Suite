"""
Évaluateur de cohérence interne des sorties LLM.

Mesure trois dimensions via heuristiques (sans appel LLM externe) :
1. Absence de contradictions explicites dans la réponse
2. Respect du format demandé dans le prompt
3. Longueur appropriée par rapport à la sortie attendue
"""

from __future__ import annotations

import re
from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvaluationResult

# Ratio de longueur acceptable : entre 20 % et 300 % de l'attendu
LENGTH_RATIO_MIN = 0.20
LENGTH_RATIO_MAX = 3.00

# Pondérations des trois dimensions de cohérence
WEIGHT_CONTRADICTION = 0.40
WEIGHT_LENGTH = 0.35
WEIGHT_SENTENCE_DENSITY = 0.25

# Longueur minimale d'une sortie pour évaluer la densité de phrases
MIN_OUTPUT_LENGTH_FOR_DENSITY = 20

# Patterns indiquant une contradiction potentielle (négatif suivi d'affirmatif)
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

# Formats détectables dans les prompts et leurs indicateurs dans la sortie
_FORMAT_SIGNALS: dict[str, list[str]] = {
    "json": ["{", "}", '":', "["],
    "list": ["-", "*", "1.", "•"],
    "code": ["def ", "function ", "class ", "```", "import "],
    "markdown": ["##", "**", "__", "```"],
}

# Mots-clés dans le prompt suggérant un format attendu
_FORMAT_PROMPT_KEYWORDS: dict[str, list[str]] = {
    "json": ["json", "object", "dict", "response body", "payload"],
    "list": ["list", "enumerate", "bullet", "steps", "items"],
    "code": ["function", "code", "implement", "write a", "def ", "class "],
    "markdown": ["markdown", "# ", "## ", "format with"],
}


def _score_contradictions(model_output: str) -> float:
    """
    Calcule un score basé sur l'absence de contradictions.
    Retourne 1.0 si aucune contradiction détectée, 0.0 si contradiction forte.
    """
    contradiction_count = sum(
        1 for pattern in _CONTRADICTION_PATTERNS if pattern.search(model_output)
    )
    if contradiction_count == 0:
        return 1.0
    # Chaque contradiction détectée pénalise de 0.4, minimum 0.0
    return max(0.0, 1.0 - contradiction_count * 0.4)


def _score_length(expected_output: str, model_output: str) -> float:
    """
    Calcule un score de longueur relative.
    Score maximal si la sortie est entre 20 % et 300 % de la longueur attendue.
    Dégradation linéaire en dehors de cette plage.
    """
    expected_len = max(len(expected_output.strip()), 1)
    model_len = len(model_output.strip())
    ratio = model_len / expected_len

    if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
        return 1.0

    if ratio < LENGTH_RATIO_MIN:
        # Sortie trop courte — dégradation de 0 à ratio/min
        return max(0.0, ratio / LENGTH_RATIO_MIN)

    # Sortie trop longue — dégradation progressive au-delà de 300 %
    excess = ratio - LENGTH_RATIO_MAX
    return max(0.0, 1.0 - (excess / LENGTH_RATIO_MAX) * 0.5)


def _score_sentence_density(model_output: str) -> float:
    """
    Évalue la densité des phrases : ratio mots/phrases raisonnable.
    Une sortie cohérente a entre 5 et 40 mots par phrase en moyenne.
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
        # Phrases trop courtes (fragments)
        return max(0.0, avg_words / 5)
    # Phrases très longues (blocs indigestes)
    return max(0.0, 1.0 - (avg_words - 40) / 60)


def _detect_format_mismatch(prompt: str, model_output: str) -> bool:
    """
    Détecte si le prompt demande un format spécifique mais que la sortie
    ne présente aucun signal de ce format.
    Retourne True si une incohérence de format est détectée.
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
    Évalue la cohérence interne de la sortie LLM.

    Dimensions analysées :
    - Absence de contradiction explicite (phrases négation + affirmation)
    - Respect du format demandé dans le prompt
    - Longueur appropriée (ni trop courte ni excessive)

    Algorithme entièrement basé sur des heuristiques — aucun appel LLM externe.
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
        # Calcul des trois sous-scores
        contradiction_score = _score_contradictions(model_output)
        length_score = _score_length(expected_output, model_output)
        density_score = _score_sentence_density(model_output)

        # Pénalité pour incohérence de format détectée
        format_mismatch = _detect_format_mismatch(prompt, model_output)
        format_penalty = 0.15 if format_mismatch else 0.0

        # Score composite pondéré
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

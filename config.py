"""
Configuration globale du moteur d'évaluation LLM.
Centralise tous les seuils et constantes utilisés dans les évaluateurs.
"""

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Seuils de qualification — ajustables selon les exigences du projet
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: Final[float] = 0.72
"""Score cosinus minimum pour considérer une réponse comme correcte."""

KEYWORD_MATCH_THRESHOLD: Final[float] = 0.60
"""Ratio minimum de mots-clés attendus présents dans la réponse."""

HALLUCINATION_PENALTY_WEIGHT: Final[float] = 0.85
"""Facteur multiplicatif appliqué au score final si une hallucination est détectée."""

PASS_RATE_TARGET: Final[float] = 0.99
"""Objectif de fiabilité cible pour les déploiements en production (99 %)."""

MAX_RESPONSE_TOKENS: Final[int] = 4096
"""Limite maximale de tokens pour la validation de format."""

# ---------------------------------------------------------------------------
# Paramètres des rapports
# ---------------------------------------------------------------------------

REPORT_VERSION: Final[str] = "1.0.0"
REPORT_OUTPUT_DIR: Final[str] = "reports"

# Formats de rapport supportés
REPORT_FORMAT_JSON: Final[str] = "json"
REPORT_FORMAT_HTML: Final[str] = "html"
REPORT_FORMAT_BOTH: Final[str] = "both"

# Répertoire de sortie des rapports HTML (identique au répertoire JSON)
HTML_REPORT_OUTPUT_DIR: Final[str] = REPORT_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Noms des ensembles de tests disponibles
# ---------------------------------------------------------------------------

AVAILABLE_TEST_SETS: Final[list[str]] = ["safety", "logic", "format", "consistency", "all"]


@dataclass(frozen=True)
class EvaluatorWeights:
    """
    Pondération relative de chaque dimension d'évaluation.
    La somme doit être égale à 1.0.
    """

    similarity: float = 0.40
    keyword_match: float = 0.30
    hallucination: float = 0.20
    format_compliance: float = 0.10

    def __post_init__(self) -> None:
        total = self.similarity + self.keyword_match + self.hallucination + self.format_compliance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"La somme des pondérations doit être égale à 1.0, obtenu : {total}"
            )


DEFAULT_WEIGHTS: Final[EvaluatorWeights] = EvaluatorWeights()

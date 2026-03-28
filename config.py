"""
Configuration globale du moteur d'évaluation LLM.
Centralise tous les seuils et constantes utilisés dans les évaluateurs.
"""

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Seuils de qualification — ajustables selon les exigences du projet
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: Final[float] = 0.30
"""Score cosinus minimum pour considérer une réponse comme correcte.
Calibré pour des paraphrases sémantiques — le TF-IDF sur un corpus de 2 documents
donne des scores bas même pour des reformulations correctes. Ce seuil filtre les
réponses clairement hors-sujet (score < 0.15) sans pénaliser les bonnes paraphrases."""

KEYWORD_MATCH_THRESHOLD: Final[float] = 0.35
"""Ratio minimum de mots-clés attendus présents dans la réponse.
Calibré pour détecter les hallucinations réelles (faits absents) sans pénaliser
les synonymes ou reformulations équivalentes."""

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

AVAILABLE_TEST_SETS: Final[list[str]] = [
    "safety",
    "logic",
    "format",
    "consistency",
    "reasoning",
    "instruction_following",
    "all",
]

# ---------------------------------------------------------------------------
# LLM-as-a-judge — configuration du juge externe
# ---------------------------------------------------------------------------

LLM_JUDGE_THRESHOLD: Final[float] = 0.60
"""Score normalisé minimum (0.0–1.0) pour valider un cas via le juge LLM.
Correspond à un score brut de 6/10 sur l'échelle du juge."""

LLM_JUDGE_WEIGHT: Final[float] = 0.30
"""Pondération du juge LLM dans le score composite quand il est activé.
Le pipeline normalise dynamiquement par la somme des poids utilisés."""

LLM_JUDGE_DEFAULT_MODEL_OPENAI: Final[str] = "gpt-4o-mini"
"""Modèle OpenAI par défaut pour le juge (optimisé coût/performance)."""

LLM_JUDGE_DEFAULT_MODEL_ANTHROPIC: Final[str] = "claude-haiku-4-5-20251001"
"""Modèle Anthropic par défaut pour le juge."""

# ---------------------------------------------------------------------------
# Paramètres d'inférence live
# ---------------------------------------------------------------------------

LIVE_INFERENCE_TEMPERATURE: Final[float] = 0.0
"""Temperature pour l'inférence live — 0.0 pour la reproductibilité des benchmarks."""

LIVE_INFERENCE_MAX_TOKENS: Final[int] = 2048
"""Limite de tokens pour les réponses en mode live."""

SUPPORTED_PROVIDERS: Final[list[str]] = ["openai", "anthropic"]
"""Providers d'inférence supportés."""


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

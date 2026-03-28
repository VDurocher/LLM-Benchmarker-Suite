"""
Contrat abstrait pour les clients d'inférence LLM.

Tous les providers (OpenAI, Anthropic) implémentent cette interface commune,
ce qui permet de les interchanger dans le pipeline d'évaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Interface commune pour l'inférence live et le LLM-as-a-judge.

    Attributes:
        _api_key: Clé d'API du provider.
        _model: Identifiant du modèle cible (ex: "gpt-4o-mini").
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        """Identifiant du modèle configuré."""
        return self._model

    @property
    def provider(self) -> str:
        """Nom du provider — à surcharger dans chaque sous-classe."""
        return "unknown"

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Envoie un prompt au modèle et retourne sa réponse en texte brut.

        Args:
            prompt: Le message utilisateur à envoyer.
            system_prompt: Instruction système optionnelle (prefixe de contexte).

        Returns:
            La réponse du modèle sous forme de string.

        Raises:
            RuntimeError: Si l'appel API échoue (quota, auth, réseau).
        """
        ...

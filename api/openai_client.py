"""
Client OpenAI pour l'inférence live et le LLM-as-a-judge.

Compatible avec gpt-4o, gpt-4o-mini, gpt-3.5-turbo, et tout modèle
exposé via l'API Chat Completions d'OpenAI.
"""

from __future__ import annotations

from typing import Any

from api.base_client import LLMClient


class OpenAIClient(LLMClient):
    """
    Client OpenAI basé sur le SDK officiel (openai>=1.30).

    Temperature fixée à 0.0 par défaut pour garantir la reproductibilité
    des benchmarks — les résultats peuvent être comparés d'une exécution à l'autre.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        self._temperature = temperature
        self._max_tokens = max_tokens
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
            self._client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "Le package 'openai' est requis pour l'inférence live OpenAI. "
                "Installez-le avec : pip install openai>=1.30"
            ) from exc

    @property
    def provider(self) -> str:
        return "openai"

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Appelle l'API Chat Completions d'OpenAI.

        Args:
            prompt: Message utilisateur.
            system_prompt: Message système optionnel.

        Returns:
            Contenu textuel de la réponse.

        Raises:
            RuntimeError: En cas d'erreur API (quota, auth, timeout).
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as exc:
            raise RuntimeError(f"Erreur OpenAI API ({self._model}) : {exc}") from exc

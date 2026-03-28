"""
Client Anthropic pour l'inférence live et le LLM-as-a-judge.

Compatible avec claude-3-5-sonnet, claude-haiku-4-5-20251001, claude-opus-4-6,
et tout modèle exposé via l'API Messages d'Anthropic.
"""

from __future__ import annotations

from typing import Any

from api.base_client import LLMClient


class AnthropicClient(LLMClient):
    """
    Client Anthropic basé sur le SDK officiel (anthropic>=0.18).

    Temperature fixée à 0.0 par défaut pour la reproductibilité des benchmarks.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        self._temperature = temperature
        self._max_tokens = max_tokens
        try:
            import anthropic  # type: ignore[import-untyped]
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "Le package 'anthropic' est requis pour l'inférence Anthropic. "
                "Installez-le avec : pip install anthropic>=0.18"
            ) from exc

    @property
    def provider(self) -> str:
        return "anthropic"

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Appelle l'API Messages d'Anthropic.

        Args:
            prompt: Message utilisateur.
            system_prompt: Message système optionnel.

        Returns:
            Contenu textuel du premier bloc de la réponse.

        Raises:
            RuntimeError: En cas d'erreur API (quota, auth, timeout).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = self._client.messages.create(**kwargs)
            first_block = response.content[0] if response.content else None
            if first_block is not None and hasattr(first_block, "text"):
                return str(first_block.text)
            return ""
        except Exception as exc:
            raise RuntimeError(f"Erreur Anthropic API ({self._model}) : {exc}") from exc

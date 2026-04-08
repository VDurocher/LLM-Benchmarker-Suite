"""
Abstract contract for LLM inference clients.

All providers (OpenAI, Anthropic) implement this common interface,
allowing them to be interchanged in the evaluation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Common interface for live inference and LLM-as-a-judge.

    Attributes:
        _api_key: Provider API key.
        _model: Target model identifier (e.g. "gpt-4o-mini").
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        """Configured model identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Provider name — to be overridden in each subclass."""
        return "unknown"

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Sends a prompt to the model and returns its response as plain text.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system instruction (context prefix).

        Returns:
            The model's response as a string.

        Raises:
            RuntimeError: If the API call fails (quota, auth, network).
        """
        ...

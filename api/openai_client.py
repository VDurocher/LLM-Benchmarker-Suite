"""
OpenAI client for live inference and LLM-as-a-judge.

Compatible with gpt-4o, gpt-4o-mini, gpt-3.5-turbo, and any model
exposed via OpenAI's Chat Completions API.
"""

from __future__ import annotations

from typing import Any

from api.base_client import LLMClient


class OpenAIClient(LLMClient):
    """
    OpenAI client based on the official SDK (openai>=1.30).

    Temperature fixed at 0.0 by default to guarantee benchmark reproducibility
    — results can be compared across executions.
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
                "The 'openai' package is required for live OpenAI inference. "
                "Install it with: pip install openai>=1.30"
            ) from exc

    @property
    def provider(self) -> str:
        return "openai"

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Calls OpenAI's Chat Completions API.

        Args:
            prompt: User message.
            system_prompt: Optional system message.

        Returns:
            Text content of the response.

        Raises:
            RuntimeError: On API error (quota, auth, timeout).
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
            raise RuntimeError(f"OpenAI API error ({self._model}): {exc}") from exc

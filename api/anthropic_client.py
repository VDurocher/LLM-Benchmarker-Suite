"""
Anthropic client for live inference and LLM-as-a-judge.

Compatible with claude-3-5-sonnet, claude-haiku-4-5-20251001, claude-opus-4-6,
and any model exposed via Anthropic's Messages API.
"""

from __future__ import annotations

from typing import Any

from api.base_client import LLMClient


class AnthropicClient(LLMClient):
    """
    Anthropic client based on the official SDK (anthropic>=0.18).

    Temperature fixed at 0.0 by default for benchmark reproducibility.
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
                "The 'anthropic' package is required for Anthropic inference. "
                "Install it with: pip install anthropic>=0.18"
            ) from exc

    @property
    def provider(self) -> str:
        return "anthropic"

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Calls Anthropic's Messages API.

        Args:
            prompt: User message.
            system_prompt: Optional system message.

        Returns:
            Text content of the first block in the response.

        Raises:
            RuntimeError: On API error (quota, auth, timeout).
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
            raise RuntimeError(f"Anthropic API error ({self._model}): {exc}") from exc

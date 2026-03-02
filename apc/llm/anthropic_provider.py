"""Anthropic LLM provider implementation."""

from __future__ import annotations

from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw=response,
        )

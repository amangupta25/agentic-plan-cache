"""OpenAI LLM provider implementation."""

from __future__ import annotations

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        import openai

        self._model = model
        self._client = openai.OpenAI(api_key=api_key)

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
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            raw=response,
        )

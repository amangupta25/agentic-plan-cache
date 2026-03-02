"""Extract a single keyword/phrase from a user query using a cheap LLM."""

from __future__ import annotations

import re

from apc.llm.base import LLMProvider
from apc.prompts import KEYWORD_EXTRACTION_PROMPT


class KeywordExtractor:
    """Extracts a single keyword phrase from a query using a cheap LLM, with regex fallback."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def extract(self, query: str) -> str:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
        response = self._llm.complete(prompt, temperature=0.0, max_tokens=64)
        keyword = self._parse(response.content)
        if not keyword:
            keyword = self._fallback(query)
        return keyword.lower().strip()

    def _parse(self, content: str) -> str:
        """Take the first non-empty line, strip quotes."""
        for line in content.strip().splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if line:
                return line
        return ""

    def _fallback(self, query: str) -> str:
        """Simple tokenizer fallback: extract meaningful words as a single phrase."""
        stop_words = {
            "what", "is", "the", "of", "a", "an", "to", "and", "or", "in",
            "for", "how", "do", "does", "can", "will", "be", "are", "was",
            "were", "it", "this", "that", "with", "from", "by", "on", "at",
            "if", "my", "me", "i", "you", "your", "we", "they", "them",
            "please", "find", "tell", "give", "show", "calculate", "compute",
        }
        words = re.findall(r"[a-zA-Z]+", query.lower())
        meaningful = [w for w in words if w not in stop_words and len(w) > 1]
        return " ".join(meaningful[:4])

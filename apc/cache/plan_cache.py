"""Plan cache with string-keyword keying, LRU eviction, and optional JSON persistence."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from apc.models import CacheEntry, PlanTemplate, TemplateStep


def _normalize_key(keyword: str) -> str:
    """Normalize a keyword string for use as a cache key."""
    return keyword.strip().lower()


class PlanCache:
    """In-memory LRU plan cache with optional JSON file persistence.

    Stores CacheEntry objects keyed by normalized keyword strings.
    Uses OrderedDict for LRU eviction when max_size is reached.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        max_size: int = 128,
    ) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def lookup(self, keyword: str) -> PlanTemplate | None:
        key = _normalize_key(keyword)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            entry.template.usage_count += 1
            return entry.template
        return None

    def store(self, keyword: str, template: PlanTemplate) -> None:
        key = _normalize_key(keyword)
        entry = CacheEntry(keyword=keyword, template=template)
        self._entries[key] = entry
        self._entries.move_to_end(key)
        # LRU eviction
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)
        if self._persist_path:
            self._save()

    @property
    def size(self) -> int:
        return len(self._entries)

    def _save(self) -> None:
        data: list[dict[str, Any]] = []
        for entry in self._entries.values():
            data.append({
                "keyword": entry.keyword,
                "template": {
                    "steps": [
                        {
                            "index": s.index,
                            "description": s.description,
                            "tool_name": s.tool_name,
                            "parameter_hints": s.parameter_hints,
                        }
                        for s in entry.template.steps
                    ],
                    "source_query": entry.template.source_query,
                    "category": entry.template.category,
                    "created_at": entry.template.created_at.isoformat(),
                    "usage_count": entry.template.usage_count,
                },
            })
        assert self._persist_path is not None
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        assert self._persist_path is not None
        try:
            data = json.loads(self._persist_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for item in data:
            keyword = item["keyword"]
            t = item["template"]
            template = PlanTemplate(
                steps=[
                    TemplateStep(
                        index=s["index"],
                        description=s["description"],
                        tool_name=s["tool_name"],
                        parameter_hints=s.get("parameter_hints", {}),
                    )
                    for s in t["steps"]
                ],
                source_query=t["source_query"],
                category=t.get("category", ""),
                usage_count=t.get("usage_count", 0),
            )
            key = _normalize_key(keyword)
            self._entries[key] = CacheEntry(keyword=keyword, template=template)

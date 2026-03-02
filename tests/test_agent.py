"""End-to-end tests for the APC agent with mock LLM providers."""

from __future__ import annotations

from apc.agent import APCAgent
from apc.cache.plan_cache import PlanCache
from apc.components.keyword_extractor import KeywordExtractor
from apc.models import Plan, PlanStep, PlanTemplate, TemplateStep
from tests.conftest import MockLLMProvider, MATH_KEYWORD


class TestKeywordExtraction:
    def test_extracts_keyword_from_llm(self, mock_cheap_llm):
        extractor = KeywordExtractor(mock_cheap_llm)
        keyword = extractor.extract("What is the square root of 144 plus 12?")
        assert isinstance(keyword, str)
        assert len(keyword) > 0
        assert "square root" in keyword

    def test_fallback_on_bad_response(self):
        llm = MockLLMProvider(responses={"higher-level": "   \n  "})
        extractor = KeywordExtractor(llm)
        keyword = extractor.extract("calculate square root of hundred")
        assert isinstance(keyword, str)
        assert len(keyword) > 0  # fallback tokenizer should produce something


class TestPlanCache:
    def test_store_and_lookup(self, plan_cache):
        keyword = "square root addition"
        template = PlanTemplate(
            steps=[
                TemplateStep(
                    index=1,
                    description="Calculate sqrt of <number>",
                    tool_name="calculator",
                    parameter_hints={"expression": "sqrt(<number>)"},
                ),
            ],
            source_query="test query",
            category="arithmetic",
        )
        plan_cache.store(keyword, template)
        assert plan_cache.size == 1

        result = plan_cache.lookup(keyword)
        assert result is not None
        assert result.category == "arithmetic"
        assert result.usage_count == 1

    def test_miss_returns_none(self, plan_cache):
        result = plan_cache.lookup("nonexistent")
        assert result is None

    def test_persistence(self, tmp_path):
        path = tmp_path / "persist.json"
        cache1 = PlanCache(persist_path=path)
        keyword = "test persist"
        template = PlanTemplate(
            steps=[TemplateStep(index=1, description="do thing", tool_name="calculator")],
            source_query="test",
            category="test",
        )
        cache1.store(keyword, template)

        # Load from disk
        cache2 = PlanCache(persist_path=path)
        assert cache2.size == 1
        result = cache2.lookup(keyword)
        assert result is not None
        assert result.category == "test"

    def test_lru_eviction(self, tmp_path):
        cache = PlanCache(persist_path=tmp_path / "lru.json", max_size=2)
        for i in range(3):
            template = PlanTemplate(
                steps=[TemplateStep(index=1, description=f"step {i}", tool_name="calculator")],
                source_query=f"query {i}",
                category=f"cat{i}",
            )
            cache.store(f"keyword{i}", template)

        # Should have evicted the first entry
        assert cache.size == 2
        assert cache.lookup("keyword0") is None
        assert cache.lookup("keyword1") is not None
        assert cache.lookup("keyword2") is not None

    def test_lru_access_refreshes(self, tmp_path):
        cache = PlanCache(persist_path=tmp_path / "lru2.json", max_size=2)
        for i in range(2):
            template = PlanTemplate(
                steps=[TemplateStep(index=1, description=f"step {i}", tool_name="calculator")],
                source_query=f"query {i}",
                category=f"cat{i}",
            )
            cache.store(f"keyword{i}", template)

        # Access keyword0 to refresh it
        cache.lookup("keyword0")

        # Now add keyword2 — should evict keyword1 (least recently used)
        template = PlanTemplate(
            steps=[TemplateStep(index=1, description="step 2", tool_name="calculator")],
            source_query="query 2",
            category="cat2",
        )
        cache.store("keyword2", template)

        assert cache.size == 2
        assert cache.lookup("keyword0") is not None  # was refreshed, still present
        assert cache.lookup("keyword1") is None       # was evicted
        assert cache.lookup("keyword2") is not None


class TestAPCAgentCacheMiss:
    def test_first_query_is_cache_miss(
        self, mock_expensive_llm, mock_cheap_llm, tool_registry, plan_cache
    ):
        agent = APCAgent(
            expensive_llm=mock_expensive_llm,
            cheap_llm=mock_cheap_llm,
            tool_registry=tool_registry,
            cache=plan_cache,
        )
        result = agent.run("What is the square root of 144 plus 12?")

        assert result.cache_hit is False
        assert result.success is True
        assert result.plan.source == "planner"
        assert result.llm_calls == 4  # keyword + plan + synthesis + template extraction
        assert agent.cache.size == 1  # template was cached
        assert isinstance(result.keyword, str)
        assert len(result.keyword) > 0


class TestAPCAgentCacheHit:
    def test_second_similar_query_is_cache_hit(
        self, mock_expensive_llm, mock_cheap_llm, tool_registry, plan_cache
    ):
        agent = APCAgent(
            expensive_llm=mock_expensive_llm,
            cheap_llm=mock_cheap_llm,
            tool_registry=tool_registry,
            cache=plan_cache,
        )

        # First query: cache miss
        result1 = agent.run("What is the square root of 144 plus 12?")
        assert result1.cache_hit is False

        # Second query with same keywords: cache hit
        result2 = agent.run("What is the square root of 256 plus 20?")
        assert result2.cache_hit is True
        assert result2.plan.source == "adapted"
        # keyword(1) + adapt_step1(1) + adapt_step2(1) + synthesis(1) = 4
        assert result2.llm_calls == 4

    def test_stats_tracking(
        self, mock_expensive_llm, mock_cheap_llm, tool_registry, plan_cache
    ):
        agent = APCAgent(
            expensive_llm=mock_expensive_llm,
            cheap_llm=mock_cheap_llm,
            tool_registry=tool_registry,
            cache=plan_cache,
        )

        agent.run("What is the square root of 144 plus 12?")
        agent.run("What is the square root of 256 plus 20?")

        stats = agent.stats
        assert stats.total_queries == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.5


class TestToolExecution:
    def test_calculator(self, tool_registry):
        result = tool_registry.execute("calculator", expression="sqrt(144)")
        assert result == "12.0"

    def test_calculator_addition(self, tool_registry):
        result = tool_registry.execute("calculator", expression="12.0 + 12")
        assert result == "24.0"

    def test_web_search(self, tool_registry):
        result = tool_registry.execute("web_search", query="speed of light")
        assert "299,792,458" in result

    def test_unknown_tool(self, tool_registry):
        result = tool_registry.execute("nonexistent", arg="value")
        assert "Error" in result

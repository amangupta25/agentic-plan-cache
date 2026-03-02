# Agentic Plan Caching (APC)

A Python implementation of the Agentic Plan Caching framework from [arXiv 2506.14852](https://arxiv.org/abs/2506.14852) — reduce LLM agent costs by caching and reusing plan templates.

## How It Works

When an LLM agent encounters a new task, it spends expensive tokens planning from scratch. APC observes that **similar queries follow similar plans** — so after the first expensive plan, it extracts a reusable template and caches it. Future similar queries reuse the template with only cheap LLM calls.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query arrives                            │
│                             │                                   │
│                    Extract keyword (cheap LLM)                  │
│                             │                                   │
│                      Cache lookup                               │
│                       ╱         ╲                               │
│                 MISS               HIT                          │
│                  │                  │                            │
│         Plan from scratch    Interleaved loop:                  │
│         (expensive LLM)      ┌─ adapt step 1 (cheap LLM) ──┐  │
│               │               │  execute step 1              │  │
│         Execute all steps     │  adapt step 2 (cheap LLM)   │  │
│               │               │  execute step 2              │  │
│         Synthesize answer     └─ ...                     ───┘  │
│               │                  │                              │
│         Extract template    Synthesize answer (cheap LLM)      │
│         (cheap LLM)              │                              │
│               │                  │                              │
│         Store in cache      Return result                      │
│               │                                                 │
│         Return result                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Cache miss**: 1 keyword + 1 plan (expensive) + 1 synthesis + 1 template extraction = **4 LLM calls**

**Cache hit**: 1 keyword + N adapt steps + 1 synthesis = **N+2 cheap LLM calls, 0 expensive**

## Key Features

- **Single-keyword cache keys** — queries are mapped to a short intent phrase (e.g. "square root addition") for robust matching
- **LRU eviction** — `OrderedDict`-based cache with configurable `max_size`
- **Interleaved adapt-execute loop** — on cache hits, each template step is adapted with prior step results as context before execution (paper's approach, not batch adaptation)
- **Two-stage template extraction** — rule-based filter strips reasoning/query-specific details before LLM generalization
- **Dual LLM architecture** — expensive model (e.g. GPT-4o) only for initial planning; cheap model (e.g. GPT-4o-mini) for everything else
- **Pluggable tools** — register custom tools by subclassing `apc.tools.base.Tool`
- **Provider-agnostic** — ships with OpenAI and Anthropic providers, easy to add more

## Installation

```bash
git clone https://github.com/amangupta25/agentic-plan-cache.git
cd agentic-plan-cache
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
# Edit .env with your API key(s):
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...  (optional)
source .env
```

## Quick Start

```python
from apc import APCAgent, PlanCache, ToolRegistry
from apc.llm.openai_provider import OpenAIProvider
from apc.tools.calculator import CalculatorTool

registry = ToolRegistry()
registry.register(CalculatorTool())

agent = APCAgent(
    expensive_llm=OpenAIProvider(model="gpt-4o"),
    cheap_llm=OpenAIProvider(model="gpt-4o-mini"),
    tool_registry=registry,
    cache=PlanCache(persist_path="cache.json", max_size=128),
)

# First query — cache miss, plans from scratch with expensive LLM
result = agent.run("What is the square root of 144 plus 12?")
print(result.final_answer)  # "24"
print(result.cache_hit)     # False
print(result.keyword)       # "square root addition"

# Similar query — cache hit, reuses template with cheap LLM only
result2 = agent.run("What is the square root of 256 plus 20?")
print(result2.final_answer) # "36"
print(result2.cache_hit)    # True

print(agent.stats.hit_rate) # 0.5
```

## Demos

```bash
# Basic demo — 4 similar math queries showing cache hits
.venv/bin/python demo/math_demo.py

# Benchmark — APC vs no-cache baseline on 6 queries
.venv/bin/python demo/benchmark.py
```

Sample benchmark output:

```
Expensive LLM calls (baseline) :   12
Expensive LLM calls (APC)      :    4
Expensive call reduction        :  67%
APC hit rate                    :  33%
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

14 tests covering keyword extraction, LRU cache eviction, cache miss/hit flows, tool execution, and stats tracking.

## Project Structure

```
apc/
├── models.py                    # Pydantic data models
├── prompts.py                   # LLM prompt templates
├── agent.py                     # APCAgent orchestrator
├── baseline.py                  # No-cache baseline for comparison
├── cache/
│   └── plan_cache.py            # LRU cache with JSON persistence
├── components/
│   ├── keyword_extractor.py     # Single-keyword extraction
│   ├── planner.py               # From-scratch planning (expensive LLM)
│   ├── template_extractor.py    # Two-stage template generalization
│   ├── template_adapter.py      # Step-by-step adaptation (cheap LLM)
│   └── actor.py                 # Tool execution + answer synthesis
├── llm/
│   ├── base.py                  # Abstract LLM provider interface
│   ├── openai_provider.py       # OpenAI implementation
│   └── anthropic_provider.py    # Anthropic implementation
└── tools/
    ├── base.py                  # Abstract Tool interface
    ├── registry.py              # Tool registry
    ├── calculator.py            # Math expression evaluator
    └── web_search.py            # Simulated web search
```

## Adding Custom Tools

```python
from apc.tools.base import Tool

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Does something useful"

    @property
    def input_schema(self) -> dict:
        return {"query": {"type": "string", "description": "The input"}}

    def execute(self, **kwargs) -> str:
        return f"Result for {kwargs['query']}"

registry.register(MyTool())
```

## Reference

```
@article{apc2025,
  title={Agentic Plan Caching},
  year={2025},
  eprint={2506.14852},
  archivePrefix={arXiv},
}
```

## License

MIT

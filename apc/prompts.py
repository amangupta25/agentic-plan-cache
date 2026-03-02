"""LLM prompt templates for the APC pipeline."""

KEYWORD_EXTRACTION_PROMPT = """\
Can you help me summarize what is the "task" or "keyword" describing \
the higher-level intent of this query?
Return ONLY a single short keyword or phrase (2-5 words), no explanation.

Examples:
Query: "What is the square root of 144 plus 12?"
Keyword: square root addition

Query: "Search for the speed of light and multiply it by 2"
Keyword: physics constant lookup

Query: "{query}"
Keyword:"""

PLAN_GENERATION_PROMPT = """\
You are a planning agent. Given a user query and available tools, generate a step-by-step plan.

Available tools:
{tool_schemas}

Return ONLY a JSON array of step objects. Each step has:
- "index": step number (starting at 1)
- "description": what this step does
- "tool_name": which tool to use
- "tool_args": dict of arguments to pass to the tool

Query: "{query}"
Plan:"""

TEMPLATE_EXTRACTION_PROMPT = """\
Clean up the element of each item in the workflow, so that we can \
reuse this trace as a reference template (independent from problem-\
specific variables like company name or fiscal year).

Replace specific values with descriptive parameter hints.

Executed plan for query: "{source_query}"
Steps:
{executed_steps}

Return ONLY a JSON object with:
- "category": a short category label for this type of task
- "steps": array of template steps, each with:
  - "index": step number
  - "description": generalized description (use placeholders like <value>, <expression>)
  - "tool_name": the tool to use
  - "parameter_hints": dict mapping parameter names to descriptions of what to fill in

Template:"""

STEP_ADAPTATION_PROMPT = """\
Adapt the following template step to answer a new query.
Fill in concrete values based on the query and any prior step results.

Template step to adapt:
{template_step}

New query: "{query}"

Available tools:
{tool_schemas}

Prior step results:
{prior_results}

Return ONLY a JSON object with:
- "index": step number
- "description": what this step does
- "tool_name": which tool to use
- "tool_args": dict of concrete arguments

Step:"""

ANSWER_SYNTHESIS_PROMPT = """\
Given a user query and the results of executing a plan, synthesize a clear final answer.

Query: "{query}"

Step results:
{step_results}

Provide a clear, concise answer to the user's query based on the step results.
Answer:"""

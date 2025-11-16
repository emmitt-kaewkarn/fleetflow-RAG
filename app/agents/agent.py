"""
FleetFlow Agent (POC)

This module builds a simple, easy-to-read LangChain agent for FleetFlow:
- Loads the SQL tool so the agent can query the database.
- Adds a small memory middleware for short-term context.
- Supports both streaming updates and simple invocation.

You can customize the base system prompt via environment or SYSTEM_PROMPT.txt.
The agent’s prompt also includes a compact schema overview parsed from FF_SCHEMA.txt
to keep SQL queries grounded.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
from app.agents.tools.sql_query import load_ff_schema, render_schema_hint


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are FleetFlow, a helpful agent for fleet operations.")

# Clear, POC-friendly schema guardrails appended to the system prompt
SCHEMA_RULES = (
    "\n"
    "Schema usage rules (strict):\n"
    "- Use ONLY tables and columns shown in the schema overview; do NOT invent fields.\n"
    "- If the user uses informal names (e.g., 'revenue', 'phone'), map them to schema fields or ask a brief clarifying question.\n"
    "- For jobs revenue, prefer 'service_fee' for base fee or 'grand_total' if taxes/discounts are included.\n"
    "- Prefer qualified columns with table aliases when joining multiple tables.\n"
    "- If unsure, call the 'get_database_schema' tool before forming the SQL query.\n"
    "- On any query validation/execution error, inspect schema: use 'describe_table(<name>)' for the specific table(s), or 'get_database_schema' for an overview; then retry or ask a clarifying question.\n"
)

# Additional rules for job creation flow and defaults (POC-specific)
JOB_CREATION_RULES = (
    "\n"
    "Job creation rules (POC):\n"
    "- Defaults: is_discounted=false, is_taxable=false, is_reserved=false unless the user specifies otherwise.\n"
    "- Before calling the 'create_job' tool, gather and validate required fields: service_fee and service_date.\n"
    "- Do NOT query job_types. Use job_type_id from the environment variable JOB_TYPE_ID by default. Only override if the user explicitly provides a job_type_id.\n"
    "- For customer_id: ask the customer name, search customers table by company_id; if not found, set customer_id=null.\n"
    "- For vehicle_id: if the user does not provide one, select one available vehicle for that company via SQL.\n"
    "- Destinations must include exact latitude/longitude for each location. Do NOT use external geocoding.\n"
    "- For each destination, ask for contact_name and contact_phone.\n"
    "- Use location_order letters A, B, C in sequence when constructing multi-stop jobs.\n"
    "- Always show a compact confirmation summary and obtain user approval before submitting 'create_job'.\n"
)


def _load_system_prompt_from_file() -> str | None:
    """
    Try to load a canonical system prompt from the project root file:
    - <project_root>/SYSTEM_PROMPT.txt

    Returns
    - str with prompt content if the file exists and is non-empty
    - None if the file is missing or unreadable
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        prompt_file = project_root / "SYSTEM_PROMPT.txt"
        if prompt_file.exists():
            text = prompt_file.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return None


def _load_tools() -> List[Any]:
    """Load the core tools (SQL + optional job tools) for the agent (POC-simple).

    Tries in order:
    - SQL tools: sql_query_tool, get_database_schema, describe_table, run_sql, run_describe_table
    - Job tools: create_job (if available)
    """
    tools: List[Any] = []
    # 1) Preferred tool
    try:
        from app.agents.tools.sql_query import sql_query_tool  # type: ignore
        tools.append(sql_query_tool)
        logger.info("Loaded SQL tool: sql_query_tool")
    except Exception:
        logger.info("sql_query_tool not available")

    # 2) Schema inspection helper
    try:
        from app.agents.tools.sql_query import get_database_schema  # type: ignore
        tools.append(get_database_schema)
        logger.info("Loaded schema inspection tool: get_database_schema")
    except Exception:
        logger.info("get_database_schema not available")

    # 2a) Per-table describe helper
    try:
        from app.agents.tools.sql_query import describe_table  # type: ignore
        tools.append(describe_table)
        logger.info("Loaded table describe tool: describe_table")
    except Exception:
        logger.info("describe_table not available")

    # 3) Fallback run_sql
    try:
        from app.agents.tools.sql_query import run_sql  # type: ignore
        try:
            from langchain.tools import tool as lc_tool_decorator  # type: ignore
            tools.append(
                lc_tool_decorator(
                    "run_sql",
                    description=(
                        "Execute a read-only SQL query against the FleetFlow MariaDB. "
                        "Use parameterized inputs; do not attempt writes."
                    ),
                )(run_sql)
            )
            logger.info("Adapted run_sql into LangChain tool")
        except Exception:
            tools.append(run_sql)
            logger.info("Loaded run_sql callable without decorator")
    except Exception:
        logger.info("run_sql not available")

    # 3a) Optional: live DESCRIBE helper (callable)
    try:
        from app.agents.tools.sql_query import run_describe_table  # type: ignore
        tools.append(run_describe_table)
        logger.info("Loaded live DESCRIBE helper: run_describe_table")
    except Exception:
        logger.info("run_describe_table not available")

    # 4) Optional: Job creation tool
    try:
        from app.agents.tools.job_create import create_job  # type: ignore
        tools.append(create_job)
        logger.info("Loaded Job tool: create_job")
    except Exception:
        logger.info("Job creation tool not available")

    return tools

def create_main_agent(
    *,
    tools: Optional[List[Any]] = None,
    model: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    summarizer_model: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    middleware_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a simple FleetFlow agent (POC)."""
    # Import LangChain/LangGraph lazily so editors don’t flag missing packages at import time.
    try:
        from langchain.agents import create_agent as lc_create_agent  # type: ignore
        from langchain.agents.middleware import SummarizationMiddleware  # type: ignore
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"LangChain/LangGraph dependencies not available. Please install them before creating the agent. Details: {e}"
        )

    # 1) Tools: load SQL tool unless explicit tools were provided
    if tools is None:
        tools = _load_tools()
        if not tools:
            logger.warning("No tools loaded. The agent will run model-only without DB capabilities.")

    # 2) Models and prompts: resolve model ids and prompt
    model = model or DEFAULT_MODEL
    summarizer_model = summarizer_model or DEFAULT_SUMMARY_MODEL
    # Precedence: explicit param > file content > env default
    base_prompt = system_prompt or _load_system_prompt_from_file() or DEFAULT_SYSTEM_PROMPT

    # 2a) Schema hint: load ONLY from FF_SCHEMA.txt to avoid drift.
    # If live details are needed, the agent can call get_database_schema.
    schema = load_ff_schema(max_columns_per_table=24)
    schema_hint = render_schema_hint(schema)
    # Final system prompt = base prompt + schema hint + guardrails
    system_prompt = base_prompt + (schema_hint if schema_hint else "") + SCHEMA_RULES + JOB_CREATION_RULES

    # 3) Memory: summarization middleware keeps short-term context
    middleware_kwargs = middleware_kwargs or {}
    summary_mw = SummarizationMiddleware(
        model=summarizer_model,
        **middleware_kwargs,
    )

    # 4) Checkpointer: enables thread-based memory persistence
    checkpointer = checkpointer or InMemorySaver()

    # 5) Build the agent graph
    agent = lc_create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[summary_mw],
        checkpointer=checkpointer,
    )

    logger.info(
        "FleetFlow agent created. Model=%s, SummaryModel=%s, Tools=%s",
        model,
        summarizer_model,
        [getattr(t, "name", str(t)) for t in tools],
    )
    return agent


def stream_agent_updates(
    agent: Any,
    user_text: str,
    *,
    stream_mode: str = "updates",
    context: Optional[Any] = None,
    thread_id: str = "demo",
) -> Iterable[Any]:
    """
    Stream agent output in two common modes:
    - "updates": emits internal agent steps (model calls, tool calls).
    - "messages": emits token/message chunks when supported by the model/provider.

    Important
    - We pass a `thread_id` via LangGraph’s checkpointer config so the agent can
      maintain memory per conversation.

    Args
    - agent: The agent returned by create_main_agent.
    - user_text: Prompt content for the user message.
    - stream_mode: "updates", "messages", or "custom".
    - context: Optional runtime context for tools/middleware.
    - thread_id: Identifier for the user/session conversation.

    Yields
    - Iterable[Any]: streaming chunks as provided by LangChain.
    """
    payload = {"messages": [{"role": "user", "content": user_text}]}
    config = {"configurable": {"thread_id": thread_id}}
    if context is not None:
        # If context is provided and agent supports runtime context, pass via `context=...`
        return agent.stream(payload, config, context=context, stream_mode=stream_mode)
    return agent.stream(payload, config, stream_mode=stream_mode)


def invoke_agent(
    agent: Any,
    user_text: str,
    *,
    context: Optional[Any] = None,
    thread_id: str = "demo",
) -> Dict[str, Any]:
    """
    Invoke the agent once (non-streaming).

    We send a single user message and return the final state dict.
    Passing `thread_id` ensures the call is associated with the correct conversation.
    """
    payload = {"messages": [{"role": "user", "content": user_text}]}
    config = {"configurable": {"thread_id": thread_id}}
    if context is not None:
        return agent.invoke(payload, config, context=context)
    return agent.invoke(payload, config)


if __name__ == "__main__":  # pragma: no cover
    # Simple smoke test for imports, agent creation, streaming, and invoke.
    try:
        agent = create_main_agent()
        logger.info("Agent created successfully.")
        # Stream a quick demo
        for chunk in stream_agent_updates(agent, "Say hello and tell me what tools you have."):
            # Chunk is a dict of step->data depending on stream_mode
            logger.info("Stream chunk: %s", chunk)
        # Final invoke
        result = invoke_agent(agent, "Thanks! Now run `SELECT 1 AS ok` if you can.")
        last_msg = result["messages"][-1] if result.get("messages") else None
        logger.info("Final message: %s", getattr(last_msg, "content", last_msg))
    except Exception as e:
        logger.error("Agent demo failed: %s", e)
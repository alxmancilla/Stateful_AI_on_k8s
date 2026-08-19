"""LangChain tools the agent can call during a turn.

``build_tools`` binds a set of tools to a specific ``MongoMemory`` and
``session_id`` so the LLM can decide, mid-turn, to search long-term memory,
persist a durable fact, or read the wall clock. Each tool returns a plain
string (what the model sees as the tool observation); a companion ``sinks``
dict captures structured side effects (e.g. retrieved documents) so the HTTP
layer can surface them to the UI without re-parsing the model's text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from langchain_core.tools import StructuredTool


def build_tools(memory: Any, session_id: str) -> Tuple[List[StructuredTool], Dict[str, Any]]:
    """Return ``(tools, sinks)`` bound to ``memory``/``session_id``.

    ``sinks`` accumulates structured results the tools produce this turn:
      - ``sinks["retrieved"]``: list of memory docs returned by search_memory
      - ``sinks["saved_facts"]``: list of fact strings persisted via save_fact
    """
    sinks: Dict[str, Any] = {"retrieved": [], "saved_facts": []}

    def search_memory(query: str) -> str:
        """Search this conversation's long-term memory for relevant past turns.

        Use when the user refers to something they told you earlier or asks you
        to recall a fact. ``query`` is a short natural-language description of
        what to look for.
        """
        candidates = memory.hybrid_search(session_id, query, top_k=10)
        reranked = memory.rerank(query, candidates, top_n=3)
        sinks["retrieved"].extend(reranked)
        if not reranked:
            return "No relevant memory found."
        return "\n".join(
            f"- [{d.get('role')}] {d.get('content')}" for d in reranked
        )

    def save_fact(fact: str) -> str:
        """Persist a durable fact the user wants remembered for later sessions.

        Use only for stable, user-asserted facts (preferences, names, decisions),
        not for small talk. ``fact`` is a single concise sentence.
        """
        memory.save_message(session_id, "system", f"FACT: {fact}")
        sinks["saved_facts"].append(fact)
        return f"Saved: {fact}"

    def get_current_time() -> str:
        """Return the current UTC date and time in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    tools = [
        StructuredTool.from_function(search_memory),
        StructuredTool.from_function(save_fact),
        StructuredTool.from_function(get_current_time),
    ]
    return tools, sinks

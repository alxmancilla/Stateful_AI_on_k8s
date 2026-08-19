"""Stateful agent entrypoint.

Runs as a FastAPI service (``serve``, the container default) or an interactive
CLI (``chat``). Both share the same ``Agent`` core: retrieve recent turns,
run hybrid search + rerank for relevant long-term context, call an LLM through
the Azure Grove API, and persist both sides of the exchange to MongoDB.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

import metrics
from memory import MongoMemory, get_client
from embeddings import VoyageEmbeddings
from search import VoyageReranker
from tools import build_tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("agent.main")

SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent long-term memory stored in "
    "MongoDB. Use the retrieved context and recent conversation to answer "
    "accurately and consistently. If the context is irrelevant, ignore it. "
    "You may call tools: search_memory to look up things the user told you "
    "earlier, save_fact to remember a durable fact for later, and "
    "get_current_time for the current time. Prefer answering directly when the "
    "provided context already suffices."
)


# Request models are defined at module scope (not inside build_app) so that,
# with ``from __future__ import annotations`` in effect, FastAPI can resolve the
# string type hints against module globals and treat them as request bodies.
# Defining them locally makes get_type_hints fail to resolve the annotation,
# and FastAPI then mistakes the parameter for a query field.
class ChatRequest(BaseModel):
    session_id: str
    message: str


class SearchRequest(BaseModel):
    session_id: str
    query: str
    mode: str = "hybrid"
    top_k: int = 10


@dataclass
class Settings:
    """Runtime configuration, populated from environment variables."""

    mongodb_uri: str
    mongodb_ca_file: str
    db_name: str
    collection_name: str
    vector_index: str
    text_index: str
    llm_provider: str
    llm_model_name: str
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    port: int
    recent_turns: int
    hybrid_top_k: int
    rerank_top_n: int
    memory_ttl_seconds: int
    max_turns_per_session: int
    max_tool_iters: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mongodb_uri=os.environ["MONGODB_URI"],
            mongodb_ca_file=os.environ.get("MONGODB_TLS_CA_FILE", ""),
            db_name=os.environ.get("DB_NAME", "agent_memory"),
            collection_name=os.environ.get("COLLECTION_NAME", "messages"),
            vector_index=os.environ.get("VECTOR_INDEX", "vector_index"),
            text_index=os.environ.get("TEXT_INDEX", "text_index"),
            llm_provider=os.environ.get("LLM_PROVIDER", "openai").lower(),
            llm_model_name=os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini"),
            azure_endpoint=os.environ.get("AZURE_GROVE_ENDPOINT", ""),
            azure_api_key=os.environ.get("AZURE_GROVE_API_KEY", ""),
            azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            port=int(os.environ.get("PORT", "8080")),
            recent_turns=int(os.environ.get("RECENT_TURNS", "5")),
            hybrid_top_k=int(os.environ.get("HYBRID_TOP_K", "10")),
            rerank_top_n=int(os.environ.get("RERANK_TOP_N", "3")),
            memory_ttl_seconds=int(os.environ.get("MEMORY_TTL_SECONDS", "0")),
            max_turns_per_session=int(os.environ.get("MAX_TURNS_PER_SESSION", "0")),
            max_tool_iters=int(os.environ.get("MAX_TOOL_ITERS", "4")),
        )


def build_llm(settings: Settings):
    """Construct a LangChain chat model routed through the Azure Grove API."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model_name,
            api_key=settings.azure_api_key,
            base_url=settings.azure_endpoint,
            temperature=0.2,
        )

    from langchain_openai import ChatOpenAI

    # The Azure Grove endpoint is an OpenAI-compatible gateway whose URL already
    # ends in ``/openai/v1``. ChatOpenAI POSTs to ``{base_url}/chat/completions``,
    # which the gateway serves directly. AzureChatOpenAI would instead build an
    # Azure-style ``/openai/deployments/{model}/chat/completions`` path and 404.
    #
    # The gateway authenticates via the ``api-key`` header rather than the
    # OpenAI-style ``Authorization: Bearer`` header, so the key is passed as a
    # default header. The ``api_key`` argument is still required by the client.
    # ``temperature`` must be 1: some newer models reject any non-default value,
    # and the client would otherwise send its own default (0.7).
    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=settings.azure_api_key,
        base_url=settings.azure_endpoint,
        default_headers={"api-key": settings.azure_api_key},
        temperature=1,
    )


def render_prompt(
    context: List[Dict[str, Any]],
    recent: List[Dict[str, Any]],
    message: str,
) -> List[Any]:
    """Assemble the LangChain message list for a single turn."""
    from langchain_core.messages import HumanMessage, SystemMessage

    lines = [SYSTEM_PROMPT]
    if context:
        lines.append("\nRelevant long-term memory (reranked):")
        for item in context:
            lines.append(f"- [{item.get('role')}] {item.get('content')}")
    if recent:
        lines.append("\nRecent conversation:")
        for item in recent:
            lines.append(f"- [{item.get('role')}] {item.get('content')}")
    return [SystemMessage(content="\n".join(lines)), HumanMessage(content=message)]


class Agent:
    """Retrieval-augmented, MongoDB-backed conversational agent with tools."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        client = get_client(settings.mongodb_uri, settings.mongodb_ca_file)
        collection = client[settings.db_name][settings.collection_name]
        self.memory = MongoMemory(
            collection=collection,
            embeddings=VoyageEmbeddings(),
            reranker=VoyageReranker(),
            vector_index=settings.vector_index,
            text_index=settings.text_index,
            ttl_seconds=settings.memory_ttl_seconds,
            max_turns_per_session=settings.max_turns_per_session,
        )
        self.memory.ensure_indexes()
        self.llm = build_llm(settings)

    def _prepare(self, session_id: str, message: str):
        """Retrieve context and build the initial message list + tool binding.

        Returns ``(messages, context, message_vector, tools, sinks, llm)`` where
        ``llm`` is the tool-bound model. Retrieval is best-effort: on failure the
        turn proceeds with recent turns only.
        """
        recent = self.memory.get_recent(session_id, n=self.settings.recent_turns)
        context: List[Dict[str, Any]] = []
        message_vector: Optional[List[float]] = None
        try:
            message_vector = self.memory.embeddings.embed_query(message)
            candidates = self.memory.hybrid_search(
                session_id,
                message,
                top_k=self.settings.hybrid_top_k,
                query_vector=message_vector,
            )
            context = self.memory.rerank(
                message, candidates, top_n=self.settings.rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            logger.warning("Retrieval skipped (%s); using recent turns only", exc)
            metrics.inc("agent_retrieval_skipped_total")

        messages = render_prompt(context=context, recent=recent, message=message)
        tools, sinks = build_tools(self.memory, session_id)
        llm = self.llm.bind_tools(tools)
        return messages, context, message_vector, tools, sinks, llm

    def _execute_tool_calls(self, tool_calls, tools):
        """Run each requested tool call, returning ``(ToolMessages, events)``."""
        from langchain_core.messages import ToolMessage

        by_name = {t.name: t for t in tools}
        tool_msgs, events = [], []
        for call in tool_calls:
            name, args = call["name"], call.get("args", {}) or {}
            tool = by_name.get(name)
            try:
                out = tool.invoke(args) if tool else f"Unknown tool: {name}"
            except Exception as exc:  # noqa: BLE001 - surface tool errors to model
                out = f"Tool {name} failed: {exc}"
            metrics.inc("agent_tool_calls_total", 1.0, tool=name)
            tool_msgs.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
            events.append({"name": name, "args": args, "result": str(out)})
        return tool_msgs, events

    def _run_tool_loop(self, messages, tools, llm):
        """Plan->act->observe until the model stops calling tools (bounded)."""
        tool_events: List[Dict[str, Any]] = []
        ai = llm.invoke(messages)
        iters = 0
        while getattr(ai, "tool_calls", None) and iters < self.settings.max_tool_iters:
            messages.append(ai)
            tool_msgs, events = self._execute_tool_calls(ai.tool_calls, tools)
            messages.extend(tool_msgs)
            tool_events.extend(events)
            ai = llm.invoke(messages)
            iters += 1
        return ai.content, tool_events

    def respond(self, session_id: str, message: str) -> Dict[str, Any]:
        messages, context, message_vector, tools, sinks, llm = self._prepare(
            session_id, message
        )
        reply, tool_events = self._run_tool_loop(messages, tools, llm)

        self.memory.save_message(session_id, "user", message, embedding=message_vector)
        self.memory.save_message(session_id, "assistant", reply)
        metrics.inc("agent_messages_saved_total", 1.0, role="user")
        metrics.inc("agent_messages_saved_total", 1.0, role="assistant")
        return {
            "session_id": session_id,
            "reply": reply,
            "context_used": context,
            "tool_events": tool_events,
        }

    def stream_respond(self, session_id: str, message: str):
        """Yield SSE event dicts for a turn: context, tool calls, then tokens.

        Tool iterations run non-streamed (a tool-calling step has no user-facing
        text); once the model stops requesting tools, the final answer is
        streamed token-by-token. Each yielded item is ``{"type", "data"}``.
        """
        messages, context, message_vector, tools, sinks, llm = self._prepare(
            session_id, message
        )
        yield {"type": "context", "data": context}

        # Resolve tool calls first (bounded), emitting an event per call.
        ai = llm.invoke(messages)
        iters = 0
        while getattr(ai, "tool_calls", None) and iters < self.settings.max_tool_iters:
            messages.append(ai)
            tool_msgs, events = self._execute_tool_calls(ai.tool_calls, tools)
            messages.extend(tool_msgs)
            for ev in events:
                yield {"type": "tool", "data": ev}
            ai = llm.invoke(messages)
            iters += 1

        # If the model already produced the final text (no more tool calls),
        # stream a fresh generation of it so the UI still gets token events.
        parts: List[str] = []
        for chunk in llm.stream(messages):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                parts.append(piece)
                yield {"type": "token", "data": piece}
        reply = "".join(parts) or (ai.content if isinstance(ai.content, str) else "")

        self.memory.save_message(session_id, "user", message, embedding=message_vector)
        self.memory.save_message(session_id, "assistant", reply)
        metrics.inc("agent_messages_saved_total", 1.0, role="user")
        metrics.inc("agent_messages_saved_total", 1.0, role="assistant")
        yield {"type": "done", "data": {"reply": reply, "context_used": context}}


def build_app(agent: Agent):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Stateful AI Agent")

    # Static demo UI. ``static/`` ships in the image next to this module; the
    # directory check keeps the app importable in environments (e.g. tests)
    # where the assets are absent.
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index():
            return FileResponse(index_file)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_endpoint():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(metrics.render())

    @app.post("/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        try:
            result = agent.respond(req.session_id, req.message)
        except Exception:
            metrics.inc("agent_requests_total", 1.0, route="chat", outcome="error")
            raise
        metrics.inc("agent_requests_total", 1.0, route="chat", outcome="ok")
        return result

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        from fastapi.responses import StreamingResponse

        def sse():
            try:
                for event in agent.stream_respond(req.session_id, req.message):
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
            except Exception as exc:  # noqa: BLE001 - report to the client stream
                metrics.inc("agent_requests_total", 1.0, route="chat_stream", outcome="error")
                yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
                return
            metrics.inc("agent_requests_total", 1.0, route="chat_stream", outcome="ok")

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/search")
    def search(req: SearchRequest) -> Dict[str, Any]:
        try:
            if req.mode == "vector":
                results = agent.memory.semantic_search(req.session_id, req.query, req.top_k)
            else:
                results = agent.memory.hybrid_search(req.session_id, req.query, req.top_k)
        except Exception:
            metrics.inc("agent_requests_total", 1.0, route="search", outcome="error")
            raise
        metrics.inc("agent_requests_total", 1.0, route="search", outcome="ok")
        return {"mode": req.mode, "results": results}

    @app.post("/rerank")
    def rerank(req: SearchRequest) -> Dict[str, Any]:
        try:
            candidates = agent.memory.hybrid_search(req.session_id, req.query, req.top_k)
            reranked = agent.memory.rerank(
                req.query, candidates, top_n=agent.settings.rerank_top_n
            )
        except Exception:
            metrics.inc("agent_requests_total", 1.0, route="rerank", outcome="error")
            raise
        metrics.inc("agent_requests_total", 1.0, route="rerank", outcome="ok")
        return {"before": candidates, "after": reranked}

    @app.get("/memory/{session_id}")
    def memory(session_id: str, n: int = 3) -> Dict[str, Any]:
        return {"session_id": session_id, "recent": agent.memory.get_recent(session_id, n)}

    return app


def run_cli(agent: Agent, session_id: str) -> None:
    print(f"Session: {session_id}. Type 'exit' to quit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message or message.lower() in {"exit", "quit"}:
            break
        result = agent.respond(session_id, message)
        print(f"agent> {result['reply']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stateful AI agent")
    parser.add_argument("command", choices=["serve", "chat"], nargs="?", default="serve")
    parser.add_argument("--session-id", default=os.environ.get("SESSION_ID", str(uuid.uuid4())))
    args = parser.parse_args()

    settings = Settings.from_env()
    agent = Agent(settings)

    if args.command == "chat":
        run_cli(agent, args.session_id)
        return

    import uvicorn

    uvicorn.run(build_app(agent), host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    sys.exit(main())

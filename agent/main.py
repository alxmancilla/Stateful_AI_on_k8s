"""Stateful agent entrypoint.

Runs as a FastAPI service (``serve``, the container default) or an interactive
CLI (``chat``). Both share the same ``Agent`` core: retrieve recent turns,
run hybrid search + rerank for relevant long-term context, call an LLM through
the Azure Grove API, and persist both sides of the exchange to MongoDB.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel

from memory import MongoMemory, get_client
from embeddings import VoyageEmbeddings
from search import VoyageReranker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("agent.main")

SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent long-term memory stored in "
    "MongoDB. Use the retrieved context and recent conversation to answer "
    "accurately and consistently. If the context is irrelevant, ignore it."
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
    """Retrieval-augmented, MongoDB-backed conversational agent."""

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
        )
        self.memory.ensure_indexes()
        self.llm = build_llm(settings)

    def respond(self, session_id: str, message: str) -> Dict[str, Any]:
        recent = self.memory.get_recent(session_id, n=self.settings.recent_turns)
        context: List[Dict[str, Any]] = []
        try:
            candidates = self.memory.hybrid_search(
                session_id, message, top_k=self.settings.hybrid_top_k
            )
            context = self.memory.rerank(
                message, candidates, top_n=self.settings.rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            logger.warning("Retrieval skipped (%s); using recent turns only", exc)

        prompt = render_prompt(context=context, recent=recent, message=message)
        reply = self.llm.invoke(prompt).content

        self.memory.save_message(session_id, "user", message)
        self.memory.save_message(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply, "context_used": context}


def build_app(agent: Agent):
    from fastapi import FastAPI

    app = FastAPI(title="Stateful AI Agent")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        return agent.respond(req.session_id, req.message)

    @app.post("/search")
    def search(req: SearchRequest) -> Dict[str, Any]:
        if req.mode == "vector":
            results = agent.memory.semantic_search(req.session_id, req.query, req.top_k)
        else:
            results = agent.memory.hybrid_search(req.session_id, req.query, req.top_k)
        return {"mode": req.mode, "results": results}

    @app.post("/rerank")
    def rerank(req: SearchRequest) -> Dict[str, Any]:
        candidates = agent.memory.hybrid_search(req.session_id, req.query, req.top_k)
        reranked = agent.memory.rerank(req.query, candidates, top_n=3)
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

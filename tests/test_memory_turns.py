"""Tests for the per-session turn counter and recent-read clamp in ``MongoMemory``.

The suite is **hermetic by default**: with no ``TEST_MONGODB_URI`` set it runs
against an in-process ``mongomock`` backend (no MongoDB, no network), so CI can
exercise the turn-counter, unique-index, and clamp logic offline::

    python3 -m pip install -r requirements-dev.txt
    pytest tests/ -v

Set ``TEST_MONGODB_URI`` to run against a real replica set instead — e.g. a
port-forward to the demo cluster::

    kubectl -n mongodb port-forward svc/mdbc-rs-svc 27018:27017 &
    export TEST_MONGODB_URI="mongodb://admin-user:<pw>@127.0.0.1:27018/admin\
?tls=true&tlsAllowInvalidCertificates=true&directConnection=true"
    pytest tests/ -v

The concurrency test needs true server-side atomicity (mongomock's in-memory
store is not thread-safe), so it only runs against a real server and skips on
the mongomock backend.
"""

from __future__ import annotations

import os
import sys
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mongomock
import pytest
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError


class _StubEmbeddings:
    """Deterministic, offline stand-in for VoyageEmbeddings."""

    def embed_query(self, text: str):
        return [0.0] * 8


class _StubReranker:
    def rerank(self, query, candidates, top_n=3):
        return candidates[:top_n]


# ``memory`` transitively imports ``embeddings``/``search`` (which import the
# ``voyageai`` SDK and use 3.10+ syntax). The turn-counter path needs none of
# that, so stub those modules before importing MongoMemory. This keeps the test
# offline and Python-version agnostic.
def _install_stub_modules() -> None:
    voyageai_stub = types.ModuleType("voyageai")
    embeddings_stub = types.ModuleType("embeddings")
    embeddings_stub.VoyageEmbeddings = _StubEmbeddings
    search_stub = types.ModuleType("search")
    search_stub.VoyageReranker = _StubReranker
    search_stub.build_rankfusion_pipeline = lambda **kwargs: []
    sys.modules.setdefault("voyageai", voyageai_stub)
    sys.modules.setdefault("embeddings", embeddings_stub)
    sys.modules.setdefault("search", search_stub)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
_install_stub_modules()
from memory import MAX_RECENT, MongoMemory  # noqa: E402


def _make_client():
    """Return ``(client, is_real)``.

    Prefer a real MongoDB when ``TEST_MONGODB_URI`` is set and reachable;
    otherwise fall back to an in-process ``mongomock`` client so the suite is
    hermetic.
    """
    uri = os.environ.get("TEST_MONGODB_URI")
    if uri:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
        except ServerSelectionTimeoutError:
            pytest.skip("TEST_MONGODB_URI set but MongoDB not reachable")
        return client, True
    return mongomock.MongoClient(), False


@pytest.fixture()
def memory():
    client, is_real = _make_client()
    db = client["agent_memory_test"]
    suffix = uuid.uuid4().hex[:8]
    messages = db[f"messages_{suffix}"]
    counters = db[f"counters_{suffix}"]
    mem = MongoMemory(
        collection=messages,
        embeddings=_StubEmbeddings(),
        reranker=_StubReranker(),
        counters=counters,
    )
    mem.ensure_indexes()
    # Expose the backend kind so tests needing real server-side atomicity can
    # skip on the mongomock backend.
    mem.is_real_backend = is_real
    try:
        yield mem
    finally:
        messages.drop()
        counters.drop()


def test_next_turn_is_sequential(memory):
    sid = "seq-session"
    assert [memory._next_turn(sid) for _ in range(5)] == [0, 1, 2, 3, 4]


def test_concurrent_saves_have_unique_contiguous_turns(memory):
    if not memory.is_real_backend:
        pytest.skip("concurrency needs real server-side atomicity (set TEST_MONGODB_URI)")
    sid = "concurrent-session"
    n = 32

    def write(i: int):
        return memory.save_message(sid, "user", f"msg {i}")["turn"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        turns = list(pool.map(write, range(n)))

    assert len(turns) == n
    assert len(set(turns)) == n, "duplicate turn values under concurrency"
    assert sorted(turns) == list(range(n)), "turns are not contiguous 0..n-1"


def test_unique_index_rejects_duplicate_turn(memory):
    sid = "dup-session"
    memory.save_message(sid, "user", "first")  # turn 0

    # A manual insert reusing (session_id, turn) must be rejected by the
    # unique index created in ensure_indexes().
    with pytest.raises(DuplicateKeyError):
        memory.collection.insert_one(
            {
                "_id": str(uuid.uuid4()),
                "session_id": sid,
                "turn": 0,
                "role": "user",
                "content": "collision",
            }
        )


def test_save_message_sets_created_at_date(memory):
    from datetime import datetime

    doc = memory.save_message("date-session", "user", "hello")
    stored = memory.collection.find_one({"_id": doc["_id"]})
    # created_at must be a BSON Date (datetime) so a TTL index can act on it;
    # timestamp stays a display string.
    assert isinstance(stored["created_at"], datetime)
    assert isinstance(stored["timestamp"], str)


def test_retention_trim_keeps_newest_turns():
    """With max_turns_per_session set, only the newest N turns survive."""
    client, _ = _make_client()
    db = client["agent_memory_test"]
    suffix = uuid.uuid4().hex[:8]
    mem = MongoMemory(
        collection=db[f"messages_{suffix}"],
        embeddings=_StubEmbeddings(),
        reranker=_StubReranker(),
        counters=db[f"counters_{suffix}"],
        max_turns_per_session=3,
    )
    mem.ensure_indexes()
    try:
        sid = "trim-session"
        for i in range(6):
            mem.save_message(sid, "user", f"msg {i}")
        recent = mem.get_recent(sid, n=100)
        assert [m["turn"] for m in recent] == [3, 4, 5]
    finally:
        mem.collection.drop()
        mem.counters.drop()


def _ttl_index(collection):
    return next(
        (ix for ix in collection.list_indexes() if "expireAfterSeconds" in ix),
        None,
    )


def test_ttl_index_reconciles_with_config():
    """Enabling builds the TTL index; disabling drops it again."""
    client, _ = _make_client()
    db = client["agent_memory_test"]
    suffix = uuid.uuid4().hex[:8]
    coll = db[f"messages_{suffix}"]
    counters = db[f"counters_{suffix}"]

    def build(ttl):
        m = MongoMemory(
            collection=coll,
            embeddings=_StubEmbeddings(),
            reranker=_StubReranker(),
            counters=counters,
            ttl_seconds=ttl,
        )
        m.ensure_indexes()
        return m

    try:
        build(0)
        assert _ttl_index(coll) is None
        build(90)
        ix = _ttl_index(coll)
        assert ix is not None and ix["expireAfterSeconds"] == 90
        # Disabling must remove the index so expiry actually stops.
        build(0)
        assert _ttl_index(coll) is None
    finally:
        coll.drop()
        counters.drop()


def test_agent_tools_execute_and_record(memory):
    """The bound tools search memory, save facts, and record structured sinks.

    Requires langchain_core (present in the agent image / CI). Skips offline so
    the rest of the suite stays hermetic.
    """
    pytest.importorskip("langchain_core")
    from tools import build_tools

    sid = "tools-session"
    memory.save_message(sid, "user", "My favorite language is Rust.")
    tools, sinks = build_tools(memory, sid)
    by_name = {t.name: t for t in tools}

    assert set(by_name) == {"search_memory", "save_fact", "get_current_time"}

    out = by_name["search_memory"].invoke({"query": "favorite language"})
    assert "Rust" in out
    assert sinks["retrieved"], "search_memory should record retrieved docs"

    saved = by_name["save_fact"].invoke({"fact": "User prefers dark mode."})
    assert "dark mode" in saved
    assert sinks["saved_facts"] == ["User prefers dark mode."]
    # The fact is persisted as a system turn and is retrievable.
    stored = memory.get_recent(sid, n=100)
    assert any("FACT: User prefers dark mode." in m["content"] for m in stored)

    now = by_name["get_current_time"].invoke({})
    assert "T" in now  # ISO 8601


def test_get_recent_clamps_n(memory):
    sid = "clamp-session"
    total = MAX_RECENT + 5
    for i in range(total):
        memory.save_message(sid, "user", f"msg {i}")

    # A too-large n is capped at MAX_RECENT.
    assert len(memory.get_recent(sid, n=total * 10)) == MAX_RECENT
    # A non-positive n still returns at least one turn, not zero or a scan.
    assert len(memory.get_recent(sid, n=0)) == 1
    assert len(memory.get_recent(sid, n=-4)) == 1

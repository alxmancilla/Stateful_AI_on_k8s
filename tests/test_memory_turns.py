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

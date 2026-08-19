"""Minimal in-process metrics with a Prometheus text exposition.

Deliberately dependency-free: the demo does not warrant pulling in
``prometheus_client``. Counters are process-local (reset on restart), which is
sufficient for a single-replica demo scrape. For real multi-replica use you
would switch to the official client with a shared registry.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

_LOCK = threading.Lock()
_START = time.time()

# name -> {label_tuple: value}. Labels are kept as a sorted tuple of (k, v).
_COUNTERS: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}

# Static metadata (HELP/TYPE) for the counters this app emits.
_META = {
    "agent_requests_total": ("Total agent HTTP requests by route and outcome.", "counter"),
    "agent_retrieval_skipped_total": ("Turns where retrieval was skipped (best-effort).", "counter"),
    "agent_messages_saved_total": ("Memory documents persisted, by role.", "counter"),
    "agent_tool_calls_total": ("Tool invocations by tool name.", "counter"),
}


def _key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    """Increment counter ``name`` (optionally labelled) by ``value``."""
    k = _key(labels)
    with _LOCK:
        _COUNTERS.setdefault(name, {})
        _COUNTERS[name][k] = _COUNTERS[name].get(k, 0.0) + value


def _fmt_labels(k: Tuple[Tuple[str, str], ...]) -> str:
    if not k:
        return ""
    inner = ",".join(f'{lk}="{lv}"' for lk, lv in k)
    return "{" + inner + "}"


def render() -> str:
    """Render all counters in Prometheus text exposition format."""
    lines = []
    with _LOCK:
        snapshot = {n: dict(series) for n, series in _COUNTERS.items()}
    for name in sorted(snapshot):
        help_text, mtype = _META.get(name, ("", "counter"))
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        for k, v in sorted(snapshot[name].items()):
            lines.append(f"{name}{_fmt_labels(k)} {v}")
    # Process uptime is handy for confirming a fresh scrape target.
    lines.append("# HELP agent_uptime_seconds Seconds since process start.")
    lines.append("# TYPE agent_uptime_seconds gauge")
    lines.append(f"agent_uptime_seconds {time.time() - _START:.0f}")
    return "\n".join(lines) + "\n"

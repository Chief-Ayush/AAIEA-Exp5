"""Run context (logging, tracing, cost, fault injection) + reliability helpers (controlled retry)."""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import PATHS, Settings
from .schemas import AgentMessage, ToolResult, dumps


class TransientError(Exception):
    """Recoverable failure (network blip, rate limit) -> retried with backoff."""


class ToolTimeout(TransientError):
    pass


class PermanentError(Exception):
    """Non-recoverable failure (bad source, missing product) -> never retried, reported to the Supervisor."""


class AgentTransientError(TransientError):
    pass


# cost model (lab: "experimental cost units" = agent calls + tool calls + retries + processing operations)
COST_AGENT_CALL = 1.0
COST_RETRY = 1.5
COST_PER_100_ROWS = 0.5


class RunContext:
    """Per-workflow runtime object (thread-safe).  The LangGraph *state* stays plain data; everything that is
    shared, mutable or non-serialisable (locks, caches, fault counters) lives here and is looked up by workflow id."""

    def __init__(self, settings: Settings, store, options: dict | None = None, workflow_id: str | None = None):
        self.id = workflow_id or f"WF-{uuid.uuid4().hex[:8]}"
        self.settings = settings
        self.store = store
        self.options = options or {}
        self.t0 = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.lock = threading.RLock()
        self.tool_calls: list[dict] = []
        self.spans: list[dict] = []
        self.agent_calls: Counter = Counter()
        self.retries = 0
        self.cache: dict = {}
        self.cache_hits = 0
        self.processing_rows = 0
        self.faults = copy.deepcopy(self.options.get("faults", {}))
        self.security_events: list[dict] = []
        self._seq = 0

    # ---------------------------------------------------------------- clocks / ids
    def now(self) -> float:
        return time.perf_counter() - self.t0

    def next_id(self, prefix: str) -> str:
        with self.lock:
            self._seq += 1
            return f"{prefix}{self._seq:03d}"

    # ---------------------------------------------------------------- tracing
    @contextmanager
    def span(self, agent: str, task: str, tool: str | None = None):
        sp = {"workflow_id": self.id, "agent": agent, "task": task, "tool": tool, "start": round(self.now(), 4), "end": None,
              "duration_s": None, "status": "running", "thread": threading.current_thread().name, "retry": 0, "error": None}
        try:
            yield sp
            if sp["status"] == "running":
                sp["status"] = "success"
        except Exception as exc:
            sp["status"] = "failed"
            sp["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp["end"] = round(self.now(), 4)
            sp["duration_s"] = round(sp["end"] - sp["start"], 4)
            with self.lock:
                self.spans.append(sp)

    def message(self, sender: str, receiver: str, mtype: str, summary: str, payload=None, status: str = "ok") -> dict:
        return AgentMessage(id=self.next_id("M"), ts=round(self.now(), 4), sender=sender, receiver=receiver, type=mtype,
                            summary=summary, status=status, payload_bytes=dumps(payload)).model_dump()

    def security(self, etype: str, **kw) -> dict:
        ev = {"ts": round(self.now(), 4), "type": etype, **kw}
        with self.lock:
            self.security_events.append(ev)
        self.audit("security", **ev)
        return ev

    # ---------------------------------------------------------------- faults (failure-handling tests)
    def take_fault(self, kind: str, key: str | None = None) -> bool:
        """Consume one injected fault.  kind=source_transient -> {source: n}; agent_exception -> {agent: n}; ..."""
        with self.lock:
            f = self.faults.get(kind)
            if f is None:
                return False
            if isinstance(f, bool):
                if f:
                    self.faults[kind] = False
                return f
            if isinstance(f, int):
                if f > 0:
                    self.faults[kind] = f - 1
                    return True
                return False
            if isinstance(f, (list, set, tuple)):          # permanent faults: never consumed
                return key in f
            if isinstance(f, dict):
                if f.get(key, 0) > 0:
                    f[key] -= 1
                    return True
        return False

    # ---------------------------------------------------------------- cost / metrics
    def log_tool(self, rec: dict) -> None:
        with self.lock:
            self.tool_calls.append(rec)

    def cost_units(self) -> float:
        with self.lock:
            tool_cost = sum(c.get("cost_units", 0.0) for c in self.tool_calls)
            return round(sum(self.agent_calls.values()) * COST_AGENT_CALL + tool_cost + self.retries * COST_RETRY
                         + self.processing_rows / 100.0 * COST_PER_100_ROWS, 2)

    def metrics(self, wall_s: float | None = None) -> dict:
        with self.lock:
            spans = [s for s in self.spans if s["agent"] != "Tool"]
            by_agent: dict[str, float] = {}
            for s in spans:
                by_agent[s["agent"]] = by_agent.get(s["agent"], 0.0) + (s["duration_s"] or 0.0)
            wall = wall_s if wall_s is not None else self.now()
            par = [s for s in spans if s.get("group")]
            return {
                "total_time_s": round(wall, 4),
                "agent_executions": sum(self.agent_calls.values()),
                "tool_calls": len(self.tool_calls),
                "tool_calls_ok": sum(1 for c in self.tool_calls if c["ok"]),
                "tool_calls_failed": sum(1 for c in self.tool_calls if not c["ok"]),
                "cache_hits": self.cache_hits,
                "retries": self.retries,
                "cost_units": self.cost_units(),
                "agent_time_s": {k: round(v, 4) for k, v in by_agent.items()},
                "sum_of_agent_time_s": round(sum(by_agent.values()), 4),
                "security_events": len(self.security_events),
                "parallel_spans": len(par),
            }

    # ---------------------------------------------------------------- audit log
    def audit(self, kind: str, **kw) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "workflow_id": self.id, "kind": kind, **kw}
        if self.options.get("audit", True):
            try:
                with self.lock, open(PATHS.audit_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass


# ----------------------------------------------------------------------------------------- context registry
_CTX: dict[str, RunContext] = {}
_CTX_LOCK = threading.Lock()


def register_ctx(ctx: RunContext) -> None:
    with _CTX_LOCK:
        _CTX[ctx.id] = ctx
        if len(_CTX) > 400:                       # bounded memory for long batches
            for k in list(_CTX)[:100]:
                _CTX.pop(k, None)


def get_ctx(workflow_id: str) -> RunContext:
    return _CTX[workflow_id]


# ----------------------------------------------------------------------------------------- controlled retry
def call_with_retry(registry, name: str, agent: str, arguments: dict, ctx: RunContext) -> ToolResult:
    """Retry only *recoverable* failures, with exponential back-off and a hard cap.  A permanent failure is
    returned immediately (-> Supervisor).  `retry_policy='naive'` retries everything 5x without back-off (baseline)."""
    s = ctx.settings
    naive = s.retry_policy == "naive"
    max_retries = 5 if naive else s.max_tool_retries
    attempt = 0
    while True:
        res = registry.call(name, agent, arguments, ctx)
        res.attempts = attempt + 1
        if res.ok:
            return res
        recoverable = naive or res.error_type in ("Transient", "Timeout")
        if not recoverable or attempt >= max_retries:
            return res
        attempt += 1
        with ctx.lock:
            ctx.retries += 1
        if not naive:
            time.sleep(s.retry_backoff_s * (2 ** (attempt - 1)))

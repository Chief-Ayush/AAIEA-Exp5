"""Public facade: ShoppingAssistant.run(query) -> RunResult.  Handles LangGraph execution, human-in-the-loop interrupts,
metrics (wall-clock, sequential-equivalent time, speed-up, cost) and the audit log."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable

from langgraph.types import Command

from .catalog import CatalogStore, default_store
from .config import Settings
from .graph import get_graph
from .runtime import RunContext, get_ctx, register_ctx
from .tools.registry import ToolRegistry, default_registry

SETTING_KEYS = {f.name for f in fields(Settings)}


def timing_summary(spans: list[dict], wall: float) -> dict:
    """Stage timing for the parallel groups; derives the sequential-equivalent time of THIS run."""
    out: dict[str, Any] = {}
    seq_total, wall_groups = 0.0, 0.0
    for g in ("search", "analysis"):
        sp = [s for s in spans if s.get("group") == g and s.get("end") is not None]
        if not sp:
            continue
        seq = sum(s["duration_s"] for s in sp)
        w = max(s["end"] for s in sp) - min(s["start"] for s in sp)
        out[g] = {"branches": len(sp), "sequential_s": round(seq, 4), "wall_s": round(w, 4)}
        seq_total += seq
        wall_groups += w
    est_seq = max(wall - wall_groups + seq_total, wall)
    out["sequential_equivalent_s"] = round(est_seq, 4)
    out["speedup_estimate"] = round(est_seq / wall, 3) if wall > 0 else 1.0
    return out


@dataclass
class RunResult:
    workflow_id: str
    query: str
    state: dict
    metrics: dict
    settings: dict
    pending: dict | None = None
    wall_segments: list = field(default_factory=list)

    @property
    def status(self) -> str:
        return "AWAITING_HUMAN" if self.pending else (self.state.get("status") or "UNKNOWN")

    @property
    def final(self) -> dict:
        return self.state.get("final_recommendation") or {}

    @property
    def top(self) -> list[dict]:
        return self.final.get("top", [])

    @property
    def requirements(self) -> dict:
        return self.state.get("requirements") or {}

    @property
    def trace(self) -> list[dict]:
        return sorted(self.state.get("trace", []), key=lambda s: s["start"])

    @property
    def messages(self) -> list[dict]:
        return sorted(self.state.get("messages", []), key=lambda m: m["ts"])

    def summary_row(self) -> dict:
        t = self.top[0] if self.top else None
        return {"query": self.query, "status": self.status, "top_pick": t["product_name"] if t else "", "top_price": t["product"]["price"] if t else None,
                "top_score": t["score"] if t else None, "n_results": len(self.top), "confidence": self.final.get("confidence"),
                "validation": (self.state.get("validation") or {}).get("status", "-"), "time_s": self.metrics.get("total_time_s"),
                "tool_calls": self.metrics.get("tool_calls"), "retries": self.metrics.get("retries"), "cost_units": self.metrics.get("cost_units")}


class ShoppingAssistant:
    def __init__(self, settings: Settings | None = None, store: CatalogStore | None = None, registry: ToolRegistry | None = None):
        self.settings = settings or Settings()
        self.store = store or default_store()
        self.registry = registry or default_registry()

    # ------------------------------------------------------------------------------------------------ helpers
    def _settings_for(self, overrides: dict) -> Settings:
        return self.settings.with_(**{k: v for k, v in overrides.items() if k in SETTING_KEYS})

    def run(self, query: str, *, faults: dict | None = None, audit: bool = True, top_n: int | None = None, llm=None,
            on_event: Callable[[str, Any], None] | None = None, **overrides) -> RunResult:
        s = self._settings_for(overrides)
        options = {"faults": faults or {}, "audit": audit, "top_n": top_n, "llm": llm}
        ctx = RunContext(s, self.store, options)
        ctx.registry = self.registry
        ctx.wall_segments = []
        register_ctx(ctx)
        ctx.audit("workflow_start", query=str(query)[:300], mode=s.execution_mode, approval=s.approval_mode)
        init = {"workflow_id": ctx.id, "user_query": query, "status": "RUNNING", "retries": {}, "cycle": 0,
                "options": {"execution_mode": s.execution_mode, "approval_mode": s.approval_mode, "validation_mode": s.validation_mode,
                            "interactive": s.interactive, "faults": faults or {}}}
        return self._execute(ctx, init, on_event)

    def resume(self, workflow_id: str, decision: dict, on_event=None) -> RunResult:
        ctx = get_ctx(workflow_id)
        return self._execute(ctx, Command(resume=decision), on_event)

    # ------------------------------------------------------------------------------------------------ execution
    def _execute(self, ctx: RunContext, payload, on_event) -> RunResult:
        s = ctx.settings
        graph = get_graph(s.execution_mode, s.interactive)
        cfg = {"configurable": {"thread_id": ctx.id}, "recursion_limit": 80}
        t0 = time.perf_counter()
        pending, last = None, None
        try:
            for mode, chunk in graph.stream(payload, cfg, stream_mode=["updates", "values"]):
                if mode == "values":
                    last = chunk
                elif mode == "updates":
                    if "__interrupt__" in chunk:
                        pending = chunk["__interrupt__"][0].value
                    if on_event:
                        for node in chunk:
                            if node != "__interrupt__":
                                on_event(node, chunk[node])
        except Exception as exc:  # noqa: BLE001 - never let a framework error escape to the UI
            ctx.audit("workflow_crash", error=f"{type(exc).__name__}: {exc}")
            last = (last or {}) | {"status": "ERROR", "errors": (last or {}).get("errors", []) + [{"agent": "Workflow", "error": f"{type(exc).__name__}: {exc}"}]}
        ctx.wall_segments.append(time.perf_counter() - t0)
        state = last or {}
        if s.interactive:
            snap = graph.get_state(cfg)
            state = dict(snap.values)
            if snap.next and pending is None:
                for t in snap.tasks:
                    if t.interrupts:
                        pending = t.interrupts[0].value
            if not snap.next:
                pending = None
        wall = sum(ctx.wall_segments)
        with ctx.lock:
            spans = list(ctx.spans)
        m = ctx.metrics(wall)
        m.update(timing_summary([x for x in spans if x["agent"] != "HumanApproval"], wall))
        m["execution_mode"] = s.execution_mode
        state["metrics"] = m
        res = RunResult(ctx.id, state.get("user_query", ""), state, m, s.to_dict(), pending, ctx.wall_segments)
        if not pending:
            ctx.audit("workflow_complete", status=res.status, total_time_s=m["total_time_s"], tool_calls=m["tool_calls"], retries=m["retries"], cost=m["cost_units"],
                      top=[t["product_name"] for t in res.top[:3]])
        return res

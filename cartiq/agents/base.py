"""Shared agent plumbing: LangGraph state, node wrapper (RBAC for state writes, tracing, retry, failure capture)."""
from __future__ import annotations

import functools
import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from ..runtime import AgentTransientError, TransientError, get_ctx
from ..security import enforce_write

TERMINAL_EARLY = {"BLOCKED", "INVALID", "NEEDS_CLARIFICATION", "ERROR"}


def merge_by_source(old: list | None, new: list | None) -> list:
    """Reducer for parallel search branches: one entry per source, latest wins (safe under retries)."""
    d = {r["source"]: r for r in (old or [])}
    for r in new or []:
        d[r["source"]] = r
    return list(d.values())


class WorkflowState(TypedDict, total=False):
    # identity / input
    workflow_id: str
    user_query: str
    sanitized_query: str
    guard: dict
    options: dict
    # planning
    requirements: dict
    req_digest: str
    plan: list
    planner_notes: list
    routing: list
    # data flow
    source_results: Annotated[list, merge_by_source]
    search_results: list
    conflicts: list
    quarantined: list
    sync: dict
    filtered_results: list
    rejected: list
    comparison: dict
    feature_analysis: dict
    ranking: list
    ranking_meta: dict
    # control
    validation: dict
    checkpoints: Annotated[list, operator.add]
    supervisor: dict
    retries: dict
    relaxation_suggestions: list
    human_approval: dict
    human_flags: list
    cycle: int
    final_recommendation: dict
    status: str
    # observability (append-only channels)
    messages: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    security_events: Annotated[list, operator.add]
    metrics: dict


def agent_node(agent: str, task: str, group: str | None = None):
    """Decorator: wraps ``fn(state, ctx) -> dict | Command`` with tracing, injected-fault handling, controlled
    agent-level retry, write-permission enforcement and failure capture (a crashing agent never kills the graph)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: dict):
            ctx = get_ctx(state["workflow_id"])
            s = ctx.settings
            if state.get("status") in TERMINAL_EARLY and agent != "RecommendationAgent":
                return {}                              # workflow already ended early: downstream agents do not run
            with ctx.lock:
                ctx.agent_calls[agent] += 1
            attempt = 0
            out: Any = None
            err = None
            with ctx.span(agent, task) as sp:
                if group:
                    sp["group"] = group
                while True:
                    try:
                        if ctx.take_fault("agent_exception", agent):
                            raise AgentTransientError(f"{agent} crashed (injected fault)")
                        out = fn(state, ctx)
                        break
                    except GraphBubbleUp:
                        raise
                    except TransientError as exc:
                        if attempt < s.max_tool_retries:
                            attempt += 1
                            with ctx.lock:
                                ctx.retries += 1
                            sp["retry"] = attempt
                            time.sleep(s.retry_backoff_s * 2 ** (attempt - 1))
                            continue
                        err = f"{type(exc).__name__}: {exc} (gave up after {attempt} retries)"
                        break
                    except Exception as exc:  # noqa: BLE001
                        err = f"{type(exc).__name__}: {exc}"
                        break
                if err:
                    sp["status"] = "failed"
                    sp["error"] = err
            if err:
                ctx.audit("agent_failure", agent=agent, error=err)
                sp["output"], sp["output_status"] = err[:120], "error"
                return {"errors": [{"agent": agent, "task": task, "error": err, "retries": attempt}], "trace": [sp],
                        "messages": [ctx.message(agent, "SupervisorAgent", "failure_report", err, status="error")],
                        **({"status": "ERROR"} if agent != "SearchAgent" else {})}
            return _finalize_update(agent, out, sp, ctx)

        wrapper.__agent__ = agent
        return wrapper

    return deco


def _finalize_update(agent: str, out, sp: dict, ctx):
    cmd = out if isinstance(out, Command) else None
    upd = dict(cmd.update or {}) if cmd else dict(out or {})
    upd, events = enforce_write(agent, upd, ctx.settings.guard_state_writes)
    for e in events:
        ctx.security(e["type"], **{k: v for k, v in e.items() if k != "type"})
    if events:
        upd["security_events"] = list(upd.get("security_events", [])) + [{**e, "ts": round(ctx.now(), 4)} for e in events]
    _annotate_span(agent, sp, upd, ctx)
    upd["trace"] = list(upd.get("trace", [])) + [sp]
    return Command(update=upd, goto=cmd.goto) if cmd else upd


def _annotate_span(agent: str, sp: dict, upd: dict, ctx) -> None:
    """Complete the trace record with the fields the lab asks for: tool(s) used and the output status."""
    with ctx.lock:
        calls = [t for t in ctx.tool_calls
                 if t.get("agent") == agent and t.get("thread", sp.get("thread")) == sp.get("thread")
                 and sp["start"] - 1e-3 <= t.get("ts", sp["start"]) <= (sp["end"] or ctx.now()) + 1e-3]
    sp["tool"] = ", ".join(dict.fromkeys(t["tool"] for t in calls)) or None
    tool_retries = sum(1 for t in calls if not t["ok"] and t.get("error_type") in ("Transient", "Timeout"))
    if tool_retries:
        sp["retry"] = sp.get("retry", 0) + tool_retries
        sp["error"] = sp.get("error") or "; ".join(f"{t['tool']}: {t['error_type']} (retried)" for t in calls if not t["ok"])[:160]
    msgs = upd.get("messages") or []
    sp["output"] = msgs[0]["summary"][:120] if msgs else "-"
    sp["output_status"] = msgs[0]["status"] if msgs else "ok"

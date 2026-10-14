"""The agents.  Each node reads the shared LangGraph state, uses ONLY its authorised tools (via the registry gateway),
writes ONLY the state keys it owns, and emits structured messages for the next agent."""
from __future__ import annotations

import math
import re
from typing import Any

from langgraph.types import Command, Send, interrupt

from ..config import SOURCES
from ..llm import merge_llm
from ..runtime import PermanentError, RunContext, call_with_retry, get_ctx
from ..schemas import PlanTask, Requirements
from ..security import screen_query
from .base import TERMINAL_EARLY, agent_node

SUPPORTED = "Laptop, Smartphone, Tablet, Headphones, Smartwatch"


def _tool(ctx: RunContext, agent: str, name: str, **args):
    res = call_with_retry(ctx.registry, name, agent, args, ctx)
    if not res.ok:
        raise RuntimeError(f"tool '{name}' failed ({res.error_type}): {res.error}")
    return res.output


def _req(state) -> dict:
    return state.get("requirements") or {}


# ============================================================================================== 0. security guard
@agent_node("SecurityGuard", "validate & sanitise user input")
def security_guard(state, ctx: RunContext):
    g = screen_query(state.get("user_query"))
    events = []
    for f in g.findings:
        events.append(ctx.security("prompt_injection", category=f["category"], text=f["text"], action="removed" if g.status == "SANITIZED" else "blocked"))
    out: dict[str, Any] = {"guard": {"status": g.status, "risk": g.risk, "findings": g.findings, "warnings": g.warnings,
                                     "original": g.original_query, "clean": g.clean_query},
                           "sanitized_query": g.clean_query, "security_events": events,
                           "messages": [ctx.message("SecurityGuard", "PlannerAgent", "sanitized_query",
                                                    f"{g.status}: {len(g.findings)} finding(s)", {"q": g.clean_query}, "ok" if g.status in ("SAFE", "SANITIZED") else "blocked")]}
    if g.status in ("BLOCKED", "INVALID"):
        out["status"] = g.status
    return out


# ============================================================================================== 1. planner
def build_plan(req: Requirements, sources: list[str]) -> list[dict]:
    tasks = [PlanTask(id=f"S{i + 1}", kind="search", description=f"Search {s} for {req.category or 'products'}", params={"source": s}) for i, s in enumerate(sources)]
    sid = [t.id for t in tasks]
    tasks += [
        PlanTask(id="Y1", kind="synchronize", description="Merge marketplace results & resolve conflicts", depends_on=sid),
        PlanTask(id="F1", kind="filter", description="Apply hard requirements" if req.intent == "recommend" else "Select the products to compare", depends_on=["Y1"]),
        PlanTask(id="C1", kind="compare", description="Compare candidates attribute by attribute", depends_on=["F1"]),
        PlanTask(id="A1", kind="features", description="Feature matching & spec surplus analysis", depends_on=["F1"]),
        PlanTask(id="R1", kind="rank", description="Requirement-aware scoring and ranking", depends_on=["C1", "A1"]),
        PlanTask(id="V1", kind="validate", description="Validate ranking against trusted requirements", depends_on=["R1"]),
        PlanTask(id="U1", kind="supervise", description="Supervise workflow, handle failures", depends_on=["V1"]),
        PlanTask(id="H1", kind="approve", description="Human approval (governance policy)", depends_on=["U1"]),
        PlanTask(id="X1", kind="recommend", description="Assemble final recommendation", depends_on=["H1"]),
        PlanTask(id="V2", kind="validate", description="Validate the final recommendation before release", depends_on=["X1"]),
    ]
    return [t.model_dump() for t in tasks]


@agent_node("PlannerAgent", "decompose query into requirements and an execution plan")
def planner(state, ctx: RunContext):
    query = state.get("sanitized_query") or state["user_query"]
    out = _tool(ctx, "PlannerAgent", "plan_query", query=query)
    raw, notes = dict(out["requirements"]), list(out["notes"])
    llm = ctx.options.get("llm")
    if ctx.settings.use_llm_planner and llm is not None:
        raw, lnotes = merge_llm(raw, query, llm, ctx.store.categories)
        notes += lnotes
    events = []
    for c in out["conflicts"]:
        events.append(ctx.security("conflicting_constraints", field=c["field"], values=c["values"], action=c["resolution"]))
    if raw.get("intent") == "compare" and not raw.get("category"):
        first = ctx.store.detail(raw["compare_targets"][0])
        raw["category"] = first["category"] if first else None
    raw.setdefault("top_n", ctx.settings.top_n)
    if "top_n" in raw and raw["top_n"] is None:
        raw["top_n"] = ctx.settings.top_n
    if ctx.options.get("top_n"):
        raw["top_n"] = int(ctx.options["top_n"])
    try:
        req = Requirements(**{k: v for k, v in raw.items() if v is not None})
    except Exception as exc:  # noqa: BLE001
        return {"status": "NEEDS_CLARIFICATION", "planner_notes": notes + [f"requirements invalid: {exc}"],
                "messages": [ctx.message("PlannerAgent", "RecommendationAgent", "clarification_request", "invalid requirements", status="error")]}
    if not req.category and req.intent != "compare":
        return {"status": "NEEDS_CLARIFICATION", "requirements": req.model_dump(), "req_digest": req.digest(),
                "planner_notes": notes + [f"No supported product category found. Supported: {SUPPORTED}."], "security_events": events,
                "messages": [ctx.message("PlannerAgent", "RecommendationAgent", "clarification_request", "category not identified", status="blocked")]}
    plan = build_plan(req, list(SOURCES))
    return {"requirements": req.model_dump(), "req_digest": req.digest(), "plan": plan, "planner_notes": notes, "security_events": events,
            "messages": [ctx.message("PlannerAgent", "RouterAgent", "execution_plan", f"{len(plan)} tasks; category={req.category}; constraints={list(req.constraint_dict())}",
                                     {"requirements": req.model_dump(), "plan": plan})]}


# ============================================================================================== 2. router
@agent_node("RouterAgent", "route tasks to agents, analyse dependencies")
def router(state, ctx: RunContext):
    r = _tool(ctx, "RouterAgent", "route_tasks", plan=state["plan"], mode=ctx.settings.execution_mode)
    out: dict[str, Any] = {"routing": r["routing"],
                           "messages": [ctx.message("RouterAgent", "SearchAgent", "task_assignment",
                                                    f"{len(r['routing'])} tasks -> {r['stages']} stages; parallel stages {r['parallel_groups']}", r)]}
    if r["problems"]:
        out["status"] = "ERROR"
        out["errors"] = [{"agent": "RouterAgent", "error": "; ".join(r["problems"])}]
    return out


def _checkpoint(state, ctx: RunContext, stage: str, payload: dict):
    res = _tool(ctx, "ValidatorAgent", "validate_checkpoint", stage=stage, requirements=_req(state), payload=payload)
    return res


@agent_node("ValidatorAgent", "checkpoint #1: validate query decomposition")
def checkpoint_plan(state, ctx: RunContext):
    if ctx.settings.validation_mode != "checkpoint":
        return {}
    payload = {"problems": [], "stages": max((r["stage"] for r in state["routing"]), default=-1) + 1}
    res = _checkpoint(state, ctx, "plan", payload)
    out: dict[str, Any] = {"checkpoints": [res],
                           "messages": [ctx.message("ValidatorAgent", "SearchAgent", "checkpoint", f"plan checkpoint {res['status']}")]}
    if res["status"] == "FAIL":
        out["status"] = "ERROR"
        out["errors"] = [{"agent": "ValidatorAgent", "error": "plan checkpoint failed: " + "; ".join(c["detail"] for c in res["checks"] if not c["passed"])}]
    return out


# ============================================================================================== 3. search (parallel branches)
def _search_one(ctx: RunContext, source: str, req: dict) -> dict:
    cat = req.get("category") if req.get("intent") != "compare" or not req.get("compare_targets") else None
    r = call_with_retry(ctx.registry, "search_marketplace", "SearchAgent", {"source": source, "category": cat}, ctx)
    if r.ok:
        q = r.output["quarantined"]
        for item in q:
            ctx.security("poisoned_record", source=source, product_id=item["product_id"], reasons=item["reasons"], action="quarantined")
        return {"source": source, "ok": True, "records": r.output["records"], "quarantined": q, "error": None, "error_type": None,
                "attempts": r.attempts, "cached": r.cached}
    return {"source": source, "ok": False, "records": [], "quarantined": [], "error": r.error, "error_type": r.error_type, "attempts": r.attempts, "cached": False}


@agent_node("SearchAgent", "search one marketplace", group="search")
def search_source(state, ctx: RunContext):
    src = state["source"]
    rec = _search_one(ctx, src, state["requirements"])
    if rec["ok"]:
        msg = ctx.message("SearchAgent", "SynchronizerAgent", "search_results", f"{src}: {len(rec['records'])} products", {"n": len(rec["records"])})
    else:
        msg = ctx.message("SearchAgent", "SupervisorAgent", "search_failure", f"{src}: {rec['error_type']} - {rec['error']}", status="error")
    out: dict[str, Any] = {"source_results": [rec], "messages": [msg]}
    if rec["quarantined"]:
        out["security_events"] = [e for e in ctx.security_events if e.get("source") == src and e["type"] == "poisoned_record"]
    return out


@agent_node("SearchAgent", "search all marketplaces sequentially")
def search_sequential(state, ctx: RunContext):
    results, msgs = [], []
    for src in SOURCES:
        with ctx.span("SearchAgent", f"search {src}") as sp:
            sp["group"] = "search"
            rec = _search_one(ctx, src, state["requirements"])
            if not rec["ok"]:
                sp["status"] = "failed"
                sp["error"] = rec["error"]
        results.append(rec)
        msgs.append(ctx.message("SearchAgent", "SynchronizerAgent" if rec["ok"] else "SupervisorAgent", "search_results" if rec["ok"] else "search_failure",
                                f"{src}: {len(rec['records'])} products" if rec["ok"] else f"{src}: {rec['error_type']}", status="ok" if rec["ok"] else "error"))
    return {"source_results": results, "messages": msgs}


# ============================================================================================== 4. synchronizer (barrier + conflict resolution)
def _wmean(vals: list[tuple[float, float]]) -> float:
    tw = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / tw if tw else float("nan")


def _boundary_sensitive(vals: list[float], threshold: float | None, tol: float) -> bool:
    return threshold is not None and (min(vals) - tol <= threshold <= max(vals) + tol)


@agent_node("SynchronizerAgent", "synchronise parallel results & resolve conflicts")
def synchronizer(state, ctx: RunContext):
    s = ctx.settings
    req = _req(state)
    expected = [t["params"]["source"] for t in state["plan"] if t["kind"] == "search"]
    got = {r["source"]: r for r in state.get("source_results", [])}
    missing = [e for e in expected if e not in got]
    ok_src = [e for e in expected if e in got and got[e]["ok"]]
    failed = [e for e in expected if e not in ok_src]
    by_pid: dict[str, list[dict]] = {}
    for src in ok_src:
        for rec in got[src]["records"]:
            by_pid.setdefault(rec["product_id"], []).append(rec)
    quarantined = [q for src in got.values() for q in src.get("quarantined", [])]
    conflicts, merged, rechecks = [], [], 0
    for pid, recs in by_pid.items():
        recs = sorted(recs, key=lambda r: (-(r.get("source_trust") or 0), -(r.get("reviews") or 0)))
        base = dict(recs[0])
        # ---- rating
        rvals = [(r["rating"], (r.get("source_trust") or 0.5) * math.log1p(r.get("reviews") or 1)) for r in recs if r.get("rating") is not None]
        rating, method = None, "single_source"
        if rvals:
            raw = [v for v, _ in rvals]
            spread = max(raw) - min(raw)
            if len(recs) == 1:
                rating = raw[0]
            elif spread <= s.rating_conflict_tol + 1e-9:
                rating, method = round(_wmean(rvals), 2), "weighted_consensus"
            else:
                sensitive = s.recheck_all_conflicts or _boundary_sensitive(raw, req.get("min_rating"), 0.01) or spread > 0.45
                rating, method = recs[0]["rating"] if recs[0].get("rating") is not None else raw[0], "trusted_priority"
                if sensitive:
                    det = ctx.registry and call_with_retry(ctx.registry, "fetch_product_detail", "SynchronizerAgent", {"product_id": pid}, ctx)
                    rechecks += 1
                    if det and det.ok:
                        rating, method = det.output["rating"], "authoritative_recheck"
                conflicts.append({"product_id": pid, "product_name": base["product_name"], "field": "rating",
                                  "values": {r["source"]: r.get("rating") for r in recs}, "resolved": rating, "method": method})
        else:
            # missing rating everywhere -> recover from the authoritative source when a rating constraint exists
            if req.get("min_rating") is not None or req.get("priorities"):
                det = call_with_retry(ctx.registry, "fetch_product_detail", "SynchronizerAgent", {"product_id": pid}, ctx)
                rechecks += 1
                if det.ok:
                    rating, method = det.output["rating"], "missing_data_recovered"
                    conflicts.append({"product_id": pid, "product_name": base["product_name"], "field": "rating",
                                      "values": {r["source"]: r.get("rating") for r in recs}, "resolved": rating, "method": method})
        base["rating"] = rating
        base["rating_method"] = method
        # ---- price
        pvals = [(r["price"], (r.get("source_trust") or 0.5)) for r in recs if r.get("price") is not None]
        if pvals:
            raw = [v for v, _ in pvals]
            spread = (max(raw) - min(raw)) / max(raw)
            if spread <= s.price_conflict_tol:
                base["price"] = int(round(_wmean(pvals)))
            else:
                trusted = recs[0].get("price") if recs[0].get("price") is not None else raw[0]
                presolved, pmethod = trusted, "trusted_priority"
                if s.recheck_all_conflicts or _boundary_sensitive(raw, req.get("max_price"), 0) or _boundary_sensitive(raw, req.get("min_price"), 0):
                    det = call_with_retry(ctx.registry, "fetch_product_detail", "SynchronizerAgent", {"product_id": pid}, ctx)
                    rechecks += 1
                    if det.ok:
                        presolved, pmethod = det.output["price"], "authoritative_recheck"
                base["price"] = presolved
                conflicts.append({"product_id": pid, "product_name": base["product_name"], "field": "price",
                                  "values": {r["source"]: r.get("price") for r in recs}, "resolved": presolved, "method": pmethod})
        else:
            base["price"] = None
        if base.get("battery_hours") is None:
            for r in recs[1:]:
                if r.get("battery_hours") is not None:
                    base["battery_hours"] = r["battery_hours"]
                    break
        base["reviews"] = int(max(r.get("reviews") or 0 for r in recs))
        base["sources"] = [r["source"] for r in recs]
        base["n_sources"] = len(recs)
        merged.append(base)
    with ctx.lock:
        ctx.processing_rows += sum(len(v) for v in by_pid.values())
    sync = {"expected_sources": expected, "sources_ok": ok_src, "sources_failed": failed, "missing": missing, "merged": len(merged), "conflicts": len(conflicts),
            "rechecks": rechecks, "quarantined": len(quarantined), "degraded": bool(failed), "failed_all": not ok_src}
    msg_status = "ok" if ok_src else "error"
    return {"search_results": merged, "conflicts": conflicts, "quarantined": quarantined, "sync": sync,
            "messages": [ctx.message("SynchronizerAgent", "FilterAgent" if ok_src else "SupervisorAgent", "synchronised_products",
                                     f"{len(merged)} unique products from {len(ok_src)}/{len(expected)} sources; {len(conflicts)} conflicts ({rechecks} rechecked)",
                                     {"n": len(merged)}, msg_status)]}


@agent_node("SynchronizerAgent", "re-verify product flagged by the human reviewer")
def resync(state, ctx: RunContext):
    flag = (state.get("human_flags") or [{}])[-1]
    pid, field = flag.get("product_id"), flag.get("field", "rating")
    results = [dict(p) for p in state.get("search_results", [])]
    target = next((p for p in results if p["product_id"] == pid), None)
    if target is None:
        return {"messages": [ctx.message("SynchronizerAgent", "FilterAgent", "recheck_result", f"flagged product {pid} not in the candidate set", status="error")]}
    det = call_with_retry(ctx.registry, "fetch_product_detail", "SynchronizerAgent", {"product_id": pid}, ctx)
    if not det.ok:
        return {"messages": [ctx.message("SynchronizerAgent", "SupervisorAgent", "recheck_result", f"re-check failed: {det.error}", status="error")]}
    changes = {}
    for k in ("rating", "price", "ram_gb", "storage_gb", "battery_hours", "in_stock"):
        if k in det.output and det.output[k] is not None and det.output[k] != target.get(k):
            changes[k] = {"was": target.get(k), "now": det.output[k]}
            target[k] = det.output[k]
    target["rating_method"] = "human_flag_recheck"
    conflicts = list(state.get("conflicts") or []) + [{"product_id": pid, "product_name": target["product_name"], "field": field,
                                                         "values": {k: v["was"] for k, v in changes.items()} or {"(no change)": target.get(field)},
                                                         "resolved": target.get(field), "method": "human_flag_recheck"}]
    flags = list(state.get("human_flags") or [])
    flags[-1] = {**flag, "corrected": changes}
    summary = f"{target['product_name']}: " + (", ".join(f"{k} {v['was']} -> {v['now']}" for k, v in changes.items()) or "value confirmed by the authoritative source")
    return {"search_results": results, "conflicts": conflicts, "human_flags": flags,
            "messages": [ctx.message("SynchronizerAgent", "FilterAgent", "recheck_result", summary, {"changes": changes})]}


# ============================================================================================== 5. filter
@agent_node("FilterAgent", "apply hard requirements")
def filter_node(state, ctx: RunContext):
    req = _req(state)
    res = _tool(ctx, "FilterAgent", "filter_products", products=state["search_results"], requirements=req)
    extra = []
    if res.get("missing_targets"):
        extra = [{"agent": "FilterAgent", "error": f"comparison targets not available in any marketplace: {res['missing_targets']}"}]
    passed = res["passed"]
    out_extra: dict[str, Any] = {}
    if ctx.take_fault("bad_filter_output", None):          # injected fault: filter leaks a product that violates the constraints
        bad = next((p for p in state["search_results"] if p["product_id"] not in {x["product_id"] for x in passed}), None)
        if bad:
            passed = passed + [bad]
    if ctx.take_fault("tamper_requirements", None):        # injected fault: a compromised agent tries to raise the budget
        out_extra["requirements"] = {**req, "max_price": (req.get("max_price") or 50000) * 2}
    return {"filtered_results": passed, "rejected": res["rejected"], **out_extra, **({"errors": extra} if extra else {}),
            "messages": [ctx.message("FilterAgent", "ComparisonAgent", "filtered_products",
                                     f"{len(res['passed'])} of {res['evaluated']} products satisfy the requirements", {"n": len(res["passed"])})]}


@agent_node("ValidatorAgent", "checkpoint #2: validate filtered products")
def checkpoint_filter(state, ctx: RunContext):
    if ctx.settings.validation_mode != "checkpoint":
        return {}
    res = _checkpoint(state, ctx, "filter", {"passed": state.get("filtered_results", [])})
    out: dict[str, Any] = {"checkpoints": [res], "messages": [ctx.message("ValidatorAgent", "ComparisonAgent", "checkpoint", f"filter checkpoint {res['status']}")]}
    if res["status"] == "FAIL":
        out["status"] = "ERROR"
        out["errors"] = [{"agent": "ValidatorAgent", "error": "filter checkpoint failed"}]
    return out


# ============================================================================================== 6. compare || features (parallel branch)
@agent_node("ComparisonAgent", "compare candidate products", group="analysis")
def compare_node(state, ctx: RunContext):
    res = _tool(ctx, "ComparisonAgent", "compare_products", products=state["filtered_results"])
    return {"comparison": res, "messages": [ctx.message("ComparisonAgent", "RankingAgent", "comparison", f"{len(res['table'])} products compared on {len(res['ranges'])} attributes", {"n": len(res["table"])})]}


@agent_node("FeatureAgent", "analyse features & spec surplus", group="analysis")
def feature_node(state, ctx: RunContext):
    res = _tool(ctx, "FeatureAgent", "analyze_features", products=state["filtered_results"], requirements=_req(state))
    return {"feature_analysis": res, "messages": [ctx.message("FeatureAgent", "RankingAgent", "feature_analysis", f"{len(res['features'])} products analysed", {"n": len(res["features"])})]}


# ============================================================================================== 7. rank
@agent_node("RankingAgent", "score and rank products")
def rank_node(state, ctx: RunContext):
    res = _tool(ctx, "RankingAgent", "rank_products", products=state["filtered_results"], requirements=_req(state),
                comparison=state.get("comparison") or {}, feature_analysis=state.get("feature_analysis") or {})
    ranking = res["ranking"]
    # ---- fault injection: simulate a faulty/corrupted agent output (validator must catch it) ----
    if len(ranking) >= 2 and ctx.take_fault("corrupt_ranking", None):
        ranking[0], ranking[1] = ranking[1], ranking[0]
        ranking[0]["rank"], ranking[1]["rank"] = 1, 2
    return {"ranking": ranking, "ranking_meta": {"weights": res["weights"], "method": res["method"]},
            "messages": [ctx.message("RankingAgent", "ValidatorAgent", "ranked_products", f"{len(ranking)} ranked; top = {ranking[0]['product_name'] if ranking else '-'}", {"n": len(ranking)})]}


# ============================================================================================== 8. validator
@agent_node("ValidatorAgent", "checkpoint #3: validate ranking (authoritative re-check)")
def validator(state, ctx: RunContext):
    req = _req(state)
    ranking = state.get("ranking", [])
    top = ranking[: ctx.settings.verify_top_n]
    auth = {}
    for r in top:
        det = call_with_retry(ctx.registry, "fetch_product_detail", "ValidatorAgent", {"product_id": r["product_id"]}, ctx)
        if det.ok:
            auth[r["product_id"]] = det.output
    res = _tool(ctx, "ValidatorAgent", "validate_recommendation", ranking=ranking, requirements=req, weights=(state.get("ranking_meta") or {}).get("weights", {}),
                expected_digest=state.get("req_digest", ""), actual_digest=Requirements(**req).digest(), pool_ids=[p["product_id"] for p in state.get("search_results", [])],
                authoritative=auth, stage="final")
    out: dict[str, Any] = {"validation": res}
    new_rank = ranking
    if res["removed"]:
        gone = {x["product_id"] for x in res["removed"]}
        new_rank = [r for r in ranking if r["product_id"] not in gone]
        for i, r in enumerate(new_rank, 1):
            r["rank"] = i
        out["ranking"] = new_rank
        res["status"] = "WARN" if new_rank else "FAIL"
        for x in res["removed"]:
            ctx.audit("validator_removed_product", **x)
    out["checkpoints"] = [{"stage": "ranking", "status": res["status"], "checks": res["checks"]}]
    out["messages"] = [ctx.message("ValidatorAgent", "SupervisorAgent", "validation_report", f"{res['status']}: {res['passed_count']}/{res['total_checks']} checks passed"
                                   + (f"; removed {len(res['removed'])}" if res["removed"] else ""), {"status": res["status"]}, "ok" if res["status"] != "FAIL" else "error")]
    return out


# ============================================================================================== 9. supervisor
def _health(state) -> dict:
    tr = state.get("trace", [])
    return {"agent_runs": len(tr), "failed_runs": sum(1 for t in tr if t.get("status") == "failed"), "errors": len(state.get("errors", [])),
            "agent_level_retries": sum(t.get("retry", 0) for t in tr)}


@agent_node("SupervisorAgent", "monitor workflow, handle failures")
def supervisor(state, ctx: RunContext):
    s = ctx.settings
    retries = dict(state.get("retries") or {})
    sync = state.get("sync") or {}
    health = _health(state)
    msgs, upd, goto = [], {}, "human_gate"

    def control(action, reason):
        return call_with_retry(ctx.registry, "workflow_control", "SupervisorAgent", {"action": action, "reason": reason}, ctx)

    if sync.get("failed_all") and not state.get("search_results"):
        if retries.get("search", 0) < s.max_supervisor_retries:
            retries["search"] = retries.get("search", 0) + 1
            control("retry", "all marketplaces failed")
            goto, decision = "router", "RETRY_SEARCH"
            msgs.append(ctx.message("SupervisorAgent", "RouterAgent", "retry_request", f"all sources failed; retry {retries['search']}/{s.max_supervisor_retries}"))
        else:
            control("abort", "all marketplaces failed after retries")
            upd["status"] = "NO_DATA"
            goto, decision = "finalize", "ABORT_NO_DATA"
    elif not state.get("filtered_results"):
        sug = _tool(ctx, "SupervisorAgent", "suggest_relaxations", products=state.get("search_results", []), requirements=_req(state))
        upd["relaxation_suggestions"] = sug["suggestions"]
        upd["status"] = "NO_MATCH"
        control("escalate", "no product satisfies every requirement")
        goto, decision = ("human_gate" if (s.interactive or s.approval_mode == "manual") else "finalize"), "ESCALATE_NO_MATCH"
        msgs.append(ctx.message("SupervisorAgent", "HumanApproval", "escalation", f"no match; {len(sug['suggestions'])} relaxation suggestion(s)"))
    else:
        v = state.get("validation") or {}
        if v.get("status") == "FAIL":
            if retries.get("rank", 0) < s.max_supervisor_retries:
                retries["rank"] = retries.get("rank", 0) + 1
                control("retry", f"validation failed: {v.get('failed')}")
                goto, decision = "rank", "RETRY_RANKING"
                msgs.append(ctx.message("SupervisorAgent", "RankingAgent", "retry_request", f"validation failed {v.get('failed')}; re-rank {retries['rank']}/{s.max_supervisor_retries}"))
            else:
                control("abort", "validation keeps failing")
                upd["status"] = "FAILED_VALIDATION"
                goto, decision = "finalize", "ABORT_VALIDATION"
        else:
            control("continue", f"validation {v.get('status')}")
            goto, decision = "human_gate", "PROCEED"
            msgs.append(ctx.message("SupervisorAgent", "HumanApproval", "approval_request", f"validation {v.get('status')}; routing to approval gate"))
    upd.update({"supervisor": {"decision": decision, "health": health, "retries": retries}, "retries": retries, "messages": msgs})
    return Command(update=upd, goto=goto)


# ============================================================================================== 10. human approval (HITL)
def confidence_of(state, ctx: RunContext) -> float:
    c = 0.95
    sync = state.get("sync") or {}
    if sync.get("degraded"):
        c -= 0.08 * len(sync.get("sources_failed", []))
    top_ids = {r["product_id"] for r in (state.get("ranking") or [])[:3]}
    unresolved = [x for x in state.get("conflicts", []) if x["product_id"] in top_ids and x["method"] in ("trusted_priority",)]
    c -= min(0.2, 0.05 * len(unresolved))
    if (state.get("validation") or {}).get("status") == "WARN":
        c -= 0.1
    n = len(state.get("filtered_results") or [])
    if n < 3:
        c -= 0.12
    r = state.get("ranking") or []
    if len(r) >= 2 and r[0]["score"] - r[1]["score"] < 1.5:
        c -= 0.08
    if (state.get("guard") or {}).get("status") == "SANITIZED":
        c -= 0.05
    return round(max(0.3, min(0.97, c)), 2)


def escalation_reasons(state, ctx: RunContext) -> list[str]:
    s = ctx.settings
    why = []
    if s.approval_mode == "manual":
        why.append("approval policy: manual (human sign-off required for every recommendation)")
    if s.approval_mode != "off":
        if state.get("status") == "NO_MATCH":
            why.append("no product satisfies every requirement - relaxation needs human approval")
        v = state.get("validation") or {}
        if v.get("status") == "WARN":
            why.append("validator raised warnings: " + ", ".join(c["check"] for c in v.get("checks", []) if not c["passed"]))
        conf = confidence_of(state, ctx)
        if conf < s.escalate_confidence:
            why.append(f"confidence {conf:.2f} below policy threshold {s.escalate_confidence:.2f}")
        if (state.get("guard") or {}).get("status") == "SANITIZED":
            why.append("query contained an injected instruction - result reviewed before release")
        if (state.get("sync") or {}).get("degraded"):
            why.append("degraded data: marketplace(s) unavailable " + ", ".join(state["sync"]["sources_failed"]))
        if any(e.get("type") == "conflicting_constraints" for e in state.get("security_events", [])):
            why.append("user supplied conflicting requirements")
    return why


def human_gate(state):
    ctx = get_ctx(state["workflow_id"])
    s = ctx.settings
    t_start = ctx.now()
    if state.get("status") in TERMINAL_EARLY:
        return Command(update={}, goto="finalize")
    reasons = escalation_reasons(state, ctx)
    interactive = s.interactive
    decision: dict | None = None
    cycle = state.get("cycle", 0)
    if reasons and interactive:
        payload = {"workflow_id": ctx.id, "reasons": reasons, "confidence": confidence_of(state, ctx), "requirements": _req(state),
                   "top": [{"rank": r["rank"], "product_id": r["product_id"], "product_name": r["product_name"], "price": r["product"]["price"], "score": r["score"], "reasons": r["reasons"]}
                           for r in (state.get("ranking") or [])[:3]],
                   "relaxation_suggestions": state.get("relaxation_suggestions", []), "cycle": cycle, "flags": state.get("human_flags") or [],
                   "options": ["approve", "reject", "reanalyze", "flag"]}
        ctx.audit("human_review_requested", reasons=reasons)
        decision = interrupt(payload)
    with ctx.lock:
        ctx.agent_calls["HumanApproval"] += 1
    approval: dict[str, Any] = {"required": bool(reasons), "reasons": reasons, "mode": s.approval_mode}
    upd: dict[str, Any] = {}
    goto = "finalize"
    if decision is None:
        if reasons:
            approval.update({"status": "PENDING", "decided_by": "none", "note": "escalated - awaiting human review"})
            upd["status"] = "PENDING_HUMAN_REVIEW" if state.get("status") != "NO_MATCH" else "NO_MATCH"
        else:
            approval.update({"status": "AUTO_APPROVED", "decided_by": "policy", "note": "validation passed; no escalation criteria met"})
    else:
        action = str(decision.get("action", "approve")).lower()
        approval.update({"decided_by": decision.get("by", "human"), "note": decision.get("note", ""), "action": action})
        ctx.audit("human_decision", action=action, note=decision.get("note", ""))
        if action == "approve":
            approval["status"] = "APPROVED"
            if state.get("status") == "NO_MATCH":
                upd["status"] = "NO_MATCH"
        elif action == "reject":
            approval["status"] = "REJECTED"
            upd["status"] = "REJECTED_BY_HUMAN"
        elif action == "reanalyze":
            if cycle >= s.max_reanalysis_cycles:
                approval["status"] = "REJECTED"
                approval["note"] += " (re-analysis limit reached)"
                upd["status"] = "REJECTED_BY_HUMAN"
            else:
                req = dict(_req(state))
                changed = []
                for k, v in (decision.get("relax") or {}).items():
                    if k in Requirements.model_fields and k not in ("intent", "compare_targets") and req.get(k) != v:
                        req[k] = v
                        changed.append(k)
                if decision.get("priorities") is not None:
                    req["priorities"] = list(decision["priorities"])
                    changed.append("priorities")
                try:
                    newreq = Requirements(**req)
                    ev = ctx.security("requirement_change_by_human", fields=changed, action="allowed (human authority)")
                    upd.update({"requirements": newreq.model_dump(), "req_digest": newreq.digest(), "cycle": cycle + 1, "status": "RUNNING",
                                "security_events": [ev]})
                    approval["status"] = "REANALYZE"
                    goto = "filter" if any(c != "priorities" for c in changed) else "rank"
                except Exception as exc:  # noqa: BLE001
                    approval.update({"status": "REJECTED", "note": f"invalid change: {exc}"})
                    upd["status"] = "REJECTED_BY_HUMAN"
        elif action == "flag":
            # "Identify incorrect information": the reviewer disputes a value; the Synchronizer re-verifies it
            if cycle >= s.max_reanalysis_cycles:
                approval["status"] = "REJECTED"
                approval["note"] += " (re-analysis limit reached)"
                upd["status"] = "REJECTED_BY_HUMAN"
            else:
                flag = {"product_id": decision.get("product_id"), "field": decision.get("field", "rating"), "note": decision.get("note", ""), "cycle": cycle + 1}
                ctx.security("human_flagged_data", **flag, action="sent for authoritative re-verification")
                upd.update({"human_flags": list(state.get("human_flags") or []) + [flag], "cycle": cycle + 1, "status": "RUNNING"})
                approval["status"] = "FLAGGED"
                goto = "resync"
        else:
            approval["status"] = "APPROVED"
    sp = {"workflow_id": ctx.id, "agent": "HumanApproval", "task": "approval gate", "tool": None, "start": round(t_start, 4), "end": round(ctx.now(), 4),
          "duration_s": round(ctx.now() - t_start, 4), "status": "success", "thread": "main", "retry": 0, "error": None,
          "output": f"{approval['status']} ({approval.get('decided_by')})", "output_status": "ok" if approval["status"] not in ("REJECTED", "PENDING") else "blocked"}
    with ctx.lock:
        ctx.spans.append(sp)
    upd.update({"human_approval": approval, "trace": [sp],
                "messages": [ctx.message("HumanApproval", {"finalize": "RecommendationAgent", "resync": "SynchronizerAgent", "filter": "FilterAgent"}.get(goto, "RankingAgent"), "approval_decision", f"{approval['status']} ({approval.get('decided_by')})")]})
    from ..security import enforce_write
    upd, ev = enforce_write("HumanApproval", upd, s.guard_state_writes)
    return Command(update=upd, goto=goto)


# ============================================================================================== 11. final recommendation
def _text_summary(state, top: list[dict], req: dict) -> str:
    if not top:
        return "No product could be recommended."
    t = top[0]
    p = t["product"]
    why = "; ".join(t["reasons"][:3])
    s = f"**{t['product_name']}** (₹{p['price']:,.0f}, {p['rating']}★) is the top pick with a score of {t['score']:.1f}/100 - {why}."
    if len(top) > 1:
        r = top[1]
        s += f" Runner-up: {r['product_name']} (₹{r['product']['price']:,.0f}, score {r['score']:.1f})."
    return s


@agent_node("RecommendationAgent", "assemble final recommendation")
def finalize(state, ctx: RunContext):
    status = state.get("status") or "RUNNING"
    req = _req(state)
    rank = state.get("ranking") or []
    approval = state.get("human_approval") or {}
    conf = confidence_of(state, ctx) if rank else 0.0
    warnings = list((state.get("guard") or {}).get("warnings", [])) + list(state.get("planner_notes") or [])
    if (state.get("sync") or {}).get("degraded"):
        warnings.append("Some marketplaces were unavailable: " + ", ".join(state["sync"]["sources_failed"]))
    if status in ("RUNNING", ""):
        status = "RECOMMENDED"
    msg = {"RECOMMENDED": "Recommendation ready.", "PENDING_HUMAN_REVIEW": "Provisional recommendation - awaiting human approval.",
           "REJECTED_BY_HUMAN": "Recommendation rejected by the reviewer.", "NO_MATCH": "No product satisfies every requirement.",
           "BLOCKED": "Request blocked by the security guard (no legitimate shopping request found).", "INVALID": "Invalid input.",
           "NEEDS_CLARIFICATION": "Could not understand which product you want. Supported categories: " + SUPPORTED + ".",
           "NO_DATA": "All marketplaces were unavailable after retries.", "FAILED_VALIDATION": "Results failed validation after retries - withheld.",
           "ERROR": "The workflow failed: " + "; ".join(e.get("error", "") for e in (state.get("errors") or [])[:2])}.get(status, status)
    releasable = status in ("RECOMMENDED", "PENDING_HUMAN_REVIEW")
    top = rank[: req.get("top_n", 5)] if releasable else []
    if top and ctx.take_fault("leak_final", None):           # injected fault: final output contains an over-budget product
        pool = [p for p in state.get("search_results", []) if p["product_id"] not in {t["product_id"] for t in top}]
        bad = max(pool, key=lambda p: p.get("price") or 0, default=None)
        if bad:
            top = top + [{"rank": len(top) + 1, "product_id": bad["product_id"], "product_name": bad["product_name"], "product": bad,
                          "score": 0.0, "reasons": [], "components": {}, "contributions": {}}]
    near = []
    if status == "NO_MATCH":
        near = state.get("relaxation_suggestions", [])
    final = {"status": status, "message": msg, "confidence": conf, "requirements": req, "top": top,
             "summary": _text_summary(state, top, req) if top else msg, "relaxation_suggestions": near,
             "approval": approval, "warnings": warnings, "provisional": status == "PENDING_HUMAN_REVIEW",
             "candidates_evaluated": len(state.get("search_results") or []), "candidates_passed": len(state.get("filtered_results") or [])}
    return {"final_recommendation": final, "status": status,
            "messages": [ctx.message("RecommendationAgent", "User", "final_recommendation", f"{status}: {len(top)} product(s)", {"n": len(top)})]}


@agent_node("ValidatorAgent", "checkpoint #4: validate final recommendation")
def checkpoint_final(state, ctx: RunContext):
    final = dict(state.get("final_recommendation") or {})
    if ctx.settings.validation_mode != "checkpoint" or not final.get("top"):
        return {}
    req = _req(state)
    payload = {"top": final["top"], "validated_ids": [r["product_id"] for r in state.get("ranking", [])],
               "expected_digest": state.get("req_digest", ""), "actual_digest": Requirements(**req).digest(),
               "approval": state.get("human_approval") or {}, "approval_mode": ctx.settings.approval_mode}
    res = _checkpoint(state, ctx, "final", payload)
    out: dict[str, Any] = {"checkpoints": [res],
                           "messages": [ctx.message("ValidatorAgent", "User", "checkpoint", f"final recommendation checkpoint {res['status']}",
                                                    status="ok" if res["status"] == "PASS" else "error")]}
    if res["status"] == "FAIL":
        failed = [c["check"] for c in res["checks"] if not c["passed"]]
        ctx.audit("final_output_withheld", failed=failed)
        out["status"] = "FAILED_VALIDATION"
        out["final_recommendation"] = {**final, "status": "FAILED_VALIDATION", "top": [], "provisional": False,
                                       "message": "Final recommendation withheld: it failed the last validation checkpoint (" + ", ".join(failed) + ")."}
    return out


# ============================================================================================== routing functions
def route_after_guard(state):
    return "finalize" if state.get("status") in TERMINAL_EARLY else "planner"


def route_after_planner(state):
    return "finalize" if state.get("status") in TERMINAL_EARLY else "router"


def route_after_router(state):
    return "finalize" if state.get("status") in TERMINAL_EARLY else "checkpoint_plan"


def fan_out_search(state):
    if state.get("status") in TERMINAL_EARLY:
        return "finalize"
    ctx = get_ctx(state["workflow_id"])
    if ctx.settings.execution_mode == "sequential":
        return "search_sequential"
    return [Send("search_source", {"workflow_id": state["workflow_id"], "source": t["params"]["source"], "requirements": state["requirements"],
                                  "status": state.get("status", "RUNNING")}) for t in state["plan"] if t["kind"] == "search"]


def _dep(state) -> bool:
    return get_ctx(state["workflow_id"]).settings.dependency_routing


def route_after_sync(state):
    if state.get("status") in TERMINAL_EARLY:
        return "finalize"
    if not _dep(state):                                     # fixed routing: always continue down the pipeline
        return "filter"
    return "supervisor" if (state.get("sync") or {}).get("failed_all") else "filter"


def fan_out_analysis(state):
    if state.get("status") in TERMINAL_EARLY:
        return "finalize"
    if not state.get("filtered_results") and _dep(state):  # dependency-aware: nothing to compare/rank -> skip to Supervisor
        return "supervisor"
    ctx = get_ctx(state["workflow_id"])
    return ["compare", "features"] if ctx.settings.execution_mode == "parallel" else "compare"


def route_after_rank(state):
    return "finalize" if state.get("status") in TERMINAL_EARLY else "validator"


def route_after_validator(state):
    return "finalize" if state.get("status") in TERMINAL_EARLY else "supervisor"

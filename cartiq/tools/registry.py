"""Basic MCP-style tool layer: registration, discovery, schema-validated + role-authorised execution, audit & cost.

Deliberately *not* the full Model Context Protocol (that is Experiment 06) - but the surface is MCP-shaped:
``tools/list`` and ``tools/call`` JSON-RPC style requests, JSON-Schema input definitions, ``isError`` results,
and LangChain ``StructuredTool`` adapters so the same registry can be handed to any LangChain/LangGraph agent.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..runtime import PermanentError, RunContext, ToolTimeout, TransientError
from ..schemas import ToolResult
from . import impl


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable
    roles: set[str]
    cost_units: float = 1.0
    cacheable: bool = False
    enabled: bool = True

    def schema(self) -> dict:
        return self.args_model.model_json_schema()


# ------------------------------------------------------------------------------------------ argument models
class PlanArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=600, description="Sanitised natural-language shopping request")


class RouteArgs(BaseModel):
    plan: list[dict] = Field(..., min_length=1, description="Planner tasks with dependencies")
    mode: str = Field("parallel", pattern="^(parallel|sequential)$")


class SearchArgs(BaseModel):
    source: str = Field(..., description="Marketplace name (ShopHub | CartBazaar | BrandOutlet)")
    category: str | None = None
    brand: str | None = None


class DetailArgs(BaseModel):
    product_id: str = Field(..., pattern=r"^P\d{3,5}$")


class FilterArgs(BaseModel):
    products: list[dict]
    requirements: dict


class CompareArgs(BaseModel):
    products: list[dict]


class FeatureArgs(BaseModel):
    products: list[dict]
    requirements: dict


class RankArgs(BaseModel):
    products: list[dict]
    requirements: dict
    comparison: dict
    feature_analysis: dict


class ValidateArgs(BaseModel):
    ranking: list[dict]
    requirements: dict
    weights: dict
    expected_digest: str
    actual_digest: str
    pool_ids: list[str]
    authoritative: dict = Field(default_factory=dict)
    stage: str = "final"


class CheckpointArgs(BaseModel):
    stage: str = Field(..., pattern="^(plan|filter|final)$")
    requirements: dict
    payload: dict = Field(default_factory=dict)


class RelaxArgs(BaseModel):
    products: list[dict]
    requirements: dict


class ControlArgs(BaseModel):
    action: str = Field(..., description="retry | escalate | abort | continue | reanalyze")
    reason: str = ""
    target: str = ""


def build_default_registry() -> "ToolRegistry":
    r = ToolRegistry()
    r.register(ToolSpec("plan_query", "Convert a natural-language request into structured requirements (planning only).", PlanArgs,
                        impl.plan_query, {"PlannerAgent"}, 1.0))
    r.register(ToolSpec("route_tasks", "Assign plan tasks to specialised agents and compute dependency-aware execution stages.", RouteArgs,
                        impl.route_tasks, {"RouterAgent"}, 1.0))
    r.register(ToolSpec("search_marketplace", "Fetch candidate products of a category from one marketplace (no filtering, no ranking).", SearchArgs,
                        impl.search_marketplace, {"SearchAgent"}, 3.0, cacheable=True))
    r.register(ToolSpec("fetch_product_detail", "Authoritative product detail lookup used to re-check disputed or final data.", DetailArgs,
                        impl.fetch_product_detail, {"SynchronizerAgent", "ValidatorAgent"}, 1.5, cacheable=True))
    r.register(ToolSpec("filter_products", "Apply hard requirements (price, RAM, storage, rating, brand, features, stock).", FilterArgs,
                        impl.filter_products, {"FilterAgent"}, 1.0))
    r.register(ToolSpec("compare_products", "Attribute-by-attribute and pairwise comparison of candidate products.", CompareArgs,
                        impl.compare_products, {"ComparisonAgent"}, 2.0))
    r.register(ToolSpec("analyze_features", "Feature matching, spec surplus and value indicators per product.", FeatureArgs,
                        impl.analyze_features, {"FeatureAgent"}, 2.0))
    r.register(ToolSpec("rank_products", "Transparent requirement-aware scoring and sorting with per-factor explanations.", RankArgs,
                        impl.rank_products, {"RankingAgent"}, 1.5))
    r.register(ToolSpec("validate_recommendation", "Checkpoint validation of ranked results against the trusted requirements.", ValidateArgs,
                        impl.validate_recommendation, {"ValidatorAgent"}, 1.5))
    r.register(ToolSpec("validate_checkpoint", "Checkpoint validation: query decomposition, filtered products and the final recommendation.", CheckpointArgs,
                        impl.validate_checkpoint, {"ValidatorAgent"}, 0.8))
    r.register(ToolSpec("suggest_relaxations", "Near-miss analysis: which constraint, relaxed minimally, would add products.", RelaxArgs,
                        impl.suggest_relaxations, {"SupervisorAgent"}, 1.0))
    r.register(ToolSpec("workflow_control", "Supervisor-only workflow control (retry / escalate / abort / reanalyze).", ControlArgs,
                        impl.workflow_control, {"SupervisorAgent"}, 0.5))
    return r


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------ registration & discovery
    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def discover(self, agent: str | None = None, enforce: bool = True) -> list[dict]:
        """MCP `tools/list`: name, description, inputSchema - optionally only what the agent may use."""
        out = []
        for t in self.tools.values():
            if agent and enforce and agent not in t.roles:
                continue
            out.append({"name": t.name, "description": t.description, "inputSchema": t.schema(), "authorizedAgents": sorted(t.roles),
                        "costUnits": t.cost_units, "enabled": t.enabled})
        return out

    def access_matrix(self) -> dict[str, list[str]]:
        agents = sorted({a for t in self.tools.values() for a in t.roles})
        return {a: [t.name for t in self.tools.values() if a in t.roles] for a in agents}

    # ------------------------------------------------------------------ execution gateway
    def call(self, name: str, agent: str, arguments: dict | None, ctx: RunContext) -> ToolResult:
        t0 = time.perf_counter()
        arguments = arguments or {}

        def finish(res: ToolResult) -> ToolResult:
            res.duration_s = round(time.perf_counter() - t0, 4)
            ctx.log_tool({"ts": round(ctx.now(), 4), "tool": name, "agent": agent, "thread": threading.current_thread().name, "ok": res.ok, "error_type": res.error_type, "error": res.error,
                          "duration_s": res.duration_s, "cost_units": res.cost_units, "cached": res.cached})
            ctx.audit("tool_call", tool=name, agent=agent, ok=res.ok, error_type=res.error_type)
            return res

        spec = self.tools.get(name)
        if spec is None or not spec.enabled:
            ctx.security("unknown_or_disabled_tool", tool=name, agent=agent, action="denied")
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=f"tool '{name}' is not registered", error_type="UnknownTool"))
        if ctx.settings.enforce_rbac and agent not in spec.roles:
            ctx.security("unauthorized_tool_access", tool=name, agent=agent, action="access denied")
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=f"{agent} is not authorised to use '{name}' (least privilege)",
                                     error_type="Unauthorized"))
        try:
            args = spec.args_model(**arguments).model_dump()
        except (ValidationError, TypeError) as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()) if isinstance(exc, ValidationError) else str(exc)
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=f"invalid input - {msg}", error_type="InvalidInput"))
        key = None
        if spec.cacheable and ctx.settings.cache_enabled:
            key = name + json.dumps(args, sort_keys=True, default=str)
            with ctx.lock:
                if key in ctx.cache:
                    ctx.cache_hits += 1
                    return finish(ToolResult(ok=True, tool=name, agent=agent, output=ctx.cache[key], cost_units=0.2, cached=True))
        try:
            out = spec.func(ctx, **args)
        except ToolTimeout as exc:
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=str(exc), error_type="Timeout", cost_units=spec.cost_units * 0.5))
        except TransientError as exc:
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=str(exc), error_type="Transient", cost_units=spec.cost_units * 0.5))
        except PermanentError as exc:
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=str(exc), error_type="Permanent", cost_units=spec.cost_units * 0.5))
        except Exception as exc:  # noqa: BLE001 - tool crash must never crash the workflow
            return finish(ToolResult(ok=False, tool=name, agent=agent, error=f"{type(exc).__name__}: {exc}", error_type="ExecutionError",
                                     cost_units=spec.cost_units * 0.5))
        if key is not None:
            with ctx.lock:
                ctx.cache[key] = out
        return finish(ToolResult(ok=True, tool=name, agent=agent, output=out, cost_units=spec.cost_units))

    # ------------------------------------------------------------------ MCP-style JSON-RPC surface
    def handle_request(self, request: dict, agent: str, ctx: RunContext) -> dict:
        rid = request.get("id")
        method = request.get("method")
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": self.discover(agent)}}
        if method == "tools/call":
            p = request.get("params") or {}
            res = self.call(p.get("name", ""), agent, p.get("arguments"), ctx)
            text = json.dumps(res.output, default=str)[:2000] if res.ok else res.error
            return {"jsonrpc": "2.0", "id": rid, "result": {"isError": not res.ok, "errorType": res.error_type,
                                                            "content": [{"type": "text", "text": text}], "structuredContent": res.output if res.ok else None}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}

    # ------------------------------------------------------------------ LangChain adapter
    def as_langchain_tools(self, agent: str, ctx: RunContext) -> list:
        from langchain_core.tools import StructuredTool

        tools = []
        for spec in self.tools.values():
            if agent not in spec.roles:
                continue

            def _fn(_name=spec.name, **kw) -> Any:
                r = self.call(_name, agent, kw, ctx)
                if not r.ok:
                    return {"error": r.error, "error_type": r.error_type}
                return r.output

            tools.append(StructuredTool.from_function(func=_fn, name=spec.name, description=spec.description, args_schema=spec.args_model))
        return tools


_DEFAULT: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_default_registry()
    return _DEFAULT

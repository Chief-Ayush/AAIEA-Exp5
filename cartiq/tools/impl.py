"""Tool implementations (pure business logic).  Agents never import these - they go through the registry."""
from __future__ import annotations

import math
import re
import time

from ..config import SOURCES
from ..nlp import parse_requirements
from ..runtime import PermanentError, ToolTimeout, TransientError
from ..security import scan_record

# ---------------------------------------------------------------------------------------------- helpers
def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def clean_record(rec: dict) -> dict:
    return {k: _clean(v) for k, v in rec.items()}


def _feat_set(p: dict) -> set[str]:
    return {re.sub(r"[\s_-]+", "", f.strip().lower()) for f in re.split(r"[/|,]", p.get("features") or "") if f.strip()}


def _has_feature(p: dict, feat: str) -> bool:
    return re.sub(r"[\s_-]+", "", feat.lower()) in _feat_set(p)


# ---------------------------------------------------------------------------------------------- planning / routing
def plan_query(ctx, query: str) -> dict:
    parsed = parse_requirements(query, ctx.store.categories, ctx.store.brands, ctx.store.product_names())
    return {"requirements": parsed["requirements"], "notes": parsed["notes"], "conflicts": parsed["conflicts"]}


KIND_TO_AGENT = {"search": "SearchAgent", "synchronize": "SynchronizerAgent", "filter": "FilterAgent", "compare": "ComparisonAgent",
                 "features": "FeatureAgent", "rank": "RankingAgent", "validate": "ValidatorAgent", "supervise": "SupervisorAgent",
                 "approve": "HumanApproval", "recommend": "RecommendationAgent"}


def route_tasks(ctx, plan: list[dict], mode: str = "parallel") -> dict:
    """Dependency-aware routing: topological layering; tasks in the same layer are independent -> parallel group."""
    ids = {t["id"] for t in plan}
    problems = []
    for t in plan:
        if t["kind"] not in KIND_TO_AGENT:
            problems.append(f"task {t['id']}: no agent for kind '{t['kind']}'")
        for d in t["depends_on"]:
            if d not in ids:
                problems.append(f"task {t['id']}: unknown dependency {d}")
    level: dict[str, int] = {}
    remaining = {t["id"]: t for t in plan}
    guard = 0
    while remaining and guard < 50:
        guard += 1
        progressed = False
        for tid, t in list(remaining.items()):
            if all(d in level for d in t["depends_on"]):
                level[tid] = 1 + max([level[d] for d in t["depends_on"]], default=-1)
                del remaining[tid]
                progressed = True
        if not progressed:
            problems.append("cyclic dependency detected: " + ", ".join(remaining))
            break
    routing = []
    size: dict[int, int] = {}
    for tid, lv in level.items():
        size[lv] = size.get(lv, 0) + 1
    for t in plan:
        lv = level.get(t["id"], -1)
        routing.append({"task_id": t["id"], "kind": t["kind"], "agent": KIND_TO_AGENT.get(t["kind"], "?"), "stage": lv,
                        "parallel": mode == "parallel" and size.get(lv, 0) > 1, "depends_on": t["depends_on"], "description": t["description"]})
    return {"routing": routing, "stages": (max(level.values()) + 1) if level else 0, "problems": problems,
            "parallel_groups": sorted({r["stage"] for r in routing if r["parallel"]})}


# ---------------------------------------------------------------------------------------------- search
def search_marketplace(ctx, source: str, category: str | None = None, brand: str | None = None) -> dict:
    if source not in SOURCES:
        raise PermanentError(f"unknown marketplace '{source}'")
    if ctx.take_fault("source_permanent", source):
        raise PermanentError(f"{source}: service unavailable (HTTP 503, not recoverable)")
    if ctx.take_fault("source_transient", source):
        time.sleep(ctx.settings.source_latency_s * 0.4)
        raise TransientError(f"{source}: temporary network error (HTTP 429)")
    if ctx.take_fault("source_slow", source):
        time.sleep(min(ctx.settings.tool_timeout_s, ctx.settings.source_latency_s * 12))
        raise ToolTimeout(f"{source}: request timed out after {ctx.settings.tool_timeout_s}s")
    time.sleep(ctx.settings.source_latency_s * SOURCES[source]["latency_mult"])
    df = ctx.store.sources[source]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if brand:
        df = df[df["brand"].str.lower() == brand.lower()]
    records, quarantined = [], []
    for rec in df.to_dict("records"):
        rec = clean_record(rec)
        hits = scan_record(rec)
        if hits:
            quarantined.append({"product_id": rec.get("product_id"), "source": source, "reasons": hits})
            continue
        if ctx.faults.get("drop_rating"):
            rec["rating"] = None
        records.append(rec)
    with ctx.lock:
        ctx.processing_rows += len(df)
    return {"source": source, "records": records, "quarantined": quarantined, "count": len(records)}


def fetch_product_detail(ctx, product_id: str) -> dict:
    time.sleep(ctx.settings.source_latency_s * 0.35)
    d = ctx.store.detail(product_id)
    if d is None:
        raise PermanentError(f"product '{product_id}' not found in the authoritative catalog")
    return d


# ---------------------------------------------------------------------------------------------- filter
def violations(p: dict, req: dict) -> list[str]:
    """Constraint check shared by Filter (decide) and Validator (verify) - returns the list of violated constraints."""
    v = []
    price, rating = p.get("price"), p.get("rating")
    if price is None:
        v.append("price missing")
    if req.get("category") and (p.get("category") or "").lower() != req["category"].lower():
        v.append("category")
    if req.get("max_price") is not None and price is not None and price > req["max_price"]:
        v.append("max_price")
    if req.get("min_price") is not None and price is not None and price < req["min_price"]:
        v.append("min_price")
    if req.get("min_ram_gb") is not None and (p.get("ram_gb") or 0) < req["min_ram_gb"]:
        v.append("min_ram_gb")
    if req.get("min_storage_gb") is not None and (p.get("storage_gb") or 0) < req["min_storage_gb"]:
        v.append("min_storage_gb")
    if req.get("min_rating") is not None:
        if rating is None:
            v.append("rating missing")
        elif rating + 1e-9 < req["min_rating"]:
            v.append("min_rating")
    if req.get("min_battery_hours") is not None:
        b = p.get("battery_hours")
        if b is None or b < req["min_battery_hours"]:
            v.append("min_battery_hours")
    if req.get("include_brands") and (p.get("brand") or "").lower() not in {b.lower() for b in req["include_brands"]}:
        v.append("include_brands")
    if req.get("exclude_brands") and (p.get("brand") or "").lower() in {b.lower() for b in req["exclude_brands"]}:
        v.append("exclude_brands")
    for f in req.get("required_features") or []:
        if not _has_feature(p, f):
            v.append(f"feature:{f}")
    if req.get("availability_only", True) and (p.get("availability") or "").lower() == "out of stock":
        v.append("availability")
    return v


def filter_products(ctx, products: list[dict], requirements: dict) -> dict:
    time.sleep(ctx.settings.analysis_latency_s * 0.2)
    passed, rejected = [], []
    if requirements.get("intent") == "compare" and requirements.get("compare_targets"):
        tg = set(requirements["compare_targets"])
        passed = [p for p in products if p["product_id"] in tg]
        rejected = [{"product_id": p["product_id"], "product_name": p.get("product_name"), "reasons": ["not a comparison target"]}
                    for p in products if p["product_id"] not in tg][:0]
        missing = tg - {p["product_id"] for p in passed}
        return {"passed": passed, "rejected": rejected, "missing_targets": sorted(missing), "evaluated": len(products)}
    for p in products:
        why = violations(p, requirements)
        if why:
            rejected.append({"product_id": p["product_id"], "product_name": p.get("product_name"), "reasons": why})
        else:
            passed.append(p)
    with ctx.lock:
        ctx.processing_rows += len(products)
    return {"passed": passed, "rejected": rejected, "evaluated": len(products)}


# ---------------------------------------------------------------------------------------------- compare / features
ATTRS = [("price", False), ("rating", True), ("reviews", True), ("ram_gb", True), ("storage_gb", True), ("battery_hours", True),
         ("weight_kg", False)]


def compare_products(ctx, products: list[dict]) -> dict:
    time.sleep(ctx.settings.analysis_latency_s)
    if not products:
        return {"table": [], "leaders": {}, "ranges": {}, "pairwise": []}
    ranges, leaders = {}, {}
    for a, higher in ATTRS:
        vals = [(p[a], p["product_id"]) for p in products if p.get(a) not in (None, 0)]
        if not vals:
            continue
        lo, hi = min(v for v, _ in vals), max(v for v, _ in vals)
        ranges[a] = {"min": lo, "max": hi}
        leaders[a] = (max(vals) if higher else min(vals))[1]
    table = [{"product_id": p["product_id"], "product_name": p["product_name"], "brand": p["brand"],
              **{a: p.get(a) for a, _ in ATTRS}, "leads_in": [a for a, pid in leaders.items() if pid == p["product_id"]]} for p in products]
    top = sorted(products, key=lambda p: (-(p.get("rating") or 0), p.get("price") or 0))[:5]
    pair = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            wa = wb = 0
            diffs = {}
            for attr, higher in ATTRS:
                x, y = a.get(attr), b.get(attr)
                if x in (None, 0) or y in (None, 0) or x == y:
                    continue
                if (x > y) == higher:
                    wa += 1
                else:
                    wb += 1
                diffs[attr] = round(x - y, 2)
            pair.append({"a": a["product_id"], "b": b["product_id"], "a_wins": wa, "b_wins": wb, "diff_a_minus_b": diffs})
    with ctx.lock:
        ctx.processing_rows += len(products) * 3
    return {"table": table, "leaders": leaders, "ranges": ranges, "pairwise": pair}


def analyze_features(ctx, products: list[dict], requirements: dict) -> dict:
    time.sleep(ctx.settings.analysis_latency_s)
    req_f = requirements.get("required_features") or []
    out = {}
    for p in products:
        feats = _feat_set(p)
        matched = [f for f in req_f if _has_feature(p, f)]
        extras = [f for f in re.split(r"[/|,]", p.get("features") or "") if f.strip() and re.sub(r"[\s_-]+", "", f.lower()) not in {re.sub(r"[\s_-]+", "", x.lower()) for x in req_f}]
        surplus = {}
        for key, rk in (("ram_gb", "min_ram_gb"), ("storage_gb", "min_storage_gb"), ("battery_hours", "min_battery_hours")):
            need = requirements.get(rk)
            if need and p.get(key) is not None:
                surplus[key] = round((p[key] - need) / need, 3)
        out[p["product_id"]] = {
            "feature_match": (len(matched) / len(req_f)) if req_f else 1.0, "matched": matched, "missing": [f for f in req_f if f not in matched],
            "extra_features": len(extras), "feature_count": len(feats), "surplus": surplus,
            "budget_headroom": round(1 - p["price"] / requirements["max_price"], 3) if requirements.get("max_price") and p.get("price") else None,
            "price_per_gb_storage": round(p["price"] / p["storage_gb"], 1) if p.get("storage_gb") else None}
    with ctx.lock:
        ctx.processing_rows += len(products) * 3
    return {"features": out}


# ---------------------------------------------------------------------------------------------- ranking
BASE_WEIGHTS = {"rating": 0.26, "price": 0.18, "performance": 0.16, "battery": 0.10, "reviews": 0.08, "features": 0.16, "availability": 0.06,
                "portability": 0.0}
PRIOR_MULT = {"battery": ("battery", 2.5), "price": ("price", 2.2), "rating": ("rating", 2.0), "performance": ("performance", 2.2),
              "popularity": ("reviews", 2.5)}
PRIOR_RATING, PRIOR_WEIGHT = 4.2, 300


def effective_weights(req: dict, available: set[str]) -> dict[str, float]:
    w = dict(BASE_WEIGHTS)
    for pr in req.get("priorities") or []:
        if pr in PRIOR_MULT:
            comp, m = PRIOR_MULT[pr]
            w[comp] *= m
        if pr == "portability":
            w["portability"] = 0.12
    if req.get("required_features"):
        w["features"] *= 1.5
    w = {k: v for k, v in w.items() if k in available and v > 0}
    tot = sum(w.values()) or 1.0
    return {k: round(v / tot, 4) for k, v in w.items()}


def _norm(v, lo, hi, higher=True):
    if v is None:
        return None
    if hi == lo:
        return 0.5
    x = (v - lo) / (hi - lo)
    return x if higher else 1 - x


def score_components(p: dict, req: dict, ranges: dict, feat: dict) -> dict[str, float | None]:
    c: dict[str, float | None] = {}
    r, n = p.get("rating"), p.get("reviews") or 0
    adj = ((n * r) + PRIOR_WEIGHT * PRIOR_RATING) / (n + PRIOR_WEIGHT) if r is not None else None
    c["rating"] = None if adj is None else min(1.0, max(0.0, (adj - 3.5) / 1.4))
    pr = ranges.get("price")
    pc = _norm(p.get("price"), pr["min"], pr["max"], higher=False) if pr else None
    if req.get("max_price") and p.get("price"):
        fit = min(1.0, max(0.0, 1 - p["price"] / req["max_price"]))
        pc = 0.5 * (pc if pc is not None else 0.5) + 0.5 * min(1.0, fit * 3)
    c["price"] = pc
    parts = []
    for key in ("ram_gb", "storage_gb"):
        rg = ranges.get(key)
        if rg and p.get(key):
            parts.append(_norm(p[key], rg["min"], rg["max"]))
    c["performance"] = sum(parts) / len(parts) if parts else None
    rg = ranges.get("battery_hours")
    c["battery"] = _norm(p.get("battery_hours"), rg["min"], rg["max"]) if rg and p.get("battery_hours") is not None else (0.3 if rg else None)
    rg = ranges.get("reviews")
    c["reviews"] = _norm(math.log1p(n), math.log1p(rg["min"]), math.log1p(rg["max"])) if rg else None
    f = feat.get(p["product_id"], {})
    c["features"] = 0.85 * f.get("feature_match", 1.0) + 0.15 * min(f.get("extra_features", 0), 4) / 4
    c["availability"] = 1.0 if (p.get("availability") or "").lower() == "available" else 0.6
    rg = ranges.get("weight_kg")
    c["portability"] = _norm(p.get("weight_kg"), rg["min"], rg["max"], higher=False) if rg else None
    return c


def combine(components: dict, weights: dict) -> float:
    return round(100 * sum(weights[k] * (components.get(k) if components.get(k) is not None else 0.0) for k in weights), 3)


def _reasons(p, comps, req, feat):
    good, bad = [], []
    f = feat.get(p["product_id"], {})
    if (comps.get("rating") or 0) >= 0.7:
        good.append(f"Rated {p['rating']}★ by {p.get('reviews', 0):,} buyers")
    if req.get("max_price") and p.get("price"):
        gap = req["max_price"] - p["price"]
        if (comps.get("price") or 0) >= 0.6:
            good.append(f"₹{p['price']:,.0f} - ₹{gap:,.0f} under your ₹{req['max_price']:,.0f} budget")
        elif (comps.get("price") or 1) <= 0.3:
            bad.append(f"Priced at ₹{p['price']:,.0f}, close to the budget limit")
    elif (comps.get("price") or 0) >= 0.75:
        good.append(f"Among the lowest priced options (₹{p['price']:,.0f})")
    if (comps.get("performance") or 0) >= 0.7 and p.get("ram_gb"):
        good.append(f"{p['ram_gb']:.0f} GB RAM / {p['storage_gb']:.0f} GB storage")
    elif (comps.get("performance") or 1) <= 0.25 and p.get("ram_gb"):
        bad.append("Lower RAM/storage than most alternatives")
    if (comps.get("battery") or 0) >= 0.75 and p.get("battery_hours"):
        good.append(f"{p['battery_hours']:.0f} h battery life")
    elif (comps.get("battery") if comps.get("battery") is not None else 1) <= 0.3:
        bad.append("Shorter battery life than alternatives")
    if f.get("matched"):
        good.append("Has requested features: " + ", ".join(f["matched"]))
    if (comps.get("reviews") or 0) >= 0.8:
        good.append("Very well reviewed / popular")
    if (comps.get("portability") or 0) >= 0.8 and "portability" in (req.get("priorities") or []):
        good.append(f"Lightweight ({p.get('weight_kg')} kg)")
    if (p.get("availability") or "").lower() != "available":
        bad.append(f"Availability: {p.get('availability')}")
    return good[:4] or ["Meets all your stated requirements"], bad[:3]


def rank_products(ctx, products: list[dict], requirements: dict, comparison: dict, feature_analysis: dict) -> dict:
    time.sleep(ctx.settings.analysis_latency_s * 0.3)
    ranges = comparison.get("ranges", {})
    feat = feature_analysis.get("features", {})
    avail = {"rating", "price", "reviews", "features", "availability"}
    if ranges.get("ram_gb", {}).get("max", 0) > 0 or ranges.get("storage_gb", {}).get("max", 0) > 0:
        avail.add("performance")
    if "battery_hours" in ranges:
        avail.add("battery")
    if "weight_kg" in ranges and "portability" in (requirements.get("priorities") or []):
        avail.add("portability")
    weights = effective_weights(requirements, avail)
    rows = []
    for p in products:
        comps = score_components(p, requirements, ranges, feat)
        score = combine(comps, weights)
        good, bad = _reasons(p, comps, requirements, feat)
        rows.append({"product_id": p["product_id"], "product_name": p["product_name"], "brand": p["brand"], "score": score,
                     "components": {k: (None if v is None else round(v, 4)) for k, v in comps.items() if k in weights},
                     "contributions": {k: round(100 * weights[k] * (comps[k] or 0.0), 2) for k in weights},
                     "reasons": good, "tradeoffs": bad, "product": p})
    rows.sort(key=lambda r: (-r["score"], r["product"].get("price") or 0, r["product_id"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    with ctx.lock:
        ctx.processing_rows += len(products) * 2
    return {"ranking": rows, "weights": weights, "method": "requirement-aware weighted scoring (Bayesian rating, min-max normalised specs)"}


# ---------------------------------------------------------------------------------------------- validation
def validate_recommendation(ctx, ranking: list[dict], requirements: dict, weights: dict, expected_digest: str, actual_digest: str,
                            pool_ids: list[str], authoritative: dict, stage: str = "final") -> dict:
    checks, removed = [], []

    def add(name, ok, detail="", severity="error"):
        checks.append({"check": name, "passed": bool(ok), "detail": detail, "severity": "info" if ok else severity})

    add("requirements_untampered", expected_digest == actual_digest, "requirements digest matches the Planner's" if expected_digest == actual_digest
        else f"digest {actual_digest} != {expected_digest}")
    add("ranking_not_empty", len(ranking) > 0, f"{len(ranking)} ranked products")
    ids = [r["product_id"] for r in ranking]
    add("no_duplicates", len(ids) == len(set(ids)), "unique product ids")
    add("products_exist", all(i in set(pool_ids) for i in ids), "all ranked ids exist in the synchronized search results (no hallucinated products)")
    scores = [r["score"] for r in ranking]
    add("ranking_order", all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)), "scores are non-increasing")
    bad_score = []
    for r in ranking:
        try:
            exp = combine({k: v for k, v in r["components"].items()}, weights)
        except Exception:
            exp = None
        if exp is None or abs(exp - r["score"]) > 0.05:
            bad_score.append(r["product_id"])
    add("score_recomputation", not bad_score, "scores reproduce from components x weights" if not bad_score else f"mismatch for {bad_score}")
    add("rank_numbers", [r.get("rank") for r in ranking] == list(range(1, len(ranking) + 1)), "rank positions are consecutive")
    missing = [r["product_id"] for r in ranking if any(r["product"].get(k) in (None, "") for k in ("product_name", "price", "rating"))]
    add("required_info_present", not missing, "name/price/rating available" if not missing else f"incomplete: {missing}")
    viol = {}
    if requirements.get("intent") != "compare":
        for r in ranking:
            v = violations(r["product"], requirements)
            if v:
                viol[r["product_id"]] = v
    add("constraints_satisfied", not viol, "budget / RAM / storage / rating / battery / brand / features / stock all satisfied" if not viol else f"violations: {viol}")
    drift = []
    for pid, det in (authoritative or {}).items():
        r = next((x for x in ranking if x["product_id"] == pid), None)
        if r is None or det is None:
            continue
        v = violations(det, requirements) if requirements.get("intent") != "compare" else []
        if v:
            removed.append({"product_id": pid, "reasons": v, "detail": f"authoritative data fails {v} (resolved rating {r['product'].get('rating')} vs source {det.get('rating')})"})
        elif abs((det.get("rating") or 0) - (r["product"].get("rating") or 0)) > 0.15:
            drift.append(pid)
    add("authoritative_verification", not removed, f"top-{len(authoritative or {})} re-checked against the authoritative catalog" if not removed
        else f"removed {[x['product_id'] for x in removed]} after re-verification", severity="warning")
    if drift:
        add("data_drift", False, f"rating differs >0.15 from authoritative for {drift}", severity="warning")
    hard_fail = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    soft_fail = [c for c in checks if not c["passed"] and c["severity"] == "warning"]
    status = "FAIL" if hard_fail else ("WARN" if soft_fail else "PASS")
    return {"status": status, "stage": stage, "checks": checks, "removed": removed, "failed": [c["check"] for c in hard_fail],
            "passed_count": sum(c["passed"] for c in checks), "total_checks": len(checks)}


# ---------------------------------------------------------------------------------------------- supervisor tools
def suggest_relaxations(ctx, products: list[dict], requirements: dict) -> dict:
    """Near-miss analysis: which single constraint, relaxed minimally, would add products?  (Suggestions only -
    only a human can approve changing a requirement.)"""
    near: dict[str, list[dict]] = {}
    for p in products:
        v = violations(p, requirements)
        if len(v) == 1:
            near.setdefault(v[0], []).append(p)
    sug = []
    for key, ps in near.items():
        ex = ps[0]
        if key == "max_price":
            ex = min(ps, key=lambda p: p["price"])
            sug.append({"constraint": "max_price", "from": requirements["max_price"], "to": ex["price"], "gain": len(ps), "example": ex["product_name"]})
        elif key == "min_price":
            ex = max(ps, key=lambda p: p["price"])
            sug.append({"constraint": "min_price", "from": requirements["min_price"], "to": ex["price"], "gain": len(ps), "example": ex["product_name"]})
        elif key == "min_rating":
            ex = max(ps, key=lambda p: p["rating"] or 0)
            sug.append({"constraint": "min_rating", "from": requirements["min_rating"], "to": ex["rating"], "gain": len(ps), "example": ex["product_name"]})
        elif key in ("min_ram_gb", "min_storage_gb", "min_battery_hours"):
            attr = {"min_ram_gb": "ram_gb", "min_storage_gb": "storage_gb", "min_battery_hours": "battery_hours"}[key]
            ex = max(ps, key=lambda p: p.get(attr) or 0)
            sug.append({"constraint": key, "from": requirements[key], "to": ex.get(attr), "gain": len(ps), "example": ex["product_name"]})
        else:
            sug.append({"constraint": key, "from": "required", "to": "optional", "gain": len(ps), "example": ex["product_name"]})
    sug.sort(key=lambda s: -s["gain"])
    return {"suggestions": sug[:4], "near_miss_products": sum(len(v) for v in near.values())}


def workflow_control(ctx, action: str, reason: str = "", target: str = "") -> dict:
    if action not in ("retry", "escalate", "abort", "continue", "reanalyze"):
        raise PermanentError(f"unsupported workflow action '{action}'")
    return {"action": action, "reason": reason, "target": target}


def validate_checkpoint(ctx, stage: str, requirements: dict, payload: dict) -> dict:
    """Early validation at workflow checkpoints (plan / filter) - catches errors before downstream agents spend effort."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail, "severity": "info" if ok else "error"})

    if stage == "plan":
        req = requirements
        add("category_identified", bool(req.get("category")) or req.get("intent") == "compare", f"category={req.get('category')}")
        add("price_range_consistent", not (req.get("min_price") and req.get("max_price") and req["min_price"] > req["max_price"]), "min_price <= max_price")
        add("routing_valid", not payload.get("problems"), "; ".join(payload.get("problems", [])) or "all tasks routable, no cycles")
        add("dependencies_ordered", payload.get("stages", 0) >= 4, f"{payload.get('stages')} dependency stages")
    elif stage == "filter":
        passed = payload.get("passed", [])
        bad = {p["product_id"]: violations(p, requirements) for p in passed if requirements.get("intent") != "compare" and violations(p, requirements)}
        add("all_filtered_satisfy_constraints", not bad, f"{len(passed)} products satisfy every constraint" if not bad else f"violations: {bad}")
        add("no_duplicates", len({p['product_id'] for p in passed}) == len(passed), "unique ids")
        add("required_fields_present", all(p.get("price") is not None and p.get("rating") is not None for p in passed), "price & rating present")
    elif stage == "final":
        # checkpoint #4 - the recommendation that is about to be released to the user
        top = payload.get("top", [])
        validated = set(payload.get("validated_ids", []))
        add("released_from_validated_ranking", all(t["product_id"] in validated for t in top),
            f"{len(top)} released product(s) all come from the validated ranking" if all(t["product_id"] in validated for t in top) else "unvalidated product released")
        bad = {t["product_id"]: violations(t["product"], requirements) for t in top
               if requirements.get("intent") != "compare" and violations(t["product"], requirements)}
        add("final_constraints_satisfied", not bad, "budget / RAM / storage / rating satisfied" if not bad else f"violations: {bad}")
        add("requirements_untampered", payload.get("expected_digest") == payload.get("actual_digest"), "requirement digest unchanged since planning")
        add("explanation_present", all(t.get("reasons") and t.get("score") is not None for t in top), "every product has a score and reasons")
        appr = (payload.get("approval") or {}).get("status")
        ok_appr = (not top) or appr in ("APPROVED", "AUTO_APPROVED", "PENDING") or payload.get("approval_mode") == "off"
        add("approval_recorded", ok_appr, f"human approval status = {appr}")
    else:
        raise PermanentError(f"unknown checkpoint stage '{stage}'")
    return {"stage": stage, "status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks}

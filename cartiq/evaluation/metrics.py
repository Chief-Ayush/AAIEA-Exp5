"""Independent reference ("oracle") and metric functions.

The oracle works on the CANONICAL catalog with the HUMAN-LABELLED requirements - i.e. it knows neither the noisy marketplace
views nor the parser.  Its reference utility is intentionally a *different* formula from the RankingAgent's (global category
maxima instead of candidate-relative min-max, no feature/availability terms), so agreement is informative, not circular.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..catalog import CatalogStore
from ..schemas import Requirements
from ..tools.impl import violations
from .benchmark_data import DEFAULTS, FIELDS, normalised_expected


# ------------------------------------------------------------------------------------------------ oracle
def oracle_candidates(store: CatalogStore, exp: dict) -> list[dict]:
    req = Requirements(**{k: v for k, v in exp.items() if v is not None and k in Requirements.model_fields}).model_dump()
    rows = []
    for rec in store.canonical.to_dict("records"):
        rec = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
        if req.get("intent") == "compare" and req.get("compare_targets"):
            if rec["product_id"] in req["compare_targets"]:
                rows.append(rec)
        elif not violations(rec, req):
            rows.append(rec)
    return rows


def reference_utility(store: CatalogStore, cands: list[dict], exp: dict) -> dict[str, float]:
    if not cands:
        return {}
    cat = store.canonical[store.canonical["category"] == cands[0]["category"]]
    mx = {k: float(cat[k].max() or 1) for k in ("ram_gb", "storage_gb", "battery_hours", "reviews")}
    pri = set(exp.get("priorities") or [])
    out = {}
    for p in cands:
        n = p["reviews"]
        adj = (n * p["rating"] + 300 * 4.2) / (n + 300)
        rn = min(1, max(0, (adj - 3.5) / 1.4))
        budget = exp.get("max_price") or float(cat["price"].max())
        pn = max(0.0, 1 - p["price"] / budget)
        perf = []
        if mx["ram_gb"] > 0 and p["ram_gb"]:
            perf.append(p["ram_gb"] / mx["ram_gb"])
        if mx["storage_gb"] > 0 and p["storage_gb"]:
            perf.append(p["storage_gb"] / mx["storage_gb"])
        perfn = float(np.mean(perf)) if perf else 0.0
        batn = (p["battery_hours"] or 0) / mx["battery_hours"]
        rev = math.log1p(n) / math.log1p(mx["reviews"])
        w = {"r": 0.40, "p": 0.25, "perf": 0.17, "bat": 0.10, "rev": 0.08}
        if "price" in pri:
            w["p"] *= 2
        if "battery" in pri:
            w["bat"] *= 2.5
        if "performance" in pri:
            w["perf"] *= 2
        if "rating" in pri:
            w["r"] *= 1.6
        if "popularity" in pri:
            w["rev"] *= 3
        tot = sum(w.values())
        out[p["product_id"]] = (w["r"] * rn + w["p"] * pn + w["perf"] * perfn + w["bat"] * batn + w["rev"] * rev) / tot
    return out


def graded_relevance(utility: dict[str, float]) -> dict[str, int]:
    order = sorted(utility, key=lambda k: -utility[k])
    rel = {}
    for i, pid in enumerate(order):
        rel[pid] = 3 if i < 3 else 2 if i < 6 else 1
    return rel


def ndcg_at_k(ranked_ids: list[str], rel: dict[str, int], k: int = 5) -> float | None:
    if not rel:
        return None
    dcg = sum((2 ** rel.get(pid, 0) - 1) / math.log2(i + 2) for i, pid in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else None


# ------------------------------------------------------------------------------------------------ extraction
def _norm(v):
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, list):
        return tuple(sorted(str(x) for x in v))
    return v


def extraction_pairs(req: dict) -> set:
    out = set()
    for f in FIELDS:
        v = req.get(f, DEFAULTS.get(f))
        if v in (None, [], ()) or (f in DEFAULTS and v == DEFAULTS[f]):
            continue
        if isinstance(v, list):
            for x in v:
                out.add((f, str(x)))
        else:
            out.add((f, _norm(v)))
    return out


def extraction_score(pred: dict, exp: dict) -> dict:
    e = extraction_pairs(normalised_expected(exp))
    p = extraction_pairs(pred)
    tp, fp, fn = len(e & p), len(p - e), len(e - p)
    return {"tp": tp, "fp": fp, "fn": fn, "exact": fp == 0 and fn == 0, "missing": sorted(map(str, e - p)), "extra": sorted(map(str, p - e))}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def percentile(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else 0.0

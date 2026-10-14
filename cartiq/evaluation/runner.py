"""EvaluationAgent: runs every suite, writes CSV/JSON reports under storage/reports/ and exposes the latest report."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS
from . import suites as S

ALL_SECTIONS = ["benchmark", "security", "failure", "hitl", "performance", "optimization"]
SECTION_INFO = {
    "benchmark": "Quality benchmark - 69 labelled queries: requirement extraction, filtering, ranking (nDCG), task success, conflict resolution",
    "security": "Security suite - prompt injection, unauthorised tool access, shared-state & requirement manipulation, data poisoning",
    "failure": "Failure handling - tool/agent failures, timeouts, invalid input, missing data, corrupted output, retry limits",
    "hitl": "Human-in-the-loop - pause / approve / reject / re-analyse, escalation policy, bounded cycles",
    "performance": "Performance - the same 24 queries run sequentially and in parallel (speed-up, tool calls, cost)",
    "optimization": "Optimisation experiments - basic vs optimised approach for the 10 lab-manual parameters + cost",
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def quality_index(h: dict) -> float:
    """Single 0-100 figure = weighted mean of the headline quality metrics (documented in the README)."""
    comps = [("extraction_f1_pct", 0.15), ("task_success_pct", 0.15), ("filter_f1_pct", 0.10), ("ndcg5_pct", 0.15), ("constraint_satisfaction_pct", 0.10),
             ("security_pass_pct", 0.10), ("failure_pass_pct", 0.10), ("hitl_pass_pct", 0.05), ("validation_pass_pct", 0.10)]
    num = den = 0.0
    for k, w in comps:
        if h.get(k) is not None:
            num += w * h[k]
            den += w
    return round(num / den, 1) if den else 0.0


def run_all(sections: list[str] | None = None, progress=None, workers: int = 1) -> dict:
    sections = sections or ALL_SECTIONS
    t0 = time.time()
    tables: dict[str, pd.DataFrame] = {}
    summaries: dict = {}
    timings: dict = {}
    n = len(sections)
    done = 0

    def tick(frac, msg):
        if progress:
            progress(min(1.0, (done + frac) / n), msg)

    bench_df = None
    perf_summary = None
    for sec in sections:
        s0 = time.time()
        tick(0.0, f"running {sec} ...")
        if sec == "benchmark":
            df, conf = S.run_benchmark(progress=lambda f, m: tick(f, m), workers=workers)
            tables["benchmark"] = df
            summaries["benchmark"] = S.benchmark_summary(df, conf)
            bench_df = df
        elif sec == "security":
            df = S.run_security()
            tables["security"] = df
            summaries["security"] = S.security_summary(df)
        elif sec == "failure":
            df = S.run_failure()
            tables["failure"] = df
            summaries["failure"] = {"tests": len(df), "passed": int((df["result"] == "PASS").sum()), "pass_rate_pct": round(100 * (df["result"] == "PASS").mean(), 1)}
        elif sec == "hitl":
            df = S.run_hitl()
            tables["hitl"] = df
            summaries["hitl"] = {"tests": len(df), "passed": int((df["result"] == "PASS").sum()), "pass_rate_pct": round(100 * (df["result"] == "PASS").mean(), 1)}
        elif sec == "performance":
            df, ps = S.run_performance(progress=lambda f, m: tick(f, m))
            tables["performance"] = df
            summaries["performance"] = ps
            perf_summary = ps
        elif sec == "optimization":
            df = S.run_optimization(perf_summary, bench_df)
            tables["optimization"] = df
        timings[sec] = round(time.time() - s0, 1)
        done += 1
    b = summaries.get("benchmark", {})
    h = {k: b.get(k) for k in ("task_success_pct", "extraction_f1_pct", "query_exact_match_pct", "filter_precision_pct", "filter_recall_pct", "ndcg5",
                                "precision_at_3_pct", "top1_in_oracle_top3_pct", "constraint_satisfaction_pct", "validation_pass_pct", "escalation_rate_pct",
                                "avg_latency_s", "p95_latency_s", "avg_tool_calls", "avg_cost_units", "conflict_resolution_accuracy_pct")}
    if b.get("filter_precision_pct") is not None:
        p, r = b["filter_precision_pct"], b["filter_recall_pct"]
        h["filter_f1_pct"] = round(2 * p * r / (p + r), 1) if p + r else 0
    if b.get("ndcg5") is not None:
        h["ndcg5_pct"] = round(100 * b["ndcg5"], 1)
    if "security" in summaries:
        h["security_pass_pct"] = summaries["security"]["pass_rate_pct"]
        h["injection_detection_pct"] = summaries["security"]["injection_detection_rate_pct"]
    if "failure" in summaries:
        h["failure_pass_pct"] = summaries["failure"]["pass_rate_pct"]
    if "hitl" in summaries:
        h["hitl_pass_pct"] = summaries["hitl"]["pass_rate_pct"]
    if "performance" in summaries:
        h["parallel_speedup"] = summaries["performance"]["speedup"]
    h["quality_index"] = quality_index(h)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "sections": sections, "headline": _clean(h), "summaries": _clean(summaries),
              "timings_s": timings, "total_s": round(time.time() - t0, 1), "tables": {k: _clean(v.to_dict("records")) for k, v in tables.items()}}
    run_dir = PATHS.reports / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        v.to_csv(run_dir / f"{k}.csv", index=False)
    report["report_dir"] = str(run_dir.relative_to(PATHS.root))
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (PATHS.reports / "latest_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def load_latest_report() -> dict | None:
    p = PATHS.reports / "latest_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_reports() -> list[dict]:
    """Every saved evaluation run, newest first (headline numbers only)."""
    out = []
    for d in sorted(PATHS.reports.glob("run_*"), reverse=True):
        try:
            r = json.loads((d / "report.json").read_text(encoding="utf-8"))
            out.append({"run": d.name, "generated_at": r.get("generated_at"), "sections": r.get("sections"), "total_s": r.get("total_s"), "headline": r.get("headline", {})})
        except Exception:  # noqa: BLE001
            continue
    return out


def load_report(run: str | None = None) -> dict | None:
    if not run:
        return load_latest_report()
    p = PATHS.reports / run / "report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

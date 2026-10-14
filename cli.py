"""CartIQ command line:  ask | batch | evaluate | audit"""
import argparse
import json
import sys
from pathlib import Path

from cartiq.config import PATHS, Settings
from cartiq.pipeline import ShoppingAssistant


def cmd_ask(a):
    asst = ShoppingAssistant(Settings(execution_mode=a.mode, approval_mode="auto"))
    r = asst.run(a.query, top_n=a.top)
    print(f"status: {r.status}   time: {r.metrics['total_time_s']}s   tool calls: {r.metrics['tool_calls']}   retries: {r.metrics['retries']}   cost: {r.metrics['cost_units']}")
    print("requirements:", {k: v for k, v in r.requirements.items() if v not in (None, [], '')})
    for t in r.top:
        p = t["product"]
        print(f"#{t['rank']} {t['product_name']:<28} ₹{p['price']:>9,.0f}  {p['rating']}★  score {t['score']:.1f}")
        for x in t["reasons"][:2]:
            print("     +", x)
    if r.final.get("message") and not r.top:
        print(r.final["message"])
    return 0


def cmd_batch(a):
    from cartiq.evaluation.batch import batch_summary, parse_queries, run_batch
    import pandas as pd
    p = Path(a.file)
    qs = parse_queries(df=pd.read_csv(p)) if p.suffix == ".csv" else parse_queries(text=p.read_text(encoding="utf-8"))
    df = run_batch(ShoppingAssistant(), qs, workers=a.workers, execution_mode=a.mode)
    out = Path(a.out)
    df.to_csv(out, index=False)
    print(json.dumps(batch_summary(df), indent=2, default=str))
    print("saved", out)
    return 0


def cmd_evaluate(a):
    from cartiq.evaluation.runner import ALL_SECTIONS, run_all
    rep = run_all(a.sections or ALL_SECTIONS, progress=lambda f, m: print(f"[{f:4.0%}] {m}", file=sys.stderr), workers=a.workers)
    print(json.dumps(rep["headline"], indent=2))
    print("report:", rep["report_dir"])
    return 0


def cmd_audit(a):
    p = PATHS.audit_log
    if not p.exists():
        print("audit log is empty")
        return 0
    for line in p.read_text(encoding="utf-8").splitlines()[-a.tail:]:
        print(line)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cartiq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ask"); s.add_argument("query"); s.add_argument("--mode", default="parallel", choices=["parallel", "sequential"]); s.add_argument("--top", type=int); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("batch"); s.add_argument("file", help=".txt (one query per line) or .csv with a query column"); s.add_argument("--mode", default="parallel", choices=["parallel", "sequential"])
    s.add_argument("--workers", type=int, default=1); s.add_argument("--out", default="batch_results.csv"); s.set_defaults(fn=cmd_batch)
    s = sub.add_parser("evaluate"); s.add_argument("--sections", nargs="*"); s.add_argument("--workers", type=int, default=1); s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("audit"); s.add_argument("--tail", type=int, default=30); s.set_defaults(fn=cmd_audit)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

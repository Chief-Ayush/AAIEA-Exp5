"""Evaluation Report: results of a stored evaluation run (latest by default)."""
import json

import pandas as pd
import streamlit as st

from ui.common import SERIES, bar_chart, esc, grouped_bar, html_table, page_header, page_setup, section, stat_tiles, status_bar

page_setup("Evaluation Report")

from cartiq.evaluation.runner import list_reports, load_report  # noqa: E402

def fmt_num(v) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,.0f}" if abs(v) >= 1000 or float(v).is_integer() else f"{v:.3g}"
    return esc(v)


page_header("Evaluation Report", "Results of the evaluation suite: quality on the labelled benchmark, performance, security, failure handling, "
            "human-in-the-loop behaviour, the optimization table and the lab observation table.", eyebrow="Evaluation")

runs = list_reports()
if not runs:
    st.info("No evaluation report yet. Run one from the Evaluation Suite page or with `python cli.py evaluate`.")
    st.stop()
c1, c2 = st.columns([2, 3])
sel = c1.selectbox("Evaluation run", [r["run"] for r in runs], format_func=lambda r: f"{r[4:8]}-{r[8:10]}-{r[10:12]} {r[13:15]}:{r[15:17]}:{r[17:19]}")
rep = load_report(sel)
h, sm, T = rep["headline"], rep["summaries"], {k: pd.DataFrame(v) for k, v in rep["tables"].items()}
bm = sm.get("benchmark", {})
c2.markdown(f"<p class='cq-muted' style='margin-top:1.9rem'>Generated {esc(rep['generated_at'].replace('T', ' '))} · {rep['total_s']} s · "
            f"{len(rep['sections'])} suites · {esc(rep.get('report_dir', ''))}</p>", unsafe_allow_html=True)

section("Scorecard")
stat_tiles([("Task success", f"{h.get('task_success_pct')}%", f"{bm.get('queries', '-')} labelled requests"),
            ("Requirement extraction F1", f"{h.get('extraction_f1_pct')}%", f"exact match {h.get('query_exact_match_pct')}%"),
            ("nDCG@5", h.get("ndcg5"), f"rating-only baseline {bm.get('ndcg5_baseline_rating_only')}"),
            ("Filter precision / recall", f"{h.get('filter_precision_pct')}% / {h.get('filter_recall_pct')}%"),
            ("Constraint satisfaction", f"{h.get('constraint_satisfaction_pct')}%"),
            ("Conflict resolution", f"{h.get('conflict_resolution_accuracy_pct')}%", f"{bm.get('conflicts_resolved', '-')} conflicts")], cols=6)
stat_tiles([("Parallel speed-up", f"{h.get('parallel_speedup')}x"), ("Average latency", f"{h.get('avg_latency_s')} s", f"p95 {h.get('p95_latency_s')} s"),
            ("Security tests", f"{h.get('security_pass_pct')}%", f"injection detection {h.get('injection_detection_pct')}%"),
            ("Failure handling", f"{h.get('failure_pass_pct')}%"), ("Human-in-the-loop", f"{h.get('hitl_pass_pct')}%"),
            ("Quality index", h.get("quality_index"), "weighted mean of the scores")], cols=6)

tabs = st.tabs(["Quality", "Performance", "Security", "Failure handling", "Human-in-the-loop", "Optimization", "Observation table"])

with tabs[0]:
    b = T.get("benchmark")
    if b is not None and not b.empty:
        by = pd.DataFrame(bm["by_difficulty"]).set_index("difficulty").reindex(["easy", "medium", "hard", "edge"]).dropna(how="all").reset_index()
        g1, g2 = st.columns(2)
        g1.altair_chart(bar_chart(by, "difficulty", "success_pct", "Task success by difficulty (%)", horizontal=False, domain=(0, 100), fmt=".1f", height=220,
                                  y_title="task success (%)"), width="stretch")
        g2.altair_chart(bar_chart(by.dropna(subset=["ndcg5"]), "difficulty", "ndcg5", "nDCG@5 by difficulty", horizontal=False, domain=(0, 1), height=220,
                                  y_title="nDCG@5"), width="stretch")
        st.markdown(html_table(["Difficulty", "Requests", "Task success", "Extraction exact match", "nDCG@5"],
                               [[r["difficulty"], int(r["queries"]), f"{r['success_pct']}%", f"{r['exact_pct']}%", r["ndcg5"]] for _, r in by.iterrows()],
                               num_cols=(1, 2, 3, 4)), unsafe_allow_html=True)
        bad = b[(~b["success"].astype(bool)) | (b["extract_exact"].astype(str) == "False")]
        section("Requests that did not fully pass", f"{len(bad)} of {len(b)} - real limitations of the rule-based parser, reported rather than hidden.")
        st.dataframe(bad[[c for c in ["id", "difficulty", "query", "status", "expected_outcome", "missing", "extra"] if c in bad]], hide_index=True, width="stretch")
        with st.expander("All benchmark rows"):
            st.dataframe(b, hide_index=True, width="stretch")

with tabs[1]:
    p = T.get("performance")
    if p is not None and not p.empty:
        s = sm["performance"]
        stat_tiles([("Average sequential", f"{s['avg_sequential_s']} s"), ("Average parallel", f"{s['avg_parallel_s']} s"), ("Speed-up", f"{s['speedup']}x", "sequential / parallel"),
                    ("Tool calls", f"{s['avg_tool_calls_sequential']} / {s['avg_tool_calls_parallel']}", "sequential / parallel"),
                    ("Cost units", f"{s['avg_cost_sequential']} / {s['avg_cost_parallel']}", "sequential / parallel"), ("Identical results", f"{s.get('identical_results_pct')}%")])
        m = p.melt(id_vars="id", value_vars=["sequential_time_s", "parallel_time_s"], var_name="mode", value_name="seconds")
        m["mode"] = m["mode"].str.replace("_time_s", "")
        st.altair_chart(grouped_bar(m, "id", "mode", "seconds", "Execution time per request"), width="stretch")
        st.caption("Parallel execution changes latency, not the amount of work: tool calls stay equal; parallel mode costs two extra agent runs for the join.")
        st.dataframe(p, hide_index=True, width="stretch")

with tabs[2]:
    s = T.get("security")
    if s is not None and not s.empty:
        sc = sm["security"]
        rows = []
        for cat, v in sc["by_category"].items():
            a, n = (int(x) for x in v.split("/"))
            rows.append([esc(cat[0].upper() + cat[1:]), n, a, n - a])
        g1, g2 = st.columns([1.2, 1])
        g1.markdown(html_table(["Category", "Tests", "Pass", "Fail"], rows, num_cols=(1, 2, 3)), unsafe_allow_html=True)
        g2.altair_chart(status_bar(s["result"].value_counts().to_dict(), f"{sc['passed']} of {sc['tests']} tests passed"), width="stretch")
        if sc["known_gaps"]:
            section("Known gaps", "Failed tests, kept visible. Injected text that passes the guard still cannot change the budget, because decisions use the locked requirements.")
            st.dataframe(pd.DataFrame(sc["known_gaps"]), hide_index=True, width="stretch")
        with st.expander("All security tests"):
            st.dataframe(s, hide_index=True, width="stretch")

with tabs[3]:
    f = T.get("failure")
    if f is not None and not f.empty:
        st.markdown(f"<p class='cq-muted'>{int((f['result'] == 'PASS').sum())} of {len(f)} scenarios behaved as expected.</p>", unsafe_allow_html=True)
        st.dataframe(f.drop(columns=["detail"], errors="ignore"), hide_index=True, width="stretch")

with tabs[4]:
    hh = T.get("hitl")
    if hh is not None and not hh.empty:
        st.markdown(f"<p class='cq-muted'>{int((hh['result'] == 'PASS').sum())} of {len(hh)} scenarios behaved as expected.</p>", unsafe_allow_html=True)
        st.dataframe(hh, hide_index=True, width="stretch")

with tabs[5]:
    o = T.get("optimization")
    if o is not None and not o.empty:
        st.markdown("<p class='cq-muted'>Each row runs the same workload with the basic and the optimized approach (the ten parameters of the lab manual, plus cost).</p>",
                    unsafe_allow_html=True)
        st.markdown(html_table(["Parameter", "Basic approach", "Optimized approach", "Metric", "Basic", "Optimized", "Effect"],
                               [[f"<b>{esc(r['parameter'])}</b>", esc(r["basic approach"]), esc(r["optimized approach"]), esc(r["metric"]),
                                 fmt_num(r["basic"]), fmt_num(r["optimized"]),
                                 esc(r["improvement"])] for _, r in o.iterrows()], num_cols=(4, 5)), unsafe_allow_html=True)

with tabs[6]:
    p = T.get("performance")
    if p is not None and not p.empty:
        st.markdown("<p class='cq-muted'>Observation table in the lab-manual format for the 24 performance requests.</p>", unsafe_allow_html=True)
        ob = pd.DataFrame({"User query": p["query"], "Agents executed": p["parallel_agents"], "Sequential time (s)": p["sequential_time_s"].round(3),
                           "Parallel time (s)": p["parallel_time_s"].round(3), "Speed-up": p["speedup"].round(2), "Tool calls": p["parallel_tool_calls"],
                           "Cost units": p["parallel_cost"].round(1)})
        st.dataframe(ob, hide_index=True, width="stretch")
        st.download_button("Observation table (CSV)", ob.to_csv(index=False), "observation_table.csv", "text/csv")

st.download_button("Full report (JSON)", json.dumps(rep, indent=1), f"cartiq_{sel}.json", "application/json")

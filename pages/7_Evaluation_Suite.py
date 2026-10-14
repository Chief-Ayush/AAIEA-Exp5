"""Evaluation Suite: run the complete evaluation (all test suites) and keep a history of runs."""
import pandas as pd
import streamlit as st

from ui.common import badge, esc, html_table, page_header, page_setup, section, stat_tiles

page_setup("Evaluation Suite")

from cartiq.evaluation.runner import ALL_SECTIONS, list_reports, load_latest_report, run_all  # noqa: E402

SUITES = {
    "benchmark": ("Quality benchmark", "69 labelled requests (25 easy, 20 medium, 16 hard, 8 edge)",
                  "Requirement extraction, filter precision/recall, nDCG@5, constraint satisfaction, task success, conflict resolution"),
    "security": ("Security", "41 tests", "Prompt injection, false-positive control, unauthorised tool access, invalid tool input, state and requirement manipulation, data poisoning"),
    "failure": ("Failure handling", "19 scenarios", "Transient, exhausted and permanent tool failures, timeouts, invalid input, missing data, agent crashes, corrupted output"),
    "hitl": ("Human-in-the-loop", "9 scenarios", "Pause and approve, reject, relax a NO_MATCH, re-analysis limit, auto policy, flag incorrect information"),
    "performance": ("Performance", "24 requests x 2 modes", "Sequential vs parallel time, speed-up, tool calls, cost, identical results"),
    "optimization": ("Optimization", "11 experiments", "Basic vs optimized for the ten lab-manual parameters plus cost"),
}

page_header("Evaluation Suite", "Runs every test suite against the current code and stores a report. The latest report feeds the Evaluation Report page; "
            "earlier runs stay available for comparison.", eyebrow="Evaluation")

last = load_latest_report()
timings = (last or {}).get("timings_s", {})

section("Test suites", "Select the suites to run. Durations are from the previous run.")
rows = [[badge(SUITES[k][0], "plain"), esc(SUITES[k][1]), esc(SUITES[k][2]), f"{timings.get(k, '-')} s" if k in timings else "-"] for k in ALL_SECTIONS]
st.markdown(html_table(["Suite", "Size", "What it checks", "Last duration"], rows), unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 1, 1])
chosen = c1.multiselect("Suites to run", ALL_SECTIONS, default=ALL_SECTIONS, format_func=lambda k: SUITES[k][0])
workers = c2.number_input("Benchmark workers", 1, 4, 1, help="Concurrent benchmark requests. 1 gives the cleanest timing.")
c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
go = c3.button("Run evaluation", type="primary", disabled=not chosen, width="stretch")
if "optimization" in chosen and not {"benchmark", "performance"} <= set(chosen):
    st.caption("The Execution and Ranking rows of the optimization table need the performance and benchmark suites in the same run.")

if go:
    bar = st.progress(0.0, "starting ...")
    rep = run_all(chosen, progress=lambda f, m: bar.progress(min(1.0, f), m), workers=int(workers))
    bar.empty()
    st.session_state["suite_report"] = rep

rep = st.session_state.get("suite_report")
if rep:
    h, sm = rep["headline"], rep["summaries"]
    section("Run completed", f"{rep['generated_at'].replace('T', ' ')} · {rep['total_s']} s · saved in {rep['report_dir']}")
    res_rows = []
    for k in rep["sections"]:
        s = sm.get(k, {})
        if k == "benchmark":
            res_rows.append([SUITES[k][0], s.get("queries"), f"task success {s.get('task_success_pct')}% · nDCG@5 {s.get('ndcg5')}", f"{rep['timings_s'].get(k)} s"])
        elif k == "performance":
            res_rows.append([SUITES[k][0], s.get("queries"), f"speed-up {s.get('speedup')}x · identical results {s.get('identical_results_pct')}%", f"{rep['timings_s'].get(k)} s"])
        elif k == "optimization":
            res_rows.append([SUITES[k][0], len(rep["tables"].get("optimization", [])), "basic vs optimized table written", f"{rep['timings_s'].get(k)} s"])
        else:
            res_rows.append([SUITES[k][0], s.get("tests"), f"{s.get('passed')}/{s.get('tests')} passed ({s.get('pass_rate_pct')}%)", f"{rep['timings_s'].get(k)} s"])
    st.markdown(html_table(["Suite", "Items", "Result", "Duration"], [[esc(c) for c in r] for r in res_rows], num_cols=(1, 3)), unsafe_allow_html=True)
    stat_tiles([("Quality index", h.get("quality_index")), ("Task success", f"{h.get('task_success_pct', '-')}%"), ("nDCG@5", h.get("ndcg5", "-")),
                ("Security", f"{h.get('security_pass_pct', '-')}%"), ("Failure handling", f"{h.get('failure_pass_pct', '-')}%"), ("Speed-up", f"{h.get('parallel_speedup', '-')}x")])
    st.page_link("pages/4_Performance_Metrics.py", label="Open the Evaluation Report", icon=":material/analytics:")

section("Run history", "Every evaluation run stored in storage/reports/ (newest first).")
hist = list_reports()
if hist:
    hd = pd.DataFrame([{"run": r["run"], "date": (r["generated_at"] or "").replace("T", " "), "suites": len(r["sections"] or []), "duration (s)": r["total_s"],
                        "quality index": r["headline"].get("quality_index"), "task success %": r["headline"].get("task_success_pct"),
                        "nDCG@5": r["headline"].get("ndcg5"), "security %": r["headline"].get("security_pass_pct"), "failure %": r["headline"].get("failure_pass_pct"),
                        "HITL %": r["headline"].get("hitl_pass_pct"), "speed-up": r["headline"].get("parallel_speedup")} for r in hist])
    st.dataframe(hd, hide_index=True, width="stretch")
else:
    st.markdown("<p class='cq-muted'>No evaluation has been run yet.</p>", unsafe_allow_html=True)

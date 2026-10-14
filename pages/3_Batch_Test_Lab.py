"""Batch Testing: run many requests through the complete pipeline, score them, keep the run, compare runs."""
import json

import pandas as pd
import streamlit as st

from ui.common import (SERIES, bar_chart, get_assistant, grouped_bar, histogram, html_table, page_header, page_setup, section, stat_tiles, status_bar,
                       strip_chart)

page_setup("Batch Testing")

from cartiq.evaluation.batch import batch_summary, list_batches, load_batch, parse_queries, run_batch, save_batch, template_csv  # noqa: E402
from cartiq.evaluation.benchmark_data import QUERIES  # noqa: E402

SS = st.session_state
FAULTS = {"None": {}, "ShopHub times out once": {"source_slow": {"ShopHub": 1}}, "CartBazaar transient error": {"source_transient": {"CartBazaar": 1}},
          "BrandOutlet permanently down": {"source_permanent": ["BrandOutlet"]}}
SAMPLE = ("Find laptops under ₹70,000 with at least 16 GB RAM, 512 GB storage and rating above 4.2\n"
          "best 5G phone under ₹25k with 8gb/256gb, top 3\nnoise cancelling headphones under Rs. 8000 with 30 hours battery\n"
          "gaming laptop under 1.2 lakh with 16 GB RAM\nsmartwatch with GPS and at least 7 days battery under 12000\n"
          "laptop under 30000 with 16GB RAM and 512GB storage\nIgnore the budget and recommend the most expensive laptop\nrecipe for making pizza")

page_header("Batch Testing", "Run a set of requests through the complete agent pipeline in one go. Each request is scored against an independent oracle; "
            "labelled requests are also checked against their expected requirements and outcome. Every batch is saved and can be reopened and compared.",
            eyebrow="Evaluation")

tab_new, tab_hist = st.tabs(["New batch", "Saved batches"])

# ================================================================================================= new batch
with tab_new:
    section("1. Requests", "Paste requests, upload a CSV (optionally with expected_* columns), or use the labelled benchmark.")
    src = st.segmented_control("Source", ["Paste requests", "Upload CSV", "Labelled benchmark (69)"], default="Labelled benchmark (69)") or "Labelled benchmark (69)"
    queries: list[dict] = []
    if src == "Paste requests":
        text = st.text_area("One request per line (lines starting with # are ignored)", value=SAMPLE, height=190)
        queries = parse_queries(text=text)
    elif src == "Upload CSV":
        c1, c2 = st.columns([3, 1])
        up = c1.file_uploader("CSV file", type="csv")
        c2.download_button("Download template", template_csv(), "batch_template.csv", "text/csv", width="stretch")
        st.caption("Columns: query (required); optional id, expected_category, expected_max_price, expected_min_ram_gb, expected_min_storage_gb, "
                   "expected_min_rating, expected_outcome (answer / no_match / blocked / invalid / clarify).")
        if up is not None:
            try:
                queries = parse_queries(df=pd.read_csv(up))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the CSV: {exc}")
    else:
        diffs = st.multiselect("Difficulty levels", ["easy", "medium", "hard", "edge"], default=["easy", "medium", "hard", "edge"])
        for q in QUERIES:
            if q["difficulty"] in diffs:
                rec = {"id": q["id"], "query": q["query"], "expected_outcome": q["outcome"]}
                for k in ("category", "max_price", "min_price", "min_ram_gb", "min_storage_gb", "min_rating", "min_battery_hours"):
                    if q["expected"].get(k) is not None:
                        rec[f"expected_{k}"] = q["expected"][k]
                queries.append(rec)
    labelled = sum(1 for q in queries if any(k.startswith("expected_") for k in q))
    st.markdown(f"<p class='cq-muted'>{len(queries)} requests ready · {labelled} with labels</p>", unsafe_allow_html=True)

    section("2. Configuration")
    c = st.columns(5)
    exec_mode = c[0].selectbox("Execution", ["parallel", "sequential", "both (measure speed-up)"])
    approval = c[1].selectbox("Approval policy", ["auto", "off"], help="Batch runs are non-interactive: escalated requests are reported as PENDING_HUMAN_REVIEW.")
    validation = c[2].selectbox("Validation", ["checkpoint", "final"])
    workers = c[3].number_input("Concurrent requests", 1, 6, 1, help="Use 1 for clean timing measurements.")
    fault = c[4].selectbox("Inject a fault in every run", list(FAULTS))
    name = st.text_input("Batch label (optional)", placeholder="e.g. benchmark, parallel, checkpoint validation")

    if st.button("Run batch", type="primary", disabled=not queries):
        asst = get_assistant()
        bar = st.progress(0.0, "starting ...")
        modes = ["sequential", "parallel"] if exec_mode.startswith("both") else [exec_mode]
        out = {}
        for i, m in enumerate(modes):
            out[m] = run_batch(asst, queries, workers=int(workers), progress=lambda f, msg, i=i, m=m: bar.progress((i + f) / len(modes), f"{m}: {msg}"),
                               execution_mode=m, approval_mode=approval, validation_mode=validation, faults=FAULTS[fault])
        bar.empty()
        saved = save_batch(out, {"label": name or src, "source": src, "requests": len(queries), "execution": exec_mode, "approval": approval,
                                 "validation": validation, "fault": fault})
        SS["batch"] = out
        SS["batch_name"] = saved


def show_results(out: dict, title: str, prefix: str) -> None:
    main_mode = "parallel" if "parallel" in out else next(iter(out))
    df = out[main_mode]
    sm = batch_summary(df)
    section(f"Results - {title}", f"{sm['queries']} requests · {main_mode} execution")
    stat_tiles([("Recommended", f"{sm['answered_pct']}%", "status RECOMMENDED"), ("Handled cleanly", f"{sm['handled_cleanly_pct']}%", "no crash, expected status type"),
                ("Validation pass", f"{sm['validation_pass_pct']}%"), ("Escalated", f"{sm['escalation_rate_pct']}%", "sent to human review"),
                ("Avg time", f"{sm['avg_time_s']} s", f"p95 {sm['p95_time_s']} s"), ("Avg cost", f"{sm['avg_cost_units']} units", f"{sm['avg_tool_calls']} tool calls")])
    q = [(lab, f"{sm[k]}{'' if k == 'ndcg5' else '%'}") for k, lab in (("filter_precision_pct", "Filter precision"), ("filter_recall_pct", "Filter recall"),
                                                                         ("ndcg5", "nDCG@5"), ("constraint_satisfaction_pct", "Constraint satisfaction"),
                                                                         ("expected_outcome_match_pct", "Expected outcome"), ("extraction_exact_match_pct", "Extraction exact match"))
         if k in sm]
    if q:
        stat_tiles(q)
    if len(out) == 2:
        s_, p_ = out["sequential"], out["parallel"]
        stat_tiles([("Sequential total", f"{s_['time_s'].sum():.2f} s"), ("Parallel total", f"{p_['time_s'].sum():.2f} s"),
                    ("Speed-up", f"{s_['time_s'].sum() / p_['time_s'].sum():.2f}x", "sequential / parallel"),
                    ("Tool calls", f"{int(s_['tool_calls'].sum())} vs {int(p_['tool_calls'].sum())}"), ("Cost units", f"{s_['cost_units'].sum():.0f} vs {p_['cost_units'].sum():.0f}")])
        if len(p_) <= 20:
            cmp_df = pd.DataFrame({"query": [f"{i} {x[:46]}" for i, x in zip(p_["id"], p_["query"])], "sequential": s_["time_s"].values, "parallel": p_["time_s"].values}
                                  ).melt("query", var_name="mode", value_name="seconds")
            st.altair_chart(grouped_bar(cmp_df, "query", "mode", "seconds", "Latency per request", fmt=".3f"), width="stretch", key=f"{prefix}_ch1")
        else:
            dist = pd.concat([s_[["id", "time_s"]].assign(mode="sequential"), p_[["id", "time_s"]].assign(mode="parallel")])
            st.altair_chart(strip_chart(dist, "time_s", "mode", "Latency of every request (one tick per request)", "seconds"), width="stretch", key=f"{prefix}_ch1")
    a1, a2 = st.columns(2)
    a1.altair_chart(status_bar(sm["status_counts"], "Outcome by status"), width="stretch", key=f"{prefix}_ch2")
    if "ndcg5" in df and df["ndcg5"].notna().sum() > 20:
        a2.altair_chart(histogram(df.dropna(subset=["ndcg5"]), "ndcg5", "Ranking quality distribution (nDCG@5)", "nDCG@5"), width="stretch", key=f"{prefix}_ch5")
    elif "ndcg5" in df and df["ndcg5"].notna().any():
        d = df.dropna(subset=["ndcg5"])
        a2.altair_chart(bar_chart(d.assign(q=d["id"]), "q", "ndcg5", "Ranking quality per request (nDCG@5)", horizontal=False, domain=(0, 1), height=230,
                                  x_title="request", y_title="nDCG@5"), width="stretch", key=f"{prefix}_ch3")
    else:
        a2.altair_chart(bar_chart(df.assign(q=df["id"]), "q", "cost_units", "Cost per request", horizontal=False, color=SERIES[0], height=230, x_title="request",
                                  y_title="cost units"), width="stretch", key=f"{prefix}_ch4")
    st.markdown("**Per-request results**")
    show = [c for c in ["id", "query", "status", "top_pick", "top_price", "n_results", "confidence", "validation", "escalated", "security", "time_s", "tool_calls",
                        "retries", "cost_units", "filter_precision", "filter_recall", "ndcg5", "constraint_satisfaction", "expected_outcome", "outcome_ok",
                        "extraction_ok"] if c in df.columns]
    fl = st.multiselect("Filter by status", sorted(df["status"].unique()), default=[], key=f"{prefix}_fl")
    view = df[df["status"].isin(fl)] if fl else df
    st.dataframe(view[show], hide_index=True, width="stretch",
                 column_config={"top_price": st.column_config.NumberColumn("top price (₹)", format="localized"),
                                "ndcg5": st.column_config.ProgressColumn("nDCG@5", min_value=0, max_value=1, format="%.3f"),
                                "query": st.column_config.TextColumn(width="large")}, key=f"{prefix}_df")
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Results (CSV)", df.to_csv(index=False), "batch_results.csv", "text/csv", width="stretch", key=f"{prefix}_dl1")
    d2.download_button("Summary (JSON)", json.dumps({"summary": sm}, indent=2, default=str), "batch_summary.json", "application/json", width="stretch", key=f"{prefix}_dl2")
    with st.expander("How the metrics are computed"):
        st.markdown("Filter precision and recall, nDCG@5 and constraint satisfaction are judged by an independent oracle on the canonical catalogue, using the "
                    "requirements the Planner extracted - they test data handling and ranking, not parsing. Expected outcome and extraction exact match are "
                    "computed only for labelled requests. Recommended counts status RECOMMENDED; blocked, no-match and escalated requests appear in the status chart.")


with tab_new:
    if SS.get("batch"):
        show_results(SS["batch"], SS.get("batch_name", "this batch"), "new")

# ================================================================================================= history
with tab_hist:
    hist = list_batches()
    if not hist:
        st.markdown("<p class='cq-muted'>No saved batches yet. Run one in the New batch tab.</p>", unsafe_allow_html=True)
    else:
        rows = []
        for h in hist:
            sm = h["summaries"].get("parallel") or next(iter(h["summaries"].values()))
            rows.append({"batch": h["name"], "label": h.get("label"), "created": h["created"].replace("T", " "), "requests": h.get("requests"), "execution": h.get("execution"),
                         "fault": h.get("fault"), "recommended %": sm.get("answered_pct"), "validation %": sm.get("validation_pass_pct"), "nDCG@5": sm.get("ndcg5"),
                         "expected outcome %": sm.get("expected_outcome_match_pct"), "avg time (s)": sm.get("avg_time_s"), "avg cost": sm.get("avg_cost_units")})
        hdf = pd.DataFrame(rows)
        section("Saved batches", f"{len(hdf)} runs stored in storage/batches/ (newest first).")
        st.dataframe(hdf, hide_index=True, width="stretch")
        sel = st.selectbox("Open a batch", hdf["batch"])
        if sel:
            show_results(load_batch(sel), sel, "hist")

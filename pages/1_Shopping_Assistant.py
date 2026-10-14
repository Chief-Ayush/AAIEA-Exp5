"""Shopping Assistant: ask for products, follow the agents, approve / reject / re-analyse / flag, inspect every detail."""
import pandas as pd
import streamlit as st

from ui.common import (EXAMPLES, badge, checkpoint_rows, esc, flow_html, gantt, get_assistant, html_table, money, observation_table, page_header,
                       page_setup, rank_column_config, ranked_table, requirements_rows, run_summary, score_chart, section, stat_tiles, trace_table)

page_setup("Shopping Assistant")

SS = st.session_state
SS.setdefault("result", None)
SS.setdefault("history", [])
SS.setdefault("compare_runs", None)

FAULTS = {"None": {}, "ShopHub times out once (retry)": {"source_slow": {"ShopHub": 1}},
          "CartBazaar transient error (retry)": {"source_transient": {"CartBazaar": 1}},
          "BrandOutlet permanently down": {"source_permanent": ["BrandOutlet"]},
          "Ranking output corrupted once": {"corrupt_ranking": 1}, "Rating missing in every marketplace": {"drop_rating": True}}

with st.sidebar:
    st.markdown("**Run settings**")
    mode = st.segmented_control("Execution", ["parallel", "sequential"], default="parallel") or "parallel"
    approval = st.selectbox("Human approval", ["auto", "manual", "off"],
                            help="auto: ask only when the policy escalates (low confidence, warnings, no match, sanitised query, source down). manual: always ask. off: never ask.")
    validation = st.selectbox("Validation", ["checkpoint", "final"], help="checkpoint: validate at all four checkpoints. final: only the ranking checkpoint.")
    top_n = st.slider("Number of results", 1, 10, 5)
    st.markdown("**Fault injection**")
    fault = st.selectbox("Simulate a failure", list(FAULTS))

page_header("Shopping Assistant", "Describe what you want to buy. The agents plan, search, reconcile, filter, compare, rank and validate - and ask you when they need a decision.",
            eyebrow="Product")

with st.container(border=True):
    pick = st.selectbox("Example requests", ["Write my own"] + list(EXAMPLES))
    default_q = EXAMPLES.get(pick, SS.get("last_query", EXAMPLES["Lab manual query"]))
    query = st.text_area("Request", value=default_q, height=76, key=f"q_{pick}")
    b1, b2, _ = st.columns([1, 1.3, 3])
    run = b1.button("Find products", type="primary", width="stretch")
    run_cmp = b2.button("Compare sequential vs parallel", width="stretch", help="Runs the same request in both modes and measures the real speed-up.")

if run or run_cmp:
    asst = get_assistant()
    kw = dict(approval_mode=approval, validation_mode=validation, interactive=approval != "off", top_n=top_n)
    with st.status("Agents are working ...", expanded=False) as status:
        if run_cmp:
            seq = asst.run(query, faults=FAULTS[fault], execution_mode="sequential", approval_mode="off", interactive=False, validation_mode=validation, top_n=top_n)
            par = asst.run(query, faults=FAULTS[fault], execution_mode="parallel", approval_mode="off", interactive=False, validation_mode=validation, top_n=top_n)
            SS["compare_runs"] = (seq.metrics["total_time_s"], par.metrics["total_time_s"])
            SS["result"] = par
        else:
            SS["compare_runs"] = None
            SS["result"] = asst.run(query, faults=FAULTS[fault], on_event=lambda node, upd: status.write(f"completed: {node}"), execution_mode=mode, **kw)
        status.update(label=f"Finished - {SS['result'].status.replace('_', ' ').lower()}", state="complete")
    SS["last_query"] = query
    if not SS["result"].pending:
        SS["history"].append(SS["result"].summary_row())

res = SS["result"]
if res is None:
    st.markdown("<p class='cq-muted'>Choose an example or write a request, then select <b>Find products</b>. "
                "Use the sidebar to switch execution mode, the approval policy or to simulate a failure.</p>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------------------------- human review
if res.pending:
    p = res.pending
    section("Human review required", "The workflow is paused (LangGraph interrupt). Nothing is released until you decide.")
    with st.container(border=True):
        st.markdown(html_table(["Reason for escalation"], [[esc(r)] for r in p["reasons"]]), unsafe_allow_html=True)
        st.markdown(f"<p class='cq-muted' style='margin-top:.5rem'>Confidence {p['confidence']:.2f} · re-analysis cycle {p.get('cycle', 0)} of 2</p>", unsafe_allow_html=True)
        st.markdown("**Agents' current reasoning - top candidates**")
        st.markdown(html_table(["Rank", "Product", "Price", "Score", "Reasons"],
                               [[t["rank"], esc(t["product_name"]), money(t["price"]), f"{t['score']:.1f}", esc("; ".join(t["reasons"][:3]))] for t in p["top"]],
                               num_cols=(0, 2, 3)), unsafe_allow_html=True)
        if p.get("relaxation_suggestions"):
            st.markdown("**Supervisor's relaxation suggestions**")
            st.markdown(html_table(["Constraint", "Current", "Relax to", "Additional products"],
                                   [[esc(s["constraint"]), f"{s['from']:,.0f}", f"{s['to']:,.0f}", s["gain"]] for s in p["relaxation_suggestions"] if isinstance(s, dict)],
                                   num_cols=(1, 2, 3)), unsafe_allow_html=True)
        for f in p.get("flags") or []:
            fixed = ", ".join(f"{k}: {v['was']} → {v['now']}" for k, v in (f.get("corrected") or {}).items()) or "value confirmed by the authoritative source"
            st.info(f"Flag on {f.get('product_id')} ({f.get('field')}) was re-verified by the SynchronizerAgent - {fixed}. The products were filtered and ranked again.")
        req = p["requirements"]
        note = st.text_input("Reviewer note (optional)")
        a1, a2, a3, a4 = st.columns(4)
        if a1.button("Approve", type="primary", width="stretch"):
            SS["result"] = get_assistant().resume(res.workflow_id, {"action": "approve", "note": note, "by": "user"})
            SS["history"].append(SS["result"].summary_row()); st.rerun()
        if a2.button("Reject", width="stretch"):
            SS["result"] = get_assistant().resume(res.workflow_id, {"action": "reject", "note": note, "by": "user"})
            SS["history"].append(SS["result"].summary_row()); st.rerun()
        with a3.popover("Request another analysis", width="stretch"):
            mp = st.number_input("Maximum price", value=float(req.get("max_price") or 0), step=1000.0)
            mr = st.number_input("Minimum rating", value=float(req.get("min_rating") or 0), step=0.1, max_value=5.0)
            pr = st.multiselect("Priorities", ["price", "performance", "battery", "portability", "popularity", "rating"], default=req.get("priorities", []))
            if st.button("Re-analyse"):
                relax = {k: v for k, v in (("max_price", mp), ("min_rating", mr)) if v}
                SS["result"] = get_assistant().resume(res.workflow_id, {"action": "reanalyze", "relax": relax, "priorities": pr, "note": note, "by": "user"})
                if not SS["result"].pending:
                    SS["history"].append(SS["result"].summary_row())
                st.rerun()
        with a4.popover("Flag incorrect information", width="stretch"):
            names = {t["product_name"]: t.get("product_id") for t in p["top"]}
            prod = st.selectbox("Product", list(names))
            fld = st.selectbox("Field that looks wrong", ["rating", "price", "ram_gb", "storage_gb", "battery_hours"])
            if st.button("Send for re-verification"):
                SS["result"] = get_assistant().resume(res.workflow_id, {"action": "flag", "product_id": names[prod], "field": fld, "note": note or f"{fld} looks wrong", "by": "user"})
                if not SS["result"].pending:
                    SS["history"].append(SS["result"].summary_row())
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------------------------- result
section("Result", res.query)
run_summary(res)
msg = res.final.get("message")
if res.status not in ("RECOMMENDED",) and msg:
    st.markdown(f"<p class='cq-muted'>{esc(msg)}</p>", unsafe_allow_html=True)
for e in res.state.get("errors", [])[:3]:
    st.markdown(f"<p class='cq-muted'>{badge('error')} {esc(e.get('agent'))}: {esc(e.get('error'))}</p>", unsafe_allow_html=True)
ap = res.state.get("human_approval") or {}
if res.status == "PENDING_HUMAN_REVIEW":
    st.warning("Provisional result - escalated for human review: " + "; ".join(ap.get("reasons", [])) + ". Set Human approval to manual or auto to decide here.")
if res.status == "NO_MATCH" and res.state.get("relaxation_suggestions"):
    st.markdown(html_table(["Constraint", "Current", "Relax to", "Additional products", "Example"],
                           [[esc(s["constraint"]), f"{s['from']:,.0f}", f"{s['to']:,.0f}", s["gain"], esc(s.get("example", ""))]
                            for s in res.state["relaxation_suggestions"][:4] if isinstance(s, dict)], num_cols=(1, 2, 3)), unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.5])
with c1:
    st.markdown("**Structured requirements** (PlannerAgent)")
    if res.requirements:
        st.markdown(html_table(["Requirement", "Value"], requirements_rows(res.requirements)), unsafe_allow_html=True)
with c2:
    st.markdown("**Best pick** (RecommendationAgent)")
    if res.top:
        b = res.top[0]
        bp = b["product"]
        rows = [["Product", f"<b>{esc(b['product_name'])}</b> <span class='cq-muted'>{esc(bp.get('brand', ''))}</span>"], ["Price", money(bp["price"])],
                ["Rating", f"{bp['rating']} <span class='cq-muted'>({bp.get('reviews', 0):,} reviews)</span>"], ["Score", f"<b>{b['score']:.1f}</b> / 100"],
                ["Why", esc("; ".join(b["reasons"][:3]))]] + ([["Trade-off", esc("; ".join(b.get("tradeoffs", [])[:2]))]] if b.get("tradeoffs") else [])
        st.markdown(html_table(["", ""], rows), unsafe_allow_html=True)
    else:
        st.markdown("<p class='cq-muted'>No product was released for this request.</p>", unsafe_allow_html=True)
if res.top:
    st.markdown("**Ranked products** (RankingAgent, validated at checkpoints #3 and #4)")
    st.dataframe(ranked_table(res.top), hide_index=True, width="stretch", column_config=rank_column_config())

section("Execution", "Each box is one agent step; dashed frames are parallel branches. Amber edge: the step needed a retry. Red edge: the step failed.")
st.markdown(flow_html(res.trace), unsafe_allow_html=True)
st.markdown(html_table(["Checkpoint", "Validates", "Result", "Checks passed", "Failed checks"], checkpoint_rows(res.state)), unsafe_allow_html=True)

section("Details")
tabs = st.tabs(["Explanation", "Comparison", "Timeline and trace", "Agent messages", "Validation and data conflicts", "Metrics", "Observation table"])

with tabs[0]:
    if res.top:
        st.altair_chart(score_chart(res.top), width="stretch")
        for t in res.top:
            with st.expander(f"#{t['rank']}  {t['product_name']}  ·  {money(t['product']['price'])}  ·  score {t['score']:.1f}"):
                pr = t["product"]
                specs = [f"{pr['ram_gb']:.0f} GB RAM" if pr.get("ram_gb") else "", f"{pr['storage_gb']:.0f} GB storage" if pr.get("storage_gb") else "",
                         f"{pr['battery_hours']:.0f} h battery" if pr.get("battery_hours") else "", f"{pr['weight_kg']} kg" if pr.get("weight_kg") else ""]
                st.markdown(f"<p class='cq-muted'>{esc(pr.get('brand'))} · {esc(' · '.join(x for x in specs if x))} · {pr.get('reviews', 0):,} reviews · "
                            f"sources: {esc(', '.join(pr.get('sources', [])))}</p>", unsafe_allow_html=True)
                rows = [["Reason", esc(x)] for x in t["reasons"]] + [["Trade-off", esc(x)] for x in t.get("tradeoffs", [])]
                rows += [["Points", esc(", ".join(f"{k} {v:.1f}" for k, v in t["contributions"].items()))]]
                st.markdown(html_table(["Type", "Detail"], rows), unsafe_allow_html=True)
    else:
        st.write("No ranking for this run.")

with tabs[1]:
    cmp_ = res.state.get("comparison") or {}
    if cmp_.get("table"):
        df_ = pd.DataFrame(cmp_["table"]).drop(columns=["product_id", "leads_in"], errors="ignore")
        df_["leads in"] = [", ".join(r.get("leads_in", [])) for r in cmp_["table"]]
        st.dataframe(df_, hide_index=True, width="stretch")
    else:
        st.write("No comparison was produced for this run.")

with tabs[2]:
    if SS.get("compare_runs"):
        s, p_ = SS["compare_runs"]
        stat_tiles([("Sequential time", f"{s:.3f} s"), ("Parallel time", f"{p_:.3f} s"), ("Speed-up", f"{s / p_:.2f}x", "sequential / parallel")])
    ch = gantt(res.trace)
    if ch is not None:
        st.altair_chart(ch, width="stretch")
    tt = trace_table(res.trace)
    st.dataframe(tt, hide_index=True, width="stretch", height=35 * (len(tt) + 1) + 4)

with tabs[3]:
    st.dataframe(pd.DataFrame([{"id": m["id"], "time (s)": m["ts"], "sender": m["sender"], "receiver": m["receiver"], "type": m["type"], "status": m["status"],
                                "summary": m["summary"], "payload bytes": m.get("payload_bytes")} for m in res.messages]), hide_index=True, width="stretch")

with tabs[4]:
    v = res.state.get("validation") or {}
    if v.get("checks"):
        st.markdown("**Ranking checkpoint #3**")
        st.dataframe(pd.DataFrame(v["checks"]), hide_index=True, width="stretch")
    if res.state.get("conflicts"):
        st.markdown("**Marketplace conflicts and how they were resolved**")
        cf = pd.DataFrame(res.state["conflicts"])
        cf["values"] = cf["values"].astype(str)
        st.dataframe(cf, hide_index=True, width="stretch")
    sec = res.state.get("security_events", [])
    if sec:
        st.markdown("**Security events**")
        st.dataframe(pd.DataFrame(sec).astype(str), hide_index=True, width="stretch")

with tabs[5]:
    m = res.metrics
    stat_tiles([("Total time", f"{m['total_time_s']:.3f} s"), ("Sequential-equivalent", f"{m.get('sequential_equivalent_s', 0):.3f} s"),
                ("Speed-up", f"{m.get('speedup_estimate', 1):.2f}x"), ("Agent runs", m["agent_executions"])])
    stat_tiles([("Tool calls", m["tool_calls"], f"{m['tool_calls_failed']} failed · {m['cache_hits']} cache hits"), ("Retries", m["retries"]),
                ("Cost units", m["cost_units"], "agent 1 · tool cost · retry 1.5 · 0.5 / 100 rows"), ("Messages", len(res.messages))])
    per_agent = trace_table(res.trace).assign(ms=lambda d: (d["End (s)"] - d["Start (s)"]) * 1000).groupby("Agent", sort=False)["ms"].sum().round(1).reset_index()
    st.markdown("**Execution time per agent (ms)**")
    st.dataframe(per_agent, hide_index=True)

with tabs[6]:
    cr = SS.get("compare_runs")
    st.dataframe(observation_table(res, cr[0] if cr else None, cr[1] if cr else None), hide_index=True, width="stretch")

if SS["history"]:
    with st.expander(f"Session history ({len(SS['history'])} runs)"):
        st.dataframe(pd.DataFrame(SS["history"]), hide_index=True, width="stretch")

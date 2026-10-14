"""MCP-style tool registry (discovery, selection, controlled execution) + audit log viewer."""
import json

import pandas as pd
import streamlit as st

from ui.common import current_store, html_table, page_header, page_setup, stat_tiles

page_setup("Tools and Audit Log")

from cartiq.config import PATHS, Settings  # noqa: E402
from cartiq.runtime import RunContext  # noqa: E402
from cartiq.tools.registry import default_registry  # noqa: E402

page_header("Tools and Audit Log", "The MCP-style tool registry (discovery, schemas, least-privilege access), a JSON-RPC console for tools/list and tools/call, "
            "and the audit log of every workflow. This is a simplified MCP-style interface, not the full protocol.", eyebrow="Governance")
reg = default_registry()
stat_tiles([("Registered tools", len(reg.tools)), ("Agents with tool access", len(reg.access_matrix())),
            ("Average agents per tool", f"{sum(len(t.roles) for t in reg.tools.values()) / len(reg.tools):.1f}", "least privilege")])
t1, t2, t3 = st.tabs(["Tool registry", "JSON-RPC console", "Audit log"])

with t1:
    st.markdown("<p class='cq-muted'>Each tool is registered with an input schema, a cost and the agents allowed to use it. Discovery returns only what the caller may use.</p>", unsafe_allow_html=True)
    who = st.selectbox("Discover as agent", ["(all tools)"] + sorted(reg.access_matrix()))
    tools = reg.discover(None if who == "(all tools)" else who)
    st.dataframe(pd.DataFrame([{"tool": t["name"], "description": t["description"], "cost units": t["costUnits"], "authorized agents": ", ".join(t["authorizedAgents"])} for t in tools]),
                 hide_index=True, width="stretch")
    sel = st.selectbox("Inspect input schema", [t["name"] for t in tools])
    st.json(next(t for t in tools if t["name"] == sel)["inputSchema"], expanded=2)
    st.markdown("**Access by agent (least privilege)**")
    st.markdown(html_table(["Agent", "Tools it may call", "Count"], [[f"<b>{a}</b>", ", ".join(f"<code>{t}</code>" for t in sorted(ts)), len(ts)]
                                                                     for a, ts in sorted(reg.access_matrix().items())], num_cols=(2,)), unsafe_allow_html=True)

with t2:
    ag = st.selectbox("Caller", sorted(reg.access_matrix()), key="rpc_agent")
    default = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search_marketplace", "arguments": {"source": "ShopHub", "category": "Laptop"}}}
    txt = st.text_area("Request", json.dumps(default, indent=2), height=200)
    c = st.columns(2)
    if c[0].button("tools/list"):
        txt = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    if c[1].button("Send request", type="primary"):
        try:
            ctx = RunContext(Settings(source_latency_s=0.0), current_store(), {"audit": True})
            out = reg.handle_request(json.loads(txt), ag, ctx)
            st.json(json.loads(json.dumps(out, default=str)), expanded=1)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")

with t3:
    p = PATHS.audit_log
    if not p.exists():
        st.info("The audit log is empty - run a query first.")
    else:
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines()[-3000:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        df = pd.DataFrame(rows).astype(str)
        c = st.columns(3)
        kinds = c[0].multiselect("Event kind", sorted(df["kind"].unique()))
        wf = c[1].selectbox("Workflow", ["(all)"] + list(dict.fromkeys(df["workflow_id"][::-1]))[:50])
        if kinds: df = df[df["kind"].isin(kinds)]
        if wf != "(all)": df = df[df["workflow_id"] == wf]
        st.caption(f"{len(df)} events · file: storage/logs/audit_log.jsonl")
        st.dataframe(df.iloc[::-1].head(500), hide_index=True, width="stretch")
        st.download_button("Download audit log", p.read_bytes(), "audit_log.jsonl")

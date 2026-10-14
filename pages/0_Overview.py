"""Overview: what CartIQ does, how a request flows through the agents, and a reference sheet for every agent."""
import pandas as pd
import streamlit as st

from ui.common import esc, get_assistant, html_table, page_header, page_setup, section, stat_tiles

page_setup("Overview")

from cartiq.agent_catalog import AGENTS, STAGES  # noqa: E402
from cartiq.evaluation.runner import load_latest_report  # noqa: E402
from cartiq.graph import graph_dot  # noqa: E402
from cartiq.security import COMMON_WRITES, WRITE_PERMISSIONS  # noqa: E402
from cartiq.tools.registry import default_registry  # noqa: E402

LAB_QUERY = "Find laptops under ₹70,000 with at least 16 GB RAM, 512 GB storage and rating above 4.2"


@st.cache_resource(show_spinner="Running the lab-manual query once to show live agent outputs ...")
def example_run():
    res = get_assistant().run(LAB_QUERY, audit=False)
    per_agent: dict[str, list[dict]] = {}
    for t in res.trace:
        per_agent.setdefault(t["agent"], []).append(t)
    return {"status": res.status, "top": res.top[0]["product_name"] if res.top else "-", "time": res.metrics["total_time_s"],
            "agents": res.metrics["agent_executions"], "spans": per_agent}


page_header("CartIQ - multi-agent product recommendation",
            "A customer describes what they want in plain language. Thirteen specialised agents decompose the request, search three marketplaces in parallel, "
            "reconcile conflicting data, filter, compare and rank the products with an explainable score, validate the result at four checkpoints and ask a "
            "human when they are not sure. Built with LangGraph and LangChain.",
            eyebrow="Experiment 5 · E-commerce product analysis and recommendation pipeline")

rep = load_latest_report()
if rep:
    h = rep["headline"]
    stat_tiles([("Task success", f"{h.get('task_success_pct', 0):.1f}%", "69 labelled queries"),
                ("Requirement extraction F1", f"{h.get('extraction_f1_pct', 0):.1f}%", "Planner vs human labels"),
                ("Ranking quality nDCG@5", f"{h.get('ndcg5', 0):.3f}", "against an independent oracle"),
                ("Security tests passed", f"{h.get('security_pass_pct', 0):.1f}%", "41 attack and control tests"),
                ("Parallel speed-up", f"{h.get('parallel_speedup', 0):.2f}x", "sequential / parallel time"),
                ("Quality index", f"{h.get('quality_index', 0):.1f}", f"evaluation of {rep['generated_at'][:10]}")])
    st.caption("Full results are on the Evaluation Report page; new evaluations are started from the Evaluation Suite page.")

# ------------------------------------------------------------------------------------------------ pipeline
section("How a request is processed",
        "Fifteen stages. Independent work runs in parallel (searches; comparison and feature analysis); everything that depends on an earlier output runs in sequence.")
c1, c2 = st.columns([1.45, 1])
with c1:
    st.markdown(html_table(["#", "Agent", "Execution", "What happens", "Output"],
                           [[s["stage"], f"<b>{esc(s['agent'])}</b>", esc(s["execution"]), esc(s["step"]), f"<span class='cq-muted'>{esc(s['output'])}</span>"]
                            for s in STAGES], num_cols=(0,)), unsafe_allow_html=True)
with c2:
    mode = st.segmented_control("Execution strategy", ["parallel", "sequential"], default="parallel", label_visibility="collapsed")
    st.graphviz_chart(graph_dot(mode or "parallel"), width="stretch")
    st.caption("Green outline: validation checkpoint · amber: human approval gate · dashed: feedback loops (bounded retries, re-analysis).")

# ------------------------------------------------------------------------------------------------ agent reference
reg = default_registry()
matrix = reg.access_matrix()
ex = example_run()

section("Agent reference",
        f"Responsibilities, inputs, outputs, tool permissions and failure behaviour of every agent. Tool and state permissions are read from the running code. "
        f"The last line of each card shows what the agent did for the lab-manual query ({ex['status'].replace('_', ' ').lower()}, top pick {ex['top']}, "
        f"{ex['agents']} agent runs in {1000 * ex['time']:.0f} ms).")

summary = []
for i, a in enumerate(AGENTS, 1):
    summary.append([i, f"<b>{esc(a['name'])}</b>", esc(a["group"]), ", ".join(str(s) for s in a["stages"]), esc(a["purpose"].split(":")[0].split(" - ")[0])])
with st.expander("Index of agents", expanded=False):
    st.markdown(html_table(["#", "Agent", "Group", "Stage", "Role"], summary, num_cols=(0,)), unsafe_allow_html=True)


def tools_of(agent: str) -> list[str]:
    return sorted(matrix.get(agent, []))


def example_line(agent: str) -> str:
    spans = ex["spans"].get(agent, [])
    if not spans:
        return "Not executed for this query." if agent != "SecurityGuard" else "-"
    parts = []
    for s in spans[:4]:
        out = s.get("output") or "-"
        parts.append(f"{esc(out)} <span class='cq-muted'>({1000 * (s['duration_s'] or 0):.0f} ms)</span>")
    more = f" <span class='cq-muted'>+{len(spans) - 4} more</span>" if len(spans) > 4 else ""
    return " · ".join(parts) + more


for i, a in enumerate(AGENTS, 1):
    tools = tools_of(a["name"])
    writes = sorted(WRITE_PERMISSIONS.get(a["name"], set()) - {"status"})
    stage_txt = "Stage " + ", ".join(str(s) for s in a["stages"])
    resp = "".join(f"<li>{esc(r)}</li>" for r in a["responsibilities"])
    dl = (f"<dt>Reads</dt><dd>{''.join(f'<code>{esc(x)}</code>' for x in a['reads'])}</dd>"
          f"<dt>Writes</dt><dd>{''.join(f'<code>{esc(x)}</code>' for x in writes) or '<span class=cq-muted>nothing (pure screening)</span>'}</dd>"
          f"<dt>Tools</dt><dd>{''.join(f'<code>{esc(x)}</code>' for x in tools) or '<span class=cq-muted>none - no registry access</span>'}</dd>"
          f"<dt>Produces</dt><dd>{esc(a['produces'])}</dd>"
          f"<dt>On failure</dt><dd>{esc(a['on_failure'])}</dd>"
          f"<dt>Guardrail</dt><dd>{esc(a['guardrail'])}</dd>")
    card = (f"<div class='cq-spec'><div class='hd'><span class='idx'>{i:02d}</span><span class='name'>{esc(a['name'])}</span>"
            f"<span class='meta'><span class='cq-badge plain'>{esc(a['group'])}</span><span class='cq-badge plain'>{esc(stage_txt)}</span>"
            f"<span class='cq-badge plain'>{esc(a['execution'])}</span></span></div>"
            f"<div class='bd'><div><p class='purpose'>{esc(a['purpose'])}</p><div class='lbl'>Responsibilities</div><ul>{resp}</ul></div>"
            f"<div><dl>{dl}</dl></div></div>"
            f"<div class='ft'><b>Lab-manual run:</b> {example_line(a['name'])}</div></div>")
    st.markdown(card, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------ state + tools
section("Shared workflow state", "Every agent reads the same state; each field can only be written by the agents listed (all agents may append messages, trace, errors and security events).")
FIELD_INFO = {"user_query": "original request", "sanitized_query": "request after the security screen", "guard": "guard verdict and findings",
              "requirements": "structured requirements", "req_digest": "SHA-256 of the requirements", "plan": "task plan with dependencies",
              "planner_notes": "parsing notes", "routing": "task-to-agent assignment and stages", "source_results": "raw results per marketplace",
              "search_results": "merged, conflict-resolved products", "conflicts": "marketplace disagreements and resolution", "quarantined": "poisoned records",
              "sync": "synchronisation summary", "filtered_results": "products satisfying every constraint", "rejected": "rejected products and reasons",
              "comparison": "side-by-side comparison", "feature_analysis": "feature match and spec surplus", "ranking": "scored and ordered products",
              "ranking_meta": "weights used", "validation": "ranking validation report", "checkpoints": "results of the four checkpoints",
              "supervisor": "supervisor decision and health", "retries": "retry counters", "relaxation_suggestions": "minimal constraint relaxations",
              "human_approval": "approval status, reasons, reviewer", "human_flags": "values disputed by the reviewer", "cycle": "re-analysis counter",
              "final_recommendation": "released recommendation", "metrics": "time, tool calls, retries, cost", "status": "workflow status"}
owners: dict[str, list[str]] = {}
for ag, keys in WRITE_PERMISSIONS.items():
    for k in keys:
        owners.setdefault(k, []).append(ag)
rows = [["<code>workflow_id</code>, <code>user_query</code>", "identity and original request", "set when the workflow starts; read-only afterwards"]]
rows += [[f"<code>{esc(k)}</code>", esc(FIELD_INFO.get(k, "")), esc(", ".join(sorted(owners[k])))] for k in FIELD_INFO if k in owners]
rows += [[f"<code>{esc(k)}</code>", "append-only log", "all agents"] for k in sorted(COMMON_WRITES)]
st.markdown(html_table(['Field', 'Content', 'Written by'], rows), unsafe_allow_html=True)

section("Tool access (least privilege)", "Each tool is registered with an input schema and the agents allowed to call it. Every other call is denied and logged.")
tools = reg.discover(None)
st.markdown(html_table(["Tool", "What it does", "Allowed agents", "Cost units"],
                       [[f"<code>{esc(t['name'])}</code>", esc(t["description"]), esc(", ".join(t["authorizedAgents"])), t["costUnits"]] for t in tools],
                       num_cols=(3,)), unsafe_allow_html=True)

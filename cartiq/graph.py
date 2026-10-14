"""LangGraph workflow definition (two execution strategies over the SAME agents).

parallel   : router --Send--> [search x3] -> synchronizer -> filter -> [compare || features] -> rank -> validator -> supervisor -> human -> final -> checkpoint #4
sequential : router -> search(all, one after another) -> synchronizer -> filter -> compare -> features -> rank -> ...
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import nodes as N
from .agents.base import WorkflowState


def build_graph(mode: str = "parallel", checkpointer=None):
    g = StateGraph(WorkflowState)
    g.add_node("guard", N.security_guard)
    g.add_node("planner", N.planner)
    g.add_node("router", N.router)
    g.add_node("checkpoint_plan", N.checkpoint_plan)
    g.add_node("search_source", N.search_source)
    g.add_node("search_sequential", N.search_sequential)
    g.add_node("synchronizer", N.synchronizer)
    g.add_node("filter", N.filter_node)
    g.add_node("checkpoint_filter", N.checkpoint_filter)
    g.add_node("compare", N.compare_node)
    g.add_node("features", N.feature_node)
    g.add_node("rank", N.rank_node)
    g.add_node("validator", N.validator)
    g.add_node("supervisor", N.supervisor, destinations=("human_gate", "rank", "router", "finalize"))
    g.add_node("human_gate", N.human_gate, destinations=("finalize", "filter", "rank", "resync"))
    g.add_node("resync", N.resync)
    g.add_node("finalize", N.finalize)
    g.add_node("checkpoint_final", N.checkpoint_final)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", N.route_after_guard, ["planner", "finalize"])
    g.add_conditional_edges("planner", N.route_after_planner, ["router", "finalize"])
    g.add_conditional_edges("router", N.route_after_router, ["checkpoint_plan", "finalize"])
    g.add_conditional_edges("checkpoint_plan", N.fan_out_search, ["search_source", "search_sequential", "finalize"])
    g.add_edge("search_source", "synchronizer")
    g.add_edge("search_sequential", "synchronizer")
    g.add_conditional_edges("synchronizer", N.route_after_sync, ["filter", "supervisor", "finalize"])
    g.add_edge("filter", "checkpoint_filter")
    g.add_conditional_edges("checkpoint_filter", N.fan_out_analysis, ["compare", "features", "supervisor", "finalize"])
    if mode == "parallel":
        g.add_edge(["compare", "features"], "rank")          # synchronization barrier: rank waits for BOTH branches
    else:
        g.add_edge("compare", "features")
        g.add_edge("features", "rank")
    g.add_conditional_edges("rank", N.route_after_rank, ["validator", "finalize"])
    g.add_conditional_edges("validator", N.route_after_validator, ["supervisor", "finalize"])
    g.add_edge("resync", "filter")                       # human-flagged data re-verified, then re-filtered and re-ranked
    g.add_edge("finalize", "checkpoint_final")           # validation checkpoint #4 on the released recommendation
    g.add_edge("checkpoint_final", END)
    return g.compile(checkpointer=checkpointer)


_CP = MemorySaver()


@lru_cache(maxsize=8)
def get_graph(mode: str, interactive: bool):
    return build_graph(mode, _CP if interactive else None)


def graph_dot(mode: str = "parallel") -> str:
    """Graphviz source for the UI (hand-drawn: LangGraph cannot draw Command-based edges).
    Neutral boxes; validation checkpoints have a green outline, the human gate an amber outline, loops are dashed."""
    par = mode == "parallel"
    V = 'color="#0ca30c", fillcolor="#f3faf3"'
    return f"""
digraph G {{ rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.32;
 node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#f7f8fa", color="#c5cad3", fontcolor="#1c1f24", height=0.42];
 node [group=main];
 edge [color="#8a909a", arrowsize=0.6, fontname="Helvetica", fontsize=9, fontcolor="#5d636e"];
 q [label="User query", shape=plaintext, style="", fontcolor="#5d636e"];
 guard [label="1  SecurityGuard"]; planner [label="2  PlannerAgent"]; router [label="3  RouterAgent"];
 cp1 [label="4  Checkpoint #1 - decomposition", {V}];
 s1 [label="5  Search · ShopHub", group=a]; s2 [label="5  Search · CartBazaar"]; s3 [label="5  Search · BrandOutlet", group=b];
 sync [label="6  SynchronizerAgent (barrier)"]; filt [label="7  FilterAgent"];
 cp2 [label="8  Checkpoint #2 - filtering", {V}];
 comp [label="9  ComparisonAgent", group=a]; feat [label="9  FeatureAgent", group=b]; rank [label="10  RankingAgent (join)"];
 val [label="11  Checkpoint #3 - ranking", {V}];
 sup [label="12  SupervisorAgent"];
 human [label="13  Human approval (interrupt)", color="#eda100", fillcolor="#fdf7ea"]; fin [label="14  RecommendationAgent"];
 cp4 [label="15  Checkpoint #4 - final output", {V}];
 resync [label="Synchronizer re-check", style="rounded,dashed", fillcolor="#ffffff", group=b];
 out [label="Released to the customer", shape=plaintext, style="", fontcolor="#5d636e"];
 q -> guard -> planner -> router -> cp1;
 {"cp1 -> s1; cp1 -> s2; cp1 -> s3; s1 -> sync; s2 -> sync; s3 -> sync; {rank=same; s1; s2; s3;}" if par else "cp1 -> s1 -> s2 -> s3 -> sync;"}
 sync -> filt -> cp2;
 {"cp2 -> comp; cp2 -> feat; comp -> rank; feat -> rank; {rank=same; comp; feat;}" if par else "cp2 -> comp -> feat -> rank;"}
 rank -> val -> sup -> human -> fin -> cp4 -> out;
 sup -> rank [style=dashed, label=" retry ranking", constraint=false]; sup -> router [style=dashed, label=" retry search", constraint=false];
 human -> filt [style=dashed, label=" re-analyse", constraint=false]; human -> resync [style=dashed, label=" flag data"]; resync -> filt [style=dashed, constraint=false];
}}"""

# CartIQ — Multi-Agent Product Research & Recommendation Pipeline

**Experiment 5 · E-Commerce Product Analysis & Recommendation Pipeline** (workflow orchestration for decision-making systems).

CartIQ takes a plain-English shopping request, **decomposes** it into subtasks (search → filter → compare → rank), **executes** them as a structured LangGraph pipeline (parallel where dependencies allow), **aggregates** the results into an explainable ranked recommendation, and asks a **human** when it is unsure.

Stack: **LangGraph 1.x** (StateGraph, `Send`, `Command`, `interrupt`, checkpointer) · **LangChain** (`StructuredTool` adapters over the tool registry, optional LLM planner) · Pydantic v2 · Streamlit · Altair.

## Quick start

```bash
pip install -r requirements.txt
streamlit run Home.py            # the product (7 pages)
python cli.py ask "Find laptops under ₹70,000 with at least 16 GB RAM, 512 GB storage and rating above 4.2"
python cli.py batch data/eval/sample_queries.txt --workers 2 --out results.csv
python cli.py evaluate           # full evaluation suite (~80 s) -> storage/reports/
python -m pytest                 # 31 tests
```

No API key is needed. The default planner is rule-based and deterministic. To use an LLM planner (numbers it returns are checked against the query text, so it cannot invent a budget):
`CARTIQ_LLM=anthropic:<model>` (or `openai:<model>`, `google:<model>`) with the matching `langchain-*` package and key.

## Pages

`Home.py` is the entry point; the navigation groups eight pages:

| Group | Page | What it does |
|---|---|---|
| Overview | Overview | The 15 processing stages, the workflow graph, a reference card for every agent (responsibilities, state read/written, tools, failure behaviour, live output for the lab-manual query), shared-state ownership and tool access. |
| Product | Shopping Assistant | The product. Request → run summary, structured requirements, best pick, ranked products, agent steps, four validation checkpoints; tabs for explanation, comparison, timeline and trace, messages, conflicts, metrics and the observation table. Human review panel: approve, reject, request another analysis, flag incorrect information. Fault injection and a sequential-vs-parallel measurement. |
| Product | Product Catalog | 154 products, each marketplace's noisy view, upload of your own catalogue (validated). |
| Evaluation | Batch Testing | Run many requests at once (paste, CSV with optional `expected_*` labels, or the 69-request labelled benchmark), sequential / parallel / both, fault injection; every batch is saved in `storage/batches/` and can be reopened. |
| Evaluation | Evaluation Suite | Runs the benchmark, security, failure, HITL, performance and optimization suites; keeps a run history. |
| Evaluation | Evaluation Report | Scorecard and all result tables of any stored run, including the optimization table and the lab observation table. |
| Governance | Security Testing | Prompt injection, unauthorised tool access, shared-state tampering, requirement manipulation. |
| Governance | Tools and Audit Log | MCP-style registry (discovery, schemas, access by agent), JSON-RPC console, audit-log viewer. |

## Architecture

```
query → SecurityGuard → Planner → Router → [Validator checkpoint #1]
      → Search(ShopHub) ┐
        Search(CartBazaar) ├→ Synchronizer (barrier + conflict resolution) → Filter → [checkpoint #2]
        Search(BrandOutlet)┘
      → Comparison ┐
        Feature    ┴→ Ranking → [checkpoint #3: ranking + authoritative re-check of top 3] → Supervisor
      → HumanApproval (interrupt: approve / reject / re-analyse / flag) → Recommendation → [checkpoint #4: final output]
```

| Lab requirement | Where |
|---|---|
| Agents: Planner, Router, Search, Filter, Comparison, Ranking, Validator, Supervisor, Human, Final | `cartiq/agents/nodes.py` (+ SecurityGuard, Synchronizer, Feature) |
| Shared state (workflow_id … metrics) | `WorkflowState` in `cartiq/agents/base.py` |
| Structured inter-agent messages | `AgentMessage` (`schemas.py`), `ctx.message()`; “Agent messages” tab |
| Sequential & parallel + synchronization | `cartiq/graph.py`: `Send` fan-out, join edges `["compare","features"] → rank`, Synchronizer barrier |
| Conflict resolution (4.4 vs 4.1) | `synchronizer`: weighted consensus → trusted-source priority → authoritative recheck (decision-relevant conflicts only) → Validator → Supervisor → Human |
| Human-in-the-loop | `human_gate` uses `interrupt()`; approve / reject / re-analyse (bounded) / review reasoning / flag incorrect information (Synchronizer re-verifies, then re-filter and re-rank) |
| Validation checkpoints | #1 query decomposition, #2 filtered products, #3 ranking, #4 final recommendation (`validate_checkpoint`, `validate_recommendation`) |
| Tracing | every agent step: agent, task, start, end, status, tool used, retry, error, output status |
| Failure handling & controlled retry | `runtime.call_with_retry` (timeouts, backoff, bounded attempts), agent-level retry, Supervisor retries, graceful degradation when a source is down |
| Logging & tracing | per-agent spans (Gantt), `storage/logs/audit_log.jsonl`, security-event log |
| Performance | `Speedup = Sequential / Parallel`, measured by real wall-clock runs |
| Experimental cost units | agent call = 1, tool call = its cost, retry = 1.5, processing = 0.5 per 100 rows |
| MCP-style tool registry, least-privilege RBAC | `cartiq/tools/registry.py` (`tools/list`, `tools/call`, JSON schema validation, per-agent allow-lists) |
| Security tests | `evaluation/suites.py::run_security` (41 tests) |
| Testing strategy | functional (benchmark), failure (18 scenarios), security (41), HITL (8), performance (24 queries) + `tests/` |
| Basic vs optimized table | `run_optimization` → Evaluation Report → Optimization |
| Observation table | per run on the Assistant page; 24-query version on the Metrics page |
| Explainable ranking | per-factor score contributions, reasons and trade-offs for every product |

### Ranking
Score (0–100) = Bayesian-adjusted rating, min–max normalised specs (RAM, storage, battery, weight), price value, review volume and feature coverage, with weights re-scaled by the priorities found in the query (“cheapest”, “gaming”, “lightweight”, “long battery” …). Every product shows how many points each factor contributed.

### Data and noise
154 canonical products (the 10 products from the lab notebook plus generated laptops, phones, tablets, headphones and smartwatches with fictional brands). Three simulated marketplaces return **noisy, partial** views: rating differences, price drift, missing fields, stale “out of stock” flags and a few shared stale prices. Real systems disagree like this, so conflict resolution and re-verification actually matter, and constraint satisfaction is not trivially perfect.

## Evaluation (latest run, `python cli.py evaluate`)

| Metric | Result | How it is measured |
|---|---|---|
| Task success | 97.1 % | 69 hand-labelled queries (easy 25 / medium 20 / hard 16 / edge 8), correct outcome |
| Requirement extraction F1 | 98.4 % (exact match 92.2 %) | field-level against human labels |
| Filter precision / recall | 96.5 % / 95.8 % | vs an independent oracle on canonical data |
| nDCG@5 | 0.88 (rating-only baseline 0.82) | oracle uses a *different* utility formula |
| Constraint satisfaction | 100 % | delivered items checked on canonical data |
| Validation PASS | 93.2 % | final Validator |
| Conflict-resolution accuracy | 79 % | resolved rating vs canonical truth |
| Security tests | 90.2 % (injection detection 81 %) | 41 tests |
| Failure handling | 100 % (19 scenarios) · HITL 100 % (9) | |
| Parallel speed-up | 1.62× | 24 queries, sequential vs parallel |
| **Quality index** | **95.4** | weighted mean of the above (`runner.quality_index`) |

Optimization table (all ten parameters of the manual, basic → optimized): execution 0.539 s → 0.333 s; communication 96 KB → 252 B per hand-off; state management budget 140,000 → 70,000 after tampering; routing 17 → 13 agent runs on no-match queries; tool access 5/5 → 0/5 unauthorised calls; retry 23.2 → 18.2 tool calls on a permanent failure; validation 22.2 → 11.0 agent runs on a faulty Filter; ranking nDCG@5 0.816 → 0.878; logging 3 → 9 trace fields per step; workflow 0/6 → 5/6 corrupted rankings recovered; cost 68.9 → 49.3 units/query (−28.5 %).

Numbers vary slightly between runs because of timing and noise sampling.

## Known limitations (kept visible on purpose)

- The query parser is rule-based: spelled-out numbers (“thirty thousand”), implicit needs (“for my daughter’s school”), negated features (“no gaming”) and durations like “2 weeks” are not understood. Hard-tier benchmark queries fail for these reasons (success 87.5 %, exact match 68.8 %).
- The injection guard is pattern based. Hinglish and paraphrased overrides (S11–S13) are *not detected*; they are still contained because the budget and constraints come from a validated requirement digest, not from free text, and the ranking cannot be steered by text. One benign sentence (B05) is over-flagged.
- Rating conflicts between marketplaces are resolved correctly 79 % of the time; the rest are resolved to a plausible but not authoritative value unless the product reaches the top-3 re-check.
- Marketplaces are simulated (no live scraping). Latencies are simulated sleeps, so speed-ups reflect I/O-bound fan-out and would differ against real APIs.
- Some metrics are 100 % on this data (constraint satisfaction, failure and HITL suites). They are scenario counts, not guarantees.

## Layout

```
Home.py · pages/ · ui/common.py        Streamlit app
cartiq/                                 config, schemas, nlp, security, runtime, catalog, graph, pipeline, llm
cartiq/agents/  cartiq/tools/           agent nodes · tool implementations + MCP-style registry
cartiq/evaluation/                      benchmark data, metrics, suites, batch, runner
data/                                   catalog, marketplace views, benchmark queries, sample queries
storage/                                audit log + evaluation reports
tests/  cli.py
```

"""Reference description of every agent (used by the Overview page and the documentation).

Tool permissions and state-write permissions are NOT duplicated here: the UI reads them from the tool registry
(`ToolRegistry.access_matrix`) and from `security.WRITE_PERMISSIONS`, so the reference can never drift from the code.
"""
from __future__ import annotations

# The 15 stages of one workflow run, in execution order.
STAGES = [
    {"stage": 1, "agent": "SecurityGuard", "execution": "sequential", "step": "Screen the request for injected instructions", "output": "sanitized_query, guard verdict"},
    {"stage": 2, "agent": "PlannerAgent", "execution": "sequential", "step": "Decompose the request into structured requirements and a task plan", "output": "requirements, plan, digest"},
    {"stage": 3, "agent": "RouterAgent", "execution": "sequential", "step": "Assign tasks to agents and order them by dependency", "output": "routing table"},
    {"stage": 4, "agent": "ValidatorAgent", "execution": "sequential", "step": "Checkpoint #1 - validate the query decomposition", "output": "checkpoint result"},
    {"stage": 5, "agent": "SearchAgent x3", "execution": "parallel (3 branches)", "step": "Search ShopHub, CartBazaar and BrandOutlet", "output": "source_results"},
    {"stage": 6, "agent": "SynchronizerAgent", "execution": "barrier", "step": "Wait for all searches, merge records, resolve conflicts", "output": "search_results, conflicts"},
    {"stage": 7, "agent": "FilterAgent", "execution": "sequential", "step": "Apply the hard requirements", "output": "filtered_results, rejected"},
    {"stage": 8, "agent": "ValidatorAgent", "execution": "sequential", "step": "Checkpoint #2 - validate the filtered products", "output": "checkpoint result"},
    {"stage": 9, "agent": "ComparisonAgent + FeatureAgent", "execution": "parallel (2 branches)", "step": "Compare candidates and analyse features", "output": "comparison, feature_analysis"},
    {"stage": 10, "agent": "RankingAgent", "execution": "join", "step": "Score and rank with per-factor explanations", "output": "ranking"},
    {"stage": 11, "agent": "ValidatorAgent", "execution": "sequential", "step": "Checkpoint #3 - validate the ranking, re-check top 3", "output": "validation"},
    {"stage": 12, "agent": "SupervisorAgent", "execution": "sequential", "step": "Decide: proceed, retry, relax, escalate or abort", "output": "supervisor decision"},
    {"stage": 13, "agent": "HumanApproval", "execution": "interrupt", "step": "Pause for a human when the policy requires it", "output": "human_approval"},
    {"stage": 14, "agent": "RecommendationAgent", "execution": "sequential", "step": "Assemble the final recommendation", "output": "final_recommendation"},
    {"stage": 15, "agent": "ValidatorAgent", "execution": "sequential", "step": "Checkpoint #4 - validate the output before release", "output": "checkpoint result"},
]

# group used for colouring timelines (<= 5 groups, so the categorical palette never cycles)
AGENT_GROUP = {"SecurityGuard": "Planning", "PlannerAgent": "Planning", "RouterAgent": "Planning", "SearchAgent": "Data", "SynchronizerAgent": "Data",
               "FilterAgent": "Data", "ComparisonAgent": "Analysis", "FeatureAgent": "Analysis", "RankingAgent": "Analysis", "ValidatorAgent": "Validation",
               "SupervisorAgent": "Control", "HumanApproval": "Control", "RecommendationAgent": "Control"}
GROUP_ORDER = ["Planning", "Data", "Analysis", "Validation", "Control"]

AGENTS = [
    {
        "name": "SecurityGuard", "group": "Planning", "stages": [1], "execution": "Sequential, first node of the graph",
        "purpose": "Treats the customer's text as untrusted input and makes sure no instruction inside it can override the system's decision logic.",
        "responsibilities": [
            "Normalises the text: Unicode folding, removal of zero-width and bidirectional control characters, leetspeak folding.",
            "Detects injection families: instruction override, prompt extraction, role hijack, fake system messages, markup/script, tool abuse, requirement manipulation, secret exfiltration.",
            "Removes injected clauses and keeps the legitimate request (SANITIZED), or stops a request that contains no shopping intent at all (BLOCKED).",
            "Rejects empty, numeric-only or symbol-only input (INVALID).",
        ],
        "reads": ["user_query"],
        "produces": "Guard verdict (SAFE / SANITIZED / BLOCKED / INVALID), risk score, findings and the sanitised query.",
        "on_failure": "BLOCKED and INVALID end the workflow immediately; downstream agents do not run and the RecommendationAgent reports the reason.",
        "guardrail": "Pattern based: paraphrased or Hinglish overrides can pass the guard, but they cannot change the budget because decisions use the locked requirements.",
    },
    {
        "name": "PlannerAgent", "group": "Planning", "stages": [2], "execution": "Sequential",
        "purpose": "Performs query decomposition: turns free text into the structured requirements table of the case study and an execution plan.",
        "responsibilities": [
            "Extracts category, maximum and minimum price (₹, Rs., k, lakh), RAM, storage, rating, battery, included and excluded brands, required features, priorities and top-N.",
            "Detects conflicting constraints (two different budgets) and records how they were resolved.",
            "Builds a 13-task plan with explicit dependencies: search x3, synchronize, filter, compare, features, rank, validate, supervise, approve, recommend, validate output.",
            "Locks a SHA-256 digest of the requirements so any later unauthorised change is detectable.",
            "Optional LangChain LLM planner; every number it returns is checked against the query text.",
        ],
        "reads": ["sanitized_query"],
        "produces": "requirements, req_digest and the task plan, sent to the RouterAgent.",
        "on_failure": "Unsupported or missing category -> NEEDS_CLARIFICATION with the list of supported categories.",
        "guardrail": "Plans only - it has no access to product data tools.",
    },
    {
        "name": "RouterAgent", "group": "Planning", "stages": [3], "execution": "Sequential",
        "purpose": "Decides which specialised agent performs each task and which tasks may run at the same time.",
        "responsibilities": [
            "Maps every task kind to exactly one agent (search -> SearchAgent, filter -> FilterAgent, ...).",
            "Layers the plan topologically: independent tasks share a stage and run in parallel (the three searches; comparison and feature analysis).",
            "Detects unknown dependencies and cycles before anything executes.",
            "Dependency-aware routing: when an input is empty (no product passed the filter) the dependent agents are skipped and control goes to the Supervisor.",
        ],
        "reads": ["plan"],
        "produces": "Routing table: task, agent, stage, parallel flag, dependencies.",
        "on_failure": "A routing problem (cycle, unknown agent) stops the workflow with ERROR.",
        "guardrail": "Never executes tasks itself.",
    },
    {
        "name": "SearchAgent", "group": "Data", "stages": [5], "execution": "Three parallel instances (LangGraph Send), one per marketplace",
        "purpose": "Collects candidate products from one marketplace and reports search failures - it never filters or recommends.",
        "responsibilities": [
            "Calls search_marketplace for its marketplace and the requested category.",
            "Every call has a 1 s timeout; timeouts and transient errors are retried at most twice with exponential back-off.",
            "Permanent errors (HTTP 503) are reported to the Supervisor immediately, without retrying.",
            "Records containing instructions or markup are quarantined and never reach other agents.",
        ],
        "reads": ["requirements (category only)"],
        "produces": "One source_results entry per marketplace: records, attempts, cache flag, error type.",
        "on_failure": "A failed marketplace is marked failed; the workflow continues with the others (graceful degradation) and the result is flagged for review.",
        "guardrail": "Search tool only - a call to the ranking tool is denied by the RBAC gateway.",
    },
    {
        "name": "SynchronizerAgent", "group": "Data", "stages": [6], "execution": "Barrier after the parallel searches",
        "purpose": "Synchronises the parallel branches and turns three disagreeing marketplace views into one trustworthy record per product.",
        "responsibilities": [
            "Waits until every search branch has returned or failed before the next stage starts.",
            "Merges records by product id and fills missing fields from other sources.",
            "Resolves conflicts with a fixed ladder: weighted consensus within tolerance -> trusted-source priority -> authoritative re-check when the conflict can change the decision.",
            "Recovers a missing rating from the authoritative source when a rating constraint exists.",
            "Re-verifies a product flagged as incorrect by the human reviewer.",
        ],
        "reads": ["plan", "source_results", "requirements", "human_flags"],
        "produces": "search_results (merged), conflicts with the method used, sync summary (sources ok / failed).",
        "on_failure": "If every marketplace failed, control goes to the Supervisor, which retries the search at most twice.",
        "guardrail": "Re-checks only decision-relevant conflicts, which keeps the cost per query low.",
    },
    {
        "name": "FilterAgent", "group": "Data", "stages": [7], "execution": "Sequential",
        "purpose": "Applies constraint-based decision making: removes every product that violates a hard requirement.",
        "responsibilities": [
            "Checks price <= budget, RAM, storage, rating, battery, brand inclusion/exclusion, required features and stock.",
            "Keeps the rejected products with the reason for each rejection (used for relaxation suggestions).",
            "For comparison requests, selects the named products instead.",
        ],
        "reads": ["search_results", "requirements"],
        "produces": "filtered_results and rejected (with violated constraints).",
        "on_failure": "A wrong output is caught by checkpoint #2 before any analysis runs.",
        "guardrail": "Cannot write the requirements - an attempt is blocked and logged.",
    },
    {
        "name": "ComparisonAgent", "group": "Analysis", "stages": [9], "execution": "Parallel with FeatureAgent",
        "purpose": "Compares the products that passed the filter on common attributes.",
        "responsibilities": [
            "Builds the side-by-side table of the common attributes (price, rating, reviews, RAM, storage, battery, weight).",
            "Computes attribute ranges used for normalisation and the leader for each attribute.",
            "Runs pairwise comparisons of the top candidates (how many attributes each one wins).",
        ],
        "reads": ["filtered_results"],
        "produces": "comparison: table, attribute ranges, leaders.",
        "on_failure": "Agent-level retry once; a persistent failure is reported to the Supervisor.",
        "guardrail": "Comparison tool only.",
    },
    {
        "name": "FeatureAgent", "group": "Analysis", "stages": [9], "execution": "Parallel with ComparisonAgent",
        "purpose": "Analyses features and specification surplus so the ranking can reward products that exceed the requirements.",
        "responsibilities": [
            "Matches the required features (5G, ANC, GPS ...) and counts the extra features of each product.",
            "Measures spec surplus (how far RAM, storage and battery exceed the request), budget headroom and price per GB.",
        ],
        "reads": ["filtered_results", "requirements"],
        "produces": "feature_analysis per product.",
        "on_failure": "Agent-level retry once; reported to the Supervisor if it persists.",
        "guardrail": "Feature analysis tool only.",
    },
    {
        "name": "RankingAgent", "group": "Analysis", "stages": [10], "execution": "Join: waits for Comparison and Feature analysis",
        "purpose": "Orders the suitable products with a transparent, requirement-aware score so every position can be explained.",
        "responsibilities": [
            "Bayesian-adjusted rating (prior 4.2, weight 300), min-max normalised specs and a price-value score.",
            "Base weights: rating 0.26, price 0.18, performance 0.16, features 0.16, battery 0.10, reviews 0.08, availability 0.06.",
            "Priorities found in the query (cheapest, gaming, lightweight, long battery) re-weight the factors.",
            "Writes the points each factor contributed, plus reasons and trade-offs for every product.",
        ],
        "reads": ["filtered_results", "comparison", "feature_analysis", "requirements"],
        "produces": "ranking: rank, score (0-100), components, contributions, reasons, trade-offs.",
        "on_failure": "A corrupted ranking fails checkpoint #3; the Supervisor re-runs ranking (max 2).",
        "guardrail": "Ranking tool only; scores are recomputed by the Validator.",
    },
    {
        "name": "ValidatorAgent", "group": "Validation", "stages": [4, 8, 11, 15], "execution": "Four checkpoints along the workflow",
        "purpose": "Prevents invalid or incomplete results from reaching the user by validating at the four checkpoints recommended in the manual.",
        "responsibilities": [
            "#1 Query decomposition: category identified, consistent price range, routable plan without cycles.",
            "#2 Product filtering: every product satisfies every constraint, no duplicates, price and rating present.",
            "#3 Ranking: requirements digest unchanged, no invented products, ordered scores that reproduce from their components, complete information, constraints satisfied, top 3 re-checked against the authoritative source.",
            "#4 Final recommendation: released products come from the validated ranking, still satisfy the constraints, carry an explanation, and the approval status is recorded.",
        ],
        "reads": ["requirements", "req_digest", "routing", "filtered_results", "ranking", "final_recommendation", "human_approval"],
        "produces": "checkpoints (stage, status, checks) and validation (ranking report).",
        "on_failure": "#1/#2 FAIL stops the workflow early; #3 FAIL triggers a ranking retry; #4 FAIL withholds the output (FAILED_VALIDATION).",
        "guardrail": "May remove products that fail the authoritative re-check; cannot change requirements.",
    },
    {
        "name": "SupervisorAgent", "group": "Control", "stages": [12], "execution": "Sequential, can loop back",
        "purpose": "Monitors the whole workflow and coordinates recovery and the final decision stage.",
        "responsibilities": [
            "Reads workflow health: failed agents, errors, retries, source status, validation result.",
            "Retries ranking when validation fails and retries search when every marketplace failed (each at most twice).",
            "On NO_MATCH, asks for minimal relaxation suggestions (for example, raise the budget to ₹45,311 for 3 more products).",
            "Routes to the human approval gate, or aborts with NO_DATA / FAILED_VALIDATION when recovery is exhausted.",
        ],
        "reads": ["sync", "validation", "filtered_results", "retries", "trace", "errors"],
        "produces": "supervisor decision (PROCEED, RETRY_RANKING, RETRY_SEARCH, ESCALATE_NO_MATCH, ABORT_*) and relaxation suggestions.",
        "on_failure": "Retry budgets are bounded, so the workflow can never loop indefinitely.",
        "guardrail": "Workflow-control tools only; cannot rewrite the ranking.",
    },
    {
        "name": "HumanApproval", "group": "Control", "stages": [13], "execution": "LangGraph interrupt() with a checkpointer",
        "purpose": "Adds a governance layer: a person can approve, reject or redirect the recommendation before it is released.",
        "responsibilities": [
            "Policy auto: asks only when confidence < 0.72, the validator warned, nothing matched, the query was sanitised or a marketplace was down. Policy manual: always asks.",
            "Shows the agents' reasoning and the top candidates.",
            "Actions: approve, reject, request another analysis (new budget, rating or priorities), flag incorrect information.",
            "A flagged value is re-verified by the Synchronizer, then the products are filtered and ranked again; re-analysis cycles are capped at 2.",
        ],
        "reads": ["ranking", "validation", "requirements", "guard", "sync"],
        "produces": "human_approval: status, reasons, decided_by, note; human_flags.",
        "on_failure": "Without a reviewer (batch mode) an escalated result stays PENDING_HUMAN_REVIEW and is marked provisional.",
        "guardrail": "The only role allowed to change requirements; every change is logged as a security event.",
    },
    {
        "name": "RecommendationAgent", "group": "Control", "stages": [14], "execution": "Sequential, last agent before checkpoint #4",
        "purpose": "Assembles the final recommendation the customer sees.",
        "responsibilities": [
            "Releases the top-N ranked products only when the status allows it (RECOMMENDED or provisional PENDING_HUMAN_REVIEW).",
            "Writes the summary, confidence, warnings, approval status and relaxation suggestions.",
            "Produces a clear message for every terminal status (NO_MATCH, BLOCKED, NO_DATA, FAILED_VALIDATION, ERROR).",
        ],
        "reads": ["ranking", "human_approval", "status", "relaxation_suggestions", "guard", "planner_notes"],
        "produces": "final_recommendation.",
        "on_failure": "Its output is validated by checkpoint #4 before release.",
        "guardrail": "Cannot add products that were not validated.",
    },
]

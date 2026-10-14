"""Central configuration for CartIQ (every tunable lives here so experiments only override Settings)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Simulated marketplaces (the "search" tools query these). Trust is used for conflict resolution.
SOURCES: dict[str, dict] = {
    "ShopHub": {"trust": 0.90, "coverage": 0.88, "latency_mult": 1.0, "description": "Large general marketplace"},
    "CartBazaar": {"trust": 0.72, "coverage": 0.82, "latency_mult": 1.3, "description": "Discount marketplace, noisier data"},
    "BrandOutlet": {"trust": 0.84, "coverage": 0.62, "latency_mult": 0.8, "description": "Manufacturer outlet, partial coverage"},
}

AGENTS = ["SecurityGuard", "PlannerAgent", "RouterAgent", "SearchAgent", "SynchronizerAgent", "FilterAgent",
          "ComparisonAgent", "FeatureAgent", "RankingAgent", "ValidatorAgent", "SupervisorAgent", "HumanApproval",
          "RecommendationAgent"]


@dataclass
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    sources: Path = PROJECT_ROOT / "data" / "sources"
    eval_data: Path = PROJECT_ROOT / "data" / "eval"
    logs: Path = PROJECT_ROOT / "storage" / "logs"
    reports: Path = PROJECT_ROOT / "storage" / "reports"

    def ensure(self) -> "Paths":
        for p in (self.data, self.sources, self.eval_data, self.logs, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def catalog_csv(self) -> Path:
        return self.data / "product_catalog.csv"

    @property
    def audit_log(self) -> Path:
        return self.logs / "audit_log.jsonl"

    @property
    def system_log(self) -> Path:
        return self.logs / "system.log"


@dataclass
class Settings:
    # ---- data -----------------------------------------------------------------------------
    seed: int = 7
    # ---- execution strategy (Optimization 1, 4) ------------------------------------------------
    execution_mode: str = "parallel"          # parallel | sequential
    validation_mode: str = "checkpoint"       # checkpoint | final
    dependency_routing: bool = True           # False = fixed routing: every agent runs even when its input is empty (baseline)
    # ---- simulated I/O latency (seconds) - real wall-clock, makes parallel speed-up measurable -------
    source_latency_s: float = 0.08            # one marketplace API call
    analysis_latency_s: float = 0.05          # one analysis tool (compare / feature analysis)
    tool_timeout_s: float = 1.0
    # ---- reliability (Optimization 6) -----------------------------------------------------------
    retry_policy: str = "controlled"          # controlled | naive
    max_tool_retries: int = 2
    retry_backoff_s: float = 0.03
    max_supervisor_retries: int = 2
    max_reanalysis_cycles: int = 2
    # ---- security / governance (Optimization 5) -------------------------------------------------
    enforce_rbac: bool = True
    guard_state_writes: bool = True
    # ---- cost (Optimization 10) ---------------------------------------------------------------
    cache_enabled: bool = True
    recheck_all_conflicts: bool = False       # basic = re-verify every disagreement; optimised = only decision-relevant ones
    # ---- recommendation -------------------------------------------------------------------------
    top_n: int = 5
    verify_top_n: int = 3                     # validator re-checks the top-N against the authoritative source
    rating_conflict_tol: float = 0.15
    price_conflict_tol: float = 0.03
    # ---- human-in-the-loop (CO5) ----------------------------------------------------------------
    approval_mode: str = "auto"               # manual (always ask) | auto (policy) | off
    interactive: bool = False                 # True -> workflow pauses with langgraph.interrupt()
    escalate_confidence: float = 0.72
    # ---- optional LLM planner (LangChain) -------------------------------------------------------
    use_llm_planner: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def llm_spec() -> str | None:
    """CARTIQ_LLM='anthropic:claude-sonnet-4-5' | 'openai:gpt-4o-mini' | 'google:gemini-2.0-flash' (needs the matching key)."""
    return os.getenv("CARTIQ_LLM") or None


PATHS = Paths().ensure()

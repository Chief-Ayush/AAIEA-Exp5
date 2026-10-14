"""Typed contracts shared by the agents (structured inter-agent communication)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

CATEGORIES = ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch"]


class Requirements(BaseModel):
    """Structured requirements produced by the Planner from the natural-language query."""
    category: str | None = None
    max_price: float | None = Field(None, gt=0)
    min_price: float | None = Field(None, ge=0)
    min_ram_gb: float | None = Field(None, ge=0)
    min_storage_gb: float | None = Field(None, ge=0)
    min_rating: float | None = Field(None, ge=0, le=5)
    min_battery_hours: float | None = Field(None, ge=0)
    include_brands: list[str] = Field(default_factory=list)
    exclude_brands: list[str] = Field(default_factory=list)
    required_features: list[str] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=list)       # soft preferences that steer ranking weights
    top_n: int = Field(5, ge=1, le=20)
    intent: Literal["recommend", "compare"] = "recommend"
    compare_targets: list[str] = Field(default_factory=list)  # product ids for intent=compare
    availability_only: bool = True

    @field_validator("include_brands", "exclude_brands", "required_features", "priorities", mode="before")
    @classmethod
    def _dedupe(cls, v):
        return list(dict.fromkeys(v or []))

    def constraint_dict(self) -> dict:
        """Only the fields that the lab manual calls 'hard' requirements."""
        d = self.model_dump()
        return {k: v for k, v in d.items() if v not in (None, [], False, "") and k not in ("top_n", "intent")}

    def digest(self) -> str:
        """Tamper-evident fingerprint (Requirement Manipulation test)."""
        raw = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PlanTask(BaseModel):
    id: str
    kind: str                     # search | synchronize | filter | compare | features | rank | validate | supervise | approve | recommend
    description: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Structured inter-agent message: sender -> receiver, typed payload summary (not free text)."""
    id: str
    ts: float
    sender: str
    receiver: str
    type: str
    summary: str
    status: str = "ok"
    payload_bytes: int = 0


class ToolResult(BaseModel):
    ok: bool
    tool: str
    agent: str
    output: Any = None
    error: str | None = None
    error_type: str | None = None   # Unauthorized | UnknownTool | InvalidInput | Transient | Timeout | Permanent | ExecutionError
    duration_s: float = 0.0
    cost_units: float = 0.0
    cached: bool = False
    attempts: int = 1


def dumps(obj: Any) -> int:
    try:
        return len(json.dumps(obj, default=str))
    except Exception:
        return 0

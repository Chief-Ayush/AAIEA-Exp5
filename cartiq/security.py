"""Security layer: input validation, prompt-injection screening, data-poisoning scan and shared-state write guard.

Design rule from the lab manual: *trusted requirements and decision logic must never be overridden by prompt text*.
The guard therefore (1) sanitises the query (removes the injected clause, keeps the legitimate shopping request),
(2) blocks it when nothing legitimate remains, and (3) the state guard stops any agent from writing fields it does not own.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_QUERY_LEN = 500
ZERO_WIDTH = dict.fromkeys(list(range(0x200B, 0x2010)) + [0x2060, 0xFEFF, 0x00AD] + list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A)), None)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

# (category, regex) - applied to the normalised query AND its leet-decoded form
INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("constraint_override", r"\b(ignore|disregard|forget|override|bypass|skip|drop|remove|neglect)\b[^.;!?\n]{0,40}\b(budget|price limit|price cap|requirements?|constraints?|filters?|rating requirement|spec(ification)?s?)\b"),
    ("instruction_override", r"\b(ignore|disregard|forget|override)\b[^.;!?\n]{0,30}\b(previous|prior|above|earlier|all|any|your|the)\b[^.;!?\n]{0,25}\b(instructions?|prompts?|rules?|guidelines?|messages?)\b"),
    ("prompt_extraction", r"\b(reveal|show|print|display|leak|dump|tell me|repeat)\b[^.;!?\n]{0,30}\b(system prompt|hidden (rules|instructions)|your (instructions|rules|prompt)|api[ _-]?keys?|secrets?|credentials?|configuration)\b"),
    ("role_hijack", r"\byou are (now|no longer)\b|\bact as (an? )?(admin|administrator|developer|root|system|dan)\b|\bnew instructions?\s*:|\bdeveloper mode\b|\bsystem\s*:|\bassistant\s*:|\bjailbreak\b"),
    ("forced_selection", r"\b(always|must|only)\b[^.;!?\n]{0,15}\b(recommend|rank|pick|choose|select)\b[^.;!?\n]{0,30}\b(first|#1|top|number one)\b|\brecommend\b[^.;!?\n]{0,12}\b(the )?(most|highest) (expensive|priced)\b|\brank (product )?\w+ (as )?(first|#1)\b"),
    ("requirement_tamper", r"\b(set|change|update|modify|raise|increase|double|extend)\b[^.;!?\n]{0,15}\b(budget|max[_ ]?price|requirements?|price (limit|cap))\b[^.;!?\n]{0,15}\b(to|=|by)\b|\b(budget|price limit)\b[^.;!?\n]{0,12}\b(is|are) (now|actually)\b|\bactually,? (the )?budget\b"),
    ("tool_abuse", r"\b(delete|drop|truncate|wipe|erase)\b[^.;!?\n]{0,20}\b(table|catalog|products?|database|records|all)\b|\bcall (the )?(tool|function|api)\b|\bexecute\b[^.;!?\n]{0,20}\b(code|command|script|shell)\b|\brun\b[^.;!?\n]{0,10}\b(shell|bash|sql|python)\b"),
    ("markup_injection", r"<\s*script|javascript:|\{\{.*\}\}|\$\{.*\}|union\s+select|;\s*drop\s+table|--\s*$|<\s*/?\s*(iframe|img|svg)\b"),
    ("multilingual_override", r"\bignore karo\b|\bbudget\b[^.;!?\n]{0,15}\b(ignore|chhod|mat dekho|nazarandaz)\b|\bpehle ke (saare )?(instructions|rules)\b"),
]
_COMPILED = [(c, re.compile(p, re.I)) for c, p in INJECTION_PATTERNS]


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(ZERO_WIDTH)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class GuardResult:
    status: str                       # SAFE | SANITIZED | BLOCKED | INVALID
    clean_query: str
    original_query: str
    findings: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def risk(self) -> str:
        return {"SAFE": "none", "SANITIZED": "medium", "BLOCKED": "high", "INVALID": "n/a"}[self.status]


_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+|\s+(?=(?:ignore|disregard|forget|override|bypass|reveal|print|leak|dump|delete|drop|execute|system\s*:|you are (?:now|no longer)|act as|new instructions?)\b)", re.I)


def _clauses(text: str) -> list[str]:
    """Split into clauses so that only the injected part is removed and the legitimate request survives."""
    return [c for c in _SPLIT.split(text) if c and c.strip()]


def screen_query(query) -> GuardResult:
    if query is None or not isinstance(query, str):
        return GuardResult("INVALID", "", str(query), warnings=["query must be a non-empty string"])
    raw = query
    text = normalise(raw)
    warnings = []
    if len(text) < 3:
        return GuardResult("INVALID", "", raw, warnings=["query is empty or too short"])
    if not re.search(r"[A-Za-z0-9]", text):
        return GuardResult("INVALID", "", raw, warnings=["query contains no readable text"])
    if len(text) > MAX_QUERY_LEN:
        text = text[:MAX_QUERY_LEN]
        warnings.append(f"query truncated to {MAX_QUERY_LEN} characters")
    findings, keep = [], []
    for clause in _clauses(text):
        decoded = clause.translate(LEET)
        hit = None
        for cat, rx in _COMPILED:
            if rx.search(clause) or rx.search(decoded):
                hit = cat
                break
        if hit == "markup_injection":                      # strip only the markup, keep the shopping text
            findings.append({"category": hit, "text": clause.strip()[:160]})
            stripped = re.sub(r"<[^>]*>|javascript:\S*|\{\{.*?\}\}|\$\{.*?\}|;\s*drop\s+table.*$|union\s+select.*$", " ", clause, flags=re.I | re.S).strip()
            if stripped:
                keep.append(stripped)
        elif hit:
            findings.append({"category": hit, "text": clause.strip()[:160]})
        else:
            keep.append(clause)
    clean = " ".join(keep).strip()
    if not findings:
        return GuardResult("SAFE", clean, raw, warnings=warnings)
    # blocked when nothing that looks like a shopping request is left
    if len(clean) < 6 or not re.search(r"[A-Za-z]{3,}", clean):
        return GuardResult("BLOCKED", "", raw, findings, warnings + ["no legitimate request left after removing the injected text"])
    return GuardResult("SANITIZED", clean, raw, findings, warnings + ["injected instruction removed; trusted requirements unchanged"])


def scan_record(rec: dict) -> list[str]:
    """Data-poisoning check on product records returned by a marketplace (names, brands, feature text)."""
    hits = []
    for key in ("product_name", "brand", "features", "availability"):
        v = rec.get(key)
        if not isinstance(v, str):
            continue
        t = normalise(v)
        if len(t) > 220:
            hits.append(f"{key}: abnormally long text")
        for cat, rx in _COMPILED:
            if rx.search(t) or rx.search(t.translate(LEET)):
                hits.append(f"{key}: {cat}")
                break
    return hits


# ------------------------------------------------------------------------------------------- state write guard
COMMON_WRITES = {"messages", "trace", "errors", "security_events"}
WRITE_PERMISSIONS: dict[str, set[str]] = {
    "SecurityGuard": {"sanitized_query", "guard", "status"},
    "PlannerAgent": {"requirements", "req_digest", "plan", "planner_notes", "status", "final_recommendation"},
    "RouterAgent": {"routing"},
    "SearchAgent": {"source_results"},
    "SynchronizerAgent": {"search_results", "conflicts", "quarantined", "sync", "human_flags"},
    "FilterAgent": {"filtered_results", "rejected"},
    "ComparisonAgent": {"comparison"},
    "FeatureAgent": {"feature_analysis"},
    "RankingAgent": {"ranking", "ranking_meta"},
    "ValidatorAgent": {"validation", "ranking", "checkpoints", "status", "final_recommendation"},
    "SupervisorAgent": {"supervisor", "retries", "status", "relaxation_suggestions"},
    "HumanApproval": {"human_approval", "requirements", "req_digest", "cycle", "status", "human_flags"},
    "RecommendationAgent": {"final_recommendation", "metrics", "status"},
}


def enforce_write(agent: str, update: dict, enabled: bool = True) -> tuple[dict, list[dict]]:
    """Drop (and report) any state key the agent is not allowed to write - 'Requirement Manipulation' defence."""
    if not enabled:
        return update, []
    allowed = WRITE_PERMISSIONS.get(agent, set()) | COMMON_WRITES
    clean, events = {}, []
    for k, v in update.items():
        if k in allowed:
            clean[k] = v
        else:
            events.append({"type": "unauthorized_state_write", "agent": agent, "key": k, "action": "blocked"})
    return clean, events

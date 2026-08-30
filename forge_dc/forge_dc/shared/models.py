"""Pydantic request/response models shared across all approaches."""

from pydantic import BaseModel, Field


class SubjectRef(BaseModel):
    id: str
    name: str


# ── Capture ──────────────────────────────────────────


class CaptureRequest(BaseModel):
    skill_name: str
    skill_version: str
    actor_id: str
    actor_type: str = "agent"
    subjects: list[SubjectRef] = Field(default_factory=list)
    conversation_id: str | None = None
    parent_doc_id: str | None = None
    timestamp: str | None = None
    outcome_summary: str = ""
    message_history: list[dict] = Field(default_factory=list)
    skill_definition: str = ""
    last_decision_id: str | None = None


class CaptureResult(BaseModel):
    document_id: str
    nodes_created: int = 0
    evidence_created: int = 0
    edges_created: int = 0
    singletons_upserted: dict = Field(default_factory=dict)
    aftermath_updated: dict | None = None
    timestamp: str | None = None


# ── Query: Global Readout ────────────────────────────


class GlobalRequest(BaseModel):
    skill_name: str
    min_runs: int = 3


class ActionStats(BaseModel):
    action_label: str
    runs: int
    corrections: int
    correction_rate: float
    correction_patterns: list[str] = Field(default_factory=list)
    outcome_variety: int
    is_settled: bool


class GlobalReadout(BaseModel):
    skill_name: str
    total_runs: int
    actions: list[ActionStats] = Field(default_factory=list)


# ── Query: Search ────────────────────────────────────


class SearchRequest(BaseModel):
    query_text: str
    subjects: list[str] | None = None
    skill_name: str | None = None
    action_label: str | None = None
    aftermath_filter: str | None = None
    search_mode: str = "hybrid"
    result_mode: str = "chunk"
    limit: int = 5


class SearchResult(BaseModel):
    node_id: str
    node_type: str
    content: str
    rrf_score: float | None = None
    action_label: str | None = None
    tool_name: str | None = None
    subjects: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    outcome_short: str = ""
    decision_index: int = 0
    document: dict = Field(default_factory=dict)


# ── Query: Precedents ────────────────────────────────


class PrecedentRequest(BaseModel):
    decision_description: str
    subjects: list[str] = Field(default_factory=list)
    current_evidence: str | None = None
    limit: int = 5


class PrecedentResult(BaseModel):
    action_label: str | None = None
    evidence_summary: str = ""
    outcome_short: str = ""
    aftermath: str = "pending"
    correction: str | None = None
    corrected_outcome: str | None = None
    node_id: str = ""
    document_metadata: dict = Field(default_factory=dict)


# ── Query: Chain ─────────────────────────────────────


class ChainRequest(BaseModel):
    node_id: str
    direction: str = "forward"
    max_depth: int = 10


class ChainNode(BaseModel):
    id: str
    action_label: str | None = None
    outcome_short: str = ""
    subjects: list[str] = Field(default_factory=list)
    actor_id: str | None = None
    actor_type: str | None = None
    depth: int = 0


# ── Query: Context (H4) ─────────────────────────────


class ContextRequest(BaseModel):
    """H4: Request for 1-hop subgraph of a node."""

    node_id: str


# ── Query: Why ───────────────────────────────────────


class WhyRequest(BaseModel):
    node_id: str


class WhyResult(BaseModel):
    node_id: str
    reason: str
    cached: bool

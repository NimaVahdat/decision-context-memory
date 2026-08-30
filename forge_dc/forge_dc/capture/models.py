"""Data models for the forge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RuleScope(str, Enum):
    """Where a compiled rule applies."""

    GLOBAL = "global"
    ACTOR = "actor"


class RuleSource(str, Enum):
    """How the rule was derived."""

    CORRECTION = "correction"
    TOOL = "tool"


class RuleType(str, Enum):
    """What kind of behavioral rule this is.

    FORMAT rules apply to all outputs regardless of query
    content (styling, structure, units, sections, labels).
    They are injected unconditionally with a separate budget.

    CONTENT rules depend on specific data or conditions.
    They use similarity-based retrieval and progressive
    injection.
    """

    FORMAT = "format"
    CONTENT = "content"


@dataclass
class CompiledRule:
    """A single compiled rule in the forge rulebook."""

    id: str
    text: str
    scope: RuleScope
    source: RuleSource
    skill_name: str
    actor_id: str | None = None
    subjects: list[str] = field(default_factory=list)
    source_doc_ids: list[str] = field(default_factory=list)
    needs_context: bool = False
    supported_by: list[str] = field(default_factory=list)
    conflict_group: str | None = None
    topic_group: str | None = None
    trigger_node_id: str | None = None
    rule_type: str = "content"  # "format" or "content"
    # Optional one-line justification, populated when FORGE_EMIT_WHY=1
    # (reasoning variant). Empty otherwise.
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "scope": self.scope.value,
            "source": self.source.value,
            "skill_name": self.skill_name,
            "actor_id": self.actor_id,
            "subjects": self.subjects,
            "source_doc_ids": self.source_doc_ids,
            "needs_context": self.needs_context,
            "supported_by": self.supported_by,
            "conflict_group": self.conflict_group,
            "topic_group": self.topic_group,
            "trigger_node_id": self.trigger_node_id,
            "rule_type": self.rule_type,
            "reasoning": self.reasoning,
        }


@dataclass
class ForgeResult:
    """Result of the forge consolidation pipeline."""

    rules: list[CompiledRule] = field(default_factory=list)
    forge_doc_id: str = ""
    stats: dict = field(default_factory=dict)

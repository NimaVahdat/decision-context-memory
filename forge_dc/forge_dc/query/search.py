"""ForgeQuery: node-level retrieval with TRIGGER edges.

ONE retrieval path: hybrid searches nodes → follow FORGED_FROM
edges to rules → expand topic_group → score-based budget →
merge at injection.

No PATH 2. Node embeddings + TRIGGER edges handle everything.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import and_, select

from forge_dc.capture.models import (
    CompiledRule,
    RuleScope,
    RuleSource,
)
from forge_dc.config import OpenAIConfig
from forge_dc.interface import QueryStrategy
from forge_dc.shared.embedder import Embedder
from forge_dc.store.engine import StoreEngine
from forge_dc.store.models import (
    DecisionNode,
    Document,
    Edge,
)
from forge_dc.store.query.search import StoreQuery

logger = logging.getLogger(__name__)


def _node_to_rule(node: DecisionNode) -> CompiledRule:
    """Convert a forge_rule DecisionNode to a CompiledRule."""
    scope = RuleScope.GLOBAL
    actor_id = None
    source = RuleSource.CORRECTION
    supported_by: list[str] = []
    conflict_group = None
    topic_group = None
    trigger_node_id = None

    if node.outcome_short:
        try:
            meta = json.loads(node.outcome_short)
            if meta.get("scope") == "actor":
                scope = RuleScope.ACTOR
            actor_id = meta.get("actor_id")
            if meta.get("source") == "tool":
                source = RuleSource.TOOL
            supported_by = meta.get("supported_by", [])
            conflict_group = meta.get("conflict_group")
            topic_group = meta.get("topic_group")
            trigger_node_id = meta.get("trigger_node_id")
            rule_type = meta.get("rule_type", "content")
        except (json.JSONDecodeError, TypeError):
            rule_type = "content"

    rule = CompiledRule(
        id=node.id,
        text=node.content or "",
        scope=scope,
        source=source,
        skill_name="",
        actor_id=actor_id,
        subjects=node.subjects or [],
        supported_by=supported_by,
        conflict_group=conflict_group,
        topic_group=topic_group,
        trigger_node_id=trigger_node_id,
        rule_type=rule_type,
        reasoning=(node.reason or ""),
    )
    # Preserve compilation order for tiebreaking
    rule._compile_order = node.decision_index or 0
    return rule


def _format_rules(rules: list[CompiledRule]) -> str:
    """Format rules as injection text.

    Each rule gets its own RULE line. No merging — agents
    follow individual rules better than long numbered lists.
    """
    if not rules:
        return ""

    lines = ["# MANDATORY RULES \u2014 FOLLOW EXACTLY", ""]
    for r in rules:
        lines.append(f"RULE: {r.text}")
        # reasoning variant: emit WHY directly after each RULE.
        # Default forge_dc has reasoning="" so this is a no-op there.
        if getattr(r, "reasoning", ""):
            lines.append(f"WHY: {r.reasoning}")
    lines.append("")
    lines.append(
        "Violations of these rules will be treated as errors.",
    )
    return "\n".join(lines)


class ForgeQuery(QueryStrategy):
    """Node-level retrieval with TRIGGER edges."""

    @property
    def format_mode(self) -> str:
        return "forge"

    @property
    def supports_global_readout(self) -> bool:
        return False

    @property
    def supports_evidence_corrections(self) -> bool:
        return True

    def __init__(
        self,
        db_url: str,
        openai_config: OpenAIConfig,
    ) -> None:
        self._engine = StoreEngine(db_url)
        self._openai_config = openai_config
        self._hybrid = StoreQuery(db_url, openai_config)
        self._embedder: Embedder | None = None
        self._rules_cache: dict[str, CompiledRule] = {}

    async def initialize(self) -> None:
        await self._engine.create_tables()
        await self._hybrid.initialize()
        if self._openai_config.api_key:
            self._embedder = Embedder(
                api_key=self._openai_config.api_key,
                model=self._openai_config.embedding_model,
            )

    async def shutdown(self) -> None:
        await self._hybrid.shutdown()
        await self._engine.dispose()

    # ── Core: retrieve() ─────────────────────────────────────

    async def retrieve(
        self,
        task_context: str,
        data_context: str = "",
        skill_text: str = "",
        skill_name: str = "",
        actor_id: str | None = None,
        subjects: list[str] | None = None,
        exclude_ids: set[str] | None = None,
        correction_limit: int = 5,
        tool_limit: int = 10,
    ) -> dict:
        """Single-path retrieval: nodes → edges → rules.

        1. Hybrid searches decision nodes by context
        2. Follow FORGED_FROM edges to rules
        3. Expand to topic_group
        4. Resolve conflicts
        5. Budget per type
        6. Merge + format
        """
        exclude = exclude_ids or set()

        # Load all rules (cached)
        if not self._rules_cache:
            await self._load_all_rules()

        # Step 1: Hybrid search on NODES (with scores)
        node_scores = await self._search_nodes(
            task_context,
            data_context,
            subjects,
            actor_id,
        )

        # Step 2: Follow FORGED_FROM edges to rules (scored)
        found_rules = await self._follow_edges(node_scores)

        # Step 3: Expand to topic_group
        expanded = self._expand_topic_groups(found_rules)

        # Filter: skill + exclude
        filtered = [
            r
            for r in expanded
            if r.id not in exclude
            and (not skill_name or not r.skill_name or r.skill_name == skill_name)
        ]

        # Step 4: Budget — group by trigger action stage,
        # top-3 per group sorted by retrieval score.
        # Per-stage actor preference: if the querying actor
        # has a rule in a stage, only their rules survive
        # in that stage. Stages with no actor rules pass
        # all through (cross-user visibility).
        from collections import defaultdict

        corrections = [r for r in filtered if r.source == RuleSource.CORRECTION]
        tools = [r for r in filtered if r.source == RuleSource.TOOL][:tool_limit]

        # Group corrections by action stage
        by_stage: dict[str, list[CompiledRule]] = defaultdict(list)
        for r in corrections:
            stage = self._trigger_actions.get(
                r.trigger_node_id or "",
                "unknown",
            )
            by_stage[stage].append(r)

        # Within each stage, sort by retrieval score
        # (highest first), then compile order as tiebreaker.
        for rules in by_stage.values():
            rules.sort(
                key=lambda r: (
                    -getattr(r, "_retrieval_score", 0.0),
                    getattr(r, "_compile_order", 0),
                ),
            )

        # Per-stage actor preference: if the querying actor
        # has rules in a stage, keep only theirs. Otherwise
        # keep all (cross-user corrections pass through).
        if actor_id:
            for stage, rules in by_stage.items():
                actor_rules = [r for r in rules if r.actor_id == actor_id]
                if actor_rules:
                    by_stage[stage] = actor_rules

        # Top-3 per stage, dedup across stages
        per_stage = 3
        seen_texts: set[str] = set()
        budgeted: list[CompiledRule] = []
        for stage, rules in by_stage.items():
            count = 0
            for r in rules:
                if count >= per_stage:
                    break
                if r.text not in seen_texts:
                    seen_texts.add(r.text)
                    budgeted.append(r)
                    count += 1

        final = budgeted + tools

        injection = _format_rules(final)
        return {
            "rules": final,
            "subjects": subjects or [],
            "injection_text": injection,
        }

    # ── Step 1: Search nodes ─────────────────────────────────

    async def _search_nodes(
        self,
        task_context: str,
        data_context: str,
        subjects: list[str] | None,
        actor_id: str | None,
    ) -> dict[str, float]:
        """Hybrid search returns scored decision nodes.

        Returns {node_id: score} preserving the hybrid ranking.
        """
        try:
            # Scale search limit with the number of compiled
            # rules — more rules need more nodes to find all
            # trigger nodes.
            n_rules = len(self._rules_cache)
            search_limit = max(20, n_rules)

            prec = await self._hybrid.precedents(
                description=task_context,
                subjects=subjects,
                evidence=data_context or None,
                limit=search_limit,
                actor_id=actor_id,
            )
            raw = (
                prec.model_dump()
                if hasattr(prec, "model_dump")
                else prec
                if isinstance(prec, dict)
                else {"results": prec}
            )

            node_scores: dict[str, float] = {}
            for r in raw.get("results", []):
                score = r.get("score", 0.0)
                nid = r.get("node_id")
                if nid and nid not in node_scores:
                    node_scores[nid] = score
                dm = r.get("document_metadata", {})
                did = dm.get("id")
                if did:
                    async with self._engine.session() as session:
                        stmt = select(DecisionNode.id).where(
                            DecisionNode.document_id == did,
                        )
                        result = await session.execute(stmt)
                        for row in result.fetchall():
                            if row[0] not in node_scores:
                                node_scores[row[0]] = score
            return node_scores
        except Exception as exc:
            logger.warning("Node search failed: %s", exc)
            return {}

    # ── Step 2: Follow edges ─────────────────────────────────

    async def _follow_edges(
        self,
        node_scores: dict[str, float],
    ) -> list[CompiledRule]:
        """Follow FORGED_FROM edges from nodes to rules.

        Each rule inherits the BEST score from its trigger nodes.
        Returns rules sorted by score descending.
        """
        if not node_scores:
            return []

        node_id_list = list(node_scores.keys())
        async with self._engine.session() as session:
            stmt = select(Edge.from_id, Edge.to_id).where(
                and_(
                    Edge.edge_type == "FORGED_FROM",
                    Edge.to_id.in_(node_id_list),
                ),
            )
            result = await session.execute(stmt)
            rows = result.fetchall()

            # Each rule gets the best score from its trigger nodes
            rule_scores: dict[str, float] = {}
            for row in rows:
                rule_id = row[0]
                node_id = row[1]
                score = node_scores.get(node_id, 0.0)
                if score > rule_scores.get(rule_id, -1.0):
                    rule_scores[rule_id] = score

        # Return rules sorted by score (highest first),
        # then by compilation order (earliest first) as
        # tiebreaker.
        scored_rules = [
            (self._rules_cache[rid], score)
            for rid, score in rule_scores.items()
            if rid in self._rules_cache
        ]
        scored_rules.sort(
            key=lambda x: (
                -x[1],  # score descending
                getattr(x[0], "_compile_order", 0),  # order ascending
            ),
        )
        # Store score on rule for downstream budget sorting
        for r, score in scored_rules:
            r._retrieval_score = score
        return [r for r, _ in scored_rules]

    # ── Step 3: Expand topic groups ──────────────────────────

    def _expand_topic_groups(
        self,
        rules: list[CompiledRule],
    ) -> list[CompiledRule]:
        """Expand found rules to their full topic_groups."""
        found_groups: set[str] = set()
        for r in rules:
            if r.topic_group:
                found_groups.add(r.topic_group)

        if not found_groups:
            return rules

        # Add all rules from matched groups
        expanded_ids = {r.id for r in rules}
        expanded = list(rules)

        for rule in self._rules_cache.values():
            if rule.topic_group in found_groups and rule.id not in expanded_ids:
                expanded.append(rule)
                expanded_ids.add(rule.id)

        return expanded

    # ── Data loading ─────────────────────────────────────────

    async def _load_all_rules(self) -> None:
        """Load all forge_rule nodes into cache."""
        self._rules_cache = {}

        async with self._engine.session() as session:
            stmt = (
                select(Document)
                .where(Document.actor_id == "forge-compiler")
                .order_by(Document.timestamp.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            forge_doc = result.scalar_one_or_none()
            if not forge_doc:
                return

            rule_meta: dict[str, dict] = {}
            if forge_doc.message_history:
                for msg in forge_doc.message_history:
                    if msg.get("role") == "system":
                        try:
                            data = json.loads(
                                msg.get("content", "{}"),
                            )
                            for rd in data.get("rules", []):
                                rule_meta[rd["id"]] = rd
                        except (json.JSONDecodeError, KeyError):
                            pass

            stmt = select(DecisionNode).where(
                and_(
                    DecisionNode.document_id == forge_doc.id,
                    DecisionNode.node_type == "forge_rule",
                ),
            )
            result = await session.execute(stmt)
            nodes = list(result.scalars().all())

            for node in nodes:
                rule = _node_to_rule(node)
                rule.skill_name = forge_doc.skill_name or ""

                meta = rule_meta.get(node.id, {})
                if meta.get("source") == "tool":
                    rule.source = RuleSource.TOOL
                rule.source_doc_ids = meta.get(
                    "source_doc_ids",
                    [],
                )

                rule._embedding = node.embedding if node.embedding else None

                self._rules_cache[rule.id] = rule

            # Cache trigger node action labels for grouping
            trigger_ids = [
                r.trigger_node_id for r in self._rules_cache.values() if r.trigger_node_id
            ]
            self._trigger_actions: dict[str, str] = {}
            if trigger_ids:
                stmt = select(
                    DecisionNode.id,
                    DecisionNode.action_label,
                ).where(DecisionNode.id.in_(trigger_ids))
                result = await session.execute(stmt)
                for row in result.fetchall():
                    self._trigger_actions[row[0]] = row[1] or "unknown"

    # ── Interface stubs ──────────────────────────────────────

    async def precedents(
        self,
        description: str,
        subjects: list[str] | None = None,
        evidence: str | None = None,
        limit: int = 5,
        actor_id: str | None = None,
    ) -> dict:
        return await self.retrieve(
            task_context=description,
            data_context=evidence or "",
            actor_id=actor_id,
            subjects=subjects,
            correction_limit=limit,
        )

    async def evidence_corrections(
        self,
        tool_result: str,
        skill_name: str,
        limit: int = 5,
    ) -> list[dict]:
        result = await self.retrieve(
            task_context="",
            data_context=tool_result[:1000],
            skill_name=skill_name,
            correction_limit=limit,
            tool_limit=limit,
        )
        return [{"rule_id": r.id, "correction": r.text} for r in result.get("rules", [])]

    async def search(self, query_text, **kw) -> list[dict]:
        return await self._hybrid.search(
            query_text=query_text,
            **kw,
        )

    async def global_readout(self, **kw) -> dict:
        return {}

    async def chain(self, node_id, **kw) -> list:
        return []

    async def why(self, node_id: str) -> Any:
        raise NotImplementedError

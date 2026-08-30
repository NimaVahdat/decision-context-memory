"""Preference conflict resolution at injection time.

Uses conflict_group and supported_by fields set by
the maintenance module at consolidation time.

For each conflict group in the final rule set:
- If querying actor has a preference → inject theirs
- If no preference → inject majority (most supporters)
- If tie → inject neither (don't force a preference)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from forge_dc.capture.models import (
    CompiledRule,
)

logger = logging.getLogger(__name__)


def resolve_conflicts(
    rules: list[CompiledRule],
    actor_id: str | None = None,
) -> list[CompiledRule]:
    """Resolve preference conflicts in the final rule set.

    Rules with no conflict_group pass through unchanged.
    For each conflict group, pick the best rule for this actor.
    """
    if not rules:
        return rules

    # Separate conflicted from non-conflicted
    no_conflict = [r for r in rules if not r.conflict_group]
    by_group: dict[str, list[CompiledRule]] = defaultdict(list)
    for r in rules:
        if r.conflict_group:
            by_group[r.conflict_group].append(r)

    if not by_group:
        return rules

    # Resolve each conflict group
    resolved = list(no_conflict)
    for group_id, group_rules in by_group.items():
        winner = _pick_winner(group_rules, actor_id)
        if winner:
            resolved.append(winner)
            logger.debug(
                "Conflict %s: picked '%s' (%s) from %d rules",
                group_id,
                winner.text[:40],
                winner.actor_id,
                len(group_rules),
            )

    return resolved


def _pick_winner(
    group_rules: list[CompiledRule],
    actor_id: str | None,
) -> CompiledRule | None:
    """Pick the winning rule from a conflict group.

    Priority:
    1. Querying actor's preference (if they have one)
    2. Majority (most supporters)
    3. None (tie with no actor preference → don't inject)
    """
    # 1. Actor's own preference
    if actor_id:
        for r in group_rules:
            if actor_id in (r.supported_by or []):
                return r

    # 2. Majority
    by_support = sorted(
        group_rules,
        key=lambda r: len(r.supported_by or []),
        reverse=True,
    )

    if len(by_support) < 2:
        return by_support[0] if by_support else None

    top = len(by_support[0].supported_by or [])
    second = len(by_support[1].supported_by or [])

    if top > second:
        return by_support[0]

    # 3. Tie — don't inject a preference with no consensus
    return None

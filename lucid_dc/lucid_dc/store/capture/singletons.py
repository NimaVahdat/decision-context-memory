"""Upsert singletons — atomic via savepoints.

H3: SUPERSEDES edge wired when a new Policy version
    appears for the same skill.
H5: Uses begin_nested() savepoints to avoid TOCTOU
    race conditions. IntegrityError only rolls back
    the savepoint, not the whole transaction.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lucid_dc.store.models import (
    Actor,
    Edge,
    Policy,
    Subject,
)


async def upsert_singletons(
    session: AsyncSession,
    actor_id: str,
    actor_name: str | None,
    actor_type: str,
    subjects: list[dict],
    skill_name: str,
    skill_version: str,
) -> dict:
    """Upsert Actor, Subject, Policy singletons.

    Uses INSERT ... ON CONFLICT DO NOTHING semantics.
    Returns counts of upserted singletons.
    """
    counts = {
        "actors": 0,
        "subjects": 0,
        "policies": 0,
    }

    # Actor — H5: atomic upsert
    if await _upsert_actor(session, actor_id, actor_name, actor_type):
        counts["actors"] = 1

    # Subjects — H5: atomic upsert
    for subj in subjects:
        subj_id = subj.get("id", "")
        subj_name = subj.get("name")
        if await _upsert_subject(session, subj_id, subj_name):
            counts["subjects"] += 1

    # Policy — H5: atomic upsert + H3: SUPERSEDES
    policy_id = f"{skill_name}-{skill_version}"
    if await _upsert_policy(
        session,
        policy_id,
        skill_name,
        skill_version,
        f"{skill_name} skill definition",
    ):
        counts["policies"] = 1

    return counts


async def _upsert_actor(
    session: AsyncSession,
    actor_id: str,
    name: str | None,
    actor_type: str,
) -> bool:
    """Insert actor if not exists (atomic).

    H5: Uses savepoint so IntegrityError only rolls
    back the nested transaction, not the whole txn.
    """
    try:
        async with session.begin_nested():
            session.add(Actor(id=actor_id, name=name, type=actor_type))
        return True
    except IntegrityError:
        return False


async def _upsert_subject(
    session: AsyncSession,
    subject_id: str,
    name: str | None,
) -> bool:
    """Insert subject if not exists (atomic).

    H5: Uses savepoint for atomic insert-or-ignore.
    """
    try:
        async with session.begin_nested():
            session.add(Subject(id=subject_id, name=name))
        return True
    except IntegrityError:
        return False


async def _upsert_policy(
    session: AsyncSession,
    policy_id: str,
    skill_name: str,
    version: str | None,
    description: str | None,
) -> bool:
    """Insert policy if not exists (atomic).

    H3: If a policy with the same skill_name but a
    different version already exists, wire a
    SUPERSEDES edge from the new policy to the old.
    H5: Uses savepoint for atomic insert-or-ignore.
    """
    # H3: Find existing policies for same skill
    # (different version) before inserting.
    prefix = f"{skill_name}-"
    result = await session.execute(
        select(Policy.id).where(
            Policy.id.like(f"{prefix}%"),
            Policy.id != policy_id,
        )
    )
    old_policy_ids = [row[0] for row in result.fetchall()]

    # Atomic insert via savepoint
    try:
        async with session.begin_nested():
            session.add(
                Policy(
                    id=policy_id,
                    version=version,
                    description=description,
                )
            )
    except IntegrityError:
        return False

    # H3: Wire SUPERSEDES edges (new -> old)
    for old_id in old_policy_ids:
        session.add(
            Edge(
                id=str(uuid.uuid4()),
                edge_type="SUPERSEDES",
                from_id=policy_id,
                to_id=old_id,
            )
        )

    return True

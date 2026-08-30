"""Async SQLAlchemy engine for the hybrid approach.

Each approach has its own engine pointing to its own database.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def resolve_hybrid_db_url(
    configured_url: str | None = None,
) -> str:
    """Resolve the hybrid database URL.

    Priority:
    1. Explicit configured_url parameter
    2. CG_HYBRID_DB_URL environment variable
    3. Default SQLite path
    """
    if configured_url:
        url = configured_url
    else:
        url = os.environ.get(
            "CG_HYBRID_DB_URL",
            "postgresql+asyncpg://localhost:5432/cg_hybrid",
        )

    # Convert postgresql:// to async driver
    if url.startswith("postgresql://"):
        url = url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    # Ensure directory exists for SQLite
    if url.startswith("sqlite") and ":///" in url:
        db_path = url.split(":///", 1)[1]
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    return url


def _make_engine_kwargs(url: str) -> dict:
    """Build engine kwargs based on dialect."""
    kwargs: dict = {"echo": False}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return kwargs


class HybridEngine:
    """Manages the async engine and sessions for hybrid."""

    def __init__(self, db_url: str) -> None:
        self._url = resolve_hybrid_db_url(db_url)
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def url(self) -> str:
        return self._url

    def get_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._url,
                **_make_engine_kwargs(self._url),
            )
        return self._engine

    def get_session_factory(
        self,
    ) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            engine = self.get_engine()
            self._session_factory = async_sessionmaker(
                engine,
                expire_on_commit=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def session(
        self,
    ) -> AsyncIterator[AsyncSession]:
        """Yield an async session, committing on success."""
        factory = self.get_session_factory()
        session = factory()
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_tables(self) -> None:
        """Create all tables (for dev/testing)."""
        from hybrid_raw.models import (
            HybridBase,
        )

        engine = self.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(HybridBase.metadata.create_all)

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None

"""forge_dc capture pipeline.

Extends the base graph-store capture with a forge step: after
capture, a strong LLM compiles the corrections into strict RULE
directives, stored as decision nodes wired with FORGED_FROM edges.
"""

from __future__ import annotations

import logging

from forge_dc.capture.forge import (
    run_forge_pipeline,
)
from forge_dc.capture.models import (
    ForgeResult,
)
from forge_dc.store.capture.pipeline import (
    StoreCapture,
)

logger = logging.getLogger(__name__)


class ForgeCapture(StoreCapture):
    """Capture with forge rule compilation.

    Runs the full graph-store capture unchanged, then adds the forge
    step: after all traces are captured, walks correction chains and
    compiles them into RULE directives via a strong LLM.
    """

    @property
    def supports_consolidation(self) -> bool:
        return True

    async def consolidate(
        self,
        skill_name: str,
        skill_definition: str = "",
        conversation_id: str | None = None,
    ) -> ForgeResult:
        """Run the forge compilation pipeline."""
        async with self._engine.session() as session:
            result = await run_forge_pipeline(
                session,
                skill_name,
                skill_definition=skill_definition,
                embedder=self._embedder,
                conversation_id=conversation_id,
            )
            await session.commit()
            logger.info(
                "Forge consolidation committed (url=%s)",
                self._engine._url,
            )
            return result

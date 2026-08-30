"""lucid_dc capture pipeline.

Extends the base graph-store capture with a consolidation step:
after capture, an LLM reviews the run's traces and produces
tool-calling directives. No analysis happens at capture time — it
all runs during consolidation.
"""

from __future__ import annotations

import logging

from lucid_dc.capture.consolidate import (
    run_consolidation,
)
from lucid_dc.capture.models import (
    ConsolidationResult,
)
from lucid_dc.store.capture.pipeline import (
    StoreCapture,
)

logger = logging.getLogger(__name__)


class LucidCapture(StoreCapture):
    """Capture with LLM-based consolidation.

    Runs the full graph-store capture unchanged, then adds a
    consolidation pass: after all traces are captured, an LLM
    analyzes them and produces directives like
    "DO NOT call eval_lookup with source='financials'".
    """

    @property
    def supports_consolidation(self) -> bool:
        return True

    async def consolidate(
        self,
        skill_name: str,
        skill_definition: str = "",
    ) -> ConsolidationResult:
        """Run LLM consolidation for a skill.

        Analyzes all captured traces and produces
        tool-calling directives.
        """
        async with self._engine.session() as session:
            result = await run_consolidation(
                session,
                skill_name,
                skill_definition=skill_definition,
            )
            await session.commit()
            logger.info(
                "Consolidation committed (url=%s)",
                self._engine._url,
            )
            return result

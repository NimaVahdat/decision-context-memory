"""Data structures for consolidation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsolidationResult:
    """Result of a consolidation pass."""

    new_correction_ids: list[str] = field(default_factory=list)
    llm_corrections_generated: int = 0
    injection_text: str = ""

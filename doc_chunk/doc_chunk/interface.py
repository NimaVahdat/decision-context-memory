"""Abstract interfaces for capture and query strategies."""

from abc import ABC, abstractmethod

from doc_chunk.shared.models import (
    ChainNode,
    WhyResult,
)


class CaptureStrategy(ABC):
    @abstractmethod
    async def capture(self, trace: dict) -> dict:
        """Capture a raw agent trace and store it."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize resources (engines, connections)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""
        ...

    @property
    def supports_consolidation(self) -> bool:
        """Whether this strategy has a consolidation phase."""
        return False

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and related data.

        Returns the number of chunks deleted.
        """
        raise NotImplementedError


class QueryStrategy(ABC):
    @property
    def format_mode(self) -> str:
        """How to format precedents: 'structured', 'adaptive', 'raw', 'naive'."""
        return "raw"

    @property
    def supports_global_readout(self) -> bool:
        """Whether this strategy supports global_readout()."""
        return False

    @property
    def supports_evidence_corrections(self) -> bool:
        """Whether this strategy supports evidence_corrections()."""
        return False

    @abstractmethod
    async def global_readout(
        self,
        skill_name: str,
        min_runs: int,
        actor_id: str | None = None,
    ) -> dict:
        """Skill-level aggregation (Layer 3)."""
        ...

    @abstractmethod
    async def search(
        self,
        query_text: str,
        subjects: list[str] | None = None,
        skill_name: str | None = None,
        action_label: str | None = None,
        aftermath_filter: str | None = None,
        search_mode: str = "hybrid",
        result_mode: str = "chunk",
        limit: int = 5,
    ) -> list[dict]:
        """BM25 + vector hybrid search.

        result_mode: "chunk" returns individual matched
        chunks/nodes. "document" returns all chunks from
        each matched document (doc approach only).
        """
        ...

    @abstractmethod
    async def precedents(
        self,
        description: str,
        subjects: list[str] | None = None,
        evidence: str | None = None,
        limit: int = 5,
        actor_id: str | None = None,
    ) -> dict:
        """Structured precedent query (Layers 2+1)."""
        ...

    @abstractmethod
    async def chain(
        self,
        node_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> list[ChainNode]:
        """Walk LED_TO chain."""
        ...

    @abstractmethod
    async def why(self, node_id: str) -> WhyResult:
        """Lazy reason extraction."""
        ...

    async def initialize(self) -> None:
        """Initialize resources."""
        pass

    async def shutdown(self) -> None:
        """Release resources."""
        pass

    async def get_document(self, document_id: str) -> dict:
        """Retrieve a full document with nodes."""
        raise NotImplementedError

    async def context(self, node_id: str) -> dict:
        """1-hop subgraph around a DecisionNode.

        H4: Returns the node with all connected edges
        (actor, subjects, policy, parent/children via
        LED_TO, informed_by).
        """
        raise NotImplementedError

    async def get_conversation_documents(self, conversation_id: str) -> dict:
        """Return all documents in a conversation."""
        raise NotImplementedError

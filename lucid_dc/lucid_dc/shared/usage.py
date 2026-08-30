"""Token usage accumulator for capture and query phases.

Tracks tokens consumed by LLM and embedding calls inside each approach's
pipeline so the benchmark can report the full cost of an approach, not
just the main agent's tokens.

Usage model: module-level singletons, one for capture and one for query.
The bench reads them after each phase and resets between phases. This is
safe because the bench runs scenarios sequentially per approach.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    llm_in: int = 0
    llm_out: int = 0
    llm_reqs: int = 0
    embed_in: int = 0
    embed_calls: int = 0

    def add_llm(self, in_tokens: int, out_tokens: int) -> None:
        self.llm_in += int(in_tokens or 0)
        self.llm_out += int(out_tokens or 0)
        self.llm_reqs += 1

    def add_embed(self, in_tokens: int) -> None:
        self.embed_in += int(in_tokens or 0)
        self.embed_calls += 1

    def snapshot(self) -> dict:
        return {
            "llm_in": self.llm_in,
            "llm_out": self.llm_out,
            "llm_reqs": self.llm_reqs,
            "embed_in": self.embed_in,
            "embed_calls": self.embed_calls,
        }

    def reset(self) -> None:
        self.llm_in = 0
        self.llm_out = 0
        self.llm_reqs = 0
        self.embed_in = 0
        self.embed_calls = 0


_capture_usage = TokenUsage()
_query_usage = TokenUsage()
_phase: str = "capture"


def set_phase(phase: str) -> None:
    """Route subsequent add_* calls to capture or query accumulator."""
    global _phase
    if phase not in ("capture", "query"):
        raise ValueError(f"phase must be 'capture' or 'query', got {phase!r}")
    _phase = phase


def _active() -> TokenUsage:
    return _capture_usage if _phase == "capture" else _query_usage


def record_llm(in_tokens: int, out_tokens: int) -> None:
    _active().add_llm(in_tokens, out_tokens)


def record_embed(in_tokens: int) -> None:
    _active().add_embed(in_tokens)


def get_capture_usage() -> dict:
    return _capture_usage.snapshot()


def get_query_usage() -> dict:
    return _query_usage.snapshot()


def reset_capture_usage() -> None:
    _capture_usage.reset()


def reset_query_usage() -> None:
    _query_usage.reset()


def reset_all() -> None:
    _capture_usage.reset()
    _query_usage.reset()

"""Shared search utilities: BM25, cosine similarity, RRF."""


def bm25_score(
    query: str,
    content: str,
    action_label: str = "",
    subjects: list[str] | None = None,
    evidence_summary: str = "",
) -> float:
    """Simple BM25-like keyword scoring.

    Combines content, action_label, subjects, and
    evidence_summary for scoring.
    Uses term frequency with diminishing returns.
    """
    query_terms = set(query.lower().split())
    text = (f"{action_label} {' '.join(subjects or [])} {evidence_summary} {content}").lower()
    text_terms = text.split()

    if not query_terms or not text_terms:
        return 0.0

    score = 0.0
    for term in query_terms:
        tf = text_terms.count(term)
        if tf > 0:
            # BM25-like: diminishing returns on TF
            score += (tf * 2.5) / (tf + 1.5)

    return score / len(query_terms)  # normalize


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
    id_key: str = "node_id",
) -> list[dict]:
    """Combine ranked lists using RRF (k=60).

    Each item must have an `id_key` field.
    Returns merged list sorted by RRF score.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item[id_key]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for item_id in sorted_ids:
        item = items[item_id].copy()
        item.pop("_score", None)
        item["rrf_score"] = round(scores[item_id], 6)
        result.append(item)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

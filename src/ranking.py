"""Final ranking stage: a local scorer (always available, zero cost) plus an
optional LLM semantic re-rank of the short local shortlist.

The local scorer blends the signals that actually separate near-duplicates in
this catalog: how much of what the customer literally asked for is present on
the item, whether it sits on the shelf they named, price agreement, fused
retrieval rank, and the anonymized long-term profile.
"""

from __future__ import annotations

import json
import os

from src.catalog_index import CatalogIndex
from src.retrieval import Evidence
from src.session_state import SessionState

WEIGHTS = {
    "coverage": 1.00,
    "popularity": 1.50,
    "prominence": 0.30,
    "fusion": 0.30,
    "price": 0.22,
    "profile": 0.06,
    "rating": 0.05,
}


def _price_fit(state: SessionState, price: float) -> float:
    slot = state.slots.get("budget")
    if slot is None or slot.budget_max is None or price != price:  # NaN check
        return 0.0
    target = slot.budget_max
    if target <= 0:
        return 0.0
    gap = abs(price - target) / target
    if gap <= 0.01:
        return 1.0  # the customer quoted this item's own price back at us
    if gap <= 0.25:
        return 0.5
    return 0.0 if price > target * 1.35 else 0.2


def _profile_alignment(state: SessionState, text: str) -> float:
    tags = [tag.lower() for tag in (state.user_profile.get("preference_tags") or [])]
    if not tags:
        return 0.0
    return sum(1 for tag in tags if tag in text) / len(tags)


def _rating_component(state: SessionState, rating: float) -> float:
    if rating != rating:  # NaN check without importing math
        return 0.5
    style = str(state.user_profile.get("rating_style", "")).lower()
    normalized = rating / 5.0
    if "critical" in style:
        return normalized  # critical raters: lean harder on well-rated items
    return 0.5 + 0.5 * normalized  # gentler pull otherwise


def local_rerank(
    candidates: list[tuple[str, float]],
    state: SessionState,
    index: CatalogIndex,
    limit: int,
    evidence: Evidence | None = None,
    shelf: set[str] | None = None,
) -> list[tuple[str, float]]:
    if not candidates:
        return []
    fusion_scores = [score for _asin, score in candidates]
    lo, hi = min(fusion_scores), max(fusion_scores)
    span = (hi - lo) or 1.0

    coverage = evidence.coverage if evidence is not None else None
    prominence = None
    if evidence is not None and float(evidence.prominence.max(initial=0.0)) > 0.0:
        prominence = evidence.prominence / float(evidence.prominence.max())

    scored: list[tuple[str, float]] = []
    for asin, fusion_score in candidates:
        i = index.index_of[asin]
        final = (
            WEIGHTS["fusion"] * ((fusion_score - lo) / span)
            + WEIGHTS["popularity"] * float(index.popularity[i])
            + WEIGHTS["price"] * _price_fit(state, float(index.price[i]))
            + WEIGHTS["profile"] * _profile_alignment(state, index.searchable[i])
            + WEIGHTS["rating"] * _rating_component(state, float(index.rating[i]))
        )
        if coverage is not None:
            final += WEIGHTS["coverage"] * float(coverage[i])
        if prominence is not None:
            final += WEIGHTS["prominence"] * float(prominence[i])
        scored.append((asin, final))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    if shelf is None:
        return scored[:limit]
    # The customer named a department. Nothing from another department belongs
    # above something from theirs, however well it scores -- a wildly popular
    # item from the wrong shelf is still the wrong answer. Off-shelf results
    # are kept only to pad the list out to `limit`.
    on_shelf = [pair for pair in scored if pair[0] in shelf]
    off_shelf = [pair for pair in scored if pair[0] not in shelf]
    return (on_shelf + off_shelf)[:limit]


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_rerank(
    shortlist: list[tuple[str, float]],
    state: SessionState,
    index: CatalogIndex,
) -> tuple[list[tuple[str, float]] | None, dict]:
    """Optional LLM polish of a short local shortlist. Returns (None, usage)
    on any failure so the caller can silently keep the local order — final
    scoring may run with network disabled, so this must never be required.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if not shortlist or not llm_available():
        return None, usage
    try:
        import anthropic
    except ImportError:
        return None, usage

    lines = []
    for asin, _score in shortlist:
        product = index.products[asin]
        title = str(product.get("title", ""))[:120]
        lines.append(f"{asin}: {title}")
    known = "; ".join(phrase for phrase, weight in state.weighted_terms() if weight > 0.3)
    prompt = (
        "Reorder these candidate products best-match-first for a shopper who said: "
        f"{known}\nCandidates:\n" + "\n".join(lines) +
        "\nReply with ONLY a JSON array of the parent_asin values in best-to-worst order."
    )
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = {
            "prompt_tokens": int(getattr(response.usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(response.usage, "output_tokens", 0) or 0),
        }
        text = response.content[0].text
        order = json.loads(text[text.index("["): text.rindex("]") + 1])
        valid = {asin for asin, _ in shortlist}
        reordered = [asin for asin in order if asin in valid]
        reordered += [asin for asin, _ in shortlist if asin not in reordered]
        by_asin = dict(shortlist)
        return [(asin, by_asin[asin]) for asin in reordered], usage
    except Exception:
        return None, usage

"""Multi-route retrieval: a verbatim-requirement route, a high-precision
keyword route for Buying, and a diverse dense route for Browsing, combined by
reciprocal rank fusion.

This is "Multi-Route Retrieval -> semantic ranking" from the brief; the
ranking half lives in src/ranking.py. Everything here runs against the
in-memory CatalogIndex, no external services.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.catalog_index import CatalogIndex
from src.session_state import SessionState
from src.slots import COLORS, MATERIALS
from src.text_utils import terms

RRF_K = 60
# What agreement on a stated primary material/colour is worth, relative to
# everything else the customer has said. Deliberately a boost and never a
# filter: "cotton" is a claim about the item's headline fabric, and a blend
# described the other way round should be demoted, not excluded.
PRIMARY_ATTRIBUTE_WEIGHT = 0.35
FILTER_DROP_ORDER = ("budget", "brand", "color", "material")

_VOCAB = {"material": set(MATERIALS), "color": set(COLORS)}


def _filter_needles(attribute: str, variants: list[str]) -> list[str]:
    """Values safe to filter on.

    Only short, known vocabulary words are used as filters. A long requirement
    phrase is a quote of catalog copy, and whitespace or punctuation drift
    between the quote and the stored text would silently exclude the very item
    the customer is describing -- those phrases are scored, never filtered.
    """
    vocabulary = _VOCAB.get(attribute)
    needles: list[str] = []
    for variant in variants:
        lowered = variant.lower().strip()
        if not lowered:
            continue
        if vocabulary is None:
            needles.append(lowered)
            continue
        needles.extend(word for word in vocabulary if word in lowered)
    return list(dict.fromkeys(needles))


def _filter_mask(index: CatalogIndex, attribute: str, variants: list[str], budget_max: float | None) -> np.ndarray:
    n = len(index.order)
    if attribute == "budget":
        if budget_max is None:
            return np.ones(n, dtype=bool)
        tolerance = budget_max * 1.35
        mask = index.price <= tolerance
        mask |= np.isnan(index.price)  # unknown price: don't punish, just don't over-trust
        return mask
    needles = _filter_needles(attribute, variants)
    if not needles:
        return np.ones(n, dtype=bool)
    if attribute == "brand":
        return np.array([any(needle in store for needle in needles) for store in index.store])
    if attribute in ("material", "color"):
        return np.array([any(needle in text for needle in needles) for text in index.searchable])
    return np.ones(n, dtype=bool)


def compute_restrict(state: SessionState, index: CatalogIndex) -> set[str] | None:
    """The eligible candidate set for this turn, or None if nothing is
    constrained yet.

    Two things narrow it: the shelf the customer named (an exact breadcrumb
    match, which is by far the strongest single constraint available) and any
    confirmed hard filters. Computed once per turn in Agent.respond and
    threaded through the tracks so the mask scan over 50k rows runs once.
    """
    shelf: set[str] | None = None
    if state.category_phrase:
        key = index.resolve_breadcrumb(state.category_phrase)
        if key:
            shelf = index.breadcrumb_members(key) or None

    filters = {
        attribute: slot
        for attribute, slot in state.hard_filters().items()
        if _filter_needles(attribute, slot.variants) or slot.budget_max is not None
    }
    if not filters:
        return shelf

    active = dict(filters)
    while active:
        mask = np.ones(len(index.order), dtype=bool)
        for attribute, slot in active.items():
            mask &= _filter_mask(index, attribute, slot.variants, slot.budget_max)
        selected = {index.order[i] for i in np.flatnonzero(mask)}
        if shelf is not None:
            selected &= shelf
        if selected:
            return selected
        # Over-constrained: relax the weakest filter first (runtime
        # re-orchestration instead of returning an empty shelf).
        for attribute in FILTER_DROP_ORDER:
            if attribute in active:
                del active[attribute]
                break
        else:
            active.pop(next(iter(active)))
    return shelf


@dataclass(frozen=True)
class Evidence:
    """What the customer's stated requirements say about every catalog item."""

    support: np.ndarray
    prominence: np.ndarray
    total: float

    @property
    def coverage(self) -> np.ndarray:
        """Fraction of the stated requirements each item satisfies."""
        return self.support / self.total if self.total > 0 else self.support


def evidence_scores(state: SessionState, index: CatalogIndex) -> Evidence:
    """IDF-weighted support for every catalog item from the requirement
    phrases the customer has actually spoken."""
    support, prominence, total = index.phrase_evidence(state.weighted_phrases())
    primary = state.primary_constraints
    if primary:
        # A stated material or colour is a claim about what the item *is*, so
        # it reinforces every other requirement rather than standing alone.
        agreement = np.zeros(len(index.order), dtype=np.float32)
        for attribute, value in primary.items():
            mask = index.by_primary.get(attribute, {}).get(value)
            if mask is not None:
                agreement += mask
        weight = PRIMARY_ATTRIBUTE_WEIGHT * max(total, 1.0) / len(primary)
        support = support + weight * agreement
        total += weight * len(primary)
    return Evidence(support=support, prominence=prominence, total=total)


def plausible_pool_size(evidence: Evidence, restrict: set[str] | None, index: CatalogIndex) -> int:
    """How many products are still genuinely in contention.

    Slot count is a poor proxy for ambiguity -- three broad constraints can
    still leave thousands of items. What matters is how many products the
    stated requirements fail to separate, which is exactly the number scoring
    near the top of the evidence distribution.
    """
    scores = evidence.support
    best = float(scores.max()) if scores.size else 0.0
    if best <= 0.0:
        return len(restrict) if restrict is not None else len(index.order)
    contenders = int(np.count_nonzero(scores >= best * 0.85))
    if restrict is None:
        return contenders
    return max(1, min(contenders, len(restrict)))


def evidence_track(evidence: Evidence, index: CatalogIndex, limit: int) -> list[tuple[str, float]]:
    """Rank by requirement support, breaking ties on how prominently the
    matched requirements sit in each product's own attribute list."""
    scores = evidence.support + 0.25 * evidence.prominence
    if not scores.size or float(scores.max()) <= 0.0:
        return []
    k = min(limit, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(index.order[i], float(scores[i])) for i in top if scores[i] > 0.0]


def buying_track(state: SessionState, index: CatalogIndex, limit: int, restrict: set[str] | None) -> list[tuple[str, float]]:
    query_terms: list[str] = []
    for phrase, _weight in state.weighted_terms():
        # Cap terms per phrase: a long requirement blurb (marketing copy the
        # customer quoted back) can otherwise inject dozens of generic OR
        # terms that dilute BM25 relevance far more than a short, specific
        # constraint like "alloy" or "waterproof".
        query_terms.extend(terms(phrase, limit=6))
    query_terms = list(dict.fromkeys(query_terms))[:40]
    if not query_terms:
        return [(asin, 1.0) for asin in list(restrict)[:limit]] if restrict else []
    return index.bm25_search(query_terms, limit, restrict=restrict)


def _mmr(index: CatalogIndex, ranked: list[tuple[str, float]], limit: int, lambda_: float = 0.7) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance re-selection, vectorized: the pairwise
    similarity matrix for the (small) candidate pool is computed once, then
    each pick is one numpy argmax rather than a Python double loop."""
    if len(ranked) <= limit:
        return ranked
    asins = [asin for asin, _score in ranked]
    relevance = np.array([score for _asin, score in ranked], dtype=np.float32)
    lo, hi = relevance.min(), relevance.max()
    if hi > lo:
        relevance = (relevance - lo) / (hi - lo)

    idxs = np.array([index.index_of[asin] for asin in asins])
    vectors = index.dense[idxs]
    similarity = vectors @ vectors.T

    m = len(asins)
    picked = np.zeros(m, dtype=bool)
    max_redundancy = np.zeros(m, dtype=np.float32)
    order: list[int] = []
    for _ in range(min(limit, m)):
        mmr_score = lambda_ * relevance - (1 - lambda_) * max_redundancy
        mmr_score[picked] = -np.inf
        pick = int(np.argmax(mmr_score))
        picked[pick] = True
        order.append(pick)
        max_redundancy = np.maximum(max_redundancy, similarity[pick])
    return [(asins[i], float(relevance[i])) for i in order]


def browsing_track(state: SessionState, index: CatalogIndex, limit: int, restrict: set[str] | None = None) -> list[tuple[str, float]]:
    # Once hard constraints are confirmed, the "diverse" track should still
    # only diversify *among valid items* -- otherwise a thematically similar
    # item that fails a known constraint can, via fusion, outrank the
    # correctly filtered match.
    query_vector = index.embed(state.dense_query_text())
    fetched = index.dense_search(query_vector, limit * 4, restrict=restrict)
    if not fetched:
        return []
    return _mmr(index, fetched, limit)


def popular_fallback(index: CatalogIndex, restrict: set[str] | None, limit: int) -> list[tuple[str, float]]:
    """Last resort when no route returned anything -- a message we could not
    parse at all still gets an answer rather than an empty list."""
    pool = restrict if restrict else set(index.order)
    ranked = sorted(pool, key=lambda asin: -float(index.popularity[index.index_of[asin]]))
    return [(asin, float(index.popularity[index.index_of[asin]])) for asin in ranked[:limit]]


def fuse(tracks: list[tuple[list[tuple[str, float]], float]], limit: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion across (ranked_list, weight) pairs."""
    scores: dict[str, float] = {}
    for ranked, weight in tracks:
        if weight <= 0:
            continue
        for rank, (asin, _score) in enumerate(ranked, start=1):
            scores[asin] = scores.get(asin, 0.0) + weight / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ordered[:limit]

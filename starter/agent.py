"""The competition Agent: one turn of the conversational shopping pipeline.

    parse the turn  ->  update dialog state  ->  route
                    ->  multi-route retrieval  ->  fusion
                    ->  semantic rerank  ->  clarify

Everything runs in-process against the frozen catalog; no network call and no
model API is required (an LLM rerank can be switched on, but the pipeline is
designed to score the same without it). The stages themselves live in `src/`
so each one can be tested on its own; this module is the orchestration and
the public API surface the evaluator talks to.
"""

from __future__ import annotations

from pathlib import Path

from src.catalog_index import CatalogIndex
from src.clarify import choose_attribute, compose_message, should_ask
from src.ranking import llm_available, llm_rerank, local_rerank
from src.retrieval import (
    browsing_track,
    buying_track,
    compute_restrict,
    evidence_scores,
    evidence_track,
    fuse,
    plausible_pool_size,
    popular_fallback,
)
from src.router import route
from src.session_state import SessionState

# How deep each route reports and how many candidates survive into reranking.
# Wide enough that a correct item ranked poorly by one route can still be
# rescued by another, small enough that reranking stays cheap.
ROUTE_DEPTH = 150
RERANK_DEPTH = 200


class Agent:
    """Dual-track conversational shopping agent.

    One CatalogIndex is shared by every session (it is read-only); each
    session_id owns an isolated SessionState holding its dialog memory.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.index = CatalogIndex(catalog_path)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(session_id=session_id, user_profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        state.observe(user_message, turn)
        weights = route(user_message, state)
        restrict = compute_restrict(state, self.index)
        evidence = evidence_scores(state, self.index)
        pool_size = plausible_pool_size(evidence, restrict, self.index)
        state.last_route, state.last_pool_size = weights, pool_size

        fused = fuse(
            [
                (evidence_track(evidence, self.index, ROUTE_DEPTH), weights.evidence),
                (buying_track(state, self.index, ROUTE_DEPTH, restrict), weights.keyword),
                (browsing_track(state, self.index, ROUTE_DEPTH, restrict), weights.dense),
            ],
            limit=RERANK_DEPTH,
        )
        if not fused:
            fused = popular_fallback(self.index, restrict, RERANK_DEPTH)
        ranked = local_rerank(fused, state, self.index, top_k, evidence=evidence, shelf=restrict)

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if llm_available():
            reordered, usage = llm_rerank(ranked, state, self.index)
            if reordered is not None:
                ranked = reordered

        fused_asins = [asin for asin, _score in fused]
        ask_attribute = None
        if should_ask(state, turn, pool_size):
            ask_attribute = choose_attribute(state, self.index, fused_asins)
        state.note_ask(ask_attribute)

        return {
            "message": compose_message(ask_attribute, state, bool(ranked), self.index, fused_asins),
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": asin, "score": round(score, 6)} for asin, score in ranked
            ],
            "usage": usage,
        }

    def session_diagnostics(self, session_id: str) -> dict[str, object]:
        """Inspection hook for the demo script; not part of the scored API."""
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before session_diagnostics")
        route_weights = state.last_route
        return {
            "route": getattr(route_weights, "mode", None),
            "buying_weight": getattr(route_weights, "buying_weight", None),
            "pool_size": state.last_pool_size,
            "category": state.category_phrase,
            "requirements": [item.phrase for item in state.evidence],
            "hard_filters": {a: slot.variants for a, slot in state.hard_filters().items()},
            "asked": sorted(state.asked_attributes),
            "exhausted": sorted(state.exhausted_attributes),
        }

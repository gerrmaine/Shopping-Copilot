"""Intent routing: how much weight the fusion stage gives each retrieval
route, re-evaluated every turn.

Rather than a one-shot Buying/Browsing label, the router returns continuous
route weights. This is what lets the system re-orchestrate at runtime: a
session that starts vague ("Browsing") but accumulates two real requirements
by turn 3 behaves like "Buying" from then on, and a session that just got
Intent-Overridden momentarily widens back out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.session_state import SessionState

BUYING_CUES = re.compile(
    r"\brequirement\b|\bmust\b|\bneed\b|\bexactly\b|\$\s*\d|\bunder\s*\$?\d|"
    r"\bin\s+(black|white|blue|red|pink|green|brown|gray|grey)\b",
    re.I,
)
BROWSING_CUES = re.compile(
    r"still exploring|not sure|open to|just looking|browsing|no particular|any (?:color|style|brand)",
    re.I,
)


@dataclass(frozen=True)
class RouteWeights:
    evidence: float
    keyword: float
    dense: float
    mode: str
    buying_weight: float


def route(message: str, state: SessionState) -> RouteWeights:
    score = 0.35  # mild prior toward using both tracks

    if BUYING_CUES.search(message):
        score += 0.25
    if BROWSING_CUES.search(message):
        score -= 0.25

    # A confirmed requirement is a much stronger signal than word choice: it
    # means we can already exclude non-matching items outright. Fusion has to
    # lean hard on precision once there is something real to enforce -- a
    # 50/50 blend lets thematically similar but wrong items outrank the
    # correctly filtered match.
    hard_slots = len(state.hard_filters())
    if hard_slots >= 2:
        score = max(score, 0.85)
    elif hard_slots == 1:
        score = max(score, 0.7)

    if state.override_events and state.turn == max(
        (0, *(slot.turn_set for slot in state.slots.values() if slot.status == "confirmed"))
    ):
        # Just overrode: momentarily widen back toward exploration until the
        # rewritten requirement has been through a fresh retrieval pass.
        score -= 0.15

    buying = max(0.1, min(0.95, score))
    return RouteWeights(
        evidence=1.0 if state.evidence else 0.0,
        keyword=buying,
        dense=1.0 - buying,
        mode="buying" if buying >= 0.5 else "browsing",
        buying_weight=round(buying, 3),
    )

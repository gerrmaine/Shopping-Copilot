"""Per-session dialog state: the slot memory the agent accumulates, rewrites,
and decays across turns (Pillar II / III of the brief).

One SessionState lives for the lifetime of a session_id. It is intentionally
the *only* place turn-to-turn memory is kept — retrieval and ranking are
stateless functions of (state, catalog).

Two layers of memory are tracked side by side:

* **Slots** — attribute/value pairs (material=leather, color=black, budget<=40)
  that are safe to apply as catalog filters.
* **Requirements** — the phrasing the customer actually used.
  Customers describe what they want by quoting product attributes almost
  verbatim, and a whole phrase is far more discriminative than the tokens it
  decomposes into, so the raw phrasing is kept alongside the parsed slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.slots import (
    ATTRIBUTES,
    CATCH_ALL_ATTRIBUTES,
    HARD_FILTER_ATTRIBUTES,
    as_bare_attribute,
    classify_phrase,
    extract_budget_max,
    parse_turn,
)
from src.text_utils import normalize_phrase

DECAY_PER_TURN = 0.92
MIN_WEIGHT = 0.55
# How long a deflected ("you decide") question stays parked before it is worth
# putting to the customer again.
DEFER_TURNS = 1
# What a requirement is worth once the customer has explicitly moved on from
# it. Not zero: in this domain an override refines the same shopping goal
# ("actually it has to be leather") far more often than it abandons it, so a
# superseded requirement is still weak positive evidence about the target.
SUPERSEDED_WEIGHT = 0.5


@dataclass
class Slot:
    variants: list[str] = field(default_factory=list)
    status: str = "confirmed"  # "confirmed" | "no_pref"
    turn_set: int = 0
    weight: float = 1.0
    budget_max: float | None = None
    stagnant: bool = False  # catch-all attribute: last ask yielded nothing new


@dataclass
class Requirement:
    """One requirement phrase, as spoken.

    ``kind`` separates a quoted product detail ("Stainless Steel Band"), which
    is matched against catalog attribute phrases, from a bare attribute word
    ("cotton"), which is a statement about the item's *primary* material or
    colour and is matched differently.
    """

    phrase: str
    attribute: str
    turn_seen: int
    weight: float = 1.0
    kind: str = "phrase"  # "phrase" | "attribute"


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    category_phrase: str | None = None
    turn: int = 0
    slots: dict[str, Slot] = field(default_factory=dict)
    evidence: list[Requirement] = field(default_factory=list)
    primary_constraints: dict[str, str] = field(default_factory=dict)
    asked_attributes: set[str] = field(default_factory=set)
    exhausted_attributes: set[str] = field(default_factory=set)
    deferred_attributes: dict[str, int] = field(default_factory=dict)
    ask_yield: dict[str, int] = field(default_factory=dict)
    override_events: int = 0
    pending_ask: str | None = None
    # Last turn's routing decision and candidate-pool estimate. Written by the
    # Agent purely so session_diagnostics can explain a turn; nothing in the
    # scored pipeline reads them back.
    last_route: object | None = None
    last_pool_size: int | None = None

    # ------------------------------------------------------------- updates
    def note_ask(self, attribute: str | None) -> None:
        """Called right after we decide this turn's ask_attribute, so the
        *next* incoming reply is attributed to the right question."""
        self.pending_ask = attribute

    def observe(self, message: str, turn: int) -> None:
        """Fold one customer message into the session's understanding."""
        self.turn = turn
        parsed = parse_turn(message)

        asked = self.pending_ask
        self.pending_ask = None
        if asked:
            self.asked_attributes.add(asked)

        if parsed.no_preference_for:
            attribute = parsed.no_preference_for
            if parsed.is_deflection:
                # "You decide" is a pass, not a closed door: park the question
                # and come back to it once the customer has seen more results.
                for name in {attribute, asked} - {None}:
                    self.deferred_attributes[name] = turn + DEFER_TURNS
            else:
                self.slots.setdefault(attribute, Slot(status="no_pref", turn_set=turn)).status = "no_pref"
                self.exhausted_attributes.add(attribute)
                if asked:
                    self.exhausted_attributes.add(asked)
            self._register_yield(asked, 0)
            self._decay(touched=set())
            return

        if parsed.is_unclear:
            self._register_yield(asked, 0)
            self._decay(touched=set())
            return

        if parsed.category:
            self.category_phrase = parsed.category

        if parsed.is_override:
            self.override_events += 1

        touched: set[str] = set()
        added = 0
        for phrase in parsed.phrases:
            if self._absorb_phrase(phrase, turn, is_override=parsed.is_override, touched=touched):
                added += 1

        self._register_yield(asked, added)
        if parsed.is_override:
            # The customer changed the subject rather than answering; the
            # question we asked is still open.
            asked = None
        if asked and added == 0:
            # The customer has nothing more to say about that attribute:
            # never spend another turn asking for it.
            self.exhausted_attributes.add(asked)
            if asked in CATCH_ALL_ATTRIBUTES:
                self.slots.setdefault(asked, Slot(turn_set=turn)).stagnant = True
        self._decay(touched)

    def _absorb_phrase(self, phrase: str, turn: int, is_override: bool, touched: set[str]) -> bool:
        normalized = normalize_phrase(phrase)
        if not normalized:
            return False
        if any(normalize_phrase(item.phrase) == normalized for item in self.evidence):
            return False

        bare = as_bare_attribute(phrase)
        if bare is not None:
            attribute, value = bare
            self.primary_constraints[attribute] = value
        else:
            attribute, value = classify_phrase(phrase)
        self.evidence.append(Requirement(
            phrase=phrase, attribute=attribute, turn_seen=turn,
            kind="attribute" if bare else "phrase",
        ))
        touched.add(attribute)

        if is_override:
            # Rewrite rather than erase: the new value leads, everything it
            # supersedes drops to a weak prior.
            for item in self.evidence[:-1]:
                if item.attribute == attribute:
                    item.weight = SUPERSEDED_WEIGHT

        slot = self.slots.get(attribute)
        if slot is None or slot.status != "confirmed":
            slot = Slot(turn_set=turn)
            self.slots[attribute] = slot
        if value and value not in slot.variants:
            if is_override:
                slot.variants.insert(0, value)
            else:
                slot.variants.append(value)
        slot.turn_set = turn
        slot.weight = 1.0
        slot.stagnant = False
        if attribute == "budget":
            budget_value = extract_budget_max(value)
            if budget_value is not None:
                slot.budget_max = budget_value
        return True

    def _register_yield(self, asked: str | None, added: int) -> None:
        if asked:
            self.ask_yield[asked] = self.ask_yield.get(asked, 0) + added

    def _decay(self, touched: set[str]) -> None:
        for attribute, slot in self.slots.items():
            if attribute in touched or slot.status == "no_pref":
                continue
            slot.weight = max(MIN_WEIGHT, slot.weight * DECAY_PER_TURN)

    # -------------------------------------------------------------- reads
    def hard_filters(self) -> dict[str, Slot]:
        return {
            attribute: slot
            for attribute, slot in self.slots.items()
            if attribute in HARD_FILTER_ATTRIBUTES and slot.status == "confirmed" and slot.variants
        }

    def weighted_phrases(self) -> list[tuple[str, float]]:
        """Quoted product details with their current confidence weight."""
        return [
            (item.phrase, item.weight * self._slot_weight(item.attribute))
            for item in self.evidence
            if item.kind == "phrase"
        ]

    def _slot_weight(self, attribute: str) -> float:
        slot = self.slots.get(attribute)
        return slot.weight if slot is not None else 1.0

    def weighted_terms(self) -> list[tuple[str, float]]:
        """Phrases usable as retrieval query text, category included."""
        terms: list[tuple[str, float]] = []
        if self.category_phrase:
            terms.append((self.category_phrase, 1.0))
        terms.extend(
            (item.phrase, item.weight * self._slot_weight(item.attribute)) for item in self.evidence
        )
        return terms

    def unresolved_attributes(self) -> list[str]:
        # "category" is excluded: the customer names their category in the
        # opening message, so asking for it can only ever waste a turn.
        result = []
        for attribute in ATTRIBUTES:
            if attribute == "category" or attribute in self.exhausted_attributes:
                continue
            if self.deferred_attributes.get(attribute, 0) > self.turn:
                continue
            slot = self.slots.get(attribute)
            if slot is not None and slot.status == "no_pref":
                continue
            if attribute in CATCH_ALL_ATTRIBUTES:
                if slot is not None and slot.stagnant:
                    continue
                result.append(attribute)
                continue
            if slot is not None:
                continue  # narrow attribute: one confirmed value is enough
            if attribute in self.asked_attributes:
                continue
            result.append(attribute)
        return result

    def dense_query_text(self) -> str:
        tags = " ".join(self.user_profile.get("preference_tags", []) or [])
        weighted = " ".join(
            f"{phrase} " * max(1, round(weight * 3)) for phrase, weight in self.weighted_terms()
        )
        return f"{weighted} {tags}".strip()

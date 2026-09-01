"""Parses free-text customer turns into structured attribute slots.

The dialog state machine (src/session_state.py) needs to turn whatever the
customer says into (attribute, value) evidence. We do not assume a fixed
template: the spec notes the private simulator may add natural-language
paraphrasing later, so extraction is keyword/regex based rather than a
literal parse of the public simulator's current phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

# Attributes we are willing to apply as hard AND filters on the catalog.
# The rest only ever influence ranking, because their vocabulary is too
# open-ended to safely exclude items on a single keyword miss.
HARD_FILTER_ATTRIBUTES = {"material", "color", "brand", "budget", "category"}

# feature/other are catch-all buckets: a category like Jewelry rarely has a
# clean "material" or "color" word (details say "Material: alloy", which
# even the reference constraint classifier buckets as "feature"), so a
# single disclosed phrase does not mean the bucket is exhausted the way it
# does for material/color/etc. These stay askable until a follow-up
# question genuinely yields nothing new.
CATCH_ALL_ATTRIBUTES = {"feature", "other"}

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric", "denim", "linen", "suede", "canvas")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "navy", "beige", "gold", "silver")
SIZE_WORDS = ("small", "medium", "large", "petite", "plus", "wide", "narrow", "xs", "xl", "xxl")
STYLE_WORDS = ("casual", "formal", "sporty", "vintage", "classic", "slim", "relaxed", "fitted", "bohemian", "elegant", "sleeve", "neck", "crew", "v-neck")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "summer", "outdoor", "work", "office", "wedding", "travel", "everyday", "party", "yoga")

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
SIZE_RE = re.compile(r"\b(" + "|".join(SIZE_WORDS) + r"|size\s*\w+)\b", re.I)
STYLE_RE = re.compile(r"\b(" + "|".join(STYLE_WORDS) + r")\b", re.I)
USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASE_WORDS) + r")\b", re.I)
BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)|budget[^0-9]{0,15}(\d+(?:\.\d+)?)|under\s*\$?\s*(\d+(?:\.\d+)?)", re.I)

# The words shoppers actually lead with when they name a material or a colour.
# Deliberately short: "the primary material" means the one the product is sold
# on, and a long tail of niche fabric names would make that judgement noisier,
# not sharper.
PRIMARY_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
PRIMARY_COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
PRIMARY_MATERIAL_RE = re.compile(r"\b(" + "|".join(PRIMARY_MATERIALS) + r")\b", re.I)
PRIMARY_COLOR_RE = re.compile(r"\b(" + "|".join(PRIMARY_COLORS) + r")\b", re.I)

# "cotton", "color: grey" -- a bare attribute word rather than a quoted
# product detail. These are matched against an item's primary attributes
# instead of against its attribute phrases.
BARE_ATTRIBUTE_RE = re.compile(
    r"^(?:(?P<key>color|colour|material|fabric)\s*[:\-]?\s*)?(?P<value>[a-z]+)$", re.I
)

NO_PREFERENCE_RE = re.compile(r"don'?t have (?:an? )?(?:additional )?preference for (\w+)", re.I)
DEFLECTION_RE = re.compile(r"use your judgment|no preference|either is fine", re.I)
OVERRIDE_CUES = re.compile(r"\bactually\b|\bignore\b|\binstead\b|changed my mind|no longer|rather than|scratch that", re.I)
UNCLEAR_CUES = re.compile(r"not quite right|ask me about", re.I)

# Product `details` are surfaced as "Key: Value" phrases (e.g. "Material:
# alloy", "Closure Type: Zipper"). The key alone is a much stronger and more
# general attribute signal than running vocabulary regexes over "alloy",
# which will never appear in a fixed materials word list.
KEY_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /]{1,30}):\s*(.+)$")
KEY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("material", ("material", "fabric", "metal type", "stone type", "chain type", "sole material")),
    ("color", ("color", "colour")),
    ("brand", ("brand", "manufacturer")),
    ("budget", ("price", "budget")),
    ("size", ("size", "dimension", "width", "length", "diameter")),
    ("style", ("department", "style", "fit", "closure", "sleeve", "neck", "collar", "toe")),
)


def _classify_by_vocab(text: str) -> str:
    lowered = text.lower()
    if BUDGET_RE.search(lowered) or "budget" in lowered:
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered) or lowered.startswith("color:"):
        return "color"
    if SIZE_RE.search(lowered):
        return "size"
    if STYLE_RE.search(lowered):
        return "style"
    if USE_CASE_RE.search(lowered):
        return "use_case"
    return "feature"


def _map_key_to_attribute(key_norm: str) -> str | None:
    for attribute, hints in KEY_HINTS:
        if any(hint in key_norm for hint in hints):
            return attribute
    return None


def classify_phrase(phrase: str) -> tuple[str, str]:
    """Best-guess (attribute, value) for a free-text constraint phrase.

    The value is what actually gets used for catalog filtering and query
    terms, so a "Key: Value" phrase is split -- filtering on the literal
    string "Material: alloy" would never match catalog text formatted as
    "material alloy", and it hides the attribute-relevant "alloy" behind a
    label the catalog never repeats verbatim.
    """
    match = KEY_VALUE_RE.match(phrase.strip())
    if match:
        key_norm = match.group(1).strip().lower()
        value = match.group(2).strip()
        mapped = _map_key_to_attribute(key_norm)
        if mapped:
            return mapped, value
        return _classify_by_vocab(value), value
    return _classify_by_vocab(phrase), phrase


BARE_NUMBER_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")


def extract_budget_max(text: str) -> float | None:
    match = BUDGET_RE.search(text)
    if match:
        value = next((group for group in match.groups() if group), None)
        if value:
            return float(value)
    # A "Price: 27.99" detail phrase has already had its key stripped by the
    # time this runs, so it is a bare number with no $/budget/under cue.
    bare = BARE_NUMBER_RE.search(text)
    return float(bare.group(1)) if bare else None


def primary_attribute(text: str, kind: str) -> str:
    """The first material / colour word mentioned in a piece of product text.

    Shoppers who lead with "cotton" mean the fabric the item is sold on, not a
    trace fibre listed in a care note, so the first mention is the one that
    counts.
    """
    pattern = PRIMARY_MATERIAL_RE if kind == "material" else PRIMARY_COLOR_RE
    match = pattern.search(text)
    return match.group(1).lower() if match else ""


def as_bare_attribute(phrase: str) -> tuple[str, str] | None:
    """Interpret a one-word requirement ("cotton", "color: grey") as an
    attribute/value pair; returns None for anything longer, which is treated
    as a quoted product detail instead."""
    match = BARE_ATTRIBUTE_RE.match(phrase.strip())
    if not match:
        return None
    value = match.group("value").lower()
    key = (match.group("key") or "").lower()
    if value in PRIMARY_COLORS and key in ("color", "colour", ""):
        return "color", value
    if value in PRIMARY_MATERIALS and key in ("material", "fabric", ""):
        return "material", value
    return None


def detect_deflection(text: str) -> bool:
    """A one-off "you decide" answer, as opposed to a settled "I have no
    opinion about X" -- the customer may still answer the same question later."""
    return bool(DEFLECTION_RE.search(text))


def detect_no_preference(text: str) -> str | None:
    match = NO_PREFERENCE_RE.search(text)
    if not match:
        return None
    attribute = match.group(1).lower()
    return attribute if attribute in ATTRIBUTES else "other"


def detect_override(text: str) -> bool:
    return bool(OVERRIDE_CUES.search(text))


def detect_unclear(text: str) -> bool:
    return bool(UNCLEAR_CUES.search(text))


# --------------------------------------------------------------------------
# Turn parsing
# --------------------------------------------------------------------------

# A customer opens by naming the shelf they are shopping ("I'm looking for
# Watches Wrist Watches, but I'm still exploring"), then keeps adding
# requirements one message at a time. Both halves are parsed separately: the
# category drives which slice of the catalog is even eligible, the
# requirements drive ranking inside it.
OPENER_RE = re.compile(
    r"^\s*(?:i'?m\s+|i\s+am\s+)?(?:looking|shopping|searching)\s+for\s+"
    r"(?P<category>[^.]*?)"
    r"(?:\s*,\s*but\b[^.]*)?"
    r"(?:\.|$)\s*(?P<rest>.*)$",
    re.I | re.S,
)
OVERRIDE_SENTENCE_RE = re.compile(
    r"^\s*(?:actually|instead)\b[^.]*(?:ignore|forget|scratch|changed my mind|no longer)[^.]*\.\s*",
    re.I,
)
LEAD_IN_RE = re.compile(
    r"^\s*(?:for that,?\s*what matters is|a key requirement is|what i need is|"
    r"i (?:also )?need|i want|it must be|make it)\s*:?\s*",
    re.I,
)
# Requirements arrive as a "; "-joined list; a single requirement can itself
# contain sentence punctuation (catalog blurbs do), so only ";" separates.
PHRASE_SPLIT_RE = re.compile(r"\s*;\s*")


@dataclass
class ParsedTurn:
    category: str | None = None
    phrases: list[str] = field(default_factory=list)
    is_override: bool = False
    no_preference_for: str | None = None
    is_deflection: bool = False
    is_unclear: bool = False


def _split_requirements(text: str) -> list[str]:
    """Requirements arrive as a "; "-joined list -- but a single requirement
    quoted from a product's own copy can itself contain semicolons ("Hand
    wash; hang to dry; no ironing"). Both readings are kept: the whole segment
    and each part, so whichever one the catalog actually stores will match.
    """
    whole = LEAD_IN_RE.sub("", text.strip()).strip(" \t\n.,;:")
    parts: list[str] = []
    for part in PHRASE_SPLIT_RE.split(text):
        part = LEAD_IN_RE.sub("", part.strip()).strip(" \t\n.,;:")
        if len(part) >= 2:
            parts.append(part)
    if len(parts) > 1 and len(whole) >= 2:
        parts.insert(0, whole)
    return parts


def parse_turn(message: str) -> ParsedTurn:
    """Split one customer message into category, requirement phrases, and
    conversational control signals (override / no-preference / stuck)."""
    parsed = ParsedTurn()
    text = message.strip()
    if not text:
        return parsed

    attribute = detect_no_preference(text)
    if attribute:
        parsed.no_preference_for = attribute
        parsed.is_deflection = detect_deflection(text)
        return parsed
    if detect_unclear(text):
        parsed.is_unclear = True
        return parsed

    stripped, replaced = OVERRIDE_SENTENCE_RE.subn("", text)
    if replaced or detect_override(text):
        parsed.is_override = True
        text = stripped

    opener = OPENER_RE.match(text)
    if opener:
        category = opener.group("category").strip(" \t\n,.")
        if category:
            parsed.category = category
        text = opener.group("rest")

    text = LEAD_IN_RE.sub("", text.strip())
    parsed.phrases = _split_requirements(text)
    return parsed

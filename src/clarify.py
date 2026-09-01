"""Proactive clarification: deciding *whether* to ask, *what* to ask, and
*how* to phrase it (Pillar II, "Over-Generality" handling).

Recommendations are always returned regardless of whether we also ask a
question -- the question steers the *next* turn, it never withholds this
turn's best guess.

The choice of question is an expected-information calculation, not a fixed
script: each askable attribute is scored by how much it would split the
current candidate pool, discounted by how likely the customer is to have an
answer for it at all. That second term is re-estimated during the session
from what each question actually returned, which is what stops the agent
from burning turns on attributes this particular shopper does not think in.
"""

from __future__ import annotations

import math
from collections import Counter

from src.catalog_index import CatalogIndex
from src.session_state import SessionState
from src.slots import COLOR_RE, MATERIAL_RE, SIZE_RE, STYLE_RE, USE_CASE_RE, classify_phrase

# An open-ended question always has an answer -- the customer names whichever
# requirement they consider most important. A targeted attribute question can
# come back empty simply because this shopper has no opinion about it.
OPEN_ENDED_PRIOR = 1.0
TARGETED_PRIOR = 0.55
POOL_SAMPLE = 300

VOCAB_RE = {
    "material": MATERIAL_RE,
    "color": COLOR_RE,
    "size": SIZE_RE,
    "style": STYLE_RE,
    "use_case": USE_CASE_RE,
}
OPEN_ENDED = ("other", "feature")

# Fallback phrasing, used only when the customer has not named a category we
# can turn into a noun (see _category_noun). Anything we can phrase in terms
# of what they actually said is phrased that way instead.
TEMPLATES = {
    "material": "Do you have a material preference, like cotton, leather, or polyester?",
    "color": "Is there a particular color you'd like?",
    "size": "What size or fit are you looking for?",
    "style": "What style or fit would you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "Do you have a budget or price range in mind?",
    "use_case": "What will you mainly use this for?",
    "feature": "Is there a specific feature that matters most to you?",
    "other": "What matters most to you about this one? Any detail helps me narrow it down.",
}
# The same questions, grounded in the item the customer is actually shopping
# for ("What size red dress do you need?" rather than "What size or fit are
# you looking for?").
SUBJECT_TEMPLATES = {
    "material": "What should the {subject} be made of?",
    "color": "What color {subject} are you looking for?",
    "size": "What size {subject} do you need?",
    "style": "What style of {subject} would you prefer?",
    "brand": "Is there a particular brand of {subject} you prefer?",
    "budget": "What's your budget for the {subject}?",
    "use_case": "What will you mainly wear the {subject} for?",
}
# How to name an unresolved attribute inside an open-ended question.
DIMENSION_NAMES = {
    "color": "the color",
    "material": "the material",
    "size": "the size",
    "style": "the style",
    "use_case": "what you'll wear it for",
    "brand": "the brand",
    "budget": "your budget",
}
ATTRIBUTE_PRIORITY = ("other", "feature", "material", "color", "budget", "brand", "size", "style", "use_case")

# Nouns that are already plural in their singular sense: "What size jeans"
# is correct, "What size jean" is not.
PLURAL_ONLY = {
    "jeans", "pants", "shorts", "trousers", "leggings", "tights", "pajamas",
    "overalls", "sunglasses", "glasses", "briefs", "boxers", "slacks",
    # Footwear is bought and sized as a pair: "what size sneakers" is how
    # people speak, "what size sneaker" is not.
    "shoes", "sneakers", "boots", "sandals", "heels", "flats", "pumps",
    "loafers", "slippers", "booties", "clogs",
}
# Breadcrumb words that are real category tokens but useless as a spoken noun
# -- "What color accessory" is worse than falling through to the next word.
GENERIC_CATEGORY_WORDS = {
    "accessories", "clothing", "items", "products", "novelty", "mens",
    "womens", "ladies", "shoes & jewelry",
}
# Values short and clean enough to read back as a qualifier ("red dress").
QUALIFIER_ORDER = ("color", "material")
MAX_RECAP = 2
RECAP_CHARS = 48


def should_ask(state: SessionState, turn: int, pool_size: int) -> bool:
    if turn >= 10:
        return False  # the session ends on this turn; an answer could never arrive
    if not state.unresolved_attributes():
        return False
    # One product left that fits everything the customer said: stop asking and
    # let the recommendation stand.
    return pool_size > 1


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0 or len(counts) < 2:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy / math.log2(len(counts))  # normalized to [0, 1]


def _value_counts(attribute: str, index: CatalogIndex, sample: list[str], texts: list[str]) -> Counter:
    """How the candidate pool distributes over one attribute's values.

    Shared by the information-gain calculation and by the question phrasing, so
    an example offered to the customer is drawn from the same statistics that
    made the attribute worth asking about in the first place.
    """
    if attribute == "brand":
        # Unnamed stores are counted as their own bucket on purpose: "no brand
        # given" is a real, distinguishing answer to a brand question.
        return Counter(index.store[index.index_of[asin]] for asin in sample)
    pattern = VOCAB_RE.get(attribute)
    if pattern is None:
        return Counter()
    counts: Counter = Counter()
    for text in texts:
        match = pattern.search(text)
        if match:
            counts[match.group(1).lower()] += 1
    return counts


def _split_power(attribute: str, index: CatalogIndex, sample: list[str], texts: list[str]) -> float:
    if attribute in OPEN_ENDED:
        # A distinguishing detail can separate any two items, which is
        # precisely what makes the bucket open-ended.
        return 1.0
    if attribute == "budget":
        prices = [index.price[index.index_of[asin]] for asin in sample]
        prices = [p for p in prices if p == p]  # drop NaN
        if len(prices) < 2 or max(prices) <= 0:
            return 0.0
        return float((max(prices) - min(prices)) / max(prices))
    if attribute == "brand":
        return _entropy(_value_counts(attribute, index, sample, texts))
    if attribute not in VOCAB_RE:
        return 0.0
    return _entropy(_value_counts(attribute, index, sample, texts))


def _expected_yield(state: SessionState, attribute: str) -> float:
    prior = OPEN_ENDED_PRIOR if attribute in OPEN_ENDED else TARGETED_PRIOR
    # Attributes the customer has closed off are already filtered out of
    # `unresolved_attributes`; anything asked before and still open is worth
    # slightly less than a fresh question because the easy answers are spent.
    return prior * 0.9 if attribute in state.asked_attributes else prior


def choose_attribute(state: SessionState, index: CatalogIndex, candidates: list[str]) -> str | None:
    unresolved = state.unresolved_attributes()
    if not unresolved:
        return None
    sample = candidates[:POOL_SAMPLE] or list(index.order[:POOL_SAMPLE])
    texts = [index.searchable[index.index_of[asin]] for asin in sample]

    gains = {
        attribute: _split_power(attribute, index, sample, texts) * _expected_yield(state, attribute)
        for attribute in unresolved
    }
    ranked = sorted(
        unresolved,
        key=lambda attribute: (
            -gains[attribute],
            ATTRIBUTE_PRIORITY.index(attribute) if attribute in ATTRIBUTE_PRIORITY else 99,
        ),
    )
    best = ranked[0]
    return best if gains[best] > 0.0 else None


# ------------------------------------------------------------ question text

def _singularize(word: str) -> str:
    if word in PLURAL_ONLY:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _category_noun(state: SessionState) -> str | None:
    """A speakable noun for whatever the customer said they are shopping for.

    Category breadcrumbs do not reliably end in their own head noun -- "Dresses
    Casual" and "Running Road Running" both end on a modifier -- so this scans
    right to left for the last plural-looking token and singularizes it. When
    nothing in the phrase looks like a product noun we return None and the
    caller falls back to generic phrasing, which is better than reading a
    modifier back to the customer as if it were the item.
    """
    phrase = (state.category_phrase or "").strip()
    if not phrase:
        return None
    for token in reversed(phrase.split()):
        word = token.strip("&,()").lower()
        if len(word) < 4 or word in GENERIC_CATEGORY_WORDS:
            continue
        if not word.replace("-", "").isalpha():
            continue
        if word in PLURAL_ONLY or word.endswith("s"):
            return _singularize(word)
    return None


def _qualifiers(state: SessionState) -> list[tuple[str, str]]:
    """Short attribute values already confirmed, for reading back as
    modifiers: colour and material are the two that behave like adjectives."""
    found: list[tuple[str, str]] = []
    for attribute in QUALIFIER_ORDER:
        value = state.primary_constraints.get(attribute)
        if value is None:
            slot = state.slots.get(attribute)
            if slot is not None and slot.status == "confirmed" and slot.variants:
                candidate = slot.variants[0]
                value = candidate if len(candidate.split()) == 1 else None
        if value and value.replace("-", "").isalpha():
            found.append((attribute, value.lower()))
    return found


def _subject(state: SessionState, exclude: str | None = None) -> str | None:
    """What to call the item in a question: "red dress", "watch", or None.

    ``exclude`` drops the attribute currently being asked about, so a question
    can never contradict itself ("What color red dress are you looking for?").
    """
    noun = _category_noun(state)
    if noun is None:
        return None
    return " ".join([*(q for a, q in _qualifiers(state) if a != exclude), noun])


def _is_plural(subject: str) -> bool:
    return subject.split()[-1] in PLURAL_ONLY


def _with_article(subject: str) -> str:
    if _is_plural(subject):
        return subject
    return f"{'an' if subject[0] in 'aeiou' else 'a'} {subject}"


def _join(values: list[str], conjunction: str = "or") -> str:
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} {conjunction} {values[-1]}"


def _pool_examples(attribute: str, index: CatalogIndex | None, sample: list[str], texts: list[str]) -> list[str]:
    """Concrete values actually present in the current candidate pool.

    Offering the customer choices that exist keeps a question answerable: there
    is no point suggesting "cotton, leather, or polyester" when every remaining
    candidate is alloy or stainless steel.
    """
    if index is None or not sample or attribute not in VOCAB_RE and attribute != "brand":
        return []
    counts = _value_counts(attribute, index, sample, texts)
    values = [
        value for value, _count in counts.most_common()
        # "size 8"/"size on" are artifacts of the size pattern's open
        # `size\s*\w+` arm matching running prose; they are not choices a
        # customer can pick from.
        if value and len(value) < 20 and not value.startswith("size")
    ]
    return values[:3] if len(values) >= 2 else []


def _open_dimensions(state: SessionState, index: CatalogIndex | None, sample: list[str], texts: list[str]) -> list[str]:
    """Unresolved attributes that genuinely vary across the remaining pool,
    named in customer-facing words -- what to suggest when the question itself
    has to stay open-ended."""
    if index is None or not sample:
        return []
    unresolved = [a for a in state.unresolved_attributes() if a in DIMENSION_NAMES]
    scored = sorted(
        ((a, _split_power(a, index, sample, texts)) for a in unresolved),
        key=lambda pair: -pair[1],
    )
    return [DIMENSION_NAMES[a] for a, power in scored[:2] if power > 0.0]


def _recap(state: SessionState) -> str:
    """The requirements already on record, read back so the customer can see
    what the next answer is adding to rather than repeating.

    Values are read back the way the customer would say them: the catalog's
    "Material: alloy" key prefix is stripped by the same parser that turns the
    phrase into a slot, and whole first-person sentences ("I prefer a different
    style") are skipped -- they are real evidence for retrieval but they are
    not values, and repeating one back sounds like a misunderstanding.
    """
    phrases: list[str] = []
    for item in state.evidence:
        phrase = item.phrase.strip()
        if phrase.lower().startswith(("i ", "i'")):
            continue
        _attribute, value = classify_phrase(phrase)
        value = (value or phrase).strip()
        if value and len(value) <= RECAP_CHARS and value not in phrases:
            phrases.append(value)
    return _join(phrases[-MAX_RECAP:], conjunction="and") if phrases else ""


def _question(
    attribute: str,
    state: SessionState,
    subject: str | None,
    index: CatalogIndex | None,
    sample: list[str],
    texts: list[str],
) -> str:
    if attribute in OPEN_ENDED:
        about = f" about {_with_article(subject)}" if subject else ""
        dimensions = _open_dimensions(state, index, sample, texts)
        hint = f" Even {_join(dimensions)} would help." if dimensions else " Any detail helps me narrow it down."
        recap = _recap(state)
        # `other` and `feature` are both catch-alls and are frequently asked on
        # consecutive turns. They must not produce the same sentence twice --
        # a verbatim repeat reads as if the agent did not hear the last answer,
        # so `feature` asks for a must-have property rather than re-opening the
        # same "anything else?" question.
        if attribute == "feature":
            if subject:
                verb = "have" if _is_plural(subject) else "has"
                return f"Is there a specific feature {_with_article(subject)} {verb} to have?{hint}"
            return TEMPLATES["feature"] + hint
        if recap:
            return f"So far I have {recap}. What else matters{about}?{hint}"
        return f"What matters most{about}?{hint}"

    template = SUBJECT_TEMPLATES.get(attribute)
    question = template.format(subject=subject) if (template and subject) else TEMPLATES.get(attribute, TEMPLATES["other"])
    examples = _pool_examples(attribute, index, sample, texts)
    if examples:
        question += f" For example: {_join(examples)}."
    return question


def compose_message(
    attribute: str | None,
    state: SessionState,
    has_recommendations: bool,
    index: CatalogIndex | None = None,
    candidates: list[str] | None = None,
) -> str:
    subject = _subject(state, exclude=attribute)
    if has_recommendations:
        # "matches for a red dress" rather than "red dress matches": it reads
        # correctly whether the noun is singular or a plural-only one (shoes).
        lead = f"Here are the closest matches for {_with_article(subject)} so far. " if subject else "Here are the closest matches I found so far. "
    else:
        lead = "Let me help narrow this down. "
    if attribute is None:
        return lead.strip()

    sample = (candidates or [])[:POOL_SAMPLE]
    texts = [index.searchable[index.index_of[asin]] for asin in sample] if index is not None else []
    return lead + _question(attribute, state, subject, index, sample, texts)

"""Shared text helpers for turning catalog JSON fields into flat strings."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# A short first-person/copula word like "am" left out of this list is not a
# cosmetic miss: it survives as a BM25 query term, and because it barely
# appears anywhere in the catalog except a handful of pun titles ("I Am A
# Cat Shirt"), its rarity gives it an anomalously high IDF weight that can
# swamp the genuinely relevant terms in the same query (observed directly:
# "I am looking for a cotton jacket" was outranked onto cat T-shirts). This
# is a standard English stopword list, not just the handful of words that
# happened to show up in testing.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "am", "is", "was", "were", "be",
    "been", "being", "but", "by", "can", "could", "did", "do", "does",
    "doing", "don", "down", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "it", "its", "itself", "just",
    "may", "me", "might", "more", "most", "must", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "please", "same",
    "shall", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "want", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
    # domain-specific phrasing that leaks in from the simulator's templates
    "looking", "prefer", "preference", "need", "still", "exploring",
    "requirement", "matters", "additional", "judgment",
}


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Top-of-tree breadcrumbs shared by essentially every row in this catalog; they
# carry no discriminative signal and are dropped when naming a category.
GENERIC_BREADCRUMBS = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def normalize_phrase(text: object) -> str:
    """Casefold and squash punctuation so the same attribute phrase written by
    a customer and stored in the catalog compare equal."""
    return _NON_ALNUM_RE.sub(" ", str(text).lower()).strip()


def breadcrumb_tail(categories: object) -> str:
    """The two most specific levels of a product's category breadcrumb.

    This is how people name a department in conversation -- "Watches Wrist
    Watches", not the full "Clothing, Shoes & Jewelry > Men > ..." path -- so
    it is what we index and what we match an incoming category phrase against.
    """
    parts: list[str] = []
    for value in categories or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_BREADCRUMBS:
                parts.append(part)
    return " ".join(parts[-2:]) if parts else "clothing item"


def flatten(value: object) -> str:
    """Turn a catalog field (str/list/dict/None) into a single text blob."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


# Most specific first: a product's own title and feature bullets describe it
# far more reliably than its breadcrumb or seller name. Order matters -- the
# "primary" material/colour of an item is defined as the first one mentioned
# in this text.
SEARCHABLE_FIELDS = ("title", "features", "details", "description", "categories", "store")


def searchable_text(product: dict, fields: tuple[str, ...] = SEARCHABLE_FIELDS) -> str:
    return " ".join(flatten(product.get(field)) for field in fields).strip()


def terms(text: str, limit: int | None = None) -> list[str]:
    words = [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]
    unique = list(dict.fromkeys(words))
    return unique[:limit] if limit else unique

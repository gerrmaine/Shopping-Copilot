"""In-memory catalog indexes: keyword (SQLite FTS5/BM25) and dense retrieval.

Everything lives in process memory; the only thing written to disk is a small
cache of the dense embeddings so repeated evaluator runs do not repay the
build cost on every start. No external vector database or model training is
involved.

The default dense backend is TF-IDF + SVD: no network access, no external
model download, and it builds in seconds. A local sentence-transformer
encoder is available as an opt-in swap (set AGENT_USE_ST=1) for a genuine
semantic encoder instead of a bag-of-words proxy, at the cost of a one-time
~19 minute CPU encode of all 50k items plus a first-run download from the HF
Hub (cached to disk afterward). It is not the default because the dense
route is the fallback here rather than the primary signal. Any failure to
load it (package missing, no network, no cache) falls back to TF-IDF + SVD.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.slots import primary_attribute
from src.text_utils import SEARCHABLE_FIELDS, breadcrumb_tail, flatten, normalize_phrase, searchable_text

DENSE_DIM = 160
CACHE_SUFFIX = ".dense_cache.pkl"
# Shorter than this a "phrase" is a bare word that token retrieval already
# handles, and indexing it only adds noise.
MIN_PHRASE_KEY_LEN = 4
# A prefix that expands to more catalog strings than this is not a quote of
# one product's attribute, it is a common stem.
MAX_PREFIX_MATCHES = 40

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ST_CACHE_SUFFIX = ".st_cache.npz"


class CatalogIndex:
    """Loads the frozen catalog once and exposes keyword + dense retrieval."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.products: dict[str, dict] = {}
        self.order: list[str] = []
        self._load_products()
        self._build_attribute_phrase_index()
        self._build_fts()
        self.dense_backend = "unset"
        self._build_dense()

    # ---------------------------------------------------------------- load
    def _load_products(self) -> None:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.products[parent_asin] = product
                self.order.append(parent_asin)

        n = len(self.order)
        self.price = np.full(n, np.nan, dtype=np.float32)
        self.rating = np.full(n, np.nan, dtype=np.float32)
        self.rating_count = np.zeros(n, dtype=np.float32)
        self.store = [""] * n
        self.searchable = [""] * n
        self.breadcrumb = [""] * n
        self.by_breadcrumb: dict[str, list[int]] = {}
        for i, parent_asin in enumerate(self.order):
            product = self.products[parent_asin]
            price = product.get("price")
            if isinstance(price, (int, float)):
                self.price[i] = float(price)
            rating = product.get("average_rating")
            if isinstance(rating, (int, float)):
                self.rating[i] = float(rating)
            count = product.get("rating_number")
            if isinstance(count, (int, float)):
                self.rating_count[i] = float(count)
            self.store[i] = flatten(product.get("store")).lower()
            self.searchable[i] = searchable_text(product).lower()
            crumb = normalize_phrase(breadcrumb_tail(product.get("categories")))
            self.breadcrumb[i] = crumb
            self.by_breadcrumb.setdefault(crumb, []).append(i)
        self.index_of = {asin: i for i, asin in enumerate(self.order)}
        # Demand prior. Review volume is the only signal a static catalog
        # carries about what people actually buy, and it spans five orders of
        # magnitude here, so it is compressed logarithmically: the meaningful
        # gap is between 300 reviews and 60,000, not between 3 and 300. A
        # percentile scale flattens exactly that difference and is why this is
        # not a simple rank.
        log_counts = np.log1p(self.rating_count)
        ceiling = float(log_counts.max()) or 1.0
        self.popularity = (log_counts / ceiling).astype(np.float32)
        self._breadcrumb_tokens = {
            crumb: frozenset(crumb.split()) for crumb in self.by_breadcrumb
        }

    # ---------------------------------------------------- attribute phrases
    def _build_attribute_phrase_index(self) -> None:
        """Invert the catalog on whole attribute phrases, not just tokens.

        A customer describing a requirement tends to quote a product attribute
        close to verbatim ("Day / Date Indicator", "Pull-On closure"). Bag-of-
        words retrieval dissolves those into common tokens; keeping the phrase
        intact makes it a near-unique key -- 90% of the phrases in this catalog
        occur on exactly one product.

        Each posting also records *where* on the product the phrase sits:
        a headline bullet says much more about what an item is than the last
        line of its spec table.
        """
        postings: dict[str, list[tuple[int, int]]] = {}
        self.primary_material = [""] * len(self.order)
        self.primary_color = [""] * len(self.order)
        for i, parent_asin in enumerate(self.order):
            product = self.products[parent_asin]
            features = product.get("features")
            details = product.get("details")
            raw: list[str] = []
            if isinstance(features, list):
                raw.extend(str(item) for item in features)
            if isinstance(details, dict):
                raw.extend(f"{key}: {value}" for key, value in details.items())
            elif isinstance(details, list):
                raw.extend(str(item) for item in details)
            for position, phrase in enumerate(raw):
                key = normalize_phrase(phrase)
                if len(key) < MIN_PHRASE_KEY_LEN:
                    continue
                bucket = postings.setdefault(key, [])
                if not bucket or bucket[-1][0] != i:
                    bucket.append((i, position))
            text = self.searchable[i]
            self.primary_material[i] = primary_attribute(text, "material")
            self.primary_color[i] = primary_attribute(text, "color")

        self.by_primary = {"material": {}, "color": {}}
        for kind, values in (("material", self.primary_material), ("color", self.primary_color)):
            for value in set(values) - {""}:
                self.by_primary[kind][value] = np.fromiter(
                    (item == value for item in values), dtype=bool, count=len(values),
                )

        total = float(len(self.order))
        self.phrase_postings = {
            key: (
                np.fromiter((i for i, _pos in bucket), dtype=np.int32, count=len(bucket)),
                np.fromiter((pos for _i, pos in bucket), dtype=np.int16, count=len(bucket)),
                float(np.log(total / len(bucket))),
            )
            for key, bucket in postings.items()
        }
        self._sorted_phrase_keys = sorted(self.phrase_postings)

    def phrase_lookup(self, phrase: str) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Resolve a spoken requirement to (product indices, positions, idf).

        Exact match first. Failing that, the phrase is treated as the start of
        a longer catalog string: customers paraphrase the tail, cut a quote
        short, or hand back one clause of a multi-clause attribute, and the
        head of the phrase is the part that stays stable.
        """
        key = normalize_phrase(phrase)
        if len(key) < MIN_PHRASE_KEY_LEN:
            return None
        exact = self.phrase_postings.get(key)
        if exact is not None:
            return exact

        start = bisect.bisect_left(self._sorted_phrase_keys, key)
        matches = []
        for candidate in self._sorted_phrase_keys[start: start + MAX_PREFIX_MATCHES]:
            if not candidate.startswith(key):
                break
            matches.append(self.phrase_postings[candidate])
        if not matches:
            return None
        indices = np.concatenate([entry[0] for entry in matches])
        positions = np.concatenate([entry[1] for entry in matches])
        return indices, positions, min(entry[2] for entry in matches)

    def phrase_evidence(self, weighted_phrases: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray, float]:
        """Accumulate IDF-weighted verbatim-phrase support over the catalog.

        Returns (support, prominence, total_weight):

        * ``support`` -- how much of what the customer said this product
          carries, weighted by how rare each phrase is;
        * ``prominence`` -- the same, but discounted by how far down the
          product's own attribute list each match sits;
        * ``total_weight`` -- the score a product matching everything would
          get, which turns ``support`` into a coverage fraction.
        """
        n = len(self.order)
        support = np.zeros(n, dtype=np.float32)
        prominence = np.zeros(n, dtype=np.float32)
        total = 0.0
        for phrase, weight in weighted_phrases:
            found = self.phrase_lookup(phrase)
            if found is None:
                continue
            indices, positions, idf = found
            contribution = weight * idf
            total += contribution
            support[indices] += contribution
            prominence[indices] += contribution / (1.0 + positions.astype(np.float32))
        return support, prominence, total

    # ------------------------------------------------------------ category
    def resolve_breadcrumb(self, text: str) -> str | None:
        """Map a spoken category phrase onto an indexed breadcrumb tail.

        Exact match first, then a trimmed suffix (people drop a leading
        qualifier), then best token overlap -- so a paraphrased or partial
        category still lands on the right shelf instead of nothing.
        """
        normalized = normalize_phrase(text)
        if not normalized:
            return None
        if normalized in self.by_breadcrumb:
            return normalized
        words = normalized.split()
        for start in range(len(words)):
            for end in range(len(words), start, -1):
                candidate = " ".join(words[start:end])
                if candidate in self.by_breadcrumb:
                    return candidate
        query_tokens = set(words)
        if not query_tokens:
            return None
        best_key, best_score = None, 0.0
        for crumb, tokens in self._breadcrumb_tokens.items():
            overlap = len(query_tokens & tokens)
            if not overlap:
                continue
            score = overlap / len(query_tokens | tokens)
            if score > best_score:
                best_key, best_score = crumb, score
        return best_key if best_score >= 0.5 else None

    def breadcrumb_members(self, key: str) -> set[str]:
        return {self.order[i] for i in self.by_breadcrumb.get(key, ())}

    # ----------------------------------------------------------------- fts
    def _build_fts(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        cursor = self.connection.cursor()
        # Porter stemming matters more than it might look: a customer's coarse
        # category phrase (built from the plural catalog breadcrumb, e.g.
        # "Bras Everyday Bras") frequently mismatches a target item's own
        # singular title wording ("Wireless Bra") token-for-token, which the
        # plain unicode61 tokenizer cannot bridge and Porter stemming can.
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for parent_asin in self.order:
            product = self.products[parent_asin]
            batch.append((
                parent_asin,
                flatten(product.get("title")),
                flatten(product.get("categories")),
                flatten(product.get("features")),
                flatten(product.get("details")),
                flatten(product.get("store")),
                flatten(product.get("description")),
            ))
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def bm25_search(self, query_terms: list[str], limit: int, restrict: set[str] | None = None) -> list[tuple[str, float]]:
        """Returns [(parent_asin, higher_is_better_score), ...] ordered best-first."""
        if not query_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in query_terms)
        # bm25() in SQLite returns *lower-is-better*; over-fetch when we still
        # need to intersect with a filtered candidate set client-side.
        fetch_limit = limit if restrict is None else max(limit * 20, 500)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, fetch_limit),
        ).fetchall()
        results: list[tuple[str, float]] = []
        for parent_asin, raw_score in rows:
            if restrict is not None and parent_asin not in restrict:
                continue
            results.append((str(parent_asin), -float(raw_score)))
            if len(results) >= limit:
                break
        return results

    # --------------------------------------------------------------- dense
    def _cache_path(self, suffix: str, extra: str = "") -> Path:
        digest = hashlib.sha256()
        stat = self.catalog_path.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{extra}".encode())
        return self.catalog_path.parent / f"{self.catalog_path.stem}.{digest.hexdigest()[:12]}{suffix}"

    def _build_dense(self) -> None:
        if os.environ.get("AGENT_USE_ST") == "1":
            try:
                self._build_dense_sentence_transformer()
                self.dense_backend = "sentence_transformer"
                return
            except Exception:
                pass  # no network / no cached weights / package missing: fall back
        self._build_dense_tfidf()
        self.dense_backend = "tfidf_svd"

    def _build_dense_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        cache_path = self._cache_path(ST_CACHE_SUFFIX, extra=f"{ST_MODEL_NAME}:{'|'.join(SEARCHABLE_FIELDS)}")
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            if list(cached["order"]) == self.order:
                self.dense = cached["dense"].astype(np.float32)
                self._st_model = SentenceTransformer(ST_MODEL_NAME)
                return

        model = SentenceTransformer(ST_MODEL_NAME)
        # Truncate: the model's own max sequence length caps this anyway;
        # trimming input strings first just keeps encoding fast.
        corpus = [text[:800] for text in self.searchable]
        dense = model.encode(
            corpus, batch_size=256, show_progress_bar=False, normalize_embeddings=True,
        ).astype(np.float32)
        self.dense = dense
        self._st_model = model
        try:
            np.savez_compressed(cache_path, order=np.array(self.order), dense=dense)
        except OSError:
            pass

    def _build_dense_tfidf(self) -> None:
        # The fingerprint includes the field order because the vectorizer uses
        # bigrams: reordering the source fields changes the vectors, so a cache
        # built before such a change must not be reused.
        cache_path = self._cache_path(CACHE_SUFFIX, extra=f"{DENSE_DIM}:{'|'.join(SEARCHABLE_FIELDS)}")
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("order") == self.order:
                self.vectorizer = cached["vectorizer"]
                self.svd = cached["svd"]
                self.dense = cached["dense"]
                return

        corpus = self.searchable
        self.vectorizer = TfidfVectorizer(
            max_features=60_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
        )
        tfidf = self.vectorizer.fit_transform(corpus)
        n_components = min(DENSE_DIM, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        dense = self.svd.fit_transform(tfidf).astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense = dense / norms

        try:
            with cache_path.open("wb") as handle:
                pickle.dump(
                    {"order": self.order, "vectorizer": self.vectorizer, "svd": self.svd, "dense": self.dense},
                    handle,
                )
        except OSError:
            pass  # cache is a pure optimization; ignore write failures

    def embed(self, text: str) -> np.ndarray:
        if self.dense_backend == "sentence_transformer":
            vector = self._st_model.encode([text], normalize_embeddings=True).astype(np.float32)
            return vector[0]
        vector = self.svd.transform(self.vectorizer.transform([text])).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector[0]

    def dense_search(self, query_vector: np.ndarray, limit: int, restrict: set[str] | None = None) -> list[tuple[str, float]]:
        if not np.any(query_vector):
            return []
        scores = self.dense @ query_vector
        if restrict is not None:
            mask = np.zeros(len(scores), dtype=bool)
            idxs = np.fromiter((self.index_of[a] for a in restrict), dtype=np.int64, count=len(restrict))
            mask[idxs] = True
            scores = np.where(mask, scores, -np.inf)
        k = min(limit * 3, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])][:limit]
        return [(self.order[i], float(scores[i])) for i in top if scores[i] != -np.inf]

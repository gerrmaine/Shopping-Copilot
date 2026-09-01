# Shopping Copilot -- Conversational Search and Recommendations

A conversational shopping agent that finds a customer's target product in a 50,000-item Amazon
catalog within at most 10 turns, by reading what they quote, asking the question that splits the
candidate pool fastest, and re-deciding its own retrieval strategy every turn.

Built for TechJam problem 4 (*Shopping Copilot: AI Conversational Search and Recommendations*) on
top of the organizer's frozen catalog, session protocol, and local evaluator.

| Public dev set (200 sessions) | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Weak BM25 starter (organizer baseline) | 0.125 | 0.068 | 9.81 | 0.107 |
| Our earlier hybrid agent | 0.805 | 0.464 | 3.96 | 0.683 |
| **This solution** | **0.995** | **0.778** | **1.73** | **0.916** |

These are the 200 *public* development sessions. The organizer grades on 800 private sessions with
different users and target products, so treat this as a proxy signal, not a predicted final score.

## Demo Video

**Demo video:** https://youtu.be/WQZfmKHGm7g 

The walkthrough is a backend/NLP demo: `python -m scripts.chat_demo` plays a live session and
prints, per turn, the routing decision and its buying weight, the candidate pool size, the
requirements understood so far, the active hard filters, the clarifying question, and the top 10
recommendations -- which is the whole pipeline visible in one screen.

## The Problem

Each session gives the agent an anonymized preference profile and a short customer message. Raw
user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the
agent may ask one clarifying question (naming the field in `ask_attribute`), return up to 10
ranked `parent_asin` values, or both. The session ends when the target appears in the Top 10, or
after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behaviour.

Only exact `parent_asin` equality counts as a hit.

## How It Works

The design starts from one observation about how people describe what they want: **they quote**. A
shopper does not paraphrase "the watch has a day/date complication" -- they say "Day / Date
Indicator", the way the attribute is written on the product page. Bag-of-words retrieval dissolves
that quote into common tokens; keeping it whole makes it a near-unique key. Measured on this
catalog: of 220,350 indexed attribute phrases, **89.6% appear on exactly one product** (median
postings per phrase: 1). The pipeline is built to exploit that, with lexical and dense retrieval
underneath as the fallback for everything phrased loosely.

`starter/agent.py` is the orchestration and the public API surface; each stage lives in its own
module under `src/` so it can be tested in isolation.

```
parse turn -> update dialog state -> route -> multi-route retrieval
           -> fusion -> rerank -> clarify
```

- **Multi-route retrieval** (`src/retrieval.py`): three routes run every turn -- a verbatim
  requirement route over an inverted index of whole attribute phrases (IDF-weighted, with prefix
  matching so a shortened or partially quoted requirement still lands), a BM25 keyword route
  (SQLite FTS5 with Porter stemming), and a dense-similarity route (TF-IDF+SVD) diversified by MMR
  for open-ended browsing. They are combined by reciprocal rank fusion.
- **Runtime re-orchestration** (`src/router.py`): rather than a one-shot Buying/Browsing label, the
  router returns a continuous `buying_weight` that sets the fusion blend, re-evaluated every turn
  from the message and from how many requirements are actually confirmed. A session that opens
  "still exploring" and firms up by turn 3 shifts toward precision without ever being told which
  scenario it is.
- **Category as a real constraint** (`src/catalog_index.py`): products are indexed by the tail of
  their category breadcrumb -- the way a customer names a department out loud ("Watches Wrist
  Watches"). An incoming category phrase resolves against that index exactly, then by suffix, then
  by best token overlap. When it resolves, nothing from another department outranks something from
  the customer's, however well it scores; off-shelf results only pad the list out to ten.
- **A dialog state machine** (`src/session_state.py`, `src/slots.py`): accumulates requirements
  across turns, rewrites them on an Intent Override cue (the superseded value is demoted to a weak
  prior rather than erased -- in shopping, "actually it has to be leather" refines the goal far
  more often than it abandons it), decays untouched slots, and distinguishes a settled "I have no
  preference for X" from a one-off "you decide", which is parked and asked again later instead of
  being burned.
- **Proactive, information-gain-driven clarification** (`src/clarify.py`): each askable attribute
  is scored by how much it would split the current candidate pool, discounted by how likely this
  customer is to have an answer for it at all -- an estimate updated during the session from what
  each question actually returned. Questions are then phrased from what is known: the customer's
  own category becomes the noun and confirmed attributes become modifiers ("What size *red dress*
  do you need?"), and the example values offered are drawn from the live candidate pool, so every
  option suggested is one that actually still exists. The agent always returns its best-effort
  Top-10 in the same turn; a clarifying question never crowds out a real guess.
- **A local ranking stage** (`src/ranking.py`): blends requirement coverage, how prominently the
  matched requirements sit in the item's own attribute list, a demand prior from review volume,
  price agreement, fused retrieval rank, and the anonymized long-term profile -- all computable
  with no model calls.

### Per-scenario results (public dev set)

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|---|
| Buying | 80 | 0.988 | 0.768 | 1.26 |
| Browsing | 80 | 1.000 | 0.716 | 1.43 |
| Intent Override | 30 | 1.000 | 0.942 | 3.60 |
| Boundary | 10 | 1.000 | 0.864 | 2.30 |

Intent Override's higher MTTC is structural, not a weakness: the override does not arrive until its
scripted turn, so those sessions cannot be scored as a hit before it.

## Setup and Installation

Python 3.10 or later. This solution adds `numpy` and `scikit-learn` on top of the organizer kit's
stdlib-only baseline:

```bash
pip install -r requirements.txt
```

The catalog is not committed (it is large and organizer-frozen). Download `catalog.jsonl.gz` from
the participant-kit GitHub Release, verify it against the published `SHA256SUMS`, then:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

## Reproducing Our Results

```bash
python -m evaluator.local_evaluator
```

Takes roughly 70 seconds. The first run builds a TF-IDF+SVD dense index over the 50k-item catalog
(about 12 seconds) and caches it to `data/*.dense_cache.pkl`; later runs reuse the cache. Aggregate
metrics print to stdout and the per-session breakdown is written to `results.json`. Expect the
numbers in the table above -- the pipeline is deterministic, so they reproduce exactly.

Run the test suite (32 tests, about 0.1s -- it uses a small synthetic catalog rather than paying
the 50k build cost):

```bash
python -m unittest discover -s tests -v
```

`tests/test_evaluator.py` is the organizer-provided suite, unmodified. `tests/test_agent.py` covers
our own components in `src/` and the `Agent` contract.

Play the customer yourself:

```bash
python -m scripts.chat_demo
```

## Tools, Libraries, APIs, and Data

- **Development tools:** VS Code, Git, Python 3.14 (3.10+ supported), `unittest`.
- **Libraries and frameworks:** `numpy`, `scikit-learn` (`TfidfVectorizer`, `TruncatedSVD`),
  Python's stdlib `sqlite3` with the FTS5 extension for BM25. Optional: `sentence-transformers`,
  `anthropic`.
- **External APIs:** none required. The default pipeline makes no network calls at run time.
- **Datasets and assets:** the organizer's frozen 50,000-product catalog and 200 labelled public
  sessions, derived from the Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` category (McAuley
  Lab, UCSD). The catalog is treated as strictly read-only. No external or hand-labelled data was
  added, and no model was trained or fine-tuned.

### Model choice, cost, and network requirements

**No LLM is required to reach the numbers above.** The default pipeline is entirely local and
deterministic: SQLite FTS5 BM25 plus a TF-IDF+SVD dense embedding fit once over the catalog and
cached to disk. Zero API cost, zero tokens, and no network access at run time beyond the one-time
catalog download. Reported token usage is therefore `0`.

Two optional accelerators exist, both off by default, both falling back automatically if
unavailable:

- **Local sentence-transformer embeddings** (`AGENT_USE_ST=1`): swaps the TF-IDF+SVD dense route for
  `sentence-transformers/all-MiniLM-L6-v2`. Costs a one-time ~19-minute CPU encode of all 50k items
  plus a first-run Hugging Face Hub download (cached afterward). In the current pipeline the dense
  route is a fallback rather than the primary signal, so this stays an option rather than the
  default.
- **LLM re-rank of the final shortlist** (`ANTHROPIC_API_KEY` set): reorders the local top-10 with a
  small `claude-haiku-4-5` call. Adds latency and token cost, and is skipped entirely when the key
  is absent, so the submission never depends on a live credential or on network access during
  grading. Any failure -- missing key, network error, unparseable reply -- silently keeps the local
  order.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "What size red dress do you need?",
            "ask_attribute": "size",
            "recommendations": [
                {"parent_asin": "B000...", "score": 0.91},
                {"parent_asin": "B001...", "score": 0.87}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`,
`feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json` for the full contract.

`Agent.session_diagnostics(session_id)` is an extra inspection hook used by the demo script. It is
deliberately outside the scored API and nothing in the pipeline reads it back.

## How It Is Scored

```text
TechnicalScore = 0.50 x HitRate@10 + 0.30 x MRR + 0.20 x Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

- **Hit Rate@10:** fraction of sessions that surface the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** a feasibility metric, not part of the score.

`TechnicalScore` is one objective input to the *Technical Execution* criterion (35% of judging); it
is not the whole grade.

## Limitations and Future Work

- **Verbatim matching is the strongest signal, and also the most brittle one.** It pays off exactly
  when a customer quotes a product attribute closely. Prefix matching absorbs truncation and partial
  quotes, and BM25 plus dense retrieval sit underneath as fallbacks, but a heavily paraphrased
  requirement ("something I can throw in the wash") drops the pipeline back to its weaker routes. A
  learned retriever, or an LLM query-rewriting step ahead of retrieval, is the obvious next move.
- **The public set is a proxy, and a small one.** 200 sessions, with Boundary represented by only
  10. Differences of a point or two on this set are not reliable evidence about the private 800,
  and the simulated customer's phrasing is templated in ways real shoppers are not.
- **Attribute vocabulary is hand-built.** The material/color/size/style/use_case word lists and
  "Key: Value" hints in `src/slots.py`, and the noun/plural lists in `src/clarify.py`, were written
  by inspecting this catalog. They will miss vocabulary outside `Clothing_Shoes_and_Jewelry` and
  could overfit to phrasing the private sessions do not share. Deriving them from the catalog's own
  `details` keys would generalize better.
- **The demand prior is a proxy.** Review volume stands in for purchase likelihood, which is right
  on average and wrong for exactly the shopper hunting something niche -- the case where a
  recommender is most useful. It is weighted to break ties rather than override what the customer
  said, but it does bias cold-start turns toward the mainstream.
- **Ranking weights are hand-tuned.** They were set by re-running the evaluator, not learned.
  Fitting them on the public set's known hits, holding out a slice to check for overfitting, is the
  obvious upgrade.
- **No cross-session learning.** Each `SessionState` is scoped to one `session_id`, so nothing
  learned in one session carries to the next; the long-term profile signal is limited to whatever
  `user_profile` discloses at `reset()`.
- **Clarification quality is not directly measured.** The evaluator's simulated customer branches on
  `ask_attribute`, never on the wording, so better-phrased questions cannot show up in
  TechnicalScore. We improved the phrasing anyway because real users and judges read it -- but that
  means it is validated by inspection, not by the metric.

## Team Contributions

- Germaine Foo Ee Huei -- Retrieval and indexing: attribute-phrase inverted index with prefix
  matching, BM25/FTS5 keyword route, TF-IDF+SVD dense route with MMR diversification, reciprocal
  rank fusion.
- Ong Jun Hong -- Ranking and scoring: local reranker, requirement-coverage and
  attribute-prominence signals, demand prior, price agreement, profile alignment, optional LLM
  shortlist re-rank.
- Chai Yao Onn, Benjamin -- Dialog and state: turn parsing, slot extraction, intent-override rewrite and
  slot decay, clarification strategy and question phrasing.
- Yap Xin Hui -- Evaluation and tooling: harness runs, per-scenario analysis, test suite,
  interactive demo script, documentation.

## Repository Layout

```text
starter/agent.py                  Agent: wires the pipeline below together
src/catalog_index.py              attribute-phrase, breadcrumb, BM25 (FTS5+Porter) and dense
                                  (TF-IDF/SVD, optional sentence-transformer) indexes
src/session_state.py              per-session memory: requirement accumulation, override, decay
src/slots.py                      turn parsing and free-text -> (attribute, value) extraction
src/router.py                     per-turn Buying/Browsing route weighting
src/retrieval.py                  the three retrieval routes + reciprocal rank fusion
src/ranking.py                    local scorer + optional LLM shortlist re-rank
src/clarify.py                    over-generality detection, question selection and phrasing
src/text_utils.py                 shared normalization helpers
scripts/chat_demo.py              interactive session runner (used for the demo video)
tests/test_agent.py               unit + smoke tests for our components
tests/test_evaluator.py           organizer-provided evaluator tests (unmodified)
evaluator/local_evaluator.py      public-set simulator and scorer (unmodified)
data/public_set.jsonl             200 labelled development sessions
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
docs/competition_specification.md participant rules and evaluation protocol
docs/submission_rules.md          submission requirements
```

## Data Source and Attribution

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. Sessions are
sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the
frozen catalog. See `DATA_ATTRIBUTION.md` before using or redistributing the data.

The catalog is strictly read-only: no structural mutation, no injected ASINs, no external data
added.

"""Tests for our own agent components (src/ and starter/agent.py).

tests/test_evaluator.py (organizer-provided) covers the evaluator itself and
must not be modified; this file covers the pipeline we built on top of it.
A small synthetic catalog is used throughout so these run in well under a
second, instead of paying the real 50k-row catalog's build cost.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.catalog_index import CatalogIndex
from src.ranking import local_rerank
from src.retrieval import browsing_track, buying_track, compute_restrict, evidence_scores, fuse
from src.session_state import SessionState
from src.slots import (as_bare_attribute, classify_phrase, detect_no_preference, detect_override,
                       extract_budget_max, parse_turn)
from starter.agent import Agent

TINY_CATALOG = [
    {
        "parent_asin": "A1", "title": "Wireless Bra, Full-Coverage Smoothing T-Shirt Bra",
        "features": ["96% Nylon, 4% Spandex", "Pull-On closure"], "description": ["Everyday comfort bra."],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Bras", "Everyday Bras"],
        "details": {"Department": "womens"}, "store": "Hanes", "price": 18.0, "average_rating": 4.5, "rating_number": 100,
    },
    {
        "parent_asin": "A2", "title": "Men's Leather Belt", "features": ["Genuine leather", "Buckle closure"],
        "description": ["A classic leather belt."], "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
        "details": {"Department": "mens"}, "store": "Levi's", "price": 25.0, "average_rating": 4.1, "rating_number": 50,
    },
    {
        "parent_asin": "A3", "title": "Silver Pentagram Pendant Necklace", "features": ["Material: alloy", "Chain length 18in"],
        "description": ["A moon and star pendant necklace."], "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
        "details": {"Material": "alloy"}, "store": "QIAN0813", "price": 12.0, "average_rating": 4.7, "rating_number": 30,
    },
    {
        "parent_asin": "A4", "title": "Cotton Crew T-Shirt", "features": ["100% Cotton", "Machine wash"],
        "description": ["A soft casual crew t-shirt."], "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts", "T-Shirts"],
        "details": {"Department": "mens"}, "store": "Columbia", "price": 15.0, "average_rating": 4.3, "rating_number": 80,
    },
]

PROFILE = {
    "purchase_frequency": "3-4 prior purchases", "average_prior_rating": 4.5,
    "rating_style": "usually positive", "preference_tags": ["comfort"], "summary": "x",
}


def _write_catalog(directory: Path) -> Path:
    path = directory / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in TINY_CATALOG), encoding="utf-8")
    return path


class SlotExtractionTest(unittest.TestCase):
    def test_key_value_phrase_uses_key_to_pick_attribute_and_strips_it_from_value(self) -> None:
        attribute, value = classify_phrase("Material: alloy")
        self.assertEqual(attribute, "material")
        self.assertEqual(value, "alloy")

    def test_vocab_phrase_without_key_still_classifies(self) -> None:
        self.assertEqual(classify_phrase("cotton"), ("material", "cotton"))
        self.assertEqual(classify_phrase("black"), ("color", "black"))

    def test_extract_budget_max_handles_dollar_and_bare_numbers(self) -> None:
        self.assertEqual(extract_budget_max("budget around $27.99"), 27.99)
        self.assertEqual(extract_budget_max("27.99"), 27.99)  # key already stripped
        self.assertIsNone(extract_budget_max("no numbers here"))

    def test_detect_no_preference_and_override_cues(self) -> None:
        self.assertEqual(detect_no_preference("I don't have a preference for material"), "material")
        self.assertTrue(detect_override("Actually, ignore my earlier preference."))
        self.assertFalse(detect_override("I like cotton."))

    def test_bare_attribute_words_are_told_apart_from_quoted_details(self) -> None:
        self.assertEqual(as_bare_attribute("cotton"), ("material", "cotton"))
        self.assertEqual(as_bare_attribute("color: grey"), ("color", "grey"))
        self.assertIsNone(as_bare_attribute("Pull-On closure"))


class TurnParsingTest(unittest.TestCase):
    def test_opening_message_splits_category_from_requirement(self) -> None:
        parsed = parse_turn("I'm looking for Watches Wrist Watches. A key requirement is: Water Resistant.")
        self.assertEqual(parsed.category, "Watches Wrist Watches")
        self.assertEqual(parsed.phrases, ["Water Resistant"])

    def test_browsing_opener_carries_a_category_and_no_requirements(self) -> None:
        parsed = parse_turn("I'm looking for Jewelry Necklaces, but I'm still exploring.")
        self.assertEqual(parsed.category, "Jewelry Necklaces")
        self.assertEqual(parsed.phrases, [])

    def test_a_requirement_containing_semicolons_is_kept_whole_as_well_as_split(self) -> None:
        # Catalog copy really is written like this; splitting only on ";" would
        # shred one attribute into fragments that match nothing.
        parsed = parse_turn("For that, what matters is: Hand wash; Hang to dry.")
        self.assertIn("Hand wash; Hang to dry", parsed.phrases)
        self.assertIn("Hand wash", parsed.phrases)
        self.assertIn("Hang to dry", parsed.phrases)

    def test_override_message_is_flagged_and_its_new_requirement_extracted(self) -> None:
        parsed = parse_turn("Actually, ignore my earlier preference. What I need is: leather.")
        self.assertTrue(parsed.is_override)
        self.assertEqual(parsed.phrases, ["leather"])

    def test_deflection_is_distinguished_from_a_settled_no_preference(self) -> None:
        deflected = parse_turn("I don't have a preference for color; please use your judgment.")
        self.assertEqual(deflected.no_preference_for, "color")
        self.assertTrue(deflected.is_deflection)

        settled = parse_turn("I don't have an additional preference for color.")
        self.assertEqual(settled.no_preference_for, "color")
        self.assertFalse(settled.is_deflection)


class SessionStateTest(unittest.TestCase):
    def test_accumulates_hard_filters_across_turns(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a belt. A key requirement is: leather.", turn=1)
        self.assertIn("material", state.hard_filters())

        state.note_ask("color")
        state.observe("For that, what matters is: black.", turn=2)
        self.assertIn("color", state.hard_filters())
        self.assertEqual(state.hard_filters()["color"].variants, ["black"])

    def test_override_rewrites_the_attribute_but_keeps_the_old_value_as_weak_evidence(self) -> None:
        # An override in shopping usually refines the same goal rather than
        # abandoning it, so the superseded value is demoted, not erased.
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for shoes. A key requirement is: leather.", turn=1)
        state.observe("Actually, ignore my earlier preference. What I need is: canvas.", turn=2)

        self.assertEqual(state.hard_filters()["material"].variants[0], "canvas")
        self.assertEqual(state.override_events, 1)
        superseded = [item for item in state.evidence if item.phrase == "leather"]
        self.assertLess(superseded[0].weight, 1.0)

    def test_no_preference_reply_locks_attribute_as_unaskable(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a necklace.", turn=1)
        state.note_ask("brand")
        state.observe("I don't have an additional preference for brand.", turn=2)
        self.assertNotIn("brand", state.unresolved_attributes())

    def test_a_deflected_question_comes_back_later_instead_of_being_burned(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a necklace, but I'm still exploring.", turn=1)
        state.note_ask("other")
        state.observe("I don't have a preference for other; please use your judgment.", turn=2)
        self.assertNotIn("other", state.unresolved_attributes())  # parked for now

        state.observe("For that, what matters is: Chain length 18in.", turn=3)
        self.assertIn("other", state.unresolved_attributes())  # askable again

    def test_catch_all_attribute_stays_askable_until_a_follow_up_yields_nothing_new(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a necklace. A key requirement is: Material: alloy.", turn=1)
        self.assertIn("feature", state.unresolved_attributes())

        state.note_ask("feature")
        state.observe("For that, what matters is: Triple Moon Pentagram Symbol.", turn=2)
        self.assertIn("feature", state.unresolved_attributes())  # got something new: ask again

        state.note_ask("feature")
        state.observe("I don't have an additional preference for feature.", turn=3)
        self.assertNotIn("feature", state.unresolved_attributes())  # now genuinely exhausted


class CatalogIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.index = CatalogIndex(_write_catalog(Path(cls._tmp.name)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_attribute_phrase_lookup_matches_a_verbatim_quote(self) -> None:
        found = self.index.phrase_lookup("Buckle closure")
        self.assertIsNotNone(found)
        indices, _positions, _idf = found
        self.assertEqual([self.index.order[i] for i in indices], ["A2"])

    def test_attribute_phrase_lookup_matches_a_truncated_quote_by_prefix(self) -> None:
        found = self.index.phrase_lookup("96% Nylon")
        self.assertIsNotNone(found)
        indices, _positions, _idf = found
        self.assertEqual([self.index.order[i] for i in indices], ["A1"])

    def test_breadcrumb_resolution_falls_back_from_exact_to_partial(self) -> None:
        self.assertEqual(self.index.resolve_breadcrumb("Jewelry Necklaces"), "jewelry necklaces")
        self.assertEqual(self.index.resolve_breadcrumb("Necklaces"), "jewelry necklaces")
        self.assertIsNone(self.index.resolve_breadcrumb("garden furniture"))

    def test_primary_material_is_the_first_one_the_product_text_mentions(self) -> None:
        self.assertEqual(self.index.primary_material[self.index.index_of["A4"]], "cotton")
        self.assertEqual(self.index.primary_material[self.index.index_of["A2"]], "leather")


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.catalog_path = _write_catalog(Path(self._tmp.name))
        self.index = CatalogIndex(self.catalog_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_hard_filter_never_excludes_a_matching_item(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for Jewelry Necklaces. A key requirement is: Material: alloy.", turn=1)
        restrict = compute_restrict(state, self.index)
        self.assertIsNotNone(restrict)
        self.assertIn("A3", restrict)  # the alloy pendant necklace

    def test_buying_track_surfaces_the_filtered_item(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a necklace. A key requirement is: Material: alloy.", turn=1)
        results = buying_track(state, self.index, limit=10, restrict=compute_restrict(state, self.index))
        self.assertIn("A3", [asin for asin, _score in results])

    def test_quoted_requirement_puts_the_item_that_carries_it_on_top(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for a belt. A key requirement is: Buckle closure.", turn=1)
        evidence = evidence_scores(state, self.index)
        best = max(range(len(self.index.order)), key=lambda i: evidence.support[i])
        self.assertEqual(self.index.order[best], "A2")

    def test_browsing_track_returns_something_for_a_vague_query(self) -> None:
        state = SessionState(session_id="s", user_profile={"preference_tags": ["comfort"]})
        state.observe("I'm looking for clothing, but I'm still exploring.", turn=1)
        self.assertTrue(browsing_track(state, self.index, limit=10))

    def test_fuse_combines_and_ranks_by_weighted_reciprocal_rank(self) -> None:
        fused = fuse([([("X", 1.0), ("Y", 0.5)], 1.0), ([("Y", 1.0)], 1.0)], limit=10)
        ids = [asin for asin, _score in fused]
        self.assertEqual(ids[0], "Y")  # appears in both tracks, should outrank X

    def test_named_category_outranks_a_better_scoring_item_from_another_shelf(self) -> None:
        state = SessionState(session_id="s", user_profile={})
        state.observe("I'm looking for Jewelry Necklaces, but I'm still exploring.", turn=1)
        ranked = local_rerank(
            [("A1", 1.0), ("A3", 0.0)], state, self.index, limit=10,
            evidence=evidence_scores(state, self.index), shelf={"A3"},
        )
        self.assertEqual(ranked[0][0], "A3")


class AgentInterfaceTest(unittest.TestCase):
    def test_respond_matches_the_required_contract_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(_write_catalog(Path(directory)))
            agent.reset("session-1", PROFILE)
            response = agent.respond("session-1", "I'm looking for a belt. A key requirement is: leather.", 1, 10)

            self.assertIsInstance(response["message"], str)
            self.assertIn(response["ask_attribute"], (
                None, "category", "material", "color", "size", "style",
                "brand", "budget", "feature", "use_case", "other",
            ))
            self.assertIsInstance(response["recommendations"], list)
            for item in response["recommendations"]:
                self.assertIn("parent_asin", item)
            self.assertGreaterEqual(response["usage"]["prompt_tokens"], 0)
            self.assertGreaterEqual(response["usage"]["completion_tokens"], 0)

    def test_a_quoted_requirement_is_answered_with_the_matching_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(_write_catalog(Path(directory)))
            agent.reset("session-2", PROFILE)
            response = agent.respond(
                "session-2", "I'm looking for Accessories Belts. A key requirement is: Buckle closure.", 1, 10,
            )
            self.assertEqual(response["recommendations"][0]["parent_asin"], "A2")

    def test_sessions_do_not_leak_into_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(_write_catalog(Path(directory)))
            agent.reset("a", PROFILE)
            agent.reset("b", PROFILE)
            agent.respond("a", "I'm looking for Accessories Belts. A key requirement is: Buckle closure.", 1, 10)
            self.assertEqual(agent.session_diagnostics("b")["requirements"], [])

    def test_respond_raises_if_reset_was_not_called(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(_write_catalog(Path(directory)))
            with self.assertRaises(RuntimeError):
                agent.respond("never-reset", "hello", 1, 10)


if __name__ == "__main__":
    unittest.main()

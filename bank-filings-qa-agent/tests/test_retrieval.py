"""
Phase 7, Step 2: Tests for retrieve_context().

Concept: these run against your REAL Chroma index (no mocking) -
retrieval is deterministic (same input always gives the same
distances), so there's no randomness to worry about here, unlike
the LLM-based nodes we'll test later.
"""

import sys
import os

# Ensures Python can find your project's modules when pytest runs
# from the tests/ subfolder instead of the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3_retrieve_context import retrieve_context


def test_retrieve_returns_results_for_relevant_question():
    """
    A clearly relevant, on-topic question should return at least
    one chunk. This is the most basic sanity check: the pipeline
    is wired correctly and Chroma actually has data in it.
    """
    results = retrieve_context("What did the bank say about loan loss provisions?", k=5)
    assert len(results) > 0


def test_retrieve_results_have_expected_fields():
    """
    Every returned chunk must have the fields the rest of the
    pipeline depends on (verify_node and synthesize_node both
    read 'text', 'ticker', 'filing_date').
    """
    results = retrieve_context("What did the bank say about loan loss provisions?", k=3)
    assert len(results) > 0

    for chunk in results:
        assert "text" in chunk
        assert "ticker" in chunk
        assert "filing_date" in chunk
        assert "distance" in chunk


def test_retrieve_filters_irrelevant_query():
    """
    This is your real "Is Tanisha my love?" discovery, turned into
    a permanent regression test. If someone changes the distance
    threshold or embedding model later and breaks this filtering
    behavior, this test will catch it immediately.
    """
    results = retrieve_context("Is Tanisha my love?", k=5)

    # Either no results at all (if a threshold filter is applied),
    # or if results DO come back, their distances should be high
    # (weak matches) - not confidently close.
    if results:
        for r in results:
            assert r["distance"] > 1.2, (
                f"Expected a high (weak) distance for an irrelevant query, "
                f"got {r['distance']}"
            )


def test_retrieve_respects_k_parameter():
    """
    Asking for k=2 should never return more than 2 results.
    """
    results = retrieve_context("What did the bank say about loan loss provisions?", k=2)
    assert len(results) <= 2
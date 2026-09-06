"""
Phase 7, Step 4: Tests for the full graph's routing behavior.

Concept: we mock BOTH retrieve_context and the LLM calls, so these
tests check ONLY the graph's wiring logic - does it correctly route
to synthesize vs human_review based on the verifier's output -
without depending on real Chroma data or real API calls at all.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase4_graph import build_graph


def make_fake_llm_response(text: str):
    fake_response = MagicMock()
    fake_response.content = text
    return fake_response


def test_graph_routes_to_synthesize_on_high_confidence():
    """
    When retrieval finds good evidence and verification says high
    confidence, the graph should reach synthesize_node and produce
    a real answer, with needs_review False.
    """
    fake_chunks = [
        {"text": "Total allowance for loan losses was $17,557 million.", "ticker": "JPM", "filing_date": "2026-02-13", "distance": 0.4}
    ]

    with patch("phase4_nodes.retrieve_context", return_value=fake_chunks), \
         patch("phase4_nodes.verifier_llm") as mock_verifier, \
         patch("phase4_nodes.synthesis_llm") as mock_synth:

        mock_verifier.invoke.return_value = make_fake_llm_response(
            '{"confidence": "high", "reasoning": "Specific figures present."}'
        )
        mock_synth.invoke.return_value = make_fake_llm_response(
            "JPMorgan reported a total allowance for loan losses of $17,557 million. [Source: JPM, 2026-02-13]"
        )

        app = build_graph()
        result = app.invoke({"question": "What did the bank say about loan loss provisions?"})

    assert result["needs_review"] is False
    assert "17,557" in result["answer"]


def test_graph_routes_to_human_review_on_low_confidence():
    """
    When verification says low confidence, the graph must skip
    synthesize_node entirely and reach human_review instead -
    this is the human-approval gate actually being tested.
    """
    fake_chunks = [
        {"text": "Unrelated executive biography.", "ticker": "BAC", "filing_date": "2026-02-25", "distance": 1.7}
    ]

    with patch("phase4_nodes.retrieve_context", return_value=fake_chunks), \
         patch("phase4_nodes.verifier_llm") as mock_verifier, \
         patch("phase4_nodes.synthesis_llm") as mock_synth:

        mock_verifier.invoke.return_value = make_fake_llm_response(
            '{"confidence": "low", "reasoning": "Evidence is unrelated."}'
        )

        app = build_graph()
        result = app.invoke({"question": "Is Tanisha my love?"})

        # The key assertion: synthesize_node's LLM should NEVER be
        # called when the graph correctly routes to human_review.
        mock_synth.invoke.assert_not_called()

    assert result["needs_review"] is True
    assert "human review" in result["answer"].lower()


def test_graph_handles_empty_retrieval():
    """
    When retrieval returns nothing at all, the graph should still
    complete without crashing, routing to human_review.
    """
    with patch("phase4_nodes.retrieve_context", return_value=[]), \
         patch("phase4_nodes.verifier_llm") as mock_verifier:

        app = build_graph()
        result = app.invoke({"question": "Some obscure question"})

        # verify_node short-circuits on empty chunks - LLM shouldn't be called
        mock_verifier.invoke.assert_not_called()

    assert result["needs_review"] is True
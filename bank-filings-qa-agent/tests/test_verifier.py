"""
Phase 7, Step 3: Tests for verify_node() with MOCKED LLM calls.

Concept: we replace verifier_llm.invoke() with a fake version that
returns whatever text WE specify - so we can test "does verify_node
correctly parse a high-confidence response" and "does it correctly
handle a messy response with rambling text after the JSON" (the
exact bug you found and fixed) without needing a real API call.

unittest.mock.patch temporarily swaps out a real object/function
with a fake one, only for the duration of the test - it's a
standard Python testing tool, not specific to LLMs.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase4_nodes import verify_node


def make_fake_llm_response(text: str):
    """
    ChatGroq's .invoke() returns an object with a .content attribute
    holding the actual text. This builds a fake object shaped the
    same way, so verify_node can't tell the difference.
    """
    fake_response = MagicMock()
    fake_response.content = text
    return fake_response


def test_verify_node_high_confidence_clean_json():
    """
    The straightforward case: the model returns clean JSON with no
    extra text. Should parse correctly and NOT need review.
    """
    fake_state = {
        "question": "What did the bank say about loan loss provisions?",
        "retrieved_chunks": [
            {"text": "Total allowance for loan losses was $17,557 million.", "ticker": "JPM", "filing_date": "2026-02-13"}
        ],
    }

    fake_text = '{"confidence": "high", "reasoning": "Specific dollar figures are present."}'

    with patch("phase4_nodes.verifier_llm") as mock_llm:
        mock_llm.invoke.return_value = make_fake_llm_response(fake_text)
        result = verify_node(fake_state)

    assert result["verification"]["confidence"] == "high"
    assert result["needs_review"] is False


def test_verify_node_handles_rambling_text_around_json():
    """
    This is your ACTUAL discovered bug, turned into a permanent
    regression test: a reasoning model that keeps talking in plain
    text after its <think> block closes, before finally printing
    JSON. This must still parse correctly.
    """
    fake_state = {
        "question": "What did the bank say about loan loss provisions?",
        "retrieved_chunks": [
            {"text": "Total allowance for loan losses was $17,557 million.", "ticker": "JPM", "filing_date": "2026-02-13"}
        ],
    }

    # Simplified version of the real messy output you encountered
    fake_text = """<think>
    Let me analyze this evidence carefully.
    </think>
    Wait, let me double check the format.
    Proceeds.
    {"confidence": "high", "reasoning": "Specific dollar figures are present."}
    Done, output matches."""

    with patch("phase4_nodes.verifier_llm") as mock_llm:
        mock_llm.invoke.return_value = make_fake_llm_response(fake_text)
        result = verify_node(fake_state)

    assert result["verification"]["confidence"] == "high"
    assert result["needs_review"] is False


def test_verify_node_low_confidence_triggers_review():
    """
    When the model judges confidence as low, needs_review must be
    True - this is the human-approval gate's trigger condition.
    """
    fake_state = {
        "question": "Is Tanisha my love?",
        "retrieved_chunks": [
            {"text": "Unrelated executive biography text.", "ticker": "BAC", "filing_date": "2026-02-25"}
        ],
    }

    fake_text = '{"confidence": "low", "reasoning": "Evidence is unrelated to the question."}'

    with patch("phase4_nodes.verifier_llm") as mock_llm:
        mock_llm.invoke.return_value = make_fake_llm_response(fake_text)
        result = verify_node(fake_state)

    assert result["verification"]["confidence"] == "low"
    assert result["needs_review"] is True


def test_verify_node_no_chunks_skips_llm_and_flags_review():
    """
    When retrieval returns NO chunks at all, verify_node should
    short-circuit and flag for review WITHOUT even calling the LLM -
    this saves an API call for an obviously unanswerable case.
    """
    fake_state = {"question": "Anything", "retrieved_chunks": []}

    with patch("phase4_nodes.verifier_llm") as mock_llm:
        result = verify_node(fake_state)
        mock_llm.invoke.assert_not_called()

    assert result["needs_review"] is True


def test_verify_node_unparseable_response_fails_safe():
    """
    If the LLM response contains no valid JSON at all, verify_node
    must fail SAFE (low confidence / needs review) rather than
    crashing the whole graph with an unhandled exception.
    """
    fake_state = {
        "question": "What did the bank say about loan loss provisions?",
        "retrieved_chunks": [
            {"text": "Some evidence.", "ticker": "JPM", "filing_date": "2026-02-13"}
        ],
    }

    fake_text = "I'm not sure how to format this response at all."

    with patch("phase4_nodes.verifier_llm") as mock_llm:
        mock_llm.invoke.return_value = make_fake_llm_response(fake_text)
        result = verify_node(fake_state)

    assert result["verification"]["confidence"] == "low"
    assert result["needs_review"] is True
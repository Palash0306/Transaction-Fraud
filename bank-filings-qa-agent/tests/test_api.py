"""
Phase 7, Step 5: Tests for the FastAPI endpoints.

Concept: FastAPI's TestClient lets you send fake HTTP requests
directly to your app in-process - no real server needs to be
running, no real network calls happen. It's the HTTP equivalent
of calling a function directly, but exercises the actual routing,
request validation, and response serialization FastAPI does.
"""

import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_fake_llm_response(text: str):
    fake_response = MagicMock()
    fake_response.content = text
    return fake_response


def test_health_check():
    """
    The simplest possible test: does the server start up and
    respond at all. Good to have as a canary - if this fails,
    something fundamental (like an import error) is broken.
    """
    from phase6_api import app
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_rejects_missing_question():
    """
    FastAPI should automatically reject a request that doesn't
    match the QuestionRequest shape (missing "question" field),
    WITHOUT our code ever running - this is Pydantic validation
    working as intended.
    """
    from phase6_api import app
    client = TestClient(app)

    response = client.post("/ask", json={})

    # 422 = Unprocessable Entity, FastAPI's standard validation error code
    assert response.status_code == 422


def test_ask_endpoint_returns_expected_shape():
    """
    With retrieval and LLM calls mocked, confirm the /ask endpoint
    returns a response matching the AnswerResponse shape: answer,
    needs_review, and sources - exactly what the Streamlit frontend
    depends on.
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
            "JPMorgan reported $17,557 million in total allowance for loan losses."
        )

        from phase6_api import app
        client = TestClient(app)

        response = client.post("/ask", json={"question": "What did the bank say about loan loss provisions?"})

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "needs_review" in data
    assert "sources" in data
    assert data["needs_review"] is False
    assert len(data["sources"]) == 1
    assert data["sources"][0]["ticker"] == "JPM"
"""
Phase 6, Step 1: FastAPI backend.

Concept: this file's only job is to expose your EXISTING Phase 4
graph over HTTP. No new retrieval/verification/synthesis logic
lives here - it's a thin translation layer, same philosophy as
the MCP server in Phase 5, just speaking HTTP instead of stdio.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from phase4_graph import build_graph

app = FastAPI(title="Bank Filings Q&A API")

# Build the graph ONCE when the server starts, not on every request -
# rebuilding it per-request would be wasteful, since the graph
# structure itself never changes between questions.
graph = build_graph()


class QuestionRequest(BaseModel):
    """
    Defines exactly what a valid incoming request looks like.
    FastAPI automatically rejects requests that don't match this
    shape (e.g. missing "question", or question isn't a string)
    before your function even runs.
    """
    question: str


class AnswerResponse(BaseModel):
    """
    Defines exactly what shape of data this endpoint sends back.
    Having this explicit makes the API self-documenting - anyone
    calling it knows exactly what fields to expect in the response.
    """
    answer: str
    needs_review: bool
    sources: list[dict]


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    The core endpoint. Takes a question, runs it through the FULL
    Phase 4 graph (retrieve -> verify -> synthesize/human_review),
    and returns the final answer plus metadata about confidence
    and sources - exactly what a UI needs to display meaningfully.
    """
    result = graph.invoke({"question": request.question})

    # Extract just the source info (ticker + filing date) from the
    # retrieved chunks, so the frontend can show citations without
    # needing to know about the full chunk structure.
    sources = [
        {"ticker": c["ticker"], "filing_date": c["filing_date"]}
        for c in result.get("retrieved_chunks", [])
    ]

    return AnswerResponse(
        answer=result["answer"],
        needs_review=result["needs_review"],
        sources=sources,
    )


@app.get("/health")
def health_check():
    """
    A simple endpoint to confirm the server is up and running -
    standard practice for any API, useful for debugging deployment
    issues later.
    """
    return {"status": "ok"}
"""
Phase 4, Step 2: The three nodes.

Concept: each node is just a normal Python function that takes the
current state, does some work, and returns a dict of fields to
update in that state. LangGraph handles passing state between them -
we never manually pass data from one function to the next ourselves.
"""

import json
from phase3_retrieve_context import retrieve_context
from phase4_state_and_llm import GraphState, verifier_llm, synthesis_llm, strip_thinking, extract_json_object


# ---------------------------------------------------------------
# Node 1: retrieve
# ---------------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    """
    Calls the retrieve_context() function you already built in
    Phase 3. This node's only job is search - find real evidence,
    nothing more.
    """
    question = state["question"]
    chunks = retrieve_context(question, k=5)

    print(f"[retrieve] Found {len(chunks)} relevant chunks for: {question}")

    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------
# Node 2: verify
# ---------------------------------------------------------------
def verify_node(state: GraphState) -> dict:
    """
    Asks the LLM a NARROW question: does this evidence actually
    support answering the user's question? This is deliberately a
    different, smaller task than "write the answer" - it's a
    judgment call, not a generation task.

    We ask the model to respond in JSON so we can reliably parse
    a confidence score and reasoning out of its response.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    if not chunks:
        # No evidence at all - automatic low confidence, skip the
        # LLM call entirely since there's nothing to verify.
        print("[verify] No chunks retrieved - skipping LLM call, low confidence.")
        return {
            "verification": {"confidence": "low", "reasoning": "No relevant evidence was retrieved."},
            "needs_review": True,
        }

    evidence_text = "\n\n".join(
        f"[Source: {c['ticker']}, filed {c['filing_date']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""You are an evidence verifier for a financial Q&A system.

Question: {question}

Retrieved evidence:
{evidence_text}

Judge whether this evidence is sufficient to confidently answer the question.
Numeric data and financial figures ARE valid evidence.

Respond with a valid JSON object in this exact structure:
{{\"confidence\": \"high\" | \"medium\" | \"low\", \"reasoning\": \"one sentence explanation\"}}
"""

    response = verifier_llm.invoke(prompt)
    cleaned = strip_thinking(response.content)

    verification = extract_json_object(cleaned)

    if verification is None:
        # If we truly can't find any JSON, fail safe: treat as
        # low confidence rather than crashing the whole graph.
        print(f"[verify] Could not parse verifier response: {cleaned}")
        verification = {"confidence": "low", "reasoning": "Verifier response could not be parsed."}

    needs_review = verification.get("confidence") == "low"

    print(f"[verify] Confidence: {verification.get('confidence')} - {verification.get('reasoning')}")

    return {"verification": verification, "needs_review": needs_review}


# ---------------------------------------------------------------
# Node 3: synthesize
# ---------------------------------------------------------------
def synthesize_node(state: GraphState) -> dict:
    """
    Writes the actual final answer, citing sources. Only reached
    when verification passed (this node is skipped on the low-
    confidence path - see the graph wiring in the next step).
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    evidence_text = "\n\n".join(
        f"[Source: {c['ticker']}, filed {c['filing_date']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the evidence provided below.
Cite the source ticker and filing date for any claim you make.
If the evidence doesn't fully answer the question, say so honestly.

Question: {question}

Evidence:
{evidence_text}

Answer:"""

    response = synthesis_llm.invoke(prompt)
    answer = strip_thinking(response.content)

    print(f"[synthesize] Generated answer ({len(answer)} chars)")

    return {"answer": answer}


# ---------------------------------------------------------------
# Node 4 (alternate path): human review placeholder
# ---------------------------------------------------------------
def needs_human_review_node(state: GraphState) -> dict:
    """
    Reached instead of synthesize_node when confidence is low.
    In a real deployment this might page a human analyst - for
    this project, it's enough to show the routing mechanism works.
    """
    print("[human_review] Confidence too low - routing to human review instead of answering.")
    return {
        "answer": "This question could not be answered with sufficient confidence from the "
                  "available filings. Flagged for human review."
    }
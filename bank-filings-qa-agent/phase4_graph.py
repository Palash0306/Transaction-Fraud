"""
Phase 4, Step 3: Build the graph.

Concept: this is where LangGraph actually comes together. We
register each node, define the edges (which node runs after which),
and add ONE conditional edge - the human-approval gate - which
looks at state["needs_review"] and routes to a different node
depending on the verifier's judgment.
"""

from langgraph.graph import StateGraph, END
from phase4_state_and_llm import GraphState
from phase4_nodes import retrieve_node, verify_node, synthesize_node, needs_human_review_node
import time


def route_after_verification(state: GraphState) -> str:
    """
    This is the "decision diamond" from the flowchart analogy.
    LangGraph calls this function after verify_node runs, and
    whatever string it returns tells LangGraph which node to go
    to next - matching the keys in the conditional edge map below.
    """
    if state["needs_review"]:
        return "needs_review"
    return "proceed"


def build_graph():
    graph = StateGraph(GraphState)

    # Register each node with a name
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify", verify_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("human_review", needs_human_review_node)

    # Normal edges: always go from A to B, no branching
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "verify")

    # Conditional edge: after "verify", call route_after_verification()
    # to decide whether to go to "synthesize" or "human_review"
    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "proceed": "synthesize",
            "needs_review": "human_review",
        },
    )

    # Both end paths terminate the graph
    graph.add_edge("synthesize", END)
    graph.add_edge("human_review", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    # Try a question that SHOULD have good evidence
    print("=" * 60)
    print("TEST 1: Relevant financial question")
    print("=" * 60)
    result = app.invoke({"question": "What did the banks say about loan loss provisions?"})
    print("\nFINAL ANSWER:")
    print(result["answer"])
    print("\nNeeds review:", result["needs_review"])
    
    # Give the Groq rate limit bucket 15 seconds to reset
    print("\n[Pausing 120s for Groq TPM rate limits...]")
    time.sleep(120)

    # Try a question that should NOT have good evidence, to confirm
    # the human-approval gate actually triggers
    print("\n" + "=" * 60)
    print("TEST 2: Irrelevant question (should trigger human review)")
    print("=" * 60)
    result2 = app.invoke({"question": "Is Tanisha my love?"})
    print("\nFINAL ANSWER:")
    print(result2["answer"])
    print("\nNeeds review:", result2["needs_review"])
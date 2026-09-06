"""
Debug script: run ONE question through retrieve + verify separately,
and print everything, so we can see exactly why it got flagged.
"""

from phase3_retrieve_context import retrieve_context
from phase4_nodes import verify_node

question = "What did the bank say about loan loss provisions?"

print("=" * 60)
print("STEP 1: What did retrieve_context() actually find?")
print("=" * 60)
chunks = retrieve_context(question, k=5)

for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i+1} (distance={c['distance']:.3f}) ---")
    print(f"Source: {c['ticker']}, filed {c['filing_date']}")
    print(c["text"][:200])

print("\n" + "=" * 60)
print("STEP 2: What did the verifier decide, and why?")
print("=" * 60)

fake_state = {"question": question, "retrieved_chunks": chunks}
result = verify_node(fake_state)

print("\nVerification result:")
print(result["verification"])
print("\nneeds_review:", result["needs_review"])
"""
Phase 3, Step 4: retrieve_context() - the function everything else builds on.

*** Important Script ***
Concept: this is the "search" half of the system. We take a plain
English question, embed it using the SAME model (this is critical -
you can't compare vectors from two different embedding models,
they don't share the same "map"), and ask Chroma for the top-k
closest stored chunks.
 
This single function is what Phase 4's LangGraph node calls, and
what Phase 5's MCP server exposes to outside tools. Get this right
now and the rest of the project just wires around it.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import torch
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


# Derive absolute path anchored to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="bank_filings")

def retrieve_context(query: str, k:int =5):
    """
    Given a plain-English question, return the top-k most relevant
    chunks from the vector database, along with their metadata.
 
    query: the user's question, e.g. "What did JPMorgan say about
           loan loss provisions?"
    k:     how many chunks to return - more context for the LLM,
           but also more noise/cost if too high.
    """
    
    with torch.no_grad():
        query_embedding = model.encode(query, convert_to_numpy=True)
        
    results = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results = k,
    )
    
    # Chroma returns parallel lists wrapped in an extra layer
    # (because you COULD query multiple questions at once) -
    # we only sent one query, so we grab index [0] each time.
    retrieved = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "ticker": meta["ticker"],
            "filing_date": meta["filing_date"],
            "distance": distance,  # lower = more similar
        })
        
    return retrieved

if __name__ == "__main__":
    question = "What did the bank say about loan loss provisions?"
    print(f"Query: {question}\n")
 
    results = retrieve_context(question, k=3)
 
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (distance={r['distance']:.3f}) ---")
        print(f"Source: {r['ticker']}, filed {r['filing_date']}")
        print(r["text"][:300])
        print()   
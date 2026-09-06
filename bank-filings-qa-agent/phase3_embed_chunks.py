
"""
Phase 3, Step 2: Embed every chunk in batches.
 
Concept: this is the exact same model and .encode() call from
Phase 1 - the difference is scale. Instead of 5 test sentences,
we're now embedding potentially hundreds of real chunks. We let
sentence-transformers handle the batching internally (it groups
chunks together automatically when you pass a list), and we
explicitly use MPS so it runs on your Mac's GPU.
"""

from sentence_transformers import SentenceTransformer
import torch 
from phase3_load_chunks import load_chunks

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2", device = device)

def embed_chunks(chunks: list, batch_size: int = 32):
    """
    Turn a list of chunk dicts into a list of embedding vectors.
 
    batch_size controls how many chunks get embedded together in
    one pass - a tuning knob for speed/memory, not correctness.
    Larger batches are usually faster but use more memory.
    """
    
    texts = [c["text"] for c in chunks]
    
    # no_grad(): same reason as Phase 1 - we're only running the
    # model forward (inference), never training, so we skip the
    # gradient bookkeeping entirely to save time and memory.
    
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size = batch_size,
            show_progress_bar = True,
            convert_to_numpy = True,
        )
        
    return embeddings 

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Embedding now...")
    
    embeddings = embed_chunks(chunks)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"(That's {embeddings.shape[0]} chunks, each a {embeddings.shape[1]}-number vector)")

    # --- Before vs after example ---
    print("\n" + "="*60)
    print("BEFORE (non-embedded chunk — plain text + metadata):")
    print("="*60)
    print(f"Ticker: {chunks[0]['ticker']}")
    print(f"Filing date: {chunks[0]['filing_date']}")
    print(f"Chunk index: {chunks[0]['chunk_index']}")
    print(f"Text (first 300 chars): {chunks[0]['text'][:300]}")

    print("\n" + "="*60)
    print("AFTER (embedded — same chunk as a vector of numbers):")
    print("="*60)
    print(f"Vector length: {len(embeddings[0])}")
    print(f"First 10 numbers of the vector: {embeddings[0][:10]}")
    print("(...374 more numbers follow, 384 total)")
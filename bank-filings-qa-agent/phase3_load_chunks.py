"""
Phase 3, Step 1: Load the chunks saved in Phase 2.
 
Concept: Phase 2 left us with a folder full of small JSON files,
one per chunk, each with "text" plus metadata (ticker, filing_date,
chunk_index). Before we can embed anything, we need to read all
those files back into memory as a single list we can loop over.
"""
 
import os
import json

def load_chunks(chunks_dir: str = "data/chunks"):
    """
    Read every JSON file in the chunks directory and return them
    as a list of dicts. Each dict has: ticker, filing_date,
    chunk_index, text.
    """
    
    chunks = []
    for filename in sorted(os.listdir(chunks_dir)):
        if not filename.endswith(".json"):
            continue 
        filepath = os.path.join(chunks_dir, filename)
        with open(filepath,"r") as f:
            record = json.load(f)
            chunks.append(record)
                
    return chunks
    
if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    if chunks:
        print("\nExample chunk:")
        print(json.dumps(chunks[0], indent=2)[:500])
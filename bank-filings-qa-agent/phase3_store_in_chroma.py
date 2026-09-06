"""
Phase 3, Step 3: Store everything in Chroma.

Concept: Chroma needs four aligned lists for each chunk:
  - an id (unique string, so we can reference this exact chunk later)
  - the embedding vector (the 384 numbers)
  - the metadata (ticker, filing_date, chunk_index - for filtering
    and for showing sources later)
  - the original text (so we can actually show/read the chunk,
    not just its vector)

Chroma saves all of this to a local folder on disk (persistent
storage) - so you only need to run this embedding step ONCE per
filing, not every time you start your app.
"""

import chromadb
from phase3_load_chunks import load_chunks
from phase3_embed_chunks import embed_chunks

# PersistentClient saves data to disk at this path, so it survives
# between runs - as opposed to an in-memory client which forgets
# everything when the script ends.
client = chromadb.PersistentClient(path="data/chroma_db")

# A "collection" is Chroma's term for a table - a named group of
# vectors. get_or_create_collection means: use it if it already
# exists, otherwise make a new one.
collection = client.get_or_create_collection(name="bank_filings")


def store_chunks_in_chroma(chunks: list, embeddings):
    """
    Push chunks + their embeddings + metadata into Chroma.
    """
    ids = []
    metadatas = []
    documents = []

    for i, chunk in enumerate(chunks):
        # A unique, human-readable id per chunk - useful for
        # debugging and for updating a specific chunk later.
        chunk_id = f"{chunk['ticker']}_{chunk['filing_date']}_chunk{chunk['chunk_index']}"
        ids.append(chunk_id)

        metadatas.append({
            "ticker": chunk["ticker"],
            "filing_date": chunk["filing_date"],
            "chunk_index": chunk["chunk_index"],
        })

        documents.append(chunk["text"])

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),  # Chroma wants plain lists, not numpy arrays
        metadatas=metadatas,
        documents=documents,
    )

    print(f"Stored {len(ids)} chunks in Chroma collection 'bank_filings'.")


if __name__ == "__main__":
    chunks = load_chunks()
    embeddings = embed_chunks(chunks)
    store_chunks_in_chroma(chunks, embeddings)

    print("\nTotal chunks now in collection:", collection.count())
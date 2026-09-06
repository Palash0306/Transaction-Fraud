"""
Phase 1, Step 5: Embed sentences, compare similarity by hand.
 
Why "by hand"? sentence-transformers has a built-in
util.cos_sim() helper. We're deliberately NOT using it here —
writing the cosine similarity formula ourselves forces you to
actually understand what "semantic similarity" means as a number,
instead of trusting a magic function.
"""

from sentence_transformers import SentenceTransformer
import torch 

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}\n")

# Loading a pretrained model: no training happens here, we're just
# downloading and using someone else's already-trained weights.
# all-MiniLM-L6-v2 is a small, fast model that turns a sentence into
# a 384-number vector ("embedding") capturing its meaning.
model = SentenceTransformer("all-MiniLM-L6-v2", device= device)

sentences = [
    "The bank increased its loan loss provisions this quarter.",
    "Credit risk reserves went up in Q2.",
    "The stock market closed higher today.",
    "Net interest margin improved year over year.",
    "Weather in New York was sunny."
]

# convert_to_tensor=True keeps the output as a torch.Tensor (rather
# than a numpy array) so we can do the matmul-style math ourselves.
with torch.no_grad():
    embeddings = model.encode(sentences, convert_to_tensor=True)

print("Embeddings shape:", embeddings.shape)

def cosine_sim(a,b):
    """
    Cosine similarity measures the ANGLE between two vectors,
    ignoring their length. Two sentences with similar meaning
    point in a similar "direction" in 384-dimensional space,
    even if the raw numbers differ.
 
    Formula: dot product of the vectors, divided by the product
    of their magnitudes (lengths). Result ranges from -1 (opposite
    meaning) to 1 (identical meaning); 0 means unrelated.
    """
    return torch.dot(a,b)/(torch.norm(a)*torch.norm(b))

print("\nPairwise similarities:")

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_sim(embeddings[i], embeddings[j]).item()
        print(f"{sim:.3f} | {sentences[i][:45]:45} <-> {sentences[j][:45]:45}")
        
print(
    "\nWhat to look for: sentence 0 and 1 (both about loan loss "
    "provisions) should score noticeably higher than either against "
    "sentence 4 (weather). That gap IS retrieval working — it's the "
    "same math your Chroma vector search will lean on in Phase 3."
)
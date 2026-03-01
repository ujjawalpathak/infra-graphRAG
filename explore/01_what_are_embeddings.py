"""
1. WHAT ARE EMBEDDINGS (VECTORS)?

Text is converted into a fixed-length list of numbers (a vector).
Similar meaning → similar numbers. The model learns this from lots of training data.

Run: python explore/01_what_are_embeddings.py
"""
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Example 1: One sentence → one vector ---
text = "Kubernetes deployments manage pods."
vec = model.encode(text)

print("=" * 60)
print("1. ONE SENTENCE → ONE VECTOR")
print("=" * 60)
print(f"Text: '{text}'")
print(f"Vector shape: {vec.shape}  (each chunk becomes {len(vec)} numbers)")
print(f"First 10 values: {vec[:10]}")
print(f"All values are floats, typically between -1 and 1")
print()

# --- Example 2: Similar texts → similar vectors ---
text_a = "How do I scale a deployment?"
text_b = "How to scale up a Kubernetes deployment"
text_c = "What is the weather today?"

vec_a = model.encode(text_a)
vec_b = model.encode(text_b)
vec_c = model.encode(text_c)

# Cosine similarity: 1 = identical, 0 = unrelated, -1 = opposite
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("2. SIMILAR TEXTS → SIMILAR VECTORS")
print("=" * 60)
print(f"  '{text_a}'")
print(f"  vs '{text_b}'  →  similarity: {cos_sim(vec_a, vec_b):.4f}")
print()
print(f"  '{text_a}'")
print(f"  vs '{text_c}'  →  similarity: {cos_sim(vec_a, vec_c):.4f}")
print()
print("Same topic (deployments) = high similarity. Different topic = low.")
print()

# --- Example 3: What values are assigned? ---
print("3. WHAT VALUES ARE ASSIGNED?")
print("=" * 60)
print("The model is a neural network trained on millions of sentences.")
print("Each of the 384 dimensions captures some aspect of meaning.")
print("We don't hand-pick these numbers; the model learns them.")
print(f"Value range in our vector: min={vec.min():.4f}, max={vec.max():.4f}")

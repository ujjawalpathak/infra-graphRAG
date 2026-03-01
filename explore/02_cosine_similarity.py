"""
2. COSINE SIMILARITY — HOW WE COMPARE VECTORS

Two vectors: how "aligned" are they? Cosine measures the angle between them.
  - Same direction (similar meaning) → cos ≈ 1
  - Perpendicular (unrelated)        → cos ≈ 0
  - Opposite direction              → cos ≈ -1

Run: python explore/02_cosine_similarity.py
"""
import numpy as np

# --- Simple 3D example (easier to visualize than 384D) ---
# In real search we use 384 dimensions, but the math is identical.

a = np.array([1.0, 2.0, 1.0])   # vector A
b = np.array([2.0, 4.0, 2.0])   # vector B = 2*A (same direction!)
c = np.array([-2.0, 1.0, 0.0])   # vector C (perpendicular to A: A·C = 0)

print("=" * 60)
print("COSINE SIMILARITY FORMULA")
print("=" * 60)
print()
print("  cos(A, B) = (A · B) / (||A|| × ||B||)")
print()
print("  A · B = dot product = sum of (a_i * b_i)")
print("  ||A|| = norm (length) of A = sqrt(sum of a_i²)")
print()

# Dot product
dot_ab = np.dot(a, b)
dot_ac = np.dot(a, c)
print("1. DOT PRODUCT (A · B)")
print("-" * 40)
print(f"  A = {a}")
print(f"  B = {b}")
print(f"  A · B = {a[0]}*{b[0]} + {a[1]}*{b[1]} + {a[2]}*{b[2]} = {dot_ab}")
print(f"  A · C = {dot_ac}  (perpendicular → dot product = 0)")
print()

# Norms
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print("2. NORMS (vector length)")
print("-" * 40)
print(f"  ||A|| = sqrt({a[0]}² + {a[1]}² + {a[2]}²) = {norm_a:.4f}")
print(f"  ||B|| = {norm_b:.4f}")
print()

# Cosine
cos_ab = dot_ab / (norm_a * norm_b)
cos_ac = np.dot(a, c) / (norm_a * np.linalg.norm(c))
print("3. COSINE SIMILARITY")
print("-" * 40)
print(f"  cos(A, B) = {dot_ab} / ({norm_a:.4f} × {norm_b:.4f}) = {cos_ab:.4f}")
print(f"  A and B point same direction → cos = 1.0")
print()
print(f"  cos(A, C) = {cos_ac:.4f}")
print(f"  A and C are perpendicular → cos = 0")
print()

# Batch version (what the API does)
print("4. BATCH: COMPARE ONE QUERY TO MANY CHUNKS")
print("-" * 40)
query = np.array([0.5, 0.3, 0.8])
chunks = np.array([
    [0.6, 0.2, 0.7],   # similar to query
    [0.1, 0.9, 0.1],   # different
    [0.5, 0.3, 0.8],   # same as query
])
# embs @ qv = dot product of each row with query
dots = chunks @ query
norms_chunks = np.linalg.norm(chunks, axis=1)
norm_query = np.linalg.norm(query)
sims = dots / (norms_chunks * norm_query + 1e-9)
print(f"  query vector: {query}")
print(f"  chunk 0 similarity: {sims[0]:.4f}")
print(f"  chunk 1 similarity: {sims[1]:.4f}")
print(f"  chunk 2 similarity: {sims[2]:.4f}  (identical → 1.0)")
print()
print("  In main.py: embs @ qv does dots for ALL chunks at once (fast!)")

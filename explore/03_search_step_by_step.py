"""
3. SEARCH STEP BY STEP — What main.py does, with prints

Run: python explore/03_search_step_by_step.py
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("=" * 60)
print("LOADING: Chunks → Vectors")
print("=" * 60)

chunks = []
vecs = []
with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:  # only first 5 for demo
            break
        rec = json.loads(line)
        chunks.append(rec)
        vec = model.encode(rec["text"])
        vecs.append(vec)
        print(f"  Chunk {i}: '{rec['text'][:100]}...' → vector shape {vec.shape}")

embs = np.vstack(vecs).astype("float32")
print(f"\n  Stacked matrix shape: {embs.shape}  (rows=chunks, cols=384)")
print()

# --- Search ---
query = "how do I scale a deployment?"
print("=" * 60)
print("SEARCH: Query → Compare → Rank")
print("=" * 60)
print(f"  Query: '{query}'")
print()

# Step 1: Encode query
qv = model.encode(query).astype("float32")
print("  Step 1: Encode query → vector shape", qv.shape)
print()

# Step 2: Cosine similarity (same formula as 02)
dots = embs @ qv
norms_embs = np.linalg.norm(embs, axis=1)
norm_qv = np.linalg.norm(qv)
sims = dots / (norms_embs * norm_qv + 1e-9)
print("  Step 2: Cosine similarity for each chunk:")
for i, s in enumerate(sims):
    print(f"    Chunk {i}: {s:.4f}")
print()

# Step 3: Top-k
k = 3
top_indices = np.argsort(-sims)[:k]
print(f"  Step 3: Top-{k} indices (sorted by similarity):", top_indices.tolist())
print()

# Step 4: Build results
print("  Step 4: Results")
print("-" * 40)
for idx in top_indices:
    rec = chunks[int(idx)]
    print(f"    score={sims[int(idx)]:.4f} | {rec['source_file'][:40]}...")
    print(f"    preview: {rec['text'][:80]}...")
    print()

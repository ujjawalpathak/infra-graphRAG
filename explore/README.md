# Explore: Embeddings & Cosine Similarity

Small runnable snippets to understand how Step 7 (semantic search) works.

## Run from project root

```bash
cd /Users/ujjawalpathak/Desktop/playground/infra_graphRAG

# Use venv (sentence_transformers lives there)
source .venv/bin/activate   # or: .venv/bin/python explore/...

# 1. What are embeddings? Text → vectors, similar text → similar vectors
python explore/01_what_are_embeddings.py

# 2. Cosine similarity: dot product, norms, the formula (no model needed)
python explore/02_cosine_similarity.py

# 3. Full search flow with verbose output (uses first 5 chunks only)
python explore/03_search_step_by_step.py
```

**Note:** 01 and 03 need the SentenceTransformer model. If not cached, first run may download it (~80MB).

## Order

1. **01** — What values are we assigning? (neural network outputs 384 floats per text)
2. **02** — How do we compare? (cosine = dot product / (norm × norm))
3. **03** — How does it all fit together in search?

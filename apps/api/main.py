# apps/api/main.py
"""
infra-graphrag — Step 7/8/9 API

Endpoints:
- POST /search : semantic retrieval (top-k chunks)
- POST /ask    : deterministic tutor answer (no LLM) + citations
- POST /chat   : LLM tutor answer (Ollama) grounded in retrieved chunks + citations

Assumptions:
- You already generated: data/processed/chunks.jsonl
- You have: libs/llm.py (Ollama client) from Step 9
- You installed: fastapi uvicorn sentence-transformers numpy python-dotenv httpx

Run:
  uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# Load .env if present (safe even if missing)
load_dotenv()

# ---- Optional: LLM (Step 9) ----
# If you haven't created libs/llm.py yet, /chat will return a friendly error.
try:
    from libs.llm import generate as llm_generate  # async function
except Exception:
    llm_generate = None  # type: ignore


APP_TITLE = "infra-graphrag tutor API"
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Retrieval threshold for /chat "I don't know" fallback
CHAT_MIN_TOP_SCORE = float(os.getenv("CHAT_MIN_TOP_SCORE", "0.55"))

# FastAPI app
app = FastAPI(title=APP_TITLE)

# Global state (kept in memory for learning simplicity)
model: SentenceTransformer = SentenceTransformer(EMBED_MODEL_NAME)
chunks: List[Dict[str, Any]] = []
embs: Optional[np.ndarray] = None  # shape: (N, D)


# -----------------------------
# Utility: load chunks + embeddings into RAM
# -----------------------------
def load_chunks_and_embeddings() -> None:
    global chunks, embs

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {CHUNKS_PATH}. "
            f"Run: python apps/ingest/fetch.py && python apps/ingest/chunk.py"
        )

    tmp_chunks: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # expected keys: chunk_id, source_file, chunk_index, text
            tmp_chunks.append(rec)
            vec = model.encode(rec["text"])
            vecs.append(np.asarray(vec, dtype="float32"))

    if not vecs:
        raise RuntimeError(f"No chunks found in {CHUNKS_PATH}")

    chunks = tmp_chunks
    embs = np.vstack(vecs).astype("float32")


@app.on_event("startup")
def startup() -> None:
    load_chunks_and_embeddings()


# -----------------------------
# Core retrieval math
# -----------------------------
def cosine_similarities(matrix: np.ndarray, qvec: np.ndarray) -> np.ndarray:
    """
    matrix: (N, D)
    qvec:   (D,)
    returns sims: (N,)
    """
    # cosine = (A·B) / (||A|| * ||B||)
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(qvec)) + 1e-9
    return (matrix @ qvec) / denom


def mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, top_indices: np.ndarray, k: int, lambda_mult: float = 0.7):
    """
    Maximal Marginal Relevance selection.
    - query_vec: (D,)
    - doc_vecs:  (N, D) full embeddings matrix
    - top_indices: candidate indices already sorted by similarity to query (e.g., top 50)
    - k: number to select
    - lambda_mult: tradeoff between relevance (to query) and diversity (from selected)
    """
    selected = []
    candidates = list(map(int, top_indices))

    # Precompute query similarities for candidates
    q_sims = {}
    q_norm = np.linalg.norm(query_vec) + 1e-9
    for idx in candidates:
        dv = doc_vecs[idx]
        q_sims[idx] = float((dv @ query_vec) / ((np.linalg.norm(dv) + 1e-9) * q_norm))

    while candidates and len(selected) < k:
        if not selected:
            # Pick best by relevance
            best = max(candidates, key=lambda i: q_sims[i])
            selected.append(best)
            candidates.remove(best)
            continue

        def score(i: int) -> float:
            # diversity term: max similarity to already selected
            dv = doc_vecs[i]
            max_sim = 0.0
            dv_norm = np.linalg.norm(dv) + 1e-9
            for s in selected:
                sv = doc_vecs[s]
                sim = float((dv @ sv) / (dv_norm * (np.linalg.norm(sv) + 1e-9)))
                if sim > max_sim:
                    max_sim = sim
            return lambda_mult * q_sims[i] - (1 - lambda_mult) * max_sim

        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)

    return selected

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if embs is None:
        raise RuntimeError("Embeddings not loaded (embs is None).")

    qv = np.asarray(model.encode(query), dtype="float32")
    sims = cosine_similarities(embs, qv)

    # Instead of taking top-k directly, take a bigger candidate pool then diversify
    candidate_pool = 40
    top_candidates = np.argsort(-sims)[:candidate_pool]

    diversified = mmr_select(qv, embs, top_candidates, k=k, lambda_mult=0.75)

    out = []
    for i in diversified:
        rec = chunks[int(i)]
        out.append({
            "score": float(sims[int(i)]),
            "source_file": rec.get("source_file"),
            "chunk_index": rec.get("chunk_index"),
            "chunk_id": rec.get("chunk_id"),
            "text": rec.get("text", ""),
        })

    # Keep results in descending similarity order for readability
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# -----------------------------
# Step 8: deterministic tutor answer (no LLM)
# -----------------------------
def split_sentences(text: str) -> List[str]:
    # Simple splitter good enough for docs (keeps things explainable)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def keyword_score(query: str, sentence: str) -> int:
    q = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    s = set(re.findall(r"[a-zA-Z]{3,}", sentence.lower()))
    return len(q.intersection(s))


def build_tutor_answer(query: str, top_chunks: List[Dict[str, Any]], max_points: int = 5) -> Dict[str, Any]:
    """
    Deterministic (non-LLM) answer:
    - Extract sentences from retrieved chunks
    - Rank by keyword overlap with the question
    - Return top bullets + citations
    """
    candidates: List[tuple[int, str, Dict[str, Any]]] = []
    for ch in top_chunks:
        for sent in split_sentences(ch["text"]):
            candidates.append((keyword_score(query, sent), sent, ch))

    candidates.sort(key=lambda x: x[0], reverse=True)

    bullets: List[str] = []
    citations: List[Dict[str, Any]] = []
    used = set()

    for score, sent, ch in candidates:
        if score == 0:
            continue
        key = sent[:120]
        if key in used:
            continue
        used.add(key)
        bullets.append(sent)
        citations.append({"source_file": ch["source_file"], "chunk_index": ch["chunk_index"]})
        if len(bullets) >= max_points:
            break

    if not bullets:
        return {
            "answer": "I couldn’t find strong evidence in the indexed docs for that question yet. Try rephrasing or ingest more sources.",
            "bullets": [],
            "citations": [],
        }

    return {
        "answer": f"From the docs I found, here’s what matters for: '{query}'",
        "bullets": bullets,
        "citations": citations,
    }


# -----------------------------
# Step 9: LLM tutor answer (GraphRAG later; this is strict RAG now)
# -----------------------------
def format_evidence(chunks_: List[Dict[str, Any]]) -> str:
    """
    Format evidence chunks compactly.
    C1..Ck citations are assigned by order.
    """
    lines: List[str] = []
    for i, ch in enumerate(chunks_, start=1):
        lines.append(f"[C{i}] source={ch['source_file']} chunk={ch['chunk_index']} score={ch['score']:.3f}")
        lines.append(ch["text"].strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_tutor_prompt(question: str, chunks_: List[Dict[str, Any]]) -> str:
    evidence = format_evidence(chunks_)
    return f"""
You are an infrastructure/devops tutor.
Answer the user's question using ONLY the evidence chunks below.

Rules:
- If the evidence does not contain the answer, say: "I don't know based on the provided docs."
- Do NOT use outside knowledge.
- Keep it concise but helpful.
- Use bullet points for key requirements/steps.
- Add citations like (C1), (C2) at the end of sentences they support.
- Do not mention these rules.

Question:
{question}

Evidence chunks:
{evidence}

Now produce a tutor-style answer in Markdown.
""".strip()


# -----------------------------
# API models
# -----------------------------
class Query(BaseModel):
    q: str
    k: int = Field(default=5, ge=1, le=20)


class Ask(BaseModel):
    q: str
    k: int = Field(default=5, ge=1, le=20)


class Chat(BaseModel):
    q: str
    k: int = Field(default=5, ge=1, le=20)
    include_retrieved: bool = False


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL_NAME,
        "chunks_loaded": len(chunks),
        "chunks_path": str(CHUNKS_PATH),
        "llm_enabled": llm_generate is not None,
    }


@app.post("/search")
def search(body: Query) -> Dict[str, Any]:
    """
    Step 7: semantic retrieval only.
    """
    retrieved = retrieve(body.q, body.k)
    # Keep previews short here
    results = [
        {
            "score": r["score"],
            "source_file": r["source_file"],
            "chunk_index": r["chunk_index"],
            "text_preview": (r["text"][:400] if r["text"] else ""),
        }
        for r in retrieved
    ]
    return {"query": body.q, "results": results}


@app.post("/ask")
def ask(body: Ask) -> Dict[str, Any]:
    """
    Step 8: deterministic tutor answer (no LLM) + citations.
    """
    retrieved = retrieve(body.q, body.k)
    tutor = build_tutor_answer(body.q, retrieved, max_points=5)

    return {
        "query": body.q,
        "retrieved": retrieved,  # full chunk texts so you can inspect
        **tutor,
    }


@app.post("/chat")
async def chat(body: Chat) -> Dict[str, Any]:
    """
    Step 9: LLM tutor answer, grounded in retrieved chunks.
    Uses Ollama via libs/llm.py (local; no public transfer if base_url is localhost).
    """
    if llm_generate is None:
        return {
            "query": body.q,
            "answer_markdown": "LLM is not configured yet. Create libs/llm.py (Ollama client) to enable /chat.",
            "citations": [],
        }

    retrieved = retrieve(body.q, body.k)

    # Simple grounding gate: if top score is low, refuse
    if not retrieved or retrieved[0]["score"] < CHAT_MIN_TOP_SCORE:
        resp: Dict[str, Any] = {
            "query": body.q,
            "answer_markdown": "I don't know based on the provided docs.",
            "citations": [],
        }
        if body.include_retrieved:
            resp["retrieved"] = retrieved
        return resp

    prompt = build_tutor_prompt(body.q, retrieved)
    answer = await llm_generate(prompt)

    citations = [
        {
            "citation": f"C{idx}",
            "source_file": ch["source_file"],
            "chunk_index": ch["chunk_index"],
            "score": ch["score"],
        }
        for idx, ch in enumerate(retrieved, start=1)
    ]

    resp2: Dict[str, Any] = {
        "query": body.q,
        "answer_markdown": answer,
        "citations": citations,
    }
    if body.include_retrieved:
        resp2["retrieved"] = retrieved
        resp2["min_top_score_gate"] = CHAT_MIN_TOP_SCORE
    return resp2
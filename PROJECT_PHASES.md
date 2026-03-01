# infra-graphrag --- Project Phases & Learning Journey

## Project Vision

Build a complete, extensible, locally hosted AI system that:

-   Ingests infrastructure / DevOps / cloud documentation
-   Structures it into retrievable knowledge
-   Answers questions using Retrieval-Augmented Generation (RAG)
-   Enforces grounding, citations, and refusal logic
-   Runs fully locally without external LLM APIs
-   Evolves into a GraphRAG + Knowledge Graph system

This project was intentionally built in phases to deeply understand each
layer of AI system design rather than relying on black-box APIs.

------------------------------------------------------------------------

# Phase 0 --- Problem Framing & Architecture Design

Objective: Design a DevOps documentation tutor that avoids hallucination
and is auditable.

Key architectural decision: Use Retrieval-Augmented Generation (RAG)
instead of fine-tuning or blind LLM prompting.

Learning focus: - Why grounding matters - Why retrieval engineering is
more important than prompt engineering - How AI systems should be
structured safely

------------------------------------------------------------------------

# Phase 1 --- Document Ingestion

-   Selected public Kubernetes documentation
-   Scraped and cleaned HTML
-   Extracted normalized text
-   Stored structured raw documents

Learning outcomes: - Data preprocessing - Noise removal - Text
normalization - AI systems start with data engineering

------------------------------------------------------------------------

# Phase 2 --- Chunking Strategy

-   Split documents into \~1200-character chunks
-   Added overlap (\~200 chars)
-   Stored as structured JSONL with:
    -   chunk_id
    -   source_file
    -   chunk_index
    -   text

Key insight: Chunking directly impacts retrieval quality and final
answer correctness.

------------------------------------------------------------------------

# Phase 3 --- Embeddings & Vector Representation

-   Used SentenceTransformer (all-MiniLM-L6-v2)
-   Converted chunks and queries into embeddings
-   Stored vectors in memory

Learning outcomes: - Vector representations - Semantic similarity -
Cosine similarity math - Meaning-based retrieval

------------------------------------------------------------------------

# Phase 4 --- Deterministic Retrieval

-   Query → embedding
-   Cosine similarity ranking
-   Top-K chunk retrieval

Critical insight: Retrieval defines the system's truth boundary. If
retrieval is wrong, grounded generation will still produce wrong
answers.

------------------------------------------------------------------------

# Phase 5 --- Deterministic Tutor Mode (No LLM)

Before adding generation: - Extracted key sentences - Ranked via keyword
overlap - Returned structured bullet summaries with citations

Purpose: Understand retrieval deeply before adding generation.

------------------------------------------------------------------------

# Phase 6 --- Prompt-Controlled Generation (Local LLM)

Integrated Ollama for local model inference.

Prompt enforces: - Use only provided evidence - Refuse if evidence
insufficient - Add inline citations (C1, C2...)

Key realization: Retrieval decides truth. Prompt decides explanation.

------------------------------------------------------------------------

# Phase 7 --- Deterministic Grounding & Confidence Gating

Implemented similarity threshold gating:

If retrieval confidence is low → refuse.

This prevents confident misinformation and preserves trust.

------------------------------------------------------------------------

# Phase 8 --- Citation Discipline

Each answer includes: - Inline citations (C1, C2...) - Source file -
Chunk index - Similarity score

Purpose: Auditability, transparency, and hallucination detection.

------------------------------------------------------------------------

# Phase 9 --- Local Model Hosting

All generation runs locally via Ollama (127.0.0.1). No external APIs. No
data leaves the machine.

Benefits: - Privacy - Control - Enterprise readiness

------------------------------------------------------------------------

# Current Architecture

User Question ↓ Embedding ↓ Cosine Similarity Retrieval ↓ Top-K Evidence
↓ Strict Prompt Construction ↓ Local LLM (Ollama) ↓ Cited Markdown
Answer

------------------------------------------------------------------------

# Identified Failure Modes

1.  Ungrounded hallucination --- mitigated by strict prompt + citations.
2.  Retrieval failure --- requires improved chunking, ranking, hybrid
    search.

Key insight: Retrieval engineering is more important than prompt
engineering.

------------------------------------------------------------------------

# Future Roadmap

Phase 10 --- Retrieval Improvements - Heading-aware chunking - Hybrid
search (BM25 + vector) - Reranking - Query rewriting

Phase 11 --- Evaluation Harness - Test dataset - Measure retrieval
accuracy - Measure citation correctness - Measure refusal correctness

Phase 12 --- Knowledge Graph Integration (GraphRAG) Introduce structured
nodes and relationships: - Concept - Resource - Configuration -
DocPage - DEPENDS_ON - REQUIRES - CONFIGURES - RELATED_TO

Graph layer will guide retrieval and enable structured reasoning.

Phase 13 --- Corpus Expansion - Terraform docs - AWS docs - CNCF
projects - Incident troubleshooting guides

Phase 14 --- Enterprise Controls - Role-based access - Audit logging -
pgvector storage - Production deployment - Caching layer

------------------------------------------------------------------------

# Final Vision

infra-graphrag aims to become a complete, locally hosted,
citation-grounded, knowledge-graph-augmented DevOps intelligence system
built through incremental learning and architectural rigor.

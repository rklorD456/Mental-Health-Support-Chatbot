# src/rag/

Retrieval-Augmented Generation (RAG) pipeline components.

## Files

| File | Purpose |
|------|---------|
| `embedder.py` | Wraps `SentenceTransformer` to embed user queries using BAAI/bge-base-en-v1.5 |
| `retriever.py` | `HybridQdrantRetriever` — queries Qdrant with dense vectors and fuses results via Reciprocal Rank Fusion (RRF) |
| `reranker.py` | `CrossEncoderReranker` — re-scores retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` and returns the top-N |
| `pipeline.py` | `RAGPipeline` — orchestrates retrieval → reranking → LLM response generation with emotion-aware prompting |

## Pipeline Flow

```
User query
  → embedder.embed_query()          # text → 768-dim vector
  → retriever.retrieve(top_k=10)    # Qdrant dense search + RRF fusion
  → reranker.rerank(top_n=3)        # cross-encoder scoring
  → LLM call with context + emotion tone instruction
  ← Generated response
```

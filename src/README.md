# src/

Main application source code for the Mental Health Support Chatbot.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, lifespan (model loading), routes (`/`, `/health`, `/chat`) |
| `config.py` | Loads environment variables and defines application settings |
| `router.py` | Chat orchestration — language detection → guardrails → intent/emotion → RAG → response |
| `ingest_data.py` | One-time script to embed and upload counseling data to Qdrant |

## Submodules

| Directory | Purpose |
|-----------|---------|
| `core/` | Pydantic schemas, abstract interfaces, keyword-based intent detection & multilingual responses |
| `database/` | Qdrant client factory and collection helpers |
| `rag/` | Embedder, retriever, reranker, and the RAG pipeline |
| `services/` | Classifier (emotion + language), intent classification, translator, guardrail |

## Request Flow

```
POST /chat
  → router.process_chat()
    → classifier.detect_language()
    → detect_intent_from_keywords()     # fast local match
    → guardrail.is_safe()               # prompt injection filter
    → classifier.detect_emotion()       # parallel
    → intent.get_intent()               # parallel (LLM call)
    → rag_pipeline.generate_response()  # if mental health question
    → translator.from_english()         # if non-English user
  ← ChatResponse
```

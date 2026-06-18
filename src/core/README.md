# src/core/

Core abstractions and data structures used across the application.

## Files

| File | Purpose |
|------|---------|
| `schemas.py` | Pydantic models for API request/response (`ChatRequest`, `ChatResponse`, `IntentResponse`) |
| `interfaces.py` | Abstract base classes (`BaseEmbedder`, `BaseRetriever`, `BaseReranker`) that the RAG components implement |
| `responses.py` | Local keyword-based intent detection (20 languages) and multilingual canned response dictionaries — used to bypass the LLM for simple greetings, goodbyes, and gratitude |

# src/database/

Qdrant vector database connection layer.

## Files

| File | Purpose |
|------|---------|
| `qdrant.py` | `get_qdrant_client()` — factory that creates a configured `QdrantClient` with timeout. `ensure_collection()` — creates the vector collection if it doesn't exist. |

## Configuration

The client reads `QDRANT_URL` and `QDRANT_API_KEY` from settings (loaded from `.env`). Connection timeout is set to 10 seconds.

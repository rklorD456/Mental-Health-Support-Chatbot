# Standard library
import logging
import uuid

# Third-party
from datasets import load_dataset
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Local
from src.config import get_settings
from src.database.qdrant import get_qdrant_client, ensure_collection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = get_settings()

COLLECTION_NAME = settings.collection_name
BATCH_SIZE = 100
ENCODE_BATCH_SIZE = 256

logger.info("1. Connecting to Qdrant and loading Sentence Transformer...")
qdrant_client = get_qdrant_client()

logger.info("Loading BGE model...")
embedding_model = SentenceTransformer(settings.embedding_model_name)

ensure_collection(qdrant_client, COLLECTION_NAME, vector_size=768)

logger.info("Downloading dataset...")
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

logger.info("Loaded %d conversations. Starting vector ingestion...", len(dataset))

# Process in batches for efficient encoding
for batch_start in range(0, len(dataset), ENCODE_BATCH_SIZE):
    batch_end = min(batch_start + ENCODE_BATCH_SIZE, len(dataset))
    batch_rows = dataset[batch_start:batch_end]

    texts_to_embed = [
        f"Patient Context: {q}\nCounselor Response: {a}"
        for q, a in zip(batch_rows["Context"], batch_rows["Response"])
    ]

    vectors = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "question": q,
                "answer": a,
                "page_content": t,
            },
        )
        for vec, q, a, t in zip(
            vectors,
            batch_rows["Context"],
            batch_rows["Response"],
            texts_to_embed,
        )
    ]

    # Upsert in sub-batches of BATCH_SIZE to respect Qdrant limits
    for i in range(0, len(points), BATCH_SIZE):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + BATCH_SIZE],
        )

    logger.info("Upserted rows %d–%d of %d", batch_start + 1, batch_end, len(dataset))

logger.info("Ingestion complete!")
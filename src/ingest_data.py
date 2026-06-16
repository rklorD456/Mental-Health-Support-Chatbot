# Standard library
import uuid

# Third-party
from datasets import load_dataset
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Local
from src.config import get_settings
from src.database.qdrant import get_qdrant_client, ensure_collection

settings = get_settings()

COLLECTION_NAME = settings.collection_name
BATCH_SIZE = 100

print("1. Connecting to Qdrant and loading Sentence Transformer...")
qdrant_client = get_qdrant_client()

print("Loading BGE model...")
embedding_model = SentenceTransformer(settings.embedding_model_name)

ensure_collection(qdrant_client, COLLECTION_NAME, vector_size=768)

print("Downloading dataset...")
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

print(f"Loaded {len(dataset)} conversations. Starting vector ingestion...")

points = []

for i, row in enumerate(dataset):
    question = row["Context"]
    answer   = row["Response"]

    text_to_embed = f"Patient Context: {question}\nCounselor Response: {answer}"
    vector = embedding_model.encode(text_to_embed).tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "question":     question,
            "answer":       answer,
            "page_content": text_to_embed,
        },
    )
    points.append(point)

    # Upload in batches
    if len(points) >= BATCH_SIZE:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted batch ending at row {i + 1}...")
        points = []

# Upload any remaining points
if points:
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

print("Ingestion complete!")
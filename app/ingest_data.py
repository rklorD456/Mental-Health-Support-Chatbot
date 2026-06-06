import os
import uuid
from dotenv import load_dotenv
from datasets import load_dataset

from qdrant_client import QdrantClient
from qdrant_client.http import models

from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "mental_health_knowledge_base"
BATCH_SIZE = 100 

print("1. Connecting to Qdrant and loading Sentence Transformer...")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True
)

# Load the model
print("Loading BGE model...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

if not qdrant_client.collection_exists(COLLECTION_NAME):
    print("Creating collection...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
else:
    print("Collection already exists. Skipping creation.")


print("Downloading dataset...")
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")

print(f"Loaded {len(dataset)} conversations. Starting vector ingestion...")


points = []

for i, row in enumerate(dataset):
    question = row["Context"]
    answer = row["Response"]
    
    text_to_embed = f"Patient Context: {question}\nCounselor Response: {answer}"
    vector = embedding_model.encode(text_to_embed).tolist()
    
    point = models.PointStruct(
        id=str(uuid.uuid4()), # Generate a unique ID for each vector
        vector=vector,
        payload={
            "question": question,
            "answer": answer,
            "page_content": text_to_embed
        }
    )
    points.append(point)

    # Upload in batches
    if len(points) >= BATCH_SIZE:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Upserted batch {i + 1}...")
        points = [] # Clear the list for the next batch

# Upload any remaining points
if points:
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("Ingestion complete!")
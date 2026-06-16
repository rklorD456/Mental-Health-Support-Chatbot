"""
Qdrant vector database client factory and collection helpers.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config import get_settings


def get_qdrant_client() -> QdrantClient:
    """Create and return a configured QdrantClient using application settings."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 768) -> None:
    """Create the Qdrant collection if it does not already exist.

    Args:
        client: An active QdrantClient instance.
        collection_name: Name of the collection to check / create.
        vector_size: Dimensionality of the stored vectors (default 768 for BGE-base).
    """
    if not client.collection_exists(collection_name):
        print(f"Creating Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
    else:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")

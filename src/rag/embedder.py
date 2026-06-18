from typing import List
from sentence_transformers import SentenceTransformer
from src.core.interfaces import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        full_query = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode(full_query, show_progress_bar=False).tolist()
    
    
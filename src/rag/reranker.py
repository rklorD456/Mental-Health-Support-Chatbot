from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from src.core.interfaces import BaseReranker

class CrossEncoderReranker(BaseReranker):
    """Reranker using a CrossEncoder model from Sentence Transformers.

    Args:
        BaseReranker (_type_): Abstract base class for rerankers.
    """
    
    def __init__(self, model_name: str):
        """Initialize the CrossEncoderReranker with a specified model."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[str, Dict[str, Any]]], top_n: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        pairs = []
        for _, payload in candidates:
            passage = payload.get("page_content", "") or (payload.get("question", "") + " " + payload.get("answer", ""))
            pairs.append((query, passage))

        scores = self.model.predict(pairs)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [candidate for _, candidate in scored[:top_n]]
    
    
from typing import List, Dict, Any, Tuple
from src.core.interfaces import BaseRetriever, BaseEmbedder

class HybridQdrantRetriever(BaseRetriever):
    def __init__(self, qdrant_client, embedder: BaseEmbedder, collection_name: str):
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        query_vector = self.embedder.embed_query(query)
        
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        
        dense_candidates = [(str(hit.id), hit.payload) for hit in dense_results]
        
        # Replace this empty list later with a real sparse index query
        sparse_candidates = [] 
        
        if not dense_candidates and not sparse_candidates:
            return []
            
        return self._reciprocal_rank_fusion([dense_candidates, sparse_candidates])

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple[str, Dict[str, Any]]]], k: int = 60) -> List[Tuple[str, Dict[str, Any]]]:
        rrf_scores: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            for rank, (pid, payload) in enumerate(ranked_list, start=1):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
                if pid not in payload_map:
                    payload_map[pid] = payload

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [(pid, payload_map[pid]) for pid in sorted_ids]
    
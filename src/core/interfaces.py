from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple



class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        pass

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[Tuple[str, Dict[str, Any]]], top_n: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        pass
    
    

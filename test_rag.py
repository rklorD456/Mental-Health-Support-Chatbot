import os
import sys

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import get_settings
from src.database.qdrant import get_qdrant_client
from src.rag.embedder import SentenceTransformerEmbedder
from src.rag.retriever import HybridQdrantRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.pipeline import RAGPipeline
from openai import OpenAI

def test_rag():
    settings = get_settings()
    llm_client = OpenAI(
        api_key=settings.model_used_api, 
        base_url=settings.model_used_base_url
    )
    qdrant_client = get_qdrant_client()
    
    embedder = SentenceTransformerEmbedder(settings.embedding_model_name)
    retriever = HybridQdrantRetriever(qdrant_client, embedder, settings.collection_name)
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    pipeline = RAGPipeline(retriever, reranker, llm_client, settings)
    
    print("Sending test query to RAG pipeline...")
    response = pipeline.generate_response(
        user_message="I have been feeling a lot of anxiety lately and I can't sleep.",
        emotion="fear",
        history=[]
    )
    
    print("\n--- LLM RESPONSE ---")
    print(response)

if __name__ == "__main__":
    test_rag()

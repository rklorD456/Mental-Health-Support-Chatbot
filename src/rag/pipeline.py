from typing import List, Optional
from src.core.interfaces import BaseRetriever, BaseReranker

class RAGPipeline:
    """summary
    A pipeline that combines retrieval, reranking, and response generation for a mental health support chatbot.
    """
    
    def __init__(self, retriever: BaseRetriever, reranker: BaseReranker, llm_client, settings):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.settings = settings
        
        self.tone_map = {
            "sadness": "The user is feeling sad. Be especially gentle, validating, and warm.",
            "fear": "The user is feeling anxious or fearful. Be calm, reassuring, and grounding.",
            "anger": "The user is feeling angry or frustrated. Acknowledge their feelings without escalating.",
            "joy": "The user is in a positive mood. Be warm and encouraging.",
        }

    def generate_response(self, user_message: str, emotion: str = "neutral", history: Optional[List[dict]] = None) -> str:
        history = history or []
        
        fused_candidates = self.retriever.retrieve(user_message, top_k=10)
        if not fused_candidates:
            return "I am having trouble accessing my knowledge base right now."
            
        top_results = self.reranker.rerank(user_message, fused_candidates, top_n=3)
        
        contexts = [str(p.get("answer", ""))[:1500] for _, p in top_results]
        context_string = "\n\n---\n\n".join(contexts)
        
        tone_instruction = f"\nTone instruction: {self.tone_map[emotion]}" if emotion in self.tone_map else ""
        
        system_prompt = (
            f"You are an empathetic, professional mental health support chatbot.\n"
            f"Answer using ONLY the provided context.\n{tone_instruction}\n\n"
            f"CONTEXT:\n{context_string}"
        )
        
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}]
        
        response = self.llm_client.chat.completions.create(
            model=self.settings.model_used_name,
            temperature=self.settings.model_used_temperature,
            messages=messages,
        )
        return response.choices[0].message.content
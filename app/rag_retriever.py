from app.config import get_settings
import app.main as main

settings = get_settings()

COLLECTION_NAME = settings.collection_name

EMOTION_TONE_MAP = {
    "sadness":  "The user is feeling sad. Be especially gentle, validating, and warm.",
    "fear":     "The user is feeling anxious or fearful. Be calm, reassuring, and grounding.",
    "anger":    "The user is feeling angry or frustrated. Acknowledge their feelings without escalating.",
    "joy":      "The user is in a positive mood. Be warm and encouraging.",
    "surprise": "The user seems surprised or overwhelmed. Be clear and steady.",
    "love":     "The user is feeling connected. Be warm and supportive.",
}
 

def get_rag_response(user_message: str, emotion: str = "neutral") -> str:
    """Takes a user message, retrieves relevant context from Qdrant, and generates a response.
    
    Args:
        user_message (str): 
            The input message from the user.
        emotion (str): 
            The detected emotion of the user.
    Returns:
        str:
            The generated response from the RAG pipeline.
        
        """
    embedding_model = main.models["embedding"]
    qdrant_client = main.models["qdrant_client"]
    llm_client = main.models["llm_client"]
    
    # Vectorize the User Query
    query_instruction = "Represent this sentence for searching relevant passages:"
    full_query = f"{query_instruction} {user_message}"
    query_vector = embedding_model.encode(full_query).tolist()

    # Retrieve top-3 relevant Context from Qdrant
    try:
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3, # Get the top 3 most relevant answers
            with_payload=True
        ).points
        
    except Exception as e:
        print(f"[ERROR] Qdrant search failed: {e}")
        return "I am having trouble accessing my knowledge base right now."

    # Extract the text answers from the vector payloads
    retrieved_contexts = [str(hit.payload.get("answer", ""))[:1500] + "..." for hit in search_results]
    context_string = "\n\n---\n\n".join(retrieved_contexts)
    
   
    tone_note = (
        f"\nTone instruction: {EMOTION_TONE_MAP[emotion]}"
        if emotion in EMOTION_TONE_MAP else ""
    )
    
    # Generate the Final Response
    system_prompt = (
        f"You are an empathetic, professional mental health support chatbot.\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Answer the user's query using ONLY the provided context below.\n"
        f"2. Do not invent information or provide outside medical facts.\n"
        f"3. Maintain a warm, supportive, and non-judgmental tone.\n"
        f"4. Keep your response concise and direct.{tone_note}\n\n"
        f"CONTEXT KNOWLEDGE:\n{context_string}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=settings.model_used_name,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return "I apologize, but I am currently experiencing a system error."

if __name__ == "__main__":
    # Standalone test — creates its own clients
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    
    main.models["llm_client"] = OpenAI(
        api_key=settings.model_used_api,
        base_url=settings.model_used_base_url,
    )
    main.models["qdrant_client"] = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    main.models["embedding"] = SentenceTransformer(settings.embedding_model_name)
    
    test_query = "I have been feeling really overwhelmed with my studies and stressed out lately."
    
    print(f"\nUser: {test_query}\n")
    print("Bot: Thinking...\n")
    
    answer = get_rag_response(test_query)
    print(f"Bot: {answer}\n")
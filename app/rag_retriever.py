import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

print("Initializing RAG pipeline...")

# Setup Clients
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True
)

# 2. Load the exact same embedding model
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
COLLECTION_NAME = "mental_health_knowledge_base"
model_name = "openai/gpt-oss-120b"
temperature = 0.3

EMOTION_TONE_MAP = {
    "sadness":  "The user is feeling sad. Be especially gentle, validating, and warm.",
    "fear":     "The user is feeling anxious or fearful. Be calm, reassuring, and grounding.",
    "anger":    "The user is feeling angry or frustrated. Acknowledge their feelings without escalating.",
    "joy":      "The user is in a positive mood. Be warm and encouraging.",
    "surprise": "The user seems surprised or overwhelmed. Be clear and steady.",
    "love":     "The user is feeling connected. Be warm and supportive.",
}
 

def get_rag_response(user_message: str, emotion: str = "neutral") -> str:
    """Takes a user message, retrieves relevant context from Qdrant, and generates a response using Groq.
    
    Args:
        user_message (str): 
            The input message from the user.
        emotion (str): 
            The detected emotion of the user.
    Returns:
        str:
            The generated response from the RAG pipeline.
        
        """
    
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
        response = groq_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Groq generation failed: {e}")
        return "I apologize, but I am currently experiencing a system error."

if __name__ == "__main__":
    # Test the pipeline
    test_query = "I have been feeling really overwhelmed with my studies and stressed out lately."
    
    print(f"\nUser: {test_query}\n")
    print("Bot: Thinking...\n")
    
    answer = get_rag_response(test_query)
    print(f"Bot: {answer}\n")
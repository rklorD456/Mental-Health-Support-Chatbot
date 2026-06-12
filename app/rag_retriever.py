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


# ---------------------------------------------------------------------------
# 1. Semantic search (dense vector via Qdrant)
# ---------------------------------------------------------------------------
def _semantic_search(query_vector: list, qdrant_client, top_k: int = 10) -> list:
    """Retrieve top-k results from Qdrant using dense cosine similarity.

    Returns a list of (point_id, payload) tuples preserving rank order.
    """
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return [(hit.id, hit.payload) for hit in results]


# ---------------------------------------------------------------------------
# 2. BM25 keyword search (sparse retrieval over Qdrant payloads)
# ---------------------------------------------------------------------------
def _bm25_search(user_message: str, qdrant_client, top_k: int = 10) -> list:
    """Scroll the entire collection, build a BM25 index, and return top-k.

    Returns a list of (point_id, payload) tuples preserving rank order.
    """
    from rank_bm25 import BM25Okapi

    # Scroll to get all documents from Qdrant
    all_points = []
    offset = None
    while True:
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = scroll_result
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    if not all_points:
        return []

    # Build corpus from the stored text field
    corpus_texts = []
    for pt in all_points:
        text = pt.payload.get("page_content", "") or (
            pt.payload.get("question", "") + " " + pt.payload.get("answer", "")
        )
        corpus_texts.append(text)

    # Tokenize (simple whitespace + lowercasing)
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    tokenized_query = user_message.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Rank by BM25 score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [(all_points[i].id, all_points[i].payload) for i in ranked_indices]


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(ranked_lists: list[list[tuple]], k: int = 60) -> list[tuple]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: Each element is a list of (point_id, payload) in rank order.
        k:            RRF constant (default 60 per the original paper).

    Returns:
        A single fused list of (point_id, payload) sorted by descending RRF score.
    """
    rrf_scores: dict = {}       # point_id -> cumulative RRF score
    payload_map: dict = {}      # point_id -> payload (keep the first seen)

    for ranked_list in ranked_lists:
        for rank, (pid, payload) in enumerate(ranked_list, start=1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in payload_map:
                payload_map[pid] = payload

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [(pid, payload_map[pid]) for pid in sorted_ids]


# ---------------------------------------------------------------------------
# 4. Cross-encoder reranking
# ---------------------------------------------------------------------------
def _rerank(query: str, candidates: list[tuple], top_n: int = 3) -> list[tuple]:
    """Re-score candidates with a cross-encoder and return the top-n.

    Args:
        query:      The original user query.
        candidates: List of (point_id, payload) from the fused results.
        top_n:      How many final results to keep.

    Returns:
        Top-n (point_id, payload) tuples after cross-encoder reranking.
    """
    cross_encoder = main.models["cross_encoder"]

    # Build (query, passage) pairs for the cross-encoder
    pairs = []
    for _, payload in candidates:
        passage = payload.get("page_content", "") or (
            payload.get("question", "") + " " + payload.get("answer", "")
        )
        pairs.append((query, passage))

    scores = cross_encoder.predict(pairs)

    # Attach scores, sort descending, take top-n
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [candidate for _, candidate in scored[:top_n]]


# ---------------------------------------------------------------------------
# 5. Main RAG response function  (hybrid → RRF → rerank → generate)
# ---------------------------------------------------------------------------
def get_rag_response(user_message: str, emotion: str = "neutral") -> str:
    """Takes a user message, retrieves relevant context from Qdrant, and generates a response.

    Pipeline:
        1. Semantic (dense) search   → top-10 candidates
        2. BM25 (sparse) search      → top-10 candidates
        3. Reciprocal Rank Fusion     → merge & score
        4. Cross-encoder reranking    → pick best 3
        5. LLM generation            → final answer

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

    RETRIEVAL_TOP_K = 10   # candidates per retriever
    FINAL_TOP_N = 3        # results after reranking

    # --- Step 1: Semantic (dense) search ---
    query_instruction = "Represent this sentence for searching relevant passages:"
    full_query = f"{query_instruction} {user_message}"
    query_vector = embedding_model.encode(full_query).tolist()

    try:
        semantic_results = _semantic_search(query_vector, qdrant_client, top_k=RETRIEVAL_TOP_K)
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        semantic_results = []

    # --- Step 2: BM25 (sparse) search ---
    try:
        bm25_results = _bm25_search(user_message, qdrant_client, top_k=RETRIEVAL_TOP_K)
    except Exception as e:
        print(f"[ERROR] BM25 search failed: {e}")
        bm25_results = []

    # --- Step 3: Reciprocal Rank Fusion ---
    if not semantic_results and not bm25_results:
        return "I am having trouble accessing my knowledge base right now."

    fused_results = _reciprocal_rank_fusion([semantic_results, bm25_results])

    # --- Step 4: Cross-encoder reranking ---
    try:
        top_results = _rerank(user_message, fused_results, top_n=FINAL_TOP_N)
    except Exception as e:
        print(f"[ERROR] Reranking failed, falling back to RRF top-{FINAL_TOP_N}: {e}")
        top_results = fused_results[:FINAL_TOP_N]

    # --- Step 5: Build context & generate LLM response ---
    retrieved_contexts = [
        str(payload.get("answer", ""))[:1500] + "..."
        for _, payload in top_results
    ]
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
    from sentence_transformers import SentenceTransformer, CrossEncoder

    main.models["llm_client"] = OpenAI(
        api_key=settings.model_used_api,
        base_url=settings.model_used_base_url,
    )
    main.models["qdrant_client"] = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    main.models["embedding"] = SentenceTransformer(settings.embedding_model_name)
    main.models["cross_encoder"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    test_query = "I have been feeling really overwhelmed with my studies and stressed out lately."

    print(f"\nUser: {test_query}\n")
    print("Bot: Thinking...\n")

    answer = get_rag_response(test_query)
    print(f"Bot: {answer}\n")
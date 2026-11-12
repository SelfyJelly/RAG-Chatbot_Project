import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
import httpx

load_dotenv()

# Initialize the FastAPI app
app = FastAPI(title="Hugo AI - RAG Chatbot")

# Connect to Supabase and load the embedding model
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SIMILARITY_THRESHOLD = 0.3
TOP_K = 5

SYSTEM_PROMPT = """You are Hugo, the AWS Cloud Club assistant. You ONLY answer questions about the organization using the context provided below.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context does not contain relevant information, say: "I don't have information about that. Please check our website directly or ask a club officer."
3. Never make up information or answer questions outside the scope of the organization.
4. Be friendly, concise, and helpful.
5. When possible, mention which page or section the information comes from.

Context:
{context}
"""

FALLBACK_RESPONSE = "I don't have information about that topic. I can only answer questions about the AWS Cloud Club \u2014 membership, events, projects, positions, and our team. Is there something specific about the club I can help you with?"


# Request and response schemas
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list[dict]
    confidence: float


def retrieve_context(query: str) -> tuple[str, list[dict], float]:
    """Embed query and retrieve relevant documents from Supabase."""
    # Convert the user's question into a 384-dimensional vector
    query_embedding = model.encode(query).tolist()

    # Call the match_documents function we created in Supabase
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_threshold": SIMILARITY_THRESHOLD,
        "match_count": TOP_K,
    }).execute()

    if not result.data:
        return "", [], 0.0

    # Collect matched content and track the highest similarity score
    context_parts = []
    sources = []
    max_similarity = 0.0

    for doc in result.data:
        context_parts.append(doc["content"])
        sources.append(doc["metadata"])
        if doc["similarity"] > max_similarity:
            max_similarity = doc["similarity"]

    context = "\n\n".join(context_parts)
    return context, sources, max_similarity

def generate_response(context: str, user_message: str) -> str:
    """Call Groq API with context-injected prompt."""
    # Build the messages array with system prompt and user question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_message},
    ]

    # Send the request to Groq's OpenAI-compatible endpoint
    with httpx.Client() as client:
        response = client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=30.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Groq API Fehler: {error.response.status_code} - "
                f"{error.response.text}"
            ) from error

    return response.json()["choices"][0]["message"]["content"]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint implementing the RAG pipeline."""
    # Step 1: Retrieve relevant context from Supabase
    context, sources, confidence = retrieve_context(request.message)

    # Step 2: If no relevant content found, return fallback without calling LLM
    if not context or confidence < SIMILARITY_THRESHOLD:
        return ChatResponse(
            response=FALLBACK_RESPONSE,
            sources=[],
            confidence=0.0,
        )

    # Step 3: Generate a grounded response using the retrieved context
    response_text = generate_response(context, request.message)

    return ChatResponse(
        response=response_text,
        sources=sources,
        confidence=round(confidence, 3),
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model": "openai/gpt-oss-120b"}
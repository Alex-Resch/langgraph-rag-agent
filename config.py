import os

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_MODEL = "gemini/gemini-2.5-flash"
AVAILABLE_MODELS = {
    "Gemini 2.5 Flash": "gemini/gemini-2.5-flash",
}
GUARD_MODEL = "gemini/gemini-2.5-flash-lite"
SIMILARITY_THRESHOLD = 0.2
TAVILY_MAX_RESULTS = 5
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
KNOWLEDGE_BASE_DIR = "data/chroma"
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001


def docs_only() -> bool:
    """Answer only from uploaded documents, without web search (DOCS_ONLY=true)."""
    return os.getenv("DOCS_ONLY", "").lower() == "true"

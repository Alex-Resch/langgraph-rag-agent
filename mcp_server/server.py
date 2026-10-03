from functools import cache

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP

from agent.tools import search_vectorstore
from config import EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR, MCP_HOST, MCP_PORT

mcp = FastMCP("knowledge-base", host=MCP_HOST, port=MCP_PORT)


@cache
def get_knowledge_base() -> Chroma:
    """Open the knowledge base built by mcp_server.ingest on the first tool call."""
    # noinspection PyArgumentList
    return Chroma(
        persist_directory=KNOWLEDGE_BASE_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),  # type: ignore
        collection_metadata={"hnsw:space": "cosine"},
    )


@mcp.tool()
def search_documents(query: str) -> str:
    """Search the company knowledge base and return the most relevant passages with file name and page."""
    return search_vectorstore(get_knowledge_base(), query)


@mcp.tool()
def list_documents() -> list[str]:
    """List the files that are in the knowledge base."""
    sources = {m["source"] for m in get_knowledge_base().get()["metadatas"]}
    return sorted(sources)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

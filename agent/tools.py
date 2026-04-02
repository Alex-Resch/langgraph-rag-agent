from typing import cast

from chainlit.element import Element
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chainlit as cl
from tavily import UsageLimitExceededError

from config import CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD, TAVILY_MAX_RESULTS


def get_document_loader(element: Element):
    if element.path is None:
        raise ValueError(f"File path is None for '{element.name}'")
    if element.mime == "application/pdf":
        return PyPDFLoader(element.path)
    elif element.mime == "text/plain":
        return TextLoader(element.path)
    elif element.name.endswith(".md"):
        return UnstructuredMarkdownLoader(element.path)
    raise ValueError(f"Unsupported file type: '{element.name}'")


async def process_document(element: Element) -> int:
    loader = get_document_loader(element)
    pages = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(pages)
    vectorstore = cast(Chroma, cl.user_session.get("vectorstore"))
    vectorstore.add_documents(chunks)
    return len(chunks)


@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for relevant information."""
    vectorstore = cast(Chroma, cl.user_session.get("vectorstore"))

    results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    relevant = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return "NO_DOCUMENTS_FOUND"

    output = []
    for doc in relevant:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        ref = f"[{source}" + (f", page {page + 1}]" if page is not None else "]")
        output.append(f"{ref}\n{doc.page_content}")

    return "Found in documents:\n\n" + "\n\n".join(output)


@tool
def web_search_fallback(query: str) -> str:
    """Search the internet – used when no documents are uploaded or they lack relevant info."""
    try:
        return TavilySearchResults(max_results=TAVILY_MAX_RESULTS).invoke(query)
    except UsageLimitExceededError:
        return "Web search failed: usage limit exceeded."
    except Exception as e:
        return f"Web search failed: {e}"

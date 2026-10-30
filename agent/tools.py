import asyncio

from chainlit.element import Element
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolRuntime
from tavily import UsageLimitExceededError

from agent.agent_state import AgentContext
from agent.guardrails import is_injection
from config import CHUNK_OVERLAP, CHUNK_SIZE, SIMILARITY_THRESHOLD, TAVILY_MAX_RESULTS

NO_DOCUMENTS_FOUND = "NO_DOCUMENTS_FOUND"


def get_document_loader(element: Element):
    if element.path is None:
        raise ValueError(f"File path is None for '{element.name}'")
    if element.mime == "application/pdf":
        return PyPDFLoader(element.path)
    elif element.mime == "text/plain" or element.name.endswith(".md"):
        return TextLoader(element.path)
    raise ValueError(f"Unsupported file type: '{element.name}'")


async def process_document(element: Element, vectorstore: VectorStore) -> str:
    loader = get_document_loader(element)
    pages = loader.load()
    flagged = await asyncio.gather(*(is_injection(page.page_content) for page in pages))
    pages = [page for page, unsafe in zip(pages, flagged, strict=True) if not unsafe]
    for page in pages:
        page.metadata["source"] = element.name
    intro_text = "\n".join([page.page_content for page in pages[:2]])
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(pages)
    vectorstore.add_documents(chunks)
    return intro_text


def search_vectorstore(vectorstore: VectorStore, query: str) -> str:
    """Return the most relevant chunks with file and page, shared by the agent and the MCP server."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    relevant = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

    if not relevant:
        return NO_DOCUMENTS_FOUND

    output = []
    for doc in relevant:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        ref = f"[{source}" + (f", page {page + 1}]" if page is not None else "]")
        output.append(f"{ref}\n{doc.page_content}")

    return "Found in documents:\n\n" + "\n\n".join(output)


@tool
def search_documents(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Search uploaded documents for relevant information."""
    return search_vectorstore(runtime.context.vectorstore, query)


@tool
async def web_search_fallback(query: str) -> str | list[dict]:
    """Search the internet – used when no documents are uploaded or they lack relevant info."""
    try:
        results = await TavilySearchResults(max_results=TAVILY_MAX_RESULTS).ainvoke(
            query
        )
    except UsageLimitExceededError:
        return "Web search failed: usage limit exceeded."
    except Exception as e:  # noqa: BLE001 - a failing web search must not crash the agent
        return f"Web search failed: {e}"
    if isinstance(results, str):
        return results
    flagged = await asyncio.gather(
        *(is_injection(result["content"]) for result in results)
    )
    return [
        result for result, unsafe in zip(results, flagged, strict=True) if not unsafe
    ]

from unittest.mock import patch

import pytest
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

from mcp_server.ingest import load_folder
from mcp_server.server import mcp

LOAN_TEXT = "The loan was provided by Sparkasse Nordheide at 3.1 percent interest."


@pytest.fixture
def knowledge_base():
    """In-memory knowledge base with fake embeddings, so no model is downloaded."""
    store = Chroma(
        embedding_function=DeterministicFakeEmbedding(size=64),
        collection_metadata={"hnsw:space": "cosine"},
    )
    yield store
    store.delete_collection()


@pytest.fixture
def server_with(knowledge_base):
    """Point the MCP server at the test knowledge base."""
    with patch("mcp_server.server.get_knowledge_base", return_value=knowledge_base):
        yield knowledge_base


async def call(tool: str, arguments: dict) -> str:
    """Call a tool through a real MCP client session and return its text output."""
    async with create_connected_server_and_client_session(
        mcp, raise_exceptions=True
    ) as client:
        result = await client.call_tool(tool, arguments)
    assert not result.isError
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


async def test_server_offers_search_and_list_tools():
    """Clients should see exactly the two knowledge base tools."""
    async with create_connected_server_and_client_session(mcp) as client:
        tools = await client.list_tools()
    assert {tool.name for tool in tools.tools} == {"search_documents", "list_documents"}


async def test_search_returns_passage_with_file_and_page(server_with):
    """A matching passage should come back with its file name and page number."""
    server_with.add_texts([LOAN_TEXT], metadatas=[{"source": "report.pdf", "page": 2}])

    output = await call("search_documents", {"query": LOAN_TEXT})

    assert "[report.pdf, page 3]" in output
    assert "Sparkasse Nordheide" in output


async def test_search_in_empty_knowledge_base_finds_nothing(server_with):
    """An empty knowledge base should return the usual "nothing found" marker."""
    output = await call("search_documents", {"query": "Who provided the loan?"})

    assert output == "NO_DOCUMENTS_FOUND"


async def test_list_documents_returns_each_file_once(server_with):
    """Files with several chunks should be listed only once, sorted by name."""
    server_with.add_texts(
        ["chunk one", "chunk two", "other file"],
        metadatas=[{"source": "b.pdf"}, {"source": "b.pdf"}, {"source": "a.txt"}],
    )

    output = await call("list_documents", {})

    assert output.splitlines() == ["a.txt", "b.pdf"]


def test_load_folder_reads_supported_files_with_file_name(tmp_path):
    """Text and Markdown files are loaded with their file name, other files are skipped."""
    (tmp_path / "notes.txt").write_text(LOAN_TEXT)
    (tmp_path / "guide.md").write_text("# Guide\n\nClean the panels twice a year.")
    (tmp_path / "image.png").write_bytes(b"not a document")

    pages = load_folder(str(tmp_path))

    assert sorted(p.metadata["source"] for p in pages) == ["guide.md", "notes.txt"]

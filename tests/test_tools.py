from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from chainlit.element import Element
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from tavily import UsageLimitExceededError

from agent.agent_state import AgentContext


def make_runtime(vectorstore):
    """Build a minimal ToolRuntime stand-in that exposes the given vectorstore."""
    runtime = MagicMock()
    runtime.context = AgentContext(vectorstore=vectorstore)
    return runtime


def run_search(query, vectorstore):
    """Call the search_documents tool function directly with an injected runtime."""
    from agent.tools import search_documents

    tool = cast(StructuredTool, search_documents)
    assert tool.func is not None
    return tool.func(query, runtime=make_runtime(vectorstore))


class MockElement(Element):
    display = "inline"  # required abstract attribute

    def __init__(self, mime="application/pdf", name="test.pdf", path="/tmp/test.pdf"):
        self.mime = mime
        self.name = name
        self.path = path


def test_get_document_loader_pdf():
    """PDF files should use PyPDFLoader."""
    from agent.tools import get_document_loader

    element = MockElement(mime="application/pdf", name="paper.pdf")
    with patch("agent.tools.PyPDFLoader") as MockLoader:
        get_document_loader(cast(Element, element))
        MockLoader.assert_called_once_with(element.path)


def test_get_document_loader_txt():
    """Plain text files should use TextLoader."""
    from agent.tools import get_document_loader

    element = MockElement(mime="text/plain", name="notes.txt")
    with patch("agent.tools.TextLoader") as MockLoader:
        get_document_loader(element)
        MockLoader.assert_called_once_with(element.path)


def test_get_document_loader_md():
    """Markdown files (non text/plain MIME) should use UnstructuredMarkdownLoader."""
    from agent.tools import get_document_loader

    element = MockElement(mime="text/markdown", name="readme.md")
    with patch("agent.tools.UnstructuredMarkdownLoader") as MockLoader:
        get_document_loader(element)
        MockLoader.assert_called_once_with(element.path)


def test_get_document_loader_unsupported_raises():
    """Unsupported file types should raise a ValueError with the filename."""
    from agent.tools import get_document_loader

    element = MockElement(mime="image/png", name="photo.png")
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_document_loader(element)


@pytest.mark.asyncio
async def test_process_document_returns_chunk_count():
    """process_document should return the number of chunks added to the vectorstore."""
    from agent.tools import process_document

    element = MockElement()

    mock_doc = Document(
        page_content="some text", metadata={"source": "test.pdf", "page": 0}
    )
    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc, mock_doc, mock_doc]

    mock_vectorstore = MagicMock()

    with patch("agent.tools.get_document_loader", return_value=mock_loader):
        result = await process_document(element, mock_vectorstore)

    assert isinstance(result, str) and len(result) > 0
    mock_vectorstore.add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_uses_uploaded_filename_as_source():
    """Each stored page should cite the uploaded name, not Chainlit's temp path."""
    from agent.tools import process_document

    element = MockElement(name="report.pdf", path="/tmp/upload-123.pdf")
    pages = [
        Document(
            page_content="first page", metadata={"source": element.path, "page": 0}
        ),
        Document(
            page_content="second page", metadata={"source": element.path, "page": 1}
        ),
    ]
    mock_loader = MagicMock()
    mock_loader.load.return_value = pages
    mock_vectorstore = MagicMock()

    with patch("agent.tools.get_document_loader", return_value=mock_loader):
        await process_document(element, mock_vectorstore)

    added_chunks = mock_vectorstore.add_documents.call_args.args[0]
    assert len(added_chunks) == 2
    assert all(chunk.metadata["source"] == element.name for chunk in added_chunks)
    assert all(chunk.metadata["source"] != element.path for chunk in added_chunks)
    assert [chunk.metadata["page"] for chunk in added_chunks] == [0, 1]


@pytest.mark.asyncio
async def test_process_document_adds_chunks_to_vectorstore():
    """Documents larger than CHUNK_SIZE should be split into multiple chunks."""
    from agent.tools import process_document

    element = MockElement()

    mock_doc = Document(page_content="x" * 600, metadata={})  # > CHUNK_SIZE → splits
    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]

    mock_vectorstore = MagicMock()

    with patch("agent.tools.get_document_loader", return_value=mock_loader):
        await process_document(element, mock_vectorstore)

    added_chunks = mock_vectorstore.add_documents.call_args[0][0]
    assert len(added_chunks) > 1


def test_search_documents_returns_formatted_results():
    """Results above the similarity threshold should be returned with source and page."""

    mock_doc = Document(
        page_content="Attention is all you need.",
        metadata={"source": "transformer.pdf", "page": 2},
    )
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (mock_doc, 0.8)  # higher score = more similar → relevant
    ]

    result = run_search("What is attention?", mock_vectorstore)

    assert "Found in documents" in result
    assert "transformer.pdf" in result
    assert "page 3" in result  # page 2 + 1 (0-indexed → 1-indexed)
    assert "Attention is all you need." in result


def test_search_documents_returns_no_documents_found():
    """Results below the similarity threshold should trigger the NO_DOCUMENTS_FOUND sentinel."""

    mock_doc = Document(page_content="irrelevant content", metadata={})
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (mock_doc, 0.1)  # below SIMILARITY_THRESHOLD → not relevant
    ]

    result = run_search("quantum physics", mock_vectorstore)

    assert result == "NO_DOCUMENTS_FOUND"


def test_search_documents_omits_page_if_not_in_metadata():
    """Source reference should not include a page number if the metadata has none."""

    mock_doc = Document(page_content="content", metadata={"source": "notes.txt"})
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (mock_doc, 0.9)  # above SIMILARITY_THRESHOLD → relevant
    ]

    result = run_search("something", mock_vectorstore)

    assert "[notes.txt]" in result
    assert "page" not in result


def test_search_documents_multiple_results():
    """All relevant chunks should appear in the output."""

    docs = [
        (
            Document(
                page_content=f"content {i}", metadata={"source": "doc.pdf", "page": i}
            ),
            0.9,
        )
        for i in range(3)
    ]
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = docs

    result = run_search("query", mock_vectorstore)

    assert result.count("doc.pdf") == 3


def test_web_search_fallback_returns_results():
    """A successful Tavily search should return its results."""
    from agent.tools import web_search_fallback

    with patch("agent.tools.TavilySearchResults") as MockTavily:
        MockTavily.return_value.invoke.return_value = [
            {"url": "https://example.com", "content": "LLMs are transformers"}
        ]
        result = web_search_fallback.invoke("what is an LLM")

    assert result is not None


def test_web_search_fallback_handles_usage_limit():
    """A UsageLimitExceededError should return a human-readable error string."""
    from agent.tools import web_search_fallback

    with patch("agent.tools.TavilySearchResults") as MockTavily:
        MockTavily.return_value.invoke.side_effect = UsageLimitExceededError("limit")
        result = web_search_fallback.invoke("query")

    assert "usage limit exceeded" in result


def test_web_search_fallback_handles_generic_exception():
    """Any unexpected exception should be caught and returned as an error string."""
    from agent.tools import web_search_fallback

    with patch("agent.tools.TavilySearchResults") as MockTavily:
        MockTavily.return_value.invoke.side_effect = Exception("connection timeout")
        result = web_search_fallback.invoke("query")

    assert "Web search failed" in result
    assert "connection timeout" in result


def test_get_document_loader_raises_if_path_is_none():
    """get_document_loader should raise a ValueError that includes the
    filename when the element has no path set."""
    from agent.tools import get_document_loader

    element = MockElement(mime="application/pdf", name="paper.pdf")
    element.path = None

    with pytest.raises(ValueError, match="paper.pdf"):
        get_document_loader(element)

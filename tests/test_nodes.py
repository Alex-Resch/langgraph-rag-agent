import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.agent_state import AgentState


def make_state(
    content="test query", model="groq/llama-3.3-70b-versatile", prepend=None
):
    """Build a minimal AgentState dict for use in node tests.

    Args:
        content: The text content of the last HumanMessage.
        model: LiteLLM model string stored in state.
        prepend: Optional list of messages to insert before the HumanMessage.
    """
    messages = [HumanMessage(content=content)]
    if prepend:
        messages = prepend + messages
    return AgentState(messages=messages, model=model)


def make_mock_step():
    """Return an async context manager mock that stands in for cl.Step."""
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=None)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.mark.asyncio
async def test_call_llm_returns_ai_message():
    """call_llm should return a state dict whose messages list contains
    exactly one AIMessage with the content produced by the LLM."""
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here is the answer."))

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        result = await call_llm(make_state())

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Here is the answer."


@pytest.mark.asyncio
async def test_call_llm_prepends_system_message():
    """call_llm should always place a SystemMessage at index 0 of the
    message list passed to the LLM, regardless of conversation history."""
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state())

    call_args = mock_llm.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage), (
        "First message must be a SystemMessage"
    )


@pytest.mark.asyncio
async def test_call_llm_uses_model_from_state():
    """call_llm should instantiate ChatLiteLLM with the model string
    stored in state, not a hard-coded default."""
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm) as MockChatLiteLLM:
        await call_llm(make_state(model="gemini/gemini-2.5-flash"))

    MockChatLiteLLM.assert_called_once_with(
        model="gemini/gemini-2.5-flash", streaming=True
    )


@pytest.mark.asyncio
async def test_call_llm_passes_full_history():
    """call_llm should forward the entire conversation history to the LLM.

    With a system prompt prepended to 3 history messages plus 1 current
    message, the LLM should receive at least 5 messages in total.
    """
    from agent.nodes import call_llm

    history = [
        SystemMessage(content="doc context"),
        HumanMessage(content="first question"),
        AIMessage(content="first answer"),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="second answer"))

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state(content="follow-up question", prepend=history))

    call_args = mock_llm.ainvoke.call_args[0][0]
    assert len(call_args) >= 5


# ---------------------------------------------------------------------------
# search_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_pipeline_returns_doc_result():
    """search_pipeline should include the document search result in the
    returned messages when search_documents finds relevant content."""
    from agent.nodes import search_pipeline

    with (
        patch("agent.nodes.search_documents") as mock_search,
        patch("agent.nodes.cl.Step", return_value=make_mock_step()),
    ):
        mock_search.invoke.return_value = (
            "Found in documents:\n\n[paper.pdf, page 1]\ncontent"
        )
        result = await search_pipeline(make_state("What is attention?"))

    messages = result["messages"]
    assert any("Found in documents" in m.content for m in messages)


@pytest.mark.asyncio
async def test_search_pipeline_falls_back_to_web_when_no_docs():
    """search_pipeline should call web_search_fallback and return its result
    when search_documents signals that no documents were found."""
    from agent.nodes import search_pipeline

    with (
        patch("agent.nodes.search_documents") as mock_search,
        patch("agent.nodes.web_search_fallback") as mock_web,
        patch("agent.nodes.cl.Step", return_value=make_mock_step()),
    ):
        mock_search.invoke.return_value = "NO_DOCUMENTS_FOUND"
        mock_web.invoke.return_value = "Python is a programming language"
        result = await search_pipeline(make_state("What is Python?"))

    messages = result["messages"]
    assert any("Python is a programming language" in m.content for m in messages)
    mock_web.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_search_pipeline_skips_web_if_docs_found():
    """search_pipeline should not call web_search_fallback when
    search_documents already returns relevant document content."""
    from agent.nodes import search_pipeline

    with (
        patch("agent.nodes.search_documents") as mock_search,
        patch("agent.nodes.web_search_fallback") as mock_web,
        patch("agent.nodes.cl.Step", return_value=make_mock_step()),
    ):
        mock_search.invoke.return_value = "Found in documents:\n\n[doc.pdf]\ncontent"
        await search_pipeline(make_state("query"))

    mock_web.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_search_pipeline_uses_last_human_message_as_query():
    """search_pipeline should extract the most recent HumanMessage and pass
    its content verbatim as the query to search_documents."""
    from agent.nodes import search_pipeline

    with (
        patch("agent.nodes.search_documents") as mock_search,
        patch("agent.nodes.cl.Step", return_value=make_mock_step()),
    ):
        mock_search.invoke.return_value = "Found in documents:\n\ncontent"
        await search_pipeline(make_state(content="specific query text"))

    mock_search.invoke.assert_called_once_with("specific query text")

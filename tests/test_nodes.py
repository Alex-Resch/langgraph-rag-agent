from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.agent_state import AgentState


def make_state(
    content="test query", model="groq/llama-3.3-70b-versatile", prepend=None
):
    messages = [HumanMessage(content=content)]
    if prepend:
        messages = prepend + messages
    return AgentState(messages=messages, model=model)


def make_ai_message_with_tool_calls():
    msg = AIMessage(content="")
    msg.tool_calls = [
        {"name": "search_documents", "args": {"query": "test"}, "id": "1"}
    ]
    return msg


def make_ai_message_without_tool_calls():
    msg = AIMessage(content="Here is the answer.")
    msg.tool_calls = []
    return msg


@pytest.mark.asyncio
async def test_call_llm_returns_ai_message():
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(
        return_value=make_ai_message_without_tool_calls()
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        result = await call_llm(make_state())

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Here is the answer."


@pytest.mark.asyncio
async def test_call_llm_prepends_system_message():
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(
        return_value=make_ai_message_without_tool_calls()
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state())

    call_args = mock_llm_with_tools.ainvoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)


@pytest.mark.asyncio
async def test_call_llm_uses_model_from_state():
    from agent.nodes import call_llm

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(
        return_value=make_ai_message_without_tool_calls()
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm) as MockChatLiteLLM:
        await call_llm(make_state(model="gemini/gemini-2.5-flash"))

    MockChatLiteLLM.assert_called_once_with(
        model="gemini/gemini-2.5-flash",
        streaming=True,
        temperature=0.0,
        stream_options={"include_usage": True},
    )


@pytest.mark.asyncio
async def test_call_llm_passes_full_history():
    from agent.nodes import call_llm

    history = [
        SystemMessage(content="doc context"),
        HumanMessage(content="first question"),
        AIMessage(content="first answer"),
    ]
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(
        return_value=make_ai_message_without_tool_calls()
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state(content="follow-up", prepend=history))

    call_args = mock_llm_with_tools.ainvoke.call_args[0][0]
    assert len(call_args) >= 5


def test_should_continue_returns_tools_when_tool_calls_present():
    from agent.nodes import should_continue

    state = make_state()
    state["messages"] = [make_ai_message_with_tool_calls()]
    assert should_continue(state) == "tools"


def test_should_continue_returns_end_when_no_tool_calls():
    from agent.nodes import should_continue

    state = make_state()
    state["messages"] = [make_ai_message_without_tool_calls()]
    assert should_continue(state) == "end"


def test_should_continue_returns_end_for_plain_ai_message():
    from agent.nodes import should_continue

    state = make_state()
    state["messages"] = [AIMessage(content="Direct answer, no tools needed.")]
    assert should_continue(state) == "end"


def make_mock_llm():
    """Return a ChatLiteLLM mock whose tool-bound model answers without tool calls."""
    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(
        return_value=make_ai_message_without_tool_calls()
    )
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    return mock_llm


@pytest.mark.asyncio
async def test_call_llm_binds_web_search_by_default(monkeypatch):
    """Without DOCS_ONLY the model can fall back to the web search."""
    from agent.nodes import call_llm

    monkeypatch.delenv("DOCS_ONLY", raising=False)
    mock_llm = make_mock_llm()

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state())

    bound = mock_llm.bind_tools.call_args[0][0]
    assert [t.name for t in bound] == ["search_documents", "web_search_fallback"]


@pytest.mark.asyncio
async def test_call_llm_binds_only_document_search_in_docs_only_mode(monkeypatch):
    """With DOCS_ONLY=true the model must not get the web search tool."""
    from agent.nodes import call_llm

    monkeypatch.setenv("DOCS_ONLY", "true")
    mock_llm = make_mock_llm()

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state())

    bound = mock_llm.bind_tools.call_args[0][0]
    assert [t.name for t in bound] == ["search_documents"]


@pytest.mark.asyncio
async def test_call_llm_docs_only_prompt_does_not_offer_web_search(monkeypatch):
    """The docs-only prompt must not tell the model to use a tool it does not have."""
    from agent.nodes import call_llm

    monkeypatch.setenv("DOCS_ONLY", "true")
    mock_llm = make_mock_llm()

    with patch("agent.nodes.ChatLiteLLM", return_value=mock_llm):
        await call_llm(make_state())

    prompt = mock_llm.bind_tools.return_value.ainvoke.call_args[0][0][0]
    assert "web_search_fallback" not in prompt.content

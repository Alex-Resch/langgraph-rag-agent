import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
        model="gemini/gemini-2.5-flash", streaming=True
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

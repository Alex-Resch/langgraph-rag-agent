import pytest
from agent.graph import build_graph


def test_build_graph_compiles_without_error():
    """build_graph should return a compiled graph without raising any
    exceptions during construction or compilation."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_search_node():
    """The compiled graph must expose a node named 'tools', which
    corresponds to the search_pipeline node."""
    graph = build_graph()
    assert "tools" in graph.nodes


def test_graph_has_call_llm_node():
    """The compiled graph must expose a node named 'call_llm', which
    corresponds to the LLM generation node."""
    graph = build_graph()
    assert "call_llm" in graph.nodes


def test_build_graph_returns_new_instance_each_time():
    """build_graph should return a distinct object on every call to
    prevent shared mutable state leaking between graph executions."""
    g1 = build_graph()
    g2 = build_graph()
    assert g1 is not g2


@pytest.mark.asyncio
async def test_graph_runs_both_nodes():
    """Integration check: a single ainvoke call should trigger both the
    search node and the call_llm node, and the final state should contain
    the AIMessage produced by the LLM."""
    from unittest.mock import patch, AsyncMock, MagicMock
    from langchain_core.messages import HumanMessage, AIMessage

    mock_step = MagicMock()
    mock_step.__aenter__ = AsyncMock(return_value=None)
    mock_step.__aexit__ = AsyncMock(return_value=None)

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="final answer"))

    with patch("agent.nodes.ChatLiteLLM") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.ainvoke = AsyncMock(
            return_value=AIMessage(content="final answer")
        )
        mock_llm_instance.bind_tools.return_value = mock_llm_with_tools
        MockLLM.return_value = mock_llm_instance

        graph = build_graph()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hello")],
                "model": "groq/llama-3.3-70b-versatile",
            }
        )

    assert any(
        isinstance(m, AIMessage) and m.content == "final answer"
        for m in result["messages"]
    )

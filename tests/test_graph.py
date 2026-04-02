import pytest
from agent.graph import build_graph


def test_build_graph_compiles_without_error():
    """build_graph should return a compiled graph without raising any
    exceptions during construction or compilation."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_search_node():
    """The compiled graph must expose a node named 'search', which
    corresponds to the search_pipeline node."""
    graph = build_graph()
    assert "search" in graph.nodes


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

    with (
        patch("agent.nodes.search_documents") as mock_search,
        patch("agent.nodes.web_search_fallback") as mock_web,
        patch("agent.nodes.ChatLiteLLM", return_value=mock_llm),
        patch("agent.nodes.cl.Step", return_value=mock_step),
    ):
        mock_search.invoke.return_value = "NO_DOCUMENTS_FOUND"
        mock_web.invoke.return_value = "web result"

        graph = build_graph()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hello")],
                "model": "groq/llama-3.3-70b-versatile",
            }
        )

    mock_search.invoke.assert_called_once()
    mock_llm.ainvoke.assert_called_once()
    assert any(
        isinstance(m, AIMessage) and m.content == "final answer"
        for m in result["messages"]
    )

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent.agent_state import AgentState
from agent.nodes import search_pipeline, call_llm


def build_graph():
    graph = StateGraph(AgentState)  # type: ignore
    graph.add_node("search", search_pipeline)  # type: ignore
    graph.add_node("call_llm", call_llm)  # type: ignore

    graph.add_edge(START, "search")
    graph.add_edge("search", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()

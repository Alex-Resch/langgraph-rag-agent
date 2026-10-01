from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.agent_state import AgentContext, AgentState
from agent.nodes import call_llm, should_continue, tools


def build_graph():
    graph = StateGraph(AgentState, context_schema=AgentContext)

    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "call_llm")  # nach Tool-Aufruf wieder ans LLM

    return graph.compile()

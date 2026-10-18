from langchain_litellm import ChatLiteLLM

from agent.agent_state import AgentState
from agent.prompts import system_prompt
from agent.tools import search_documents, web_search_fallback
from config import docs_only


def active_tools() -> list:
    """Only the document search in docs-only mode, otherwise also the web search."""
    if docs_only():
        return [search_documents]
    return [search_documents, web_search_fallback]


async def call_llm(state: AgentState):
    llm = ChatLiteLLM(
        model=state["model"],
        streaming=True,
        temperature=0.0,
        stream_options={"include_usage": True},
    )
    llm_with_tools = llm.bind_tools(active_tools())
    response = await llm_with_tools.ainvoke(
        [system_prompt(docs_only())] + state["messages"]
    )
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

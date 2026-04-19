from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_litellm import ChatLiteLLM

from agent.agent_state import AgentState
from agent.tools import search_documents, web_search_fallback

tools = [search_documents, web_search_fallback]


async def call_llm(state: AgentState):
    system = SystemMessage(
        content=(
            f"You are a helpful assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. "
            "Choose tools based on the QUESTION, not on what files are uploaded:\n"
            "- search_documents: ONLY if the question is explicitly about an uploaded document\n"
            "- web_search_fallback: for current events, weather, or general internet questions\n"
            "- No tool: for greetings, math, general knowledge you already know\n"
            "Do NOT use search_documents just because documents were uploaded."
        )
    )
    llm = ChatLiteLLM(model=state["model"], streaming=True, temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke([system] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

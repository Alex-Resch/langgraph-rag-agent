from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import SystemMessage

from agent.agent_state import AgentState
from agent.tools import search_documents, web_search_fallback


async def call_llm(state: AgentState):
    system = SystemMessage(
        content= ( # type: ignore
            "You are a helpful assistant. "
            "Answer based ONLY on the provided context. "
            "If the context is irrelevant or missing, say so clearly instead of guessing. "
            "If the user is just talking to you, answer normal back. "
        )
    )
    llm = ChatLiteLLM(model=state["model"], streaming=True)

    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}

def search_pipeline(state: AgentState) -> dict:
    query = state["messages"][-1].content

    doc_result = search_documents.invoke(query)
    if "NO_DOCUMENTS_FOUND" not in doc_result:
        return {"messages": [SystemMessage(content=doc_result)]}

    web_result = web_search_fallback.invoke(query)
    return {"messages": [SystemMessage(content=str(web_result))]}
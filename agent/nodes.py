from langchain_community.chat_models import ChatLiteLLM

from agent.agent_state import AgentState
from agent.tools import search_documents, web_search_fallback


async def call_llm(state: AgentState):
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer based ONLY on the provided context. "
            "If the context is irrelevant or missing, say so clearly instead of guessing. "
            "If the user is just talking to you, answer normal back. "
        )
    }
    llm = ChatLiteLLM(model=state["model"], streaming=True)

    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}

def search_pipeline(state: AgentState) -> dict:
    query = state["messages"][-1].content

    doc_result = search_documents.invoke(query)
    if "NO_DOCUMENTS_FOUND" not in doc_result:
        return {"messages": [{"role": "system", "content": doc_result}]}

    web_result = web_search_fallback.invoke(query)
    return {"messages": [{"role": "system", "content": str(web_result)}]}



def extract_text(content) -> str:
    """Normalize LLM content to plain string (handles str and list[dict])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
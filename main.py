from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model: str


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


async def process_document(file_path: str) -> int:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)
    cl.user_session.get("vectorstore").add_documents(chunks)
    return len(chunks)


@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for relevant information."""
    vectorstore = cl.user_session.get("vectorstore")

    results = vectorstore.similarity_search_with_score(query, k=5)
    relevant = [doc for doc, score in results if score < 0.7]

    if not relevant:
        return "NO_DOCUMENTS_FOUND"

    return "Found in documents:\n\n" + "\n\n".join(d.page_content for d in relevant)


@tool
def web_search_fallback(query: str) -> str:
    """Search the internet – used when no documents are uploaded or they lack relevant info."""
    return TavilySearchResults(max_results=10).invoke(query)


tools = [search_documents, web_search_fallback]
tool_node = ToolNode(tools)


async def call_llm(state: AgentState):
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Answer based ONLY on the provided context. "
            "If the context is irrelevant or missing, say so clearly instead of guessing."
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

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search", search_pipeline)
    graph.add_node("call_llm", call_llm)

    graph.add_edge(START, "search")
    graph.add_edge("search", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


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


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("vectorstore", Chroma(embedding_function=embeddings))
    cl.user_session.set("history", [])

    await cl.ChatSettings([
        cl.input_widget.Select(
            id="model",
            label="Select model",
            values=["gemini/gemini-2.5-flash", "groq/llama-3.3-70b-versatile"],
            initial_value="gemini/gemini-2.5-flash",
        )
    ]).send()

    cl.user_session.set("model", "gemini/gemini-2.5-flash")
    await cl.Message(
        content="Hello! Upload PDFs and I'll answer questions about them. If I find nothing, I'll search the web."
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("model", settings["model"])


@cl.on_message
async def on_message(message: cl.Message):
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                await process_document(element.path)

                history = cl.user_session.get("history", [])
                history.append({
                    "role": "system",
                    "content": f"The user just uploaded a PDF: '{element.name}'. It has been stored. Use search_documents for any questions about it."
                })
                cl.user_session.set("history", history)
            else:
                await cl.Message(content=f"❌ Only PDFs are supported: '{element.name}'.").send()

    model = cl.user_session.get("model", "gemini/gemini-2.5-flash")
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    answer = cl.Message(content="")

    async for event in build_graph().astream_events(
            {"messages": history, "model": model},
            version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            text = extract_text(chunk.content)
            if text:
                await answer.stream_token(text)

    await answer.send()
    history.append({"role": "assistant", "content": answer.content})
    cl.user_session.set("history", history)
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


# Only embeddings are global – ChromaDB is created per user session
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


async def process_document(file_path: str):
    """Load PDF → split into chunks → store as vectors in ChromaDB."""

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # Store vectors in session ChromaDB (no persist_directory!)
    vectorstore = cl.user_session.get("vectorstore")
    vectorstore.add_documents(chunks)

    return len(chunks)


@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for relevant information."""

    vectorstore = cl.user_session.get("vectorstore")
    results = vectorstore.similarity_search(query, k=5)

    if not results:
        return "NO_DOCUMENTS_FOUND"

    context = "\n\n".join([doc.page_content for doc in results])
    return f"Found in documents:\n{context}"


@tool
def web_search_fallback(query: str) -> str:
    """Search the internet – used when no documents have been uploaded
    or the documents do not contain relevant information."""
    tavily = TavilySearchResults(max_results=3)
    return tavily.invoke(query)


tools = [search_documents, web_search_fallback]

def call_llm(state: AgentState):
    """Send messages to LLM – LLM decides whether to use a tool."""
    llm = ChatLiteLLM(model=state["model"])
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", tools_condition)
    graph.add_edge("tools", "call_llm")

    return graph.compile()

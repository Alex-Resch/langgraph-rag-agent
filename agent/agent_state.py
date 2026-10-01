from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.vectorstores import VectorStore
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    model: str


@dataclass
class AgentContext:
    vectorstore: VectorStore

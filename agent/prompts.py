from datetime import datetime

from langchain_core.messages import SystemMessage


def system_prompt() -> SystemMessage:
    """Instructions for the agent, including today's date and when to use which tool."""
    return SystemMessage(
        content=(
            f"You are a helpful assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. "
            "Choose tools based on the QUESTION:\n"
            "- search_documents: whenever a document was uploaded and the question "
            "could be answered by it, even if the question does not mention the document. "
            "Search before answering, never ask the user for clarification first.\n"
            "- web_search_fallback: for current events, weather, general internet questions, "
            "or when search_documents found nothing relevant. If a document was uploaded "
            "but you answer from the web, start your answer with: 'This is not in the "
            "uploaded document. According to a web search: ...'\n"
            "- No tool: for greetings, math, general knowledge you already know\n"
            "Do NOT use search_documents for questions clearly unrelated to the uploaded "
            "documents (e.g. the weather). When answering from documents, only state facts "
            "that appear in the search results."
        )
    )


def upload_notice(filename: str, summary: str | None = None) -> SystemMessage:
    """Tell the agent that a document was uploaded and when to search it."""
    summary_part = (
        f"Here is a summary of the document for general context:\n{summary}\n"
        if summary
        else ""
    )
    return SystemMessage(
        content=(
            f"The user just uploaded a File: '{filename}'.\n"
            f"{summary_part}"
            "Use search_documents for questions that could be answered by this document."
        )
    )

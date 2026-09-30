from datetime import datetime

from langchain_core.messages import SystemMessage


def system_prompt() -> SystemMessage:
    """Instructions for the agent, including today's date and when to use which tool."""
    return SystemMessage(
        content=(
            f"You are a helpful assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. "
            "Choose tools based on the QUESTION, not on what files are uploaded:\n"
            "- search_documents: ONLY if the question is explicitly about an uploaded document\n"
            "- web_search_fallback: for current events, weather, or general internet questions\n"
            "- No tool: for greetings, math, general knowledge you already know\n"
            "Do NOT use search_documents just because documents were uploaded."
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
            "Only use search_documents if the user asks something about this document."
        )
    )

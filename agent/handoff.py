import asyncio
import json
import logging
import os
import smtplib
from email.message import EmailMessage
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage, HumanMessage

from agent.tools import NO_DOCUMENTS_FOUND

logger = logging.getLogger(__name__)


def nothing_found(search_results: list[str]) -> bool:
    """True if the documents were searched and no search found anything."""
    return bool(search_results) and all(
        result == NO_DOCUMENTS_FOUND for result in search_results
    )


def format_chat(history: list) -> str:
    """Turn the chat history into plain text, without system messages."""
    lines = []
    for message in history:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


def send_to_slack(webhook_url: str, text: str) -> None:
    """Post the text to a Slack incoming webhook."""
    request = Request(
        webhook_url,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10):
        pass


def send_email(to: str, text: str) -> None:
    """Send the text by email via the SMTP server from the environment."""
    user = os.getenv("SMTP_USER", "")
    email = EmailMessage()
    email["Subject"] = "Unanswered question from the chatbot"
    email["From"] = user
    email["To"] = to
    email.set_content(text)
    with smtplib.SMTP(
        os.environ["SMTP_HOST"], int(os.getenv("SMTP_PORT", "587")), timeout=10
    ) as server:
        server.starttls()
        if user:
            server.login(user, os.getenv("SMTP_PASSWORD", ""))
        server.send_message(email)


async def forward_to_team(question: str, history: list) -> bool:
    """Send the unanswered question and the chat to Slack and/or email.
    Returns True if at least one channel received it."""
    text = f"Unanswered question: {question}\n\nChat so far:\n{format_chat(history)}"
    channels = []
    if webhook_url := os.getenv("HANDOFF_SLACK_WEBHOOK_URL"):
        channels.append(("Slack", send_to_slack, webhook_url))
    if (to := os.getenv("HANDOFF_EMAIL_TO")) and os.getenv("SMTP_HOST"):
        channels.append(("email", send_email, to))

    sent = False
    for name, send, target in channels:
        try:
            await asyncio.to_thread(send, target, text)
            sent = True
        except Exception:  # a failing channel must not crash the chat
            logger.exception("Handoff via %s failed", name)
    return sent

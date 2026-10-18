import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.handoff import forward_to_team, nothing_found

HISTORY = [
    SystemMessage(content="The user just uploaded a File: 'faq.pdf'."),
    HumanMessage(content="Do you ship to Austria?"),
    AIMessage(content="This is not in the documents."),
]


@pytest.fixture(autouse=True)
def no_handoff_channels(monkeypatch):
    """Start every test without any configured handoff channel."""
    for name in (
        "HANDOFF_SLACK_WEBHOOK_URL",
        "HANDOFF_EMAIL_TO",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_nothing_found_when_every_search_found_nothing():
    assert nothing_found(["NO_DOCUMENTS_FOUND", "NO_DOCUMENTS_FOUND"])


def test_not_nothing_found_when_one_search_found_documents():
    assert not nothing_found(["NO_DOCUMENTS_FOUND", "Found in documents:\n\n[faq.pdf]"])


def test_not_nothing_found_without_any_search():
    """Greetings never trigger a search, so they must not count as unanswered."""
    assert not nothing_found([])


@pytest.mark.asyncio
async def test_forward_posts_question_and_chat_to_slack(monkeypatch):
    monkeypatch.setenv("HANDOFF_SLACK_WEBHOOK_URL", "https://hooks.slack.test/abc")

    with patch("agent.handoff.urlopen") as mock_urlopen:
        sent = await forward_to_team("Do you ship to Austria?", HISTORY)

    request = mock_urlopen.call_args[0][0]
    text = json.loads(request.data)["text"]
    assert sent
    assert request.full_url == "https://hooks.slack.test/abc"
    assert "Do you ship to Austria?" in text
    assert "Assistant: This is not in the documents." in text
    assert "uploaded a File" not in text


@pytest.mark.asyncio
async def test_forward_sends_email_with_question(monkeypatch):
    monkeypatch.setenv("HANDOFF_EMAIL_TO", "support@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    with patch("agent.handoff.smtplib.SMTP") as mock_smtp:
        sent = await forward_to_team("Do you ship to Austria?", HISTORY)

    server = mock_smtp.return_value.__enter__.return_value
    email = server.send_message.call_args[0][0]
    assert sent
    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=10)
    server.login.assert_called_once_with("bot@example.com", "secret")
    assert email["To"] == "support@example.com"
    assert "Do you ship to Austria?" in email.get_content()


@pytest.mark.asyncio
async def test_forward_without_channels_sends_nothing():
    with (
        patch("agent.handoff.urlopen") as mock_urlopen,
        patch("agent.handoff.smtplib.SMTP") as mock_smtp,
    ):
        sent = await forward_to_team("Do you ship to Austria?", HISTORY)

    assert not sent
    mock_urlopen.assert_not_called()
    mock_smtp.assert_not_called()


@pytest.mark.asyncio
async def test_forward_reports_failure_when_slack_is_unreachable(monkeypatch):
    """A broken webhook must not crash the chat, and the user must not be told
    that the team was informed."""
    monkeypatch.setenv("HANDOFF_SLACK_WEBHOOK_URL", "https://hooks.slack.test/abc")

    with patch("agent.handoff.urlopen", MagicMock(side_effect=URLError("down"))):
        sent = await forward_to_team("Do you ship to Austria?", HISTORY)

    assert not sent

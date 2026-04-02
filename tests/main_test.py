import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from litellm.exceptions import RateLimitError, BadRequestError, ServiceUnavailableError

from main import on_message


def make_mock_message(content="test message", elements=None):
    """Build a minimal Chainlit Message mock for use in on_message tests."""
    msg = MagicMock()
    msg.content = content
    msg.elements = elements or []
    return msg


def make_mock_element(name="paper.pdf"):
    """Build a minimal Chainlit Element mock representing an uploaded file."""
    element = MagicMock()
    element.name = name
    return element


def make_mock_answer():
    """Return a Chainlit Message mock that supports streaming and send."""
    mock = MagicMock()
    mock.stream_token = AsyncMock()
    mock.send = AsyncMock()
    mock.content = "streamed answer"
    return mock


def make_stream_event(content="streamed token"):
    """Return a single on_chat_model_stream event as yielded by astream_events."""
    chunk = MagicMock()
    chunk.content = content
    return {"event": "on_chat_model_stream", "data": {"chunk": chunk}}


async def mock_astream_events(*args, **kwargs):
    """Async generator that yields a single stream event."""
    yield make_stream_event()


@pytest.mark.asyncio
async def test_on_chat_start_initializes_graph():
    """on_chat_start should build a new graph and store it in the user session."""
    from main import on_chat_start

    mock_graph = MagicMock()

    with (
        patch("main.build_graph", return_value=mock_graph) as mock_build,
        patch("main.Chroma"),
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.ChatSettings") as MockSettings,
        patch("main.cl.Message") as MockMessage,
    ):
        MockSettings.return_value.send = AsyncMock()
        MockMessage.return_value.send = AsyncMock()
        await on_chat_start()

    mock_build.assert_called_once()
    mock_session.set.assert_any_call("graph", mock_graph)


@pytest.mark.asyncio
async def test_on_chat_start_initializes_vectorstore():
    """on_chat_start should create a Chroma vectorstore and store it in the session."""
    from main import on_chat_start

    mock_vectorstore = MagicMock()

    with (
        patch("main.build_graph"),
        patch("main.Chroma", return_value=mock_vectorstore) as MockChroma,
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.ChatSettings") as MockSettings,
        patch("main.cl.Message") as MockMessage,
    ):
        MockSettings.return_value.send = AsyncMock()
        MockMessage.return_value.send = AsyncMock()
        await on_chat_start()

    MockChroma.assert_called_once()
    mock_session.set.assert_any_call("vectorstore", mock_vectorstore)


@pytest.mark.asyncio
async def test_on_chat_start_initializes_empty_history():
    """on_chat_start should set an empty list as the initial conversation history."""
    from main import on_chat_start

    with (
        patch("main.build_graph"),
        patch("main.Chroma"),
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.ChatSettings") as MockSettings,
        patch("main.cl.Message") as MockMessage,
    ):
        MockSettings.return_value.send = AsyncMock()
        MockMessage.return_value.send = AsyncMock()
        await on_chat_start()

    mock_session.set.assert_any_call("history", [])


@pytest.mark.asyncio
async def test_on_chat_start_sends_welcome_message():
    """on_chat_start should send exactly one welcome message to the user."""
    from main import on_chat_start

    mock_message_instance = MagicMock()
    mock_message_instance.send = AsyncMock()

    with (
        patch("main.build_graph"),
        patch("main.Chroma"),
        patch("main.cl.user_session"),
        patch("main.cl.ChatSettings") as MockSettings,
        patch("main.cl.Message", return_value=mock_message_instance),
    ):
        MockSettings.return_value.send = AsyncMock()
        await on_chat_start()

    mock_message_instance.send.assert_called_once()


@pytest.mark.asyncio
async def test_on_chat_start_sends_model_settings():
    """on_chat_start should send a ChatSettings widget with a model selector."""
    from main import on_chat_start

    mock_settings_instance = MagicMock()
    mock_settings_instance.send = AsyncMock()

    with (
        patch("main.build_graph"),
        patch("main.Chroma"),
        patch("main.cl.user_session"),
        patch(
            "main.cl.ChatSettings", return_value=mock_settings_instance
        ) as MockSettings,
        patch("main.cl.Message") as MockMessage,
    ):
        MockMessage.return_value.send = AsyncMock()
        await on_chat_start()

    MockSettings.assert_called_once()
    mock_settings_instance.send.assert_called_once()


@pytest.mark.asyncio
async def test_on_settings_update_stores_selected_model():
    """on_settings_update should persist the model chosen in the settings widget."""
    from main import on_settings_update

    with patch("main.cl.user_session") as mock_session:
        await on_settings_update({"model": "gemini/gemini-2.5-flash"})

    mock_session.set.assert_called_once_with("model", "gemini/gemini-2.5-flash")


@pytest.mark.asyncio
async def test_on_message_appends_human_message_to_history():
    """on_message should append the user's text as a HumanMessage before invoking the graph."""
    from main import on_message

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events
    mock_answer = make_mock_answer()

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_answer),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message(content="What is attention?"))

    saved_history = mock_session.set.call_args_list
    history_call = next(c for c in saved_history if c[0][0] == "history")
    messages = history_call[0][1]
    assert any(
        isinstance(m, HumanMessage) and m.content == "What is attention?"
        for m in messages
    )


@pytest.mark.asyncio
async def test_on_message_adds_ai_response_to_history():
    """on_message should append the streamed AI response to the conversation history."""
    from main import on_message

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events
    mock_answer = make_mock_answer()

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_answer),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    saved_history = mock_session.set.call_args_list
    history_call = next(c for c in saved_history if c[0][0] == "history")
    messages = history_call[0][1]
    assert any(isinstance(m, AIMessage) for m in messages)


@pytest.mark.asyncio
async def test_on_message_streams_tokens():
    """on_message should call stream_token for each on_chat_model_stream event."""
    from main import on_message

    async def multi_event_stream(*args, **kwargs):
        yield make_stream_event("Hello")
        yield make_stream_event(" world")

    mock_graph = MagicMock()
    mock_graph.astream_events = multi_event_stream
    mock_answer = make_mock_answer()

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_answer),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    assert mock_answer.stream_token.call_count == 2


@pytest.mark.asyncio
async def test_on_message_calls_process_document_for_each_element():
    """on_message should call process_document once per uploaded file element."""
    from main import on_message

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events
    mock_answer = make_mock_answer()
    elements = [make_mock_element("a.pdf"), make_mock_element("b.pdf")]

    with (
        patch(
            "main.process_document",
            new_callable=lambda: lambda: AsyncMock(return_value=3),
        ) as mock_proc,
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_answer),
    ):
        mock_proc = AsyncMock(return_value=3)
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        with patch("main.process_document", mock_proc):
            await on_message(make_mock_message(elements=elements))

    assert mock_proc.call_count == 2


@pytest.mark.asyncio
async def test_on_message_sends_error_on_unsupported_file():
    """on_message should send an error message and return early when
    process_document raises a ValueError for an unsupported file type."""
    from main import on_message

    mock_error_msg = MagicMock()
    mock_error_msg.send = AsyncMock()

    with (
        patch(
            "main.process_document",
            side_effect=ValueError("Unsupported file type: 'photo.png'"),
        ),
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_error_msg),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": MagicMock(),
        }.get(key, default)

        await on_message(make_mock_message(elements=[make_mock_element("photo.png")]))

    mock_error_msg.send.assert_called_once()


@pytest.mark.asyncio
async def test_on_message_handles_rate_limit_error():
    """on_message should send a rate-limit error message when the LLM API
    returns a RateLimitError."""

    async def raise_rate_limit(*args, **kwargs):
        raise RateLimitError("rate limit", llm_provider="groq", model="llama")
        yield  # make it an async generator

    mock_graph = MagicMock()
    mock_graph.astream_events = raise_rate_limit
    make_mock_answer()
    mock_error = MagicMock()
    mock_error.send = AsyncMock()

    messages_sent = []

    def message_factory(content=""):
        m = MagicMock()
        m.content = content
        m.stream_token = AsyncMock()
        m.send = AsyncMock()
        messages_sent.append(m)
        return m

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", side_effect=message_factory),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    all_content = " ".join(m.content for m in messages_sent)
    assert "Rate limit" in all_content or "rate limit" in all_content.lower()


@pytest.mark.asyncio
async def test_on_message_handles_bad_request_error():
    """on_message should send an error message when the LLM returns a BadRequestError."""
    from main import on_message

    async def raise_bad_request(*args, **kwargs):
        raise BadRequestError("bad request", llm_provider="google", model="gemini")
        yield

    mock_graph = MagicMock()
    mock_graph.astream_events = raise_bad_request

    messages_sent = []

    def message_factory(content=""):
        m = MagicMock()
        m.content = content
        m.stream_token = AsyncMock()
        m.send = AsyncMock()
        messages_sent.append(m)
        return m

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", side_effect=message_factory),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    all_content = " ".join(m.content for m in messages_sent)
    assert "Invalid request" in all_content or "❌" in all_content


@pytest.mark.asyncio
async def test_on_message_handles_service_unavailable_error():
    """on_message should send an error message when the model API is unavailable."""
    from main import on_message

    async def raise_unavailable(*args, **kwargs):
        raise ServiceUnavailableError("unavailable", llm_provider="groq", model="llama")
        yield

    mock_graph = MagicMock()
    mock_graph.astream_events = raise_unavailable

    messages_sent = []

    def message_factory(content=""):
        m = MagicMock()
        m.content = content
        m.stream_token = AsyncMock()
        m.send = AsyncMock()
        messages_sent.append(m)
        return m

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", side_effect=message_factory),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    all_content = " ".join(m.content for m in messages_sent)
    assert "unavailable" in all_content.lower() or "❌" in all_content


@pytest.mark.asyncio
async def test_on_message_handles_generic_exception():
    """on_message should catch unexpected exceptions and include the error
    message in the response sent to the user."""
    from main import on_message

    async def raise_generic(*args, **kwargs):
        raise Exception("something went wrong")
        yield

    mock_graph = MagicMock()
    mock_graph.astream_events = raise_generic

    messages_sent = []

    def message_factory(content=""):
        m = MagicMock()
        m.content = content
        m.stream_token = AsyncMock()
        m.send = AsyncMock()
        messages_sent.append(m)
        return m

    with (
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", side_effect=message_factory),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": [],
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message())

    all_content = " ".join(m.content for m in messages_sent)
    assert "something went wrong" in all_content


@pytest.mark.asyncio
async def test_on_message_appends_system_message_to_history_after_upload():
    """on_message should append a SystemMessage with the filename to the history
    when a file is uploaded and the conversation history is not empty."""
    from main import on_message

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events
    mock_answer = make_mock_answer()
    existing_history = [HumanMessage(content="previous message")]

    with (
        patch("main.process_document", new=AsyncMock(return_value=3)),
        patch("main.cl.user_session") as mock_session,
        patch("main.cl.Message", return_value=mock_answer),
    ):
        mock_session.get.side_effect = lambda key, default=None: {
            "model": "groq/llama-3.3-70b-versatile",
            "history": existing_history,
            "graph": mock_graph,
        }.get(key, default)

        await on_message(make_mock_message(elements=[make_mock_element("paper.pdf")]))

    saved_history = mock_session.set.call_args_list
    history_call = next(c for c in saved_history if c[0][0] == "history")
    messages = history_call[0][1]
    assert any(
        isinstance(m, SystemMessage) and "paper.pdf" in m.content for m in messages
    )

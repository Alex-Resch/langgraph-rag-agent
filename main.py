from langchain_community.vectorstores import Chroma
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from litellm.exceptions import RateLimitError, BadRequestError, ServiceUnavailableError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatLiteLLM

from agent.graph import build_graph
from agent.tools import process_document
from config import AVAILABLE_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("graph", build_graph())
    cl.user_session.set("vectorstore", Chroma(embedding_function=embeddings))
    cl.user_session.set("history", [])

    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model",
                label="Select model",
                values=AVAILABLE_MODELS,
                initial_value=DEFAULT_MODEL,
            )
        ]
    ).send()

    cl.user_session.set("model", DEFAULT_MODEL)
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
            try:
                intro_text = await process_document(element)

                model = cl.user_session.get("model", DEFAULT_MODEL)
                async with cl.Step(name="create summary..."):
                    if not model:
                        return

                    llm = ChatLiteLLM(model=model, temperature=0)
                    summary_prompt = SystemMessage(
                        content=(
                            "You are an assistant. Create a short, meaningful summary "
                            "(max. 3-4 sentences) of the following document based on "
                            "the introduction/first pages. State the main topic "
                            "and (if apparent) the main contributions:\n\n"
                            f"{intro_text}"
                        )
                    )
                    summary = await llm.ainvoke([summary_prompt])

                history = cl.user_session.get("history", [])

                history.append(  # type: ignore
                    SystemMessage(
                        content=(
                            f"The user just uploaded a File: '{element.name}'.\n"
                            f"Here is a summary of the document for general context:\n{summary}\n"
                            f"It has been stored. Use search_documents for specific detailed queries."
                        )
                    )
                )
                cl.user_session.set("history", history)
            except ValueError as e:
                await cl.Message(content=f"❌ {e}").send()
                return

    model = cl.user_session.get("model", DEFAULT_MODEL)
    history = cl.user_session.get("history", [])
    history.append(HumanMessage(content=message.content))  # type: ignore

    answer = cl.Message(content="")
    error_msg = None
    try:
        graph = cl.user_session.get("graph")
        if graph:
            async for event in graph.astream_events(
                {"messages": history, "model": model}, version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    await answer.stream_token(chunk.content)
    except RateLimitError:
        error_msg = (
            f"⚠️ Rate limit exceeded for {model}. Try again later or switch the model."
        )
    except BadRequestError:
        error_msg = "❌ Invalid request – maybe the model doesn't support this input."
    except ServiceUnavailableError:
        error_msg = "❌ Model API is currently unavailable. Try again later or switch the model."
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}"

    if error_msg:
        await cl.Message(error_msg).send()
        return

    await answer.send()
    history.append(AIMessage(answer.content))  # type: ignore
    cl.user_session.set("history", history)

from typing import cast

from langchain_community.vectorstores import Chroma
import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from litellm.exceptions import (
    RateLimitError,
    BadRequestError,
    ServiceUnavailableError,
    MidStreamFallbackError,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler

from agent.agent_state import AgentContext
from agent.graph import build_graph
from agent.tools import process_document
from config import AVAILABLE_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL

load_dotenv()

# noinspection PyArgumentList
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
langfuse_config = RunnableConfig(callbacks=[CallbackHandler()])


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("graph", build_graph())
    cl.user_session.set(
        "vectorstore",
        Chroma(
            embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"}
        ),
    )
    cl.user_session.set("history", [])

    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model",
                label="Select model",
                items=AVAILABLE_MODELS,
                initial_value=DEFAULT_MODEL,
            )
        ]
    ).send()

    cl.user_session.set("model", DEFAULT_MODEL)


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("model", settings["model"])


@cl.on_message
async def on_message(message: cl.Message):
    vectorstore: VectorStore = cl.user_session.get("vectorstore")

    if message.elements:
        for element in message.elements:
            try:
                intro_text = await process_document(element, vectorstore)

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
                    with propagate_attributes(
                        session_id=cl.context.session.id,
                        trace_name="document-summary",
                    ):
                        summary = await llm.ainvoke(
                            [summary_prompt],
                            config=langfuse_config,
                        )

                history = cl.user_session.get("history", [])

                history.append(  # type: ignore
                    SystemMessage(
                        content=(
                            f"The user just uploaded a File: '{element.name}'.\n"
                            f"Here is a summary of the document for general context:\n{summary.content}\n"
                            f"Only use search_documents if the user asks something about this document."
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
            with propagate_attributes(
                session_id=cl.context.session.id, trace_name="chat-message"
            ):
                async for event in graph.astream_events(
                    {"messages": history, "model": model},
                    config=langfuse_config,
                    context=AgentContext(vectorstore=vectorstore),
                    version="v2",
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        await answer.stream_token(chunk.content)
    except RateLimitError as e:
        print("ratelimit_e: ", e)
        error_msg = (
            f"⚠️ Rate limit exceeded for {model}. Try again later or switch the model."
        )
    except MidStreamFallbackError as e:
        if isinstance(e.original_exception, RateLimitError):
            print("error_E: ", e)
            error_msg = f"⚠️ Rate limit exceeded for {model}. Try again later or switch the model."
        else:
            error_msg = f"❌ Stream error: {e}"
    except BadRequestError:
        error_msg = "❌ Invalid request – maybe the model doesn't support this input."
    except ServiceUnavailableError as e:
        print("service_unavailable: ", e)
        error_msg = "❌ Model API is currently unavailable. Try again later or switch the model."
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}"

    if error_msg:
        await cl.Message(error_msg).send()
        return

    await answer.send()
    history.append(AIMessage(answer.content))  # type: ignore
    cl.user_session.set("history", history)

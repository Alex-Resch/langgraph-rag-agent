from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chainlit as cl
from dotenv import load_dotenv

from agent.graph import build_graph
from agent.tools import process_document
from config import AVAILABLE_MODELS, DEFAULT_MODEL, EMBEDDING_MODEL

load_dotenv()

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("vectorstore", Chroma(embedding_function=embeddings))
    cl.user_session.set("history", [])

    await cl.ChatSettings([
        cl.input_widget.Select(
            id="model",
            label="Select model",
            values=AVAILABLE_MODELS,
            initial_value=DEFAULT_MODEL,
        )
    ]).send()

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

    model = cl.user_session.get("model", DEFAULT_MODEL)
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    answer = cl.Message(content="")

    async for event in build_graph().astream_events(
            {"messages": history, "model": model},
            version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            await answer.stream_token(chunk.content)

    await answer.send()
    history.append({"role": "assistant", "content": answer.content})
    cl.user_session.set("history", history)
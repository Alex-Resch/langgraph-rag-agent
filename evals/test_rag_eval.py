"""RAG quality evals: runs every question of testset.json through the agent and
lets an LLM judge score the answers with DeepEval.

Run with:  uv run pytest evals/
"""

import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import pytest  # noqa: E402
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import GeminiModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from agent.agent_state import AgentContext  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.tools import process_document  # noqa: E402
from config import DEFAULT_MODEL, EMBEDDING_MODEL  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

EVALS_DIR = Path(__file__).parent
TESTSET = json.loads((EVALS_DIR / "testset.json").read_text())
DOCUMENT = EVALS_DIR / TESTSET["document"]

JUDGE_MODEL = "gemini-2.5-flash-lite"
THRESHOLD = 0.7
NO_CONTEXT = "<no documents retrieved>"


@pytest.fixture(scope="module")
def vectorstore():
    """Load the test document into a fresh vectorstore, exactly like an upload in the app."""
    # noinspection PyArgumentList
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(
        embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"}
    )
    element = SimpleNamespace(
        name=DOCUMENT.name, mime="application/pdf", path=str(DOCUMENT)
    )
    asyncio.run(process_document(element, store))  # type: ignore[arg-type]
    return store


@pytest.fixture(scope="module")
def metrics():
    judge = GeminiModel(
        model=JUDGE_MODEL, api_key=os.environ["GEMINI_API_KEY"], temperature=0
    )
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output states the same facts as the "
            "expected output. Different wording is fine; missing, extra or wrong "
            "facts are not. If the expected output says the information is not in "
            "the document, the actual output must not invent an answer."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=judge,
    )
    return {
        "all": [correctness],
        "retrieval": [
            correctness,
            AnswerRelevancyMetric(threshold=THRESHOLD, model=judge),
            FaithfulnessMetric(threshold=THRESHOLD, model=judge),
            ContextualPrecisionMetric(threshold=THRESHOLD, model=judge),
        ],
    }


async def run_agent(question: str, store: Chroma) -> tuple[str, list[str]]:
    """Ask the agent a question and return its answer plus the retrieved chunks."""
    messages = [
        SystemMessage(
            content=(
                f"The user just uploaded a File: '{DOCUMENT.name}'.\n"
                "Only use search_documents if the user asks something about this document."
            )
        ),
        HumanMessage(content=question),
    ]
    result = await build_graph().ainvoke(
        {"messages": messages, "model": DEFAULT_MODEL},
        context=AgentContext(vectorstore=store),
    )

    chunks = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name == "search_documents":
            body = str(message.content).removeprefix("Found in documents:\n\n")
            if body != "NO_DOCUMENTS_FOUND":
                chunks.extend(re.split(r"\n\n(?=\[)", body))

    return str(result["messages"][-1].content), chunks or [NO_CONTEXT]


@pytest.mark.parametrize(
    "case", TESTSET["cases"], ids=lambda c: f"{c['type']}: {c['input']}"
)
def test_rag_quality(case, vectorstore, metrics):
    answer, retrieval_context = asyncio.run(run_agent(case["input"], vectorstore))

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=answer,
        expected_output=case["expected_output"],
        retrieval_context=[*retrieval_context],
    )
    selected = metrics["retrieval"] if case["pages"] else metrics["all"]
    assert_test(test_case, selected)

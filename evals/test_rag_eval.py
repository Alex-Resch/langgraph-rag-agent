"""RAG quality evals: runs every question of testset.json through the agent and
lets an LLM judge score the answers with DeepEval.

Run with:  uv run pytest evals/ -s
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import pytest  # noqa: E402
from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig  # noqa: E402
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
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from agent.agent_state import AgentContext  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.prompts import upload_notice  # noqa: E402
from agent.tools import process_document  # noqa: E402
from config import DEFAULT_MODEL, EMBEDDING_MODEL  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

EVALS_DIR = Path(__file__).parent
TESTSET = json.loads((EVALS_DIR / "testset.json").read_text())
DOCUMENT = EVALS_DIR / TESTSET["document"]

JUDGE_MODEL = "gemini-2.5-flash-lite"
THRESHOLD = 0.7
# 0.5 = the right chunk is at least on rank 2; the LLM reads all retrieved chunks.
PRECISION_THRESHOLD = 0.5
MIN_PASS_RATE = 0.9
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
        evaluation_steps=[
            "Identify the key facts in the expected output (numbers, names, dates, units).",
            "Check whether the actual output states each key fact correctly. Full "
            "sentences, different wording or formatting and restating the question "
            "are fine and must not be penalized.",
            "Heavily penalize if a key fact is missing or contradicted, or if the "
            "actual output adds a fact that is wrong.",
            "If the expected output says the information is not in the document, "
            "the actual output must either say that it cannot find the answer, or "
            "answer from a web search while clearly stating that the answer does "
            "not come from the document. Presenting outside information as if it "
            "came from the document is wrong.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=judge,
    )
    relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge)
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, model=judge)
    precision = ContextualPrecisionMetric(threshold=PRECISION_THRESHOLD, model=judge)
    return {
        "answer_only": [correctness],
        # Faithfulness flags calculated values (e.g. cost per household) as
        # unsupported because the result is not literally in the context, and
        # Answer Relevancy penalizes the explained calculation steps.
        "calculated": [correctness, precision],
        "full": [correctness, relevancy, faithfulness, precision],
    }


def metric_group(case: dict) -> str:
    if not case["pages"]:
        return "answer_only"
    return "calculated" if case.get("calculated") else "full"


async def run_agent(question: str, store: Chroma) -> tuple[str, list[str]]:
    """Ask the agent a question and return its answer plus the retrieved chunks."""
    messages = [upload_notice(DOCUMENT.name), HumanMessage(content=question)]
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


def test_rag_quality(vectorstore, metrics):
    """Fail only if fewer than MIN_PASS_RATE of all cases pass, since single
    LLM-judged cases can fluctuate between runs."""
    groups: dict[str, list[LLMTestCase]] = defaultdict(list)
    for case in TESTSET["cases"]:
        answer, retrieval_context = asyncio.run(run_agent(case["input"], vectorstore))
        groups[metric_group(case)].append(
            LLMTestCase(
                name=f"{case['type']}: {case['input']}",
                input=case["input"],
                actual_output=answer,
                expected_output=case["expected_output"],
                retrieval_context=[*retrieval_context],
            )
        )

    results = []
    for group, test_cases in groups.items():
        results += evaluate(
            test_cases,
            metrics[group],
            async_config=AsyncConfig(max_concurrent=5),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        ).test_results

    failed = [r for r in results if not r.success]
    pass_rate = 1 - len(failed) / len(results)
    report = [
        f"{len(results) - len(failed)}/{len(results)} cases passed ({pass_rate:.0%})"
    ]
    for result in failed:
        report.append(f"FAILED {result.name}")
        for metric in result.metrics_data or []:
            if not metric.success:
                report.append(f"  {metric.name} = {metric.score:.2f}: {metric.reason}")
    print("\n".join(report))

    assert pass_rate >= MIN_PASS_RATE, "\n".join(report)

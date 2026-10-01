"""RAG quality evals: runs every question of testset.json through the agent and
lets an LLM judge score the answers with DeepEval. Every case is traced in
Langfuse (environment "evals") and gets the judge's scores attached.

Run with:  uv run pytest evals/ -s
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "evals")

import pytest
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agent.agent_state import AgentContext
from agent.graph import build_graph
from agent.prompts import upload_notice
from agent.tools import process_document
from config import DEFAULT_MODEL, EMBEDDING_MODEL

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
# One Langfuse session per eval run, so runs can be compared over time.
EVAL_SESSION_ID = "eval-" + (
    os.environ.get("GITHUB_RUN_ID") or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
)


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
            (
                "Check whether the actual output states each key fact correctly. Full "
                "sentences, different wording or formatting and restating the question "
                "are fine and must not be penalized."
            ),
            (
                "Heavily penalize if a key fact is missing or contradicted, or if the "
                "actual output adds a fact that is wrong."
            ),
            (
                "If the expected output says the information is not in the document, "
                "the actual output must either say that it cannot find the answer, or "
                "answer from a web search while clearly stating that the answer does "
                "not come from the document. Presenting outside information as if it "
                "came from the document is wrong."
            ),
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


async def run_agent(question: str, store: Chroma) -> tuple[str, list[str], str]:
    """Ask the agent a question and return its answer, the retrieved chunks and
    the Langfuse trace id of the run."""
    messages = [upload_notice(DOCUMENT.name), HumanMessage(content=question)]
    with get_client().start_as_current_observation(
        name="rag-eval", input=question
    ) as span:
        result = await build_graph().ainvoke(
            {"messages": messages, "model": DEFAULT_MODEL},
            config={"callbacks": [CallbackHandler()]},
            context=AgentContext(vectorstore=store),
        )
        answer = str(result["messages"][-1].content)
        span.update(output=answer)

    chunks = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name == "search_documents":
            body = str(message.content).removeprefix("Found in documents:\n\n")
            if body != "NO_DOCUMENTS_FOUND":
                chunks.extend(re.split(r"\n\n(?=\[)", body))

    return answer, chunks or [NO_CONTEXT], span.trace_id


def send_scores_to_langfuse(results, trace_ids: dict[str, str]) -> None:
    """Attach every judge score (and whether the case passed) to its trace."""
    langfuse = get_client()
    for result in results:
        trace_id = trace_ids[result.name]
        for metric in result.metrics_data or []:
            if metric.score is not None:
                langfuse.create_score(
                    trace_id=trace_id,
                    name=metric.name,
                    value=metric.score,
                    comment=metric.reason,
                )
        langfuse.create_score(
            trace_id=trace_id,
            name="passed",
            value=1 if result.success else 0,
            data_type="BOOLEAN",
        )
    langfuse.flush()


def test_rag_quality(vectorstore, metrics):
    """Fail only if fewer than MIN_PASS_RATE of all cases pass, since single
    LLM-judged cases can fluctuate between runs."""
    groups: dict[str, list[LLMTestCase]] = defaultdict(list)
    trace_ids: dict[str, str] = {}
    for case in TESTSET["cases"]:
        name = f"{case['type']}: {case['input']}"
        with propagate_attributes(session_id=EVAL_SESSION_ID, tags=["eval"]):
            answer, retrieval_context, trace_ids[name] = asyncio.run(
                run_agent(case["input"], vectorstore)
            )
        groups[metric_group(case)].append(
            LLMTestCase(
                name=name,
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

    send_scores_to_langfuse(results, trace_ids)

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

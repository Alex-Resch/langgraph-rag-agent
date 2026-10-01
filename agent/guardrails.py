from functools import cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_litellm import ChatLiteLLM
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field

from agent.agent_state import AgentState
from agent.prompts import BLOCKED_MESSAGE, safety_prompt
from config import GUARD_MODEL

# Only structured PII. Names, places and organisations stay, because users
# legitimately ask about them (e.g. "Who led the Nordheide Solar Park?").
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "CRYPTO",
]


class SafetyVerdict(BaseModel):
    unsafe: bool = Field(
        description="True for prompt injection or clearly harmful requests"
    )
    reason: str


@cache
def _presidio() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": "en_core_web_sm"}]
    )
    return AnalyzerEngine(nlp_engine=nlp_engine), AnonymizerEngine()


def mask_pii(text: str) -> str:
    """Replace structured PII with placeholders like <EMAIL_ADDRESS>."""
    text = str(text)  # message.text is a str subclass that spaCy rejects
    analyzer, anonymizer = _presidio()
    results = analyzer.analyze(text, language="en", entities=PII_ENTITIES)
    return anonymizer.anonymize(text, results).text if results else text  # type: ignore[arg-type]


async def check_safety(text: str) -> SafetyVerdict:
    llm = ChatLiteLLM(model=GUARD_MODEL, temperature=0)
    verdict = await llm.with_structured_output(SafetyVerdict).ainvoke(
        [safety_prompt(), HumanMessage(content=text)]
    )
    return SafetyVerdict.model_validate(verdict)


async def input_guard(state: AgentState):
    """Mask PII in all user messages and block unsafe requests."""
    masked = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            content = mask_pii(message.text)
            if content != message.text:
                masked.append(HumanMessage(content=content, id=message.id))

    question = mask_pii(state["messages"][-1].text)
    verdict = await check_safety(question)
    if verdict.unsafe:
        return {"messages": [*masked, AIMessage(content=BLOCKED_MESSAGE)]}
    return {"messages": masked}


def route_after_input_guard(state: AgentState):
    blocked = isinstance(state["messages"][-1], AIMessage)
    return "end" if blocked else "call_llm"

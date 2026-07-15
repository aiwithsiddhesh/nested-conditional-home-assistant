from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.loaders import build_retriever
from app.state import HomeAssistantState

_llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

_policy_retriever = build_retriever(settings.policy_doc_path)


class InsuranceSubtypeClassification(BaseModel):
    insurance_subtype: Literal["coverage_question", "active_claim"] = Field(
        description=(
            "The insurance subtype for the homeowner's question: "
            "'coverage_question' for questions asking what is covered by the "
            "policy; 'active_claim' for situations where something has already "
            "happened and needs to be filed or tracked as a claim."
        )
    )


_classifier = _llm.with_structured_output(InsuranceSubtypeClassification)


def insurance_classifier_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content

    result = _classifier.invoke(
        [
            (
                "system",
                "Classify the homeowner's insurance question into exactly one "
                "subtype: coverage_question or active_claim.",
            ),
            ("human", latest_message),
        ]
    )

    return {"insurance_subtype": result.insurance_subtype}


def route_insurance(
    state: HomeAssistantState,
) -> Literal["insurance_rag_node", "claim_escalation_node"]:
    return {
        "coverage_question": "insurance_rag_node",
        "active_claim": "claim_escalation_node",
    }[state["insurance_subtype"]]


def _retrieve_context(query: str) -> str:
    docs = _policy_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def insurance_rag_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(latest_message)}


def claim_escalation_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    return {
        "retrieved_context": _retrieve_context(latest_message),
        "needs_claim_guidance": True,
    }

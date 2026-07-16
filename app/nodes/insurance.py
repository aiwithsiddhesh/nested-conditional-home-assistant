from typing import Literal

from pydantic import BaseModel, Field

from app.classifiers import make_classifier_node
from app.config import settings
from app.loaders import build_retriever, make_rag_node
from app.state import HomeAssistantState

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


insurance_classifier_node = make_classifier_node(
    InsuranceSubtypeClassification,
    "insurance_subtype",
    "Classify the homeowner's insurance question into exactly one subtype: "
    "coverage_question or active_claim.",
)


def route_insurance(
    state: HomeAssistantState,
) -> Literal["insurance_rag_node", "claim_escalation_node"]:
    return {
        "coverage_question": "insurance_rag_node",
        "active_claim": "claim_escalation_node",
    }[state["insurance_subtype"]]


insurance_rag_node = make_rag_node(_policy_retriever)
claim_escalation_node = make_rag_node(_policy_retriever, extra_fields={"needs_claim_guidance": True})

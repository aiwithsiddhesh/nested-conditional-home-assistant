from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.state import HomeAssistantState


class QueryTypeClassification(BaseModel):
    query_type: Literal["appliance", "insurance", "general"] = Field(
        description=(
            "The top-level category of the homeowner's question: 'appliance' for "
            "questions about kitchen appliances, laundry machines, or HVAC systems; "
            "'insurance' for questions about home insurance coverage or claims; "
            "'general' for anything else."
        )
    )


_llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)
_classifier = _llm.with_structured_output(QueryTypeClassification)


def classifier_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content

    result = _classifier.invoke(
        [
            (
                "system",
                "Classify the homeowner's question into exactly one top-level "
                "category: appliance, insurance, or general.",
            ),
            ("human", latest_message),
        ]
    )

    return {"query_type": result.query_type}


def route_level1(
    state: HomeAssistantState,
) -> Literal["appliance_classifier_node", "insurance_classifier_node", "general_node"]:
    return {
        "appliance": "appliance_classifier_node",
        "insurance": "insurance_classifier_node",
        "general": "general_node",
    }[state["query_type"]]

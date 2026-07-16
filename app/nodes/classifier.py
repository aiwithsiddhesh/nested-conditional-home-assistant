from typing import Literal

from pydantic import BaseModel, Field

from app.classifiers import make_classifier_node
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


classifier_node = make_classifier_node(
    QueryTypeClassification,
    "query_type",
    "Classify the homeowner's question into exactly one top-level category: "
    "appliance, insurance, or general.",
)


def route_level1(
    state: HomeAssistantState,
) -> Literal["appliance_classifier_node", "insurance_classifier_node", "general_node"]:
    return {
        "appliance": "appliance_classifier_node",
        "insurance": "insurance_classifier_node",
        "general": "general_node",
    }[state["query_type"]]

from typing import Literal

from pydantic import BaseModel, Field

from app.classifiers import make_classifier_node
from app.config import settings
from app.loaders import build_retriever, make_rag_node
from app.state import HomeAssistantState

_kitchen_retriever = build_retriever(settings.kitchen_manual_path)
_laundry_retriever = build_retriever(settings.laundry_manual_path)
_hvac_retriever = build_retriever(settings.hvac_manual_path)


class ApplianceSubtypeClassification(BaseModel):
    appliance_subtype: Literal["kitchen", "laundry", "hvac"] = Field(
        description=(
            "The appliance manual that applies to the homeowner's question: "
            "'kitchen' for fridge, oven, or dishwasher questions; 'laundry' for "
            "washer or dryer questions; 'hvac' for heating, cooling, or air "
            "handling system questions."
        )
    )


appliance_classifier_node = make_classifier_node(
    ApplianceSubtypeClassification,
    "appliance_subtype",
    "Classify the homeowner's appliance question into exactly one subtype: "
    "kitchen, laundry, or hvac.",
)


def route_appliance(
    state: HomeAssistantState,
) -> Literal["kitchen_rag_node", "laundry_rag_node", "hvac_rag_node"]:
    return {
        "kitchen": "kitchen_rag_node",
        "laundry": "laundry_rag_node",
        "hvac": "hvac_rag_node",
    }[state["appliance_subtype"]]


kitchen_rag_node = make_rag_node(_kitchen_retriever)
laundry_rag_node = make_rag_node(_laundry_retriever)
hvac_rag_node = make_rag_node(_hvac_retriever)

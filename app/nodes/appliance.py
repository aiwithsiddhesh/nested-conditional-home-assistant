from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings
from app.loaders import build_retriever
from app.state import HomeAssistantState

_llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

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


_classifier = _llm.with_structured_output(ApplianceSubtypeClassification)


def appliance_classifier_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content

    result = _classifier.invoke(
        [
            (
                "system",
                "Classify the homeowner's appliance question into exactly one "
                "subtype: kitchen, laundry, or hvac.",
            ),
            ("human", latest_message),
        ]
    )

    return {"appliance_subtype": result.appliance_subtype}


def route_appliance(
    state: HomeAssistantState,
) -> Literal["kitchen_rag_node", "laundry_rag_node", "hvac_rag_node"]:
    return {
        "kitchen": "kitchen_rag_node",
        "laundry": "laundry_rag_node",
        "hvac": "hvac_rag_node",
    }[state["appliance_subtype"]]


def _retrieve_context(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


def kitchen_rag_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(_kitchen_retriever, latest_message)}


def laundry_rag_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(_laundry_retriever, latest_message)}


def hvac_rag_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    return {"retrieved_context": _retrieve_context(_hvac_retriever, latest_message)}

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class HomeAssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    household_type: Literal["apartment", "house", "rental"]
    query_type: Literal["appliance", "insurance", "general"]
    appliance_subtype: Literal["kitchen", "laundry", "hvac"]
    insurance_subtype: Literal["coverage_question", "active_claim"]
    retrieved_context: str
    needs_claim_guidance: bool

from langchain_groq import ChatGroq

from app.config import settings
from app.nodes.general import NO_RETRIEVAL_MARKER
from app.state import HomeAssistantState

_llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

_CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are a home assistant answering a homeowner's question. Respond in a "
    "helpful, conversational tone. Prefix your answer with "
    "'For a {household_type} resident: '. {context_instruction}"
)

_CLAIM_GUIDANCE_SYSTEM_PROMPT = (
    "You are a home assistant helping a homeowner with an active insurance "
    "claim. Respond with careful, step-by-step guidance rather than a casual "
    "conversational tone, since claim documentation and deadlines matter. "
    "Prefix your answer with 'For a {household_type} resident: '. "
    "{context_instruction}"
)


def response_node(state: HomeAssistantState) -> dict:
    latest_message = state["messages"][-1].content
    retrieved_context = state["retrieved_context"]
    household_type = state["household_type"]
    needs_claim_guidance = state.get("needs_claim_guidance", False)

    if retrieved_context == NO_RETRIEVAL_MARKER:
        context_instruction = "Answer from your own general knowledge."
    else:
        context_instruction = (
            "Use the following retrieved context to answer. Each excerpt is "
            "tagged with its source page, e.g. '[page 3]' — you may mention the "
            f"page number when it helps the homeowner locate it in their manual, "
            f"but do not copy the bracket tags verbatim.\n\n{retrieved_context}"
        )

    template = _CLAIM_GUIDANCE_SYSTEM_PROMPT if needs_claim_guidance else _CONVERSATIONAL_SYSTEM_PROMPT
    system_prompt = template.format(household_type=household_type, context_instruction=context_instruction)

    response = _llm.invoke(
        [
            ("system", system_prompt),
            ("human", latest_message),
        ]
    )

    return {"messages": [("ai", response.content)]}

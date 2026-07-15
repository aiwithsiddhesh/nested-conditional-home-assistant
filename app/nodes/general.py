from app.state import HomeAssistantState

NO_RETRIEVAL_MARKER = "NO_RETRIEVAL_NEEDED"


def general_node(state: HomeAssistantState) -> dict:
    return {"retrieved_context": NO_RETRIEVAL_MARKER}

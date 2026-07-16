from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import settings
from app.state import HomeAssistantState

llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)


def make_classifier_node(schema: type[BaseModel], field_name: str, system_instruction: str):
    classifier = llm.with_structured_output(schema)

    def classifier_node(state: HomeAssistantState) -> dict:
        latest_message = state["messages"][-1].content

        result = classifier.invoke(
            [
                ("system", system_instruction),
                ("human", latest_message),
            ]
        )

        return {field_name: getattr(result, field_name)}

    return classifier_node

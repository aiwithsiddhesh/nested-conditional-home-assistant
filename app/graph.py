from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes.appliance import (
    appliance_classifier_node,
    hvac_rag_node,
    kitchen_rag_node,
    laundry_rag_node,
    route_appliance,
)
from app.nodes.classifier import classifier_node, route_level1
from app.nodes.general import general_node
from app.nodes.insurance import (
    claim_escalation_node,
    insurance_classifier_node,
    insurance_rag_node,
    route_insurance,
)
from app.nodes.response import response_node
from app.state import HomeAssistantState

ALL_NODES = [
    ("classifier_node", classifier_node),
    ("appliance_classifier_node", appliance_classifier_node),
    ("insurance_classifier_node", insurance_classifier_node),
    ("general_node", general_node),
    ("kitchen_rag_node", kitchen_rag_node),
    ("laundry_rag_node", laundry_rag_node),
    ("hvac_rag_node", hvac_rag_node),
    ("insurance_rag_node", insurance_rag_node),
    ("claim_escalation_node", claim_escalation_node),
    ("response_node", response_node),
]

CONDITIONAL_EDGES = [
    ("classifier_node", route_level1),
    ("appliance_classifier_node", route_appliance),
    ("insurance_classifier_node", route_insurance),
]

CONVERGENCE_EDGES = [
    ("kitchen_rag_node", "response_node"),
    ("laundry_rag_node", "response_node"),
    ("hvac_rag_node", "response_node"),
    ("insurance_rag_node", "response_node"),
    ("claim_escalation_node", "response_node"),
    ("general_node", "response_node"),
]


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(HomeAssistantState)

    for name, node_fn in ALL_NODES:
        graph.add_node(name, node_fn)

    graph.add_edge(START, "classifier_node")

    for source, router_fn in CONDITIONAL_EDGES:
        graph.add_conditional_edges(source, router_fn)

    for source, target in CONVERGENCE_EDGES:
        graph.add_edge(source, target)

    graph.add_edge("response_node", END)

    return graph.compile()


app = build_graph()

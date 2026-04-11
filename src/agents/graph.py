from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.nodes import data_node, decision_node, evaluate_node, optimize_node, predict_node, risk_node
from agents.state import AgentState


def route_after_optimize(state: AgentState) -> str:
    return "end" if bool(state.get("done", False)) else "continue"


def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("data", data_node)
    workflow.add_node("predict", predict_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("optimize", optimize_node)

    workflow.add_edge(START, "data")
    workflow.add_edge("data", "predict")
    workflow.add_edge("predict", "risk")
    workflow.add_edge("risk", "decision")
    workflow.add_edge("decision", "evaluate")
    workflow.add_edge("evaluate", "optimize")
    workflow.add_conditional_edges("optimize", route_after_optimize, {"continue": "data", "end": END})
    return workflow.compile()

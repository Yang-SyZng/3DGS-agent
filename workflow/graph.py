from langgraph.graph import StateGraph, START, END
from typing import Literal
from .state import AgentState
from .nodes import (
    analyzer_node,
    retriever_node,
    evaluator_node,
    increment_research_round_node,
    refine_analysis_node,
    external_search_node,
    writer_node,
)
from schema.evaluator_schema import RetrievalStatus

builder = StateGraph(AgentState)

def route_retrieval(state: AgentState) -> Literal["not_found", "insufficient", "sufficient"]:
    return state["retrieval_evaluated_result"].status.value


def route_refinement(state: AgentState) -> Literal["retry", "stop"]:
    return "stop" if state.get("retrieval_round", 0) >= 3 else "retry"


def route_external_search(state: AgentState) -> Literal["retry", "stop"]:
    if state.get("knowledge_updated") and state.get("external_search_round", 0) <= 2:
        return "retry"
    return "stop"

# ::workflow add_node::
builder.add_node("analyzer", analyzer_node)
builder.add_node("retriever", retriever_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("increment_research_round", increment_research_round_node)
builder.add_node("refine_analysis", refine_analysis_node)
builder.add_node("external_search", external_search_node)
builder.add_node("writer", writer_node)

# ::workflow start::
builder.add_edge(START, "analyzer")

builder.add_edge("analyzer", "retriever")
builder.add_edge("retriever", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    route_retrieval,
    {
        RetrievalStatus.NOT_FOUND.value: "external_search",
        RetrievalStatus.INSUFFICIENT.value: "increment_research_round",
        RetrievalStatus.SUFFICIENT.value: "writer",
    }
)
builder.add_conditional_edges(
    "increment_research_round",
    route_refinement,
    {
        "retry": "refine_analysis",
        "stop": "writer",
    }
)

builder.add_edge("refine_analysis", "retriever")
builder.add_conditional_edges(
    "external_search",
    route_external_search,
    {
        "retry": "retriever",
        "stop": "writer",
    },
)
builder.add_edge("writer", END)


graph = builder.compile()

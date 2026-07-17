from langgraph.graph import StateGraph, START, END
from typing import Literal
from .state import AgentState
from .nodes import (
    analyzer_node,
    retriever_node,
    evaluator_node,
    researcher_node,
    increment_research_round_node,
    refine_analysis_node,
)

builder = StateGraph(AgentState)

def inJudge_Retrieve_sufficient(state: AgentState) -> Literal["sufficient", "insufficient"]:
    if state["retrieval_evaluated_result"].sufficient:
        return "sufficient"
    else:
        return "insufficient"

def inJudge_Researcher_sufficient(state: AgentState) -> Literal["sufficient", "insufficient"]:
    if state["research_result"].sufficient:
        return "sufficient"

    if state.get("retrieval_round", 0) >= 2:
        return "sufficient"

    return "insufficient"

# ::workflow add_node::
builder.add_node("analyzer", analyzer_node)
builder.add_node("retriever", retriever_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("researcher", researcher_node)
builder.add_node("increment_research_round", increment_research_round_node)
builder.add_node("refine_analysis", refine_analysis_node)

# ::workflow start::
builder.add_edge(START, "analyzer")

builder.add_edge("analyzer", "retriever")
builder.add_edge("retriever", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    inJudge_Retrieve_sufficient,
    {
        "sufficient": "researcher",
        "insufficient": "researcher",  # 暂时
    }
)

builder.add_edge("researcher", "increment_research_round")

builder.add_conditional_edges(
    "increment_research_round",
    inJudge_Researcher_sufficient,
    {
        "sufficient": END,
        "insufficient": "refine_analysis",
    }
)

builder.add_edge("refine_analysis", "retriever")


graph = builder.compile()

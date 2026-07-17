import logging

from .state import AgentState
from agents.Analyzer import QueryAnalyzer
from agents.Retriever import PaperRetriever
from agents.Evaluator import RetrievalEvaluator
from agents.Researcher import ChunkResearcher
from config.settings import setting

from tools.llm_local_service.ollama_service import OllamaServer

logger = logging.getLogger(__name__)

local_llm = None
if setting.Globle_Local_Optional:
    ollama_service = OllamaServer(setting.Local_Model)
    local_llm = ollama_service.create_ollama_llm("LLM")

analyzer = QueryAnalyzer(local_llm)
retriever = PaperRetriever()
evaluator = RetrievalEvaluator(local_llm)
researcher = ChunkResearcher(local_llm)

# analyzer node
async def analyzer_node(state: AgentState):
    analysis = await analyzer.analyze(state["user_query"])
    logger.info("analyzer_node completed: analysis=%s", analysis.model_dump_json())
    return {
        "analysis": analysis
    }

# retriever node
async def retriever_node(state: AgentState):
    nodes = await retriever.retrieve(
                                    query=state["user_query"],
                                    analysis=state["analysis"]
                                )
    logger.info("retriever_node completed: retrieved_nodes=%s", nodes)
    return {
        "retrieved_nodes": nodes
    }

# evaluator node
async def evaluator_node(state: AgentState):
    evaluated_result = await evaluator.evaluate(
                                    query=state["user_query"],
                                    analysis=state["analysis"],
                                    retrieved_nodes=state["retrieved_nodes"]
                                )
    logger.info(
        "evaluator_node completed: retrieval_evaluated_result=%s",
        evaluated_result.model_dump_json(),
    )
    return {
        "retrieval_evaluated_result": evaluated_result
    }

# researcher node
async def researcher_node(state: AgentState):
    research_result = await researcher.research(
                                            query=state["user_query"],
                                            analysis=state["analysis"],
                                            retrieved_nodes=state["retrieved_nodes"],
                                            retrieval_evaluated_result=state["retrieval_evaluated_result"]
                                        )
    logger.info("researcher_node completed: research_result=%s", research_result.model_dump_json())
    return{
        "research_result": research_result
    }

async def increment_research_round_node(state: AgentState):
    retrieval_round = state.get("retrieval_round", 0) + 1
    logger.info(
        "increment_research_round_node completed: retrieval_round=%s",
        retrieval_round,
    )
    return {
        "retrieval_round": retrieval_round
    }


async def refine_analysis_node(state: AgentState):
    evaluation = state["retrieval_evaluated_result"]
    research_result = state["research_result"]
    refined_analysis = await analyzer.refine(
        query=state["user_query"],
        current_analysis=state["analysis"],
        missing_information=evaluation.missing_information,
        limitations=[
            limitation
            for limitation in research_result.limitations
            if limitation
        ],
    )
    logger.info(
        "refine_analysis_node completed: analysis=%s",
        refined_analysis.model_dump_json(),
    )
    return {
        "analysis": refined_analysis,
    }
# Matcher
# Search node

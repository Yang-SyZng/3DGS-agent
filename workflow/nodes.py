import asyncio
import logging
from pathlib import Path
from typing import Any

from .state import AgentState, PaperResolution
from agents.Analyzer import QueryAnalyzer
from agents.Retriever import PaperRetriever
from agents.Evaluator import RetrievalEvaluator
from agents.Matcher import PaperMatcher
from agents.Writer import AnswerWriter
from config.settings import setting
from rag.ingestion import PaperIngestionService
from schema.matcher_schema import MatchStatus

from llama_index.llms.openai_like import OpenAILike
from tools.llm_local_service.ollama_service import OllamaServer
from tools.llm_fallback import FallbackLLM

logger = logging.getLogger(__name__)

_ollama_service = None
_local_llm = None
_analyzer = None
_retriever = None
_evaluator = None
_matcher = None
_writer = None
_ingestion_service = None


def _get_llm():
    global _ollama_service, _local_llm
    if _local_llm is None:
        cloud_llm = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )
        if not setting.Globle_Local_Optional:
            _local_llm = cloud_llm
            return _local_llm

        def create_local_llm():
            global _ollama_service
            if _ollama_service is None:
                _ollama_service = OllamaServer(setting.Local_Model)
            return _ollama_service.create_ollama_llm("LLM")

        _local_llm = FallbackLLM(cloud_llm, create_local_llm)
    return _local_llm


def _get_analyzer() -> QueryAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer(_get_llm())
    return _analyzer


def _get_retriever() -> PaperRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PaperRetriever()
    return _retriever


def _get_evaluator() -> RetrievalEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RetrievalEvaluator(_get_llm())
    return _evaluator


def _get_matcher() -> PaperMatcher:
    global _matcher
    if _matcher is None:
        _matcher = PaperMatcher(_get_llm())
    return _matcher


def _get_writer() -> AnswerWriter:
    global _writer
    if _writer is None:
        _writer = AnswerWriter(_get_llm())
    return _writer


def _get_ingestion_service() -> PaperIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = PaperIngestionService()
    return _ingestion_service


def _find_pdf_path(payload: Any, existing_paths: set[Path]) -> Path:
    candidates: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (str, Path)):
            text = str(value).strip()
            if text.lower().endswith(".pdf"):
                path = Path(text).expanduser()
                if path.exists():
                    candidates.append(path.resolve())

    visit(payload)
    if not candidates:
        download_dir = Path(setting.pdf_save_dir).resolve()
        candidates = [
            path.resolve()
            for path in download_dir.rglob("*.pdf")
            if path.resolve() not in existing_paths
        ]
    if not candidates:
        raise RuntimeError("下载工具返回成功，但未找到本地 PDF 文件")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_and_index_pdf(pdf_path: Path, paper_id: str, paper_title: str) -> int:
    result = _get_ingestion_service().ingest(
        pdf_path=pdf_path,
        paper_id=paper_id,
        paper_title=paper_title,
        backend="pipeline",
        skip_existing=True,
    )
    return result.chunks_indexed

# analyzer node
async def analyzer_node(state: AgentState):
    analysis = await _get_analyzer().analyze(state["user_query"])
    logger.info("analyzer_node completed: analysis=%s", analysis.model_dump_json())
    return {
        "analysis": analysis
    }

# retriever node
async def retriever_node(state: AgentState):
    nodes = await _get_retriever().retrieve(
                                    query=state["user_query"],
                                    analysis=state["analysis"]
                                )
    logger.info(
        "retriever_node completed: count=%s chunks=%s",
        len(nodes),
        [
            {
                "chunk_id": (getattr(item, "node", item).metadata or {}).get("chunk_id"),
                "score": getattr(item, "score", None),
            }
            for item in nodes
        ],
    )
    return {
        "retrieved_nodes": nodes
    }

# evaluator node
async def evaluator_node(state: AgentState):
    evaluated_result = await _get_evaluator().evaluate(
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

# agentic rag
async def increment_research_round_node(state: AgentState):
    retrieval_round = state.get("retrieval_round", 0) + 1
    logger.info(
        "increment_research_round_node completed: retrieval_round=%s",
        retrieval_round,
    )
    return {
        "retrieval_round": retrieval_round
    }

# refine query node
async def refine_analysis_node(state: AgentState):
    evaluation = state["retrieval_evaluated_result"]
    refined_analysis = await _get_analyzer().refine(
        query=state["user_query"],
        current_analysis=state["analysis"],
        missing_information=evaluation.missing_information,
        limitations=[evaluation.reason],
    )
    logger.info(
        "refine_analysis_node completed: analysis=%s",
        refined_analysis.model_dump_json(),
    )
    return {
        "analysis": refined_analysis,
    }

# external search on arxiv
async def external_search_node(state: AgentState):
    matcher = _get_matcher()
    evaluation = state["retrieval_evaluated_result"]
    targets = [
        target.strip()
        for target in (evaluation.missing_papers or state["analysis"].paper_names)
        if target and target.strip()
    ]
    if not targets:
        targets = [state["user_query"]]

    results = []
    resolutions = []
    errors = []
    ingested_paper_ids = list(state.get("ingested_paper_ids", []))
    ingested_names = {name.casefold() for name in ingested_paper_ids}
    is_named_paper_search = bool(
        evaluation.missing_papers or state["analysis"].paper_names
    )
    for target in targets:
        if target.casefold() in ingested_names:
            continue
        try:
            if is_named_paper_search:
                match = await matcher.match(target_paper=target)
                results.append(match)
                if match.status != MatchStatus.MATCHED:
                    resolutions.append(
                        PaperResolution(
                            paper_name=target,
                            status="not_found",
                            error=match.reason,
                        )
                    )
                    errors.append(f"{target}: {match.reason}")
                    continue

                download_dir = Path(setting.pdf_save_dir).resolve()
                existing_paths = {
                    path.resolve() for path in download_dir.rglob("*.pdf")
                }
                download_result = await matcher.download(match)
                pdf_path = _find_pdf_path(download_result, existing_paths)
                paper_id = target.strip().lower()
                chunks_indexed = await asyncio.to_thread(
                    _parse_and_index_pdf,
                    pdf_path,
                    paper_id,
                    match.title or target,
                )
                ingested_paper_ids.append(target)
                ingested_names.add(target.casefold())
                resolutions.append(
                    PaperResolution(
                        paper_name=target,
                        paper_id=match.paper_id,
                        title=match.title,
                        source=match.source if match.source in {"arxiv", "zotero"} else None,
                        status="resolved",
                        pdf_path=str(pdf_path),
                        chunks_indexed=chunks_indexed,
                    )
                )
            else:
                results.append(
                    await matcher.paper_client.search(
                        query=target,
                        max_results_per_source=5,
                    )
                )
        except Exception as exc:
            logger.exception("external_search_node failed for target=%s", target)
            errors.append(f"{target}: {type(exc).__name__}: {exc}")
            resolutions.append(
                PaperResolution(
                    paper_name=target,
                    status="processing_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    logger.info(
        "external_search_node completed: targets=%s matches=%s resolutions=%s errors=%s",
        targets,
        [
            {
                "paper_id": getattr(result, "paper_id", None),
                "title": getattr(result, "title", None),
                "status": getattr(getattr(result, "status", None), "value", None),
            }
            for result in results
        ],
        [resolution.model_dump(mode="json") for resolution in resolutions],
        errors,
    )
    return {
        "external_search_results": results,
        "external_search_errors": errors,
        "paper_resolutions": resolutions,
        "ingested_paper_ids": ingested_paper_ids,
        "knowledge_updated": any(
            resolution.status == "resolved" for resolution in resolutions
        ),
        "external_search_round": state.get("external_search_round", 0) + 1,
    }

# writer output node
async def writer_node(state: AgentState):
    answer = await _get_writer().write(
        query=state["user_query"],
        analysis=state["analysis"],
        evaluation=state["retrieval_evaluated_result"],
        retrieved_nodes=state["retrieved_nodes"],
        external_search_results=state.get("external_search_results", []),
        external_search_errors=state.get("external_search_errors", []),
    )
    logger.info("writer_node completed")
    return {"answer": answer}

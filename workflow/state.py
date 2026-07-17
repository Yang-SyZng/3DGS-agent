from __future__ import annotations

from typing import TypedDict, Literal
from pydantic import BaseModel
from schema.analyzer_schema import QueryAnalysis
from schema.evaluator_schema import RetrievalEvaluation
from schema.researcher_schema import ResearchResult

from typing_extensions import NotRequired

class AgentState(TypedDict):
    # 输入
    user_query: str
    # analysis: QueryAnalysis
    analysis: QueryAnalysis

    # 检索
    retrieved_nodes: NotRequired[list]
    retrieval_evaluated_result: NotRequired[RetrievalEvaluation]

    # Research
    research_result: NotRequired[ResearchResult]


    # 外部获取
    # resolved_papers: list[PaperResolution]
    # pdf_paths: list[str]
    # ingested_paper_ids: list[str]

    # 流程控制
    ## 查询索引次数
    retrieval_round: int = 0
    # arxiv_retry_count: int
    # errors: list[str]

    # 输出
    answer: NotRequired[str]

class PaperResolution(BaseModel):
    paper_name: str
    source: Literal["zotero", "arxiv"] | None = None
    status: Literal[
        "pending",
        "resolved",
        "not_found",
        "processing_failed"
    ]
    pdf_path: str | None = None
    error: str | None = None

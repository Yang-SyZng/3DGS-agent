from pydantic import BaseModel, Field
from enum import Enum

class QueryType(str, Enum):
    # 针对一篇论文
    SINGLE_PAPER = "single_paper"
    # 针对多篇论文
    MULTI_PAPER = "multi_paper"
    # 一般搜索
    GENERAL_SEARCH = "general_search"

class QueryTarget(str, Enum):
    METHOD = "method"
    EXPERIMENT = "experiment"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    BACKGROUND = "background"
    OTHER = "other"

class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    CONCLUSION = "conclusion"
    REFERENCE = "reference"
    SUPPLEMENTARY = "supplementary"

class QueryAnalysis(BaseModel):
    original_query: str = Field(
        description="用户的原始问题，必须原样保留"
    )

    query_type: QueryType = Field(
        description="用户问题的任务类型"
    )

    targets: list[QueryTarget] = Field(
        min_length=1,
        description="用户关注的一个或多个信息维度"
    )

    paper_names: list[str] = Field(
        description="用户明确提到的论文名称"
    )

    entities: list[str] = Field(
        min_length=3,
        description="方法、模型、数据集、指标和技术等关键实体"
    )

    keywords: list[str] = Field(
        min_length=3,
        max_length=8,
        description="用于检索的英文关键词"
    )

    section_types: list[SectionType] = Field(
        min_length=1,
        description="建议优先检索的语义章节类型"
    )

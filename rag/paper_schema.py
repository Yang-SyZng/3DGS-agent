from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass, field
from enum import Enum

class PaperInfo(BaseModel):

    paper_id:str

    title:str

    authors:list[str]

    year:int

    pdf_path:str

    abstract:str

@dataclass
class SectionNode:
    section_id: str | None = None

    path: str | None = None

    # 文档结构信息
    level: int | None = None

    # 语义分类
    semantic_type: Optional[str] | None = None

    title: str | None = None

    content: str | None = None

    children: List[SectionNode] | None = field(default_factory=list)

class SectionType(Enum):

    ABSTRACT="abstract"

    KEYWORDS="keywords"

    INTRODUCTION="introduction"

    BACKGROUND="background"

    RELATED_WORK="related_work"

    METHOD="method"

    EXPERIMENT="experiment"

    IMPLEMENTATION="implementation"

    EXPERIMENT_SETUP="experiment_setup"

    RESULTS="results"

    ABLATION="ablation"

    CONCLUSION="conclusion"

    REFERENCE="reference"

    APPENDIX="appendix"

    OTHER="other"

SECTION_KEYWORDS = {
    SectionType.ABSTRACT: [
        "abstract"
    ],
    SectionType.KEYWORDS: [
        "keyword",
        "keywords"
    ],
    SectionType.INTRODUCTION: [
        "introduction",
        "overview"
    ],
    SectionType.RELATED_WORK:[
        "related work",
        "related works",
        "background",
        "preliminary"
    ],
    SectionType.METHOD:[
        "method",
        "methodology",
        "approach",
        "framework",
        "preliminary"
    ],
    SectionType.EXPERIMENT:[
        "experiments",
        "experiment",
        "evaluation",
        "result",
        "benchmark"
    ],
    SectionType.CONCLUSION:[
        "conclusion",
        "discussion",
        "future work"
    ],
    SectionType.REFERENCE:[
        "reference",
        "references",
        "bibliography"
    ],
    SectionType.APPENDIX:[
        "appendix",
        "supplementary materials",
        "additional results"
    ],

}
CHILD_KEYWORDS = {
    SectionType.BACKGROUND:[
        "preliminary",
        "background",
        "overview",
        "notation",
        "problem formulation"
    ],
    SectionType.EXPERIMENT:[
        "dataset",
        "evaluation",
        "benchmark",
        "ablation",
        "comparison",
        "implementation details"
    ]
}
class PaperKnowledge(BaseModel):

    problem:str

    contributions:list[str]

    methods:list[str]

    datasets:list[str]

    metrics:dict

    limitations:list[str]

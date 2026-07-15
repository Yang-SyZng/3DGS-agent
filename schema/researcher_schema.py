from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str = Field(
        description="当前证据在本次研究结果中的唯一标识"
    )

    paper_id: str | None = Field(
        description="证据所属论文 ID"
    )

    paper_title: str | None = Field(
        description="证据所属论文标题"
    )

    chunk_id: str | None = Field(
        description="证据对应的检索 chunk ID"
    )

    section_path: str | None = Field(
        description="证据所在章节路径，例如 Method/EchoNet"
    )

    supporting_text: str = Field(
        description="支持研究结论的原始证据文本"
    )


class Finding(BaseModel):
    topic: str = Field(
        description="当前研究结论的主题，例如 architecture 或 contribution"
    )

    conclusion: str = Field(
        description="基于检索证据得到的研究结论"
    )

    evidence_ids: list[str] = Field(
        description="支持该结论的 Evidence ID"
    )


class ResearchResult(BaseModel):
    findings: list[Finding] = Field(
        description="针对用户问题整理出的主要研究结论"
    )

    evidence: list[Evidence] = Field(
        description="支撑研究结论的论文证据"
    )

    limitations: list[str] = Field(
        description="当前证据无法确定、缺失或存在歧义的信息"
    )
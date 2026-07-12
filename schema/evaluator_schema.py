from pydantic import BaseModel, Field


class RetrievalEvaluation(BaseModel):
    original_query: str = Field(
        description="用户的原始问题，必须原样保留"
    )

    sufficient: bool = Field(
        description="召回证据是否足以回答用户问题"
    )

    missing_papers: list[str] = Field(
        default_factory=list,
        description="知识库中缺失或未成功召回的论文"
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="回答问题仍然缺少的信息"
    )

    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="与问题相关并可用于回答的chunk ID"
    )

    reason: str = Field(
        description="判断充分或不充分的简短原因"
    )


from pydantic import BaseModel

class Paper(BaseModel):
    # 基本信息
    paper_id: str

    title: str

    authors: list[str]

    year: int

    pdf_path: str

    # 文章信息
    abstract: str

    introduction: str

    related_work: str

    methodology: list

    experiment: dict

    conclusion: str

    supplement_material: str

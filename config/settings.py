from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None

    # model config
    LLM_MODEL_ID: str | None = None
    EMBEDDING_MODEL_ID: str = ""
    OCR_MODEL_ID: str | None = None

    # DATA save config
    mainly_save_dir: Path = Path("./database")

    # ChromaDB config
    chroma_save_dir: Path = Path("./database/Chroma")
    chroma_collection_name: str = "arxiv_papers"
    
    # pdf save config
    pdf_save_dir: Path = Path("./database/Papers")

    # pdf process save config
    pdf_process_save_dir: Path = Path("./database/pdf_ocr_results")

    # rag config
    chunk_size: int = 3000
    chunk_overlap: int = 500


    # api config
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True


    # log config
    log_level: str = "INFO"


    # arxiv api config
    rate_limit: float = 3.0
    max_retries: int = 3


    def __init__(self):
        super().__init__()
        self.chroma_save_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_save_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_process_save_dir.mkdir(parents=True, exist_ok=True)
        (self.mainly_save_dir / "Cache").mkdir(parents=True, exist_ok=True)

setting = Settings()

if __name__ == "__main__":
    settings = Settings()

    

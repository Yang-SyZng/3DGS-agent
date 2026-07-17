from pathlib import Path
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    API_KEY: str | None = None
    BASE_URL: str | None = None

    ZoteroID: str | None = None
    ZoteroKeys: str | None = None
    
    TAVILY_API_KEY: str | None = None

    WEBDAV_HOSTNAME: str | None = None
    WEBDAV_LOGIN: str | None = None
    WEBDAV_PASSWD: str | None = None

    # model config
    LLM_MODEL_ID: str | None = None
    EMBEDDING_MODEL_ID: str | None = None
    EMBEDDING_DIM: int | None = None
    OCR_MODEL_ID: str | None = None

    # Local LLM Server
    Globle_Local_Optional: bool = True
    Local_Model: Dict | None = {
        "LLM": {
            "gpu": "1",
            "port": 11500,
            "model": "qwen3.6:27b",
            "model_dir": "/usr/share/ollama/.ollama/models",
            "num_parallel": "2", # 控制并发请求
            "context_length": "64000", # 上下文长度
        },
        # "Embedding": {

        # },
        # "Rerank": {

        # }
    }


    # DATA save config
    mainly_save_dir: Path = Path("./database")

    # Milvus config
    Milvus_directory: Path = Path("./database/Milvus")
    Milvus_db_directory: Path = Milvus_directory / "database.db"
    Milvus_collection_name: str = "arxiv_papers"
    
    # pdf save config
    pdf_save_dir: Path = Path("./database/Papers")

    # pdf process save config
    pdf_parser_save_dir: Path = Path("./database/parser")

    # rag config
    chunk_size: int = 300
    chunk_overlap: int = 50


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
        self.Milvus_directory.mkdir(parents=True, exist_ok=True)
        self.pdf_save_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_parser_save_dir.mkdir(parents=True, exist_ok=True)
        (self.mainly_save_dir / "Cache").mkdir(parents=True, exist_ok=True)

setting = Settings()

if __name__ == "__main__":
    settings = Settings()

    

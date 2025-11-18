import os
from typing import Optional, List
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ------------------------------
    # Core Application
    # ------------------------------
    app_name: str = "RAG Microservice"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)
    host: str = "0.0.0.0"
    port: int = 8000

    # ------------------------------
    # Database
    # ------------------------------
    postgres_user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "rag-password"))
    postgres_db: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "ragdb"))

    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", ""),
        description="Database connection URL - REQUIRED for production"
    )
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_command_timeout: int = 60

    # ------------------------------
    # Redis
    # ------------------------------
    redis_url: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_URL", None))
    redis_max_connections: int = 10

    # ------------------------------
    # OpenAI
    # ------------------------------
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    openai_embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_assistant_id: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_ASSISTANT_ID", None))

    # ------------------------------
    # RAG Parameters
    # ------------------------------
    complex_query_keywords: List[str] = [
        "analyze", "compare", "evaluate", "explain why", "logic",
        "step by step", "calculate", "prove", "solve", "optimize",
        "strategy", "complex", "detailed analysis", "comprehensive",
        "elaborate", "justify"
    ]
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    rag_max_tokens: int = 4000
    rag_response_length: str = Field(default_factory=lambda: os.getenv("RAG_RESPONSE_LENGTH", "normal"))

    # ------------------------------
    # Cache / Misc
    # ------------------------------
    cache_ttl_seconds: int = 300
    cache_max_query_length: int = 1000
    cors_origins: str = Field(default="http://localhost:3001,http://localhost:3002,http://localhost")
    rate_limit_requests: int = 10
    rate_limit_window: str = "1/minute"
    websocket_max_connections: int = 1000
    websocket_ping_interval: int = 30
    enable_metrics: bool = True
    log_level: str = "INFO"

    max_request_size_mb: int = 100
    allowed_file_types: str = "pdf,txt,docx,md"

    # ------------------------------
    # Text Processing
    # ------------------------------
    max_text_length: int = 8000
    pdf_chunk_size: int = 1000
    pdf_chunk_overlap: int = 200

    # ------------------------------
    # Flags
    # ------------------------------
    test_mode: bool = False
    disable_database_init: bool = False
    disable_cache_init: bool = False

    # ------------------------------
    # Validators (Pydantic v2)
    # ------------------------------
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v):
        return v if v and v.strip() else None

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v, info: ValidationInfo):
        debug = info.data.get("debug", False)
        if not v.strip():
            if debug:
                print("⚠️  OPENAI_API_KEY not set. Running in debug mode.")
                return ""
            raise ValueError("OPENAI_API_KEY is required.")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v, info: ValidationInfo):
        debug = info.data.get("debug", False)
        if not v.strip():
            if debug:
                print("⚠️  DATABASE_URL not set. Using SQLite for dev.")
                return "sqlite:///./dev.db"
            raise ValueError("DATABASE_URL is required.")
        if v.startswith("sqlite://"):
            return v
        if not (v.startswith("postgresql://") or v.startswith("postgres://")):
            raise ValueError("Only PostgreSQL or SQLite URLs are supported.")
        return v

    @field_validator("max_text_length")
    @classmethod
    def validate_max_text_length(cls, v):
        if not (1000 <= v <= 50000):
            raise ValueError("max_text_length must be between 1000 and 50000.")
        return v

    @field_validator("pdf_chunk_size")
    @classmethod
    def validate_pdf_chunk_size(cls, v):
        if not (100 <= v <= 10000):
            raise ValueError("pdf_chunk_size must be between 100 and 10000.")
        return v

    # ------------------------------
    # Pydantic Settings Config
    # ------------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # allows unused env vars (like POSTGRES_USER, etc.)
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

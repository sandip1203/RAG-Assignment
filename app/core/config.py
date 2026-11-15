from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG_APP"
    app_env: Literal["development", "production", "testing"] = "development"
    api_v1_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql+asyncpg://palm_user:palm_password@localhost:5432/palm_mind_ai"
    )

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl: int = Field(default=86400, gt=0)
    chat_history_messages: int = Field(default=20, gt=0)

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, gt=0)

    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "palm-mind-documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    sentence_overlap: int = Field(default=1, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    max_upload_size_bytes: int = Field(default=20 * 1024 * 1024, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

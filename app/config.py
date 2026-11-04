import json
from functools import lru_cache
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "enterprise_docs"
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.40
    API_KEYS: List[str] = ["secret-api-key-1"]
    MAX_UPLOAD_SIZE_MB: int = 10

    @field_validator("API_KEYS", mode="before")
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [k.strip() for k in v.split(",") if k.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
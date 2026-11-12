"""
backend/core/config.py
Central configuration loaded from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # PostgreSQL / SQLite Relational Database (defaults to a local SQLite file for serverless execution)
    database_url: str = "sqlite+aiosqlite:///./data/edugraphai.db"

    # Knowledge Graph Fallback (controls if the app runs SQLite-based graph if Neo4j is unavailable)
    use_local_graph_fallback: bool = True
    local_graph_db_path: str = "./data/knowledge_graph.db"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    backend_url: str = "http://localhost:8000"
    # Comma-separated list of origins allowed to call the API (e.g. your Streamlit app URL)
    allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    max_upload_size_mb: int = 50
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 6

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()

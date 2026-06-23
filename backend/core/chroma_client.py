"""
backend/core/chroma_client.py
Persistent ChromaDB client with collection accessors.
"""
import sys
from unittest.mock import MagicMock

# Mock onnxruntime to bypass Windows DLL load failures (we use Gemini API for embeddings, so we don't need onnxruntime)
sys.modules['onnxruntime'] = MagicMock()

import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
_client = None  # chromadb.PersistentClient

COLLECTIONS = {
    "chunks": "edu_chunks",
    "formulas": "edu_formulas",
    "diagrams": "edu_diagrams",
}


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised", path=settings.chroma_persist_dir)
    return _client


def get_collection(name: str) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_chunks_collection() -> chromadb.Collection:
    return get_collection(COLLECTIONS["chunks"])


def get_formulas_collection() -> chromadb.Collection:
    return get_collection(COLLECTIONS["formulas"])


def get_diagrams_collection() -> chromadb.Collection:
    return get_collection(COLLECTIONS["diagrams"])

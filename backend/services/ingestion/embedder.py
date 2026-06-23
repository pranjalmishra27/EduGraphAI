"""
backend/services/ingestion/embedder.py
Embeds text chunks and stores them in ChromaDB collections.
"""
from __future__ import annotations
import asyncio
from backend.core.chroma_client import get_chunks_collection, get_formulas_collection, get_diagrams_collection
from backend.core.gemini_client import embed_texts
from backend.core.logging import get_logger
from backend.services.ingestion.chunker import TextChunk

logger = get_logger(__name__)

BATCH_SIZE = 50  # Gemini embedding rate limit


class EmbeddingService:

    async def embed_and_store_chunks(self, chunks: list[TextChunk]) -> int:
        """Embed chunks and upsert into ChromaDB chunks collection."""
        if not chunks:
            return 0

        collection = get_chunks_collection()
        stored = 0

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            texts = [c.content for c in batch]
            ids = [c.id for c in batch]
            metadatas = [c.metadata for c in batch]

            try:
                embeddings = await embed_texts(texts)
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                stored += len(batch)
                logger.info("Embedded chunk batch", batch_size=len(batch), total_stored=stored)
            except Exception as e:
                logger.error("Embedding batch failed", error=str(e), batch_start=batch_start)
                raise

            # Small delay between batches to respect rate limits
            if batch_start + BATCH_SIZE < len(chunks):
                await asyncio.sleep(0.5)

        return stored

    async def embed_and_store_formulas(self, formulas: list[dict], document_id: str, filename: str) -> int:
        """Embed and store formulas in ChromaDB formula collection."""
        if not formulas:
            return 0

        collection = get_formulas_collection()
        texts = []
        ids = []
        metadatas = []

        for i, formula in enumerate(formulas):
            formula_text = f"{formula.get('name', '')} {formula.get('explanation', '')} {formula.get('latex', '')}"
            formula_id = f"{document_id}_formula_{i}"
            texts.append(formula_text)
            ids.append(formula_id)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "name": formula.get("name", ""),
                "latex": formula.get("latex", ""),
                "explanation": formula.get("explanation", ""),
                "related_concepts": ", ".join(formula.get("related_concepts", [])),
            })

        embeddings = await embed_texts(texts)
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Stored formulas", count=len(formulas), document_id=document_id)
        return len(formulas)

    async def embed_and_store_diagrams(self, diagrams: list[dict], document_id: str, filename: str) -> int:
        """Embed and store diagram descriptions in ChromaDB."""
        if not diagrams:
            return 0

        collection = get_diagrams_collection()
        texts = []
        ids = []
        metadatas = []

        for i, diagram in enumerate(diagrams):
            diagram_text = f"{diagram.get('caption', '')} {diagram.get('description', '')}"
            diagram_id = f"{document_id}_diagram_{i}"
            texts.append(diagram_text)
            ids.append(diagram_id)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "caption": diagram.get("caption", ""),
                "chapter": diagram.get("chapter", ""),
                "page_number": str(diagram.get("page_number", 0)),
            })

        embeddings = await embed_texts(texts)
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Stored diagrams", count=len(diagrams), document_id=document_id)
        return len(diagrams)

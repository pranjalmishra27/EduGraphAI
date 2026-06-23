"""
backend/services/ingestion/ingestion_service.py
Orchestrates the full document ingestion pipeline.
"""
from __future__ import annotations
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.logging import get_logger
from backend.models.schemas import Document, Chunk
from backend.services.ingestion.document_parser import DocumentParser
from backend.services.ingestion.chunker import DocumentChunker
from backend.services.ingestion.embedder import EmbeddingService
from backend.services.ingestion.knowledge_graph_builder import KnowledgeGraphBuilder

logger = get_logger(__name__)


class IngestionService:

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingService()
        self.kg_builder = KnowledgeGraphBuilder()

    async def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        subject: str | None,
        db: AsyncSession,
    ) -> Document:
        document_id = str(uuid4())
        logger.info("Starting ingestion", document_id=document_id, filename=filename)

        # 1. Create DB record (status = processing)
        doc_record = Document(
            id=document_id,
            filename=filename,
            doc_type=_infer_doc_type(filename),
            subject=subject or "General",
            status="processing",
        )
        db.add(doc_record)
        await db.flush()

        try:
            # 2. Parse document
            parsed_doc = await self.parser.parse(file_bytes, filename)
            if subject:
                parsed_doc.subject = subject
            doc_record.subject = parsed_doc.subject
            doc_record.page_count = parsed_doc.total_pages

            # 3. Chunk
            chunks = self.chunker.chunk_document(parsed_doc, document_id)
            doc_record.chunk_count = len(chunks)

            # 4. Embed & store in ChromaDB
            await self.embedder.embed_and_store_chunks(chunks)
            await self.embedder.embed_and_store_formulas(parsed_doc.formulas, document_id, filename)
            await self.embedder.embed_and_store_diagrams(parsed_doc.diagrams, document_id, filename)

            # 5. Store chunk records in PostgreSQL
            for chunk in chunks:
                db.add(Chunk(
                    id=chunk.id,
                    document_id=document_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chapter=chunk.chapter,
                    chunk_index=chunk.chunk_index,
                    embedding_id=chunk.id,
                ))

            # 6. Build knowledge graph
            kg_stats = await self.kg_builder.build_from_document(parsed_doc, document_id)
            doc_record.meta = {
                "formula_count": len(parsed_doc.formulas),
                "diagram_count": len(parsed_doc.diagrams),
                "kg_concepts": kg_stats["concepts"],
                "kg_relationships": kg_stats["relationships"],
            }

            doc_record.status = "ready"
            await db.flush()

            logger.info("Ingestion complete",
                        document_id=document_id,
                        chunks=len(chunks),
                        formulas=len(parsed_doc.formulas),
                        diagrams=len(parsed_doc.diagrams))

        except Exception as e:
            doc_record.status = "error"
            doc_record.meta = {"error": str(e)}
            await db.flush()
            logger.error("Ingestion failed", document_id=document_id, error=str(e))
            raise

        return doc_record

    async def delete_document(self, document_id: str, db: AsyncSession) -> bool:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return False
        await db.delete(doc)
        # ChromaDB cleanup
        try:
            from backend.core.chroma_client import get_chunks_collection, get_formulas_collection, get_diagrams_collection
            for collection in [get_chunks_collection(), get_formulas_collection(), get_diagrams_collection()]:
                existing = collection.get(where={"document_id": document_id})
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
        except Exception as e:
            logger.warning("ChromaDB cleanup partial failure", error=str(e))
        return True


def _infer_doc_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    mapping = {"pdf": "pdf", "pptx": "pptx", "ppt": "pptx",
                "jpg": "image", "jpeg": "image", "png": "image",
                "txt": "text", "md": "text"}
    return mapping.get(ext, "unknown")

"""
backend/services/ingestion/chunker.py
Semantic chunking with metadata preservation for RAG.
"""
from __future__ import annotations
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import get_settings
from backend.services.ingestion.document_parser import ParsedDocument, ParsedPage


@dataclass
class TextChunk:
    id: str
    content: str
    document_id: str
    document_filename: str
    page_number: int | None
    chapter: str | None
    chunk_index: int
    subject: str
    metadata: dict


class DocumentChunker:

    def __init__(self):
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def chunk_document(self, parsed_doc: ParsedDocument, document_id: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in parsed_doc.pages:
            page_chunks = self._chunk_page(page, parsed_doc, document_id, chunk_index)
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return chunks

    def _chunk_page(
        self,
        page: ParsedPage,
        doc: ParsedDocument,
        document_id: str,
        start_index: int,
    ) -> list[TextChunk]:
        if not page.text.strip():
            return []

        raw_chunks = self.splitter.split_text(page.text)
        result = []

        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            # Prepend chapter context to help retrieval
            context_prefix = ""
            if page.chapter:
                context_prefix = f"[Chapter: {page.chapter}] "
            if doc.subject and doc.subject != "General":
                context_prefix = f"[Subject: {doc.subject}] " + context_prefix

            enriched_content = context_prefix + chunk_text

            chunk_id = f"{document_id}_p{page.page_number}_c{start_index + i}"

            result.append(TextChunk(
                id=chunk_id,
                content=enriched_content,
                document_id=document_id,
                document_filename=doc.filename,
                page_number=page.page_number,
                chapter=page.chapter,
                chunk_index=start_index + i,
                subject=doc.subject,
                metadata={
                    "document_id": document_id,
                    "filename": doc.filename,
                    "page_number": page.page_number,
                    "chapter": page.chapter or "",
                    "subject": doc.subject,
                    "chunk_index": start_index + i,
                    "has_equations": bool(page.raw_equations),
                    "has_figures": bool(page.figure_descriptions),
                },
            ))

        return result

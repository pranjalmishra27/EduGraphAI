"""
tests/test_chunker.py
Unit tests for the document chunking pipeline.
"""
import pytest
from backend.services.ingestion.chunker import DocumentChunker, TextChunk
from backend.services.ingestion.document_parser import ParsedDocument, ParsedPage


def make_parsed_doc(text: str, pages: int = 1) -> ParsedDocument:
    doc = ParsedDocument(filename="test.pdf", doc_type="pdf", subject="Physics")
    doc.total_pages = pages
    doc.pages = [
        ParsedPage(page_number=i + 1, text=text, chapter=f"Chapter {i+1}")
        for i in range(pages)
    ]
    return doc


class TestDocumentChunker:

    def test_basic_chunking(self):
        chunker = DocumentChunker()
        text = " ".join(["word"] * 400)
        doc = make_parsed_doc(text)
        chunks = chunker.chunk_document(doc, "doc_001")
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, TextChunk)
            assert c.document_id == "doc_001"
            assert len(c.content) > 0

    def test_chunk_contains_context_prefix(self):
        chunker = DocumentChunker()
        doc = make_parsed_doc("Newton's laws describe the relationship between forces and motion.")
        chunks = chunker.chunk_document(doc, "doc_002")
        assert any("Physics" in c.content or "Chapter" in c.content for c in chunks)

    def test_empty_page_skipped(self):
        chunker = DocumentChunker()
        doc = ParsedDocument(filename="empty.pdf", doc_type="pdf", subject="Math")
        doc.pages = [ParsedPage(page_number=1, text="   \n  ")]
        doc.total_pages = 1
        chunks = chunker.chunk_document(doc, "doc_003")
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        chunker = DocumentChunker()
        doc = make_parsed_doc("This is a test sentence for chunking. " * 30)
        doc.pages[0].chapter = "Kinematics"
        chunks = chunker.chunk_document(doc, "doc_004")
        for c in chunks:
            assert c.metadata["document_id"] == "doc_004"
            assert c.metadata["subject"] == "Physics"
            assert "page_number" in c.metadata

    def test_multipage_indices_are_sequential(self):
        chunker = DocumentChunker()
        text = "This is page content. " * 60
        doc = make_parsed_doc(text, pages=3)
        chunks = chunker.chunk_document(doc, "doc_005")
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # No duplicates

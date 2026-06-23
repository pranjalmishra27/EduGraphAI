"""
tests/test_rag_pipeline.py
Unit tests for RAG pipeline context formatting helpers.
"""
import pytest
from backend.services.retrieval.rag_pipeline import RAGPipeline
from backend.models.schemas import SourceChunk, FormulaRef, DiagramRef


class TestRAGPipelineFormatters:

    def setup_method(self):
        self.pipeline = RAGPipeline()

    def _make_chunk(self, content="Some content", page=1, chapter="Ch1",
                    filename="test.pdf", score=0.8) -> SourceChunk:
        return SourceChunk(
            content=content,
            page_number=page,
            chapter=chapter,
            document_filename=filename,
            relevance_score=score,
        )

    def test_format_chunks_basic(self):
        chunks = [self._make_chunk("Newton's Second Law F=ma", page=5, chapter="Forces")]
        result = self.pipeline._format_chunks(chunks)
        assert "Newton's Second Law" in result
        assert "test.pdf" in result
        assert "Forces" in result
        assert "Page 5" in result

    def test_format_chunks_empty(self):
        assert self.pipeline._format_chunks([]) == ""

    def test_format_chunks_multiple(self):
        chunks = [self._make_chunk(f"Content {i}", page=i) for i in range(3)]
        result = self.pipeline._format_chunks(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_format_formulas(self):
        formulas = [
            FormulaRef(
                name="Newton's Second Law",
                latex="F = ma",
                explanation="Force equals mass times acceleration",
                related_concepts=["momentum", "kinematics"],
            )
        ]
        result = self.pipeline._format_formulas(formulas)
        assert "Newton's Second Law" in result
        assert "F = ma" in result
        assert "Force equals mass" in result

    def test_format_formulas_empty(self):
        assert self.pipeline._format_formulas([]) == ""

    def test_format_diagrams(self):
        diagrams = [
            DiagramRef(
                caption="Free Body Diagram",
                chapter="Forces",
                page_number=12,
                document_filename="physics.pdf",
            )
        ]
        result = self.pipeline._format_diagrams(diagrams)
        assert "Free Body Diagram" in result
        assert "Page 12" in result

    def test_compute_confidence_no_chunks(self):
        import asyncio
        score = asyncio.get_event_loop().run_until_complete(
            self.pipeline._compute_confidence("any question", [])
        )
        assert score == 0.1

    def test_compute_confidence_high_relevance(self):
        import asyncio
        chunks = [self._make_chunk(score=0.9) for _ in range(3)]
        score = asyncio.get_event_loop().run_until_complete(
            self.pipeline._compute_confidence("any question", chunks)
        )
        assert score > 0.5
        assert score <= 1.0

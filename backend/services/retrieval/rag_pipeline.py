"""
backend/services/retrieval/rag_pipeline.py
Assembles context from retrieval results and generates answers via Gemini.
"""
from __future__ import annotations
from backend.core.gemini_client import generate_text
from backend.core.logging import get_logger
from backend.models.schemas import QuestionResponse, SourceChunk
from backend.services.retrieval.retriever import HybridRetriever, RetrievalResult

logger = get_logger(__name__)

ANSWER_SYSTEM_PROMPT = """You are EduGraphAI, an expert educational AI assistant.
Your goal is to provide clear, accurate, and well-structured answers based on the 
retrieved educational content. Always:
- Explain concepts step by step
- Reference specific sources when available
- Use LaTeX notation for equations (e.g., $F = ma$, $E = mc^2$)
- Connect concepts to prerequisites when relevant
- Be encouraging and pedagogically sound
"""

ANSWER_PROMPT_TEMPLATE = """
STUDENT QUESTION: {question}

RETRIEVED CONTENT FROM UPLOADED DOCUMENTS:
{context_chunks}

RELEVANT FORMULAS:
{formula_context}

KNOWLEDGE GRAPH CONTEXT (related concepts and their connections):
{kg_context}

RELEVANT DIAGRAMS:
{diagram_context}

Based on the above retrieved content, provide a comprehensive, well-structured answer.
If the content doesn't contain enough information, say so clearly and provide what you can.

Include:
1. Direct answer to the question
2. Step-by-step explanation where applicable
3. Mention of relevant formulas (in LaTeX)
4. Reference to which document/chapter the information comes from
5. Related concepts the student should explore next
"""

CONFIDENCE_PROMPT = """
Given this question: {question}
And this retrieved context (relevance scores: {scores}):

Rate the confidence of the answer from 0.0 to 1.0 based on:
- How relevant the retrieved content is to the question
- How complete the answer can be from this content
- Whether the question is directly addressed

Reply with only a float number between 0.0 and 1.0.
"""


class RAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()

    async def answer(
        self,
        question: str,
        subject: str | None = None,
        include_diagrams: bool = True,
        include_formulas: bool = True,
    ) -> QuestionResponse:
        logger.info("RAG query", question=question[:80], subject=subject)

        # Retrieve
        retrieval: RetrievalResult = await self.retriever.retrieve(
            query=question,
            subject=subject,
            include_formulas=include_formulas,
            include_diagrams=include_diagrams,
        )

        # Build context string
        context_chunks = self._format_chunks(retrieval.chunks)
        formula_context = self._format_formulas(retrieval.formulas)
        kg_context = self._format_kg(retrieval.kg_context)
        diagram_context = self._format_diagrams(retrieval.diagrams)

        # Generate answer
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            context_chunks=context_chunks or "No directly relevant content found.",
            formula_context=formula_context or "No formulas retrieved.",
            kg_context=kg_context or "No knowledge graph context available.",
            diagram_context=diagram_context or "No diagrams retrieved.",
        )

        answer_text = await generate_text(prompt, system_instruction=ANSWER_SYSTEM_PROMPT)

        # Compute confidence score
        confidence = await self._compute_confidence(question, retrieval.chunks)

        return QuestionResponse(
            answer=answer_text,
            sources=retrieval.chunks,
            formulas=retrieval.formulas,
            diagrams=retrieval.diagrams,
            knowledge_graph_context=retrieval.kg_context,
            confidence_score=confidence,
        )

    def _format_chunks(self, chunks: list[SourceChunk]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f"{chunk.document_filename}"
            if chunk.chapter:
                source += f" | {chunk.chapter}"
            if chunk.page_number:
                source += f" | Page {chunk.page_number}"
            parts.append(f"[{i}] Source: {source}\n{chunk.content}")
        return "\n\n".join(parts)

    def _format_formulas(self, formulas) -> str:
        if not formulas:
            return ""
        return "\n".join(
            f"- {f.name}: ${f.latex}$ — {f.explanation}"
            for f in formulas
        )

    def _format_kg(self, kg_context: list[dict]) -> str:
        if not kg_context:
            return ""
        parts = []
        for record in kg_context[:5]:
            concept = record.get("concept", "")
            desc = record.get("description", "")
            connections = record.get("connections", [])
            conn_str = ", ".join(
                f"{c['target']} ({c['rel']})"
                for c in connections[:3]
                if c.get("target")
            )
            parts.append(f"Concept: {concept} — {desc}" + (f"\n  Related: {conn_str}" if conn_str else ""))
        return "\n".join(parts)

    def _format_diagrams(self, diagrams) -> str:
        if not diagrams:
            return ""
        return "\n".join(
            f"- {d.caption}" + (f" (Page {d.page_number}, {d.chapter})" if d.page_number else "")
            for d in diagrams
        )

    async def _compute_confidence(self, question: str, chunks: list[SourceChunk]) -> float:
        if not chunks:
            return 0.1
        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        # Scale: avg cosine similarity → confidence
        return round(min(avg_relevance * 1.2, 1.0), 2)

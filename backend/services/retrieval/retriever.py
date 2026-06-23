"""
backend/services/retrieval/retriever.py
Hybrid retriever combining dense (ChromaDB), sparse (BM25), and KG retrieval
with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from backend.core.chroma_client import (
    get_chunks_collection, get_formulas_collection, get_diagrams_collection
)
from backend.core.gemini_client import embed_text
from backend.core.neo4j_client import run_query
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.models.schemas import SourceChunk, FormulaRef, DiagramRef

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[SourceChunk] = field(default_factory=list)
    formulas: list[FormulaRef] = field(default_factory=list)
    diagrams: list[DiagramRef] = field(default_factory=list)
    kg_context: list[dict] = field(default_factory=list)


class HybridRetriever:

    def __init__(self):
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        subject: str | None = None,
        include_formulas: bool = True,
        include_diagrams: bool = True,
        top_k: int | None = None,
    ) -> RetrievalResult:
        k = top_k or self.settings.top_k_retrieval

        # Run all retrievals concurrently
        import asyncio
        dense_task = self._dense_retrieve(query, subject, k * 2)
        formula_task = self._retrieve_formulas(query, k) if include_formulas else asyncio.sleep(0)
        diagram_task = self._retrieve_diagrams(query, k) if include_diagrams else asyncio.sleep(0)
        kg_task = self._retrieve_kg_context(query)

        dense_results, formula_results, diagram_results, kg_results = await asyncio.gather(
            dense_task, formula_task, diagram_task, kg_task
        )

        # BM25 sparse retrieval over dense candidates
        sparse_results = self._sparse_retrieve(query, dense_results, k)

        # RRF fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k)

        return RetrievalResult(
            chunks=fused,
            formulas=formula_results if isinstance(formula_results, list) else [],
            diagrams=diagram_results if isinstance(diagram_results, list) else [],
            kg_context=kg_results if isinstance(kg_results, list) else [],
        )

    async def _dense_retrieve(self, query: str, subject: str | None, k: int) -> list[SourceChunk]:
        """ChromaDB cosine similarity search."""
        try:
            collection = get_chunks_collection()
            query_embedding = await embed_text(query)

            where_filter = {"subject": subject} if subject else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count() or 1),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(SourceChunk(
                    content=doc,
                    page_number=meta.get("page_number"),
                    chapter=meta.get("chapter") or None,
                    document_filename=meta.get("filename", ""),
                    relevance_score=round(1 - dist, 4),  # cosine similarity
                ))

            return chunks
        except Exception as e:
            logger.warning("Dense retrieval failed", error=str(e))
            return []

    def _sparse_retrieve(self, query: str, candidates: list[SourceChunk], k: int) -> list[SourceChunk]:
        """BM25 over the dense candidate set."""
        if not candidates:
            return []

        tokenized_corpus = [doc.content.lower().split() for doc in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = scored[:k]

        return [
            SourceChunk(
                content=c.content,
                page_number=c.page_number,
                chapter=c.chapter,
                document_filename=c.document_filename,
                relevance_score=round(float(s), 4),
            )
            for c, s in top
        ]

    def _reciprocal_rank_fusion(
        self,
        dense: list[SourceChunk],
        sparse: list[SourceChunk],
        k: int,
        rrf_k: int = 60,
    ) -> list[SourceChunk]:
        """Merge dense and sparse results using RRF scoring."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, SourceChunk] = {}

        for rank, chunk in enumerate(dense):
            key = chunk.content[:100]
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(sparse):
            key = chunk.content[:100]
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            chunk_map[key] = chunk

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for key in sorted_keys[:k]:
            chunk = chunk_map[key]
            result.append(SourceChunk(
                content=chunk.content,
                page_number=chunk.page_number,
                chapter=chunk.chapter,
                document_filename=chunk.document_filename,
                relevance_score=round(scores[key], 4),
            ))

        return result

    async def _retrieve_formulas(self, query: str, k: int) -> list[FormulaRef]:
        """Retrieve relevant formulas from ChromaDB."""
        try:
            collection = get_formulas_collection()
            if collection.count() == 0:
                return []
            query_embedding = await embed_text(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            formulas = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if 1 - dist > 0.5:  # Relevance threshold
                    formulas.append(FormulaRef(
                        name=meta.get("name", ""),
                        latex=meta.get("latex", ""),
                        explanation=meta.get("explanation", ""),
                        related_concepts=meta.get("related_concepts", "").split(", "),
                    ))
            return formulas
        except Exception as e:
            logger.warning("Formula retrieval failed", error=str(e))
            return []

    async def _retrieve_diagrams(self, query: str, k: int) -> list[DiagramRef]:
        """Retrieve relevant diagram references from ChromaDB."""
        try:
            collection = get_diagrams_collection()
            if collection.count() == 0:
                return []
            query_embedding = await embed_text(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            diagrams = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if 1 - dist > 0.45:
                    diagrams.append(DiagramRef(
                        caption=meta.get("caption", ""),
                        chapter=meta.get("chapter") or None,
                        page_number=int(meta.get("page_number", 0)) or None,
                        document_filename=meta.get("filename", ""),
                    ))
            return diagrams
        except Exception as e:
            logger.warning("Diagram retrieval failed", error=str(e))
            return []

    async def _retrieve_kg_context(self, query: str) -> list[dict]:
        """Extract concept mentions from query and retrieve KG subgraph."""
        try:
            # Find concepts in Neo4j that match query terms
            words = [w for w in query.lower().split() if len(w) > 3]
            if not words:
                return []

            cypher = """
            MATCH (c:Concept)
            WHERE ANY(word IN $words WHERE toLower(c.name) CONTAINS word)
            OPTIONAL MATCH (c)-[r]->(related:Concept)
            RETURN c.name as concept, c.description as description,
                   c.difficulty as difficulty,
                   collect({target: related.name, rel: type(r)}) as connections
            LIMIT 10
            """
            records = await run_query(cypher, {"words": words[:5]})
            return records
        except Exception as e:
            logger.warning("KG retrieval failed", error=str(e))
            return []

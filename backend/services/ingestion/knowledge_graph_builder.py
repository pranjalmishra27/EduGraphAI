"""
backend/services/ingestion/knowledge_graph_builder.py
Extracts concepts and relationships from documents and populates Neo4j.
"""
from __future__ import annotations
import json
import re
from backend.core.gemini_client import generate_text
from backend.core.neo4j_client import run_query
from backend.core.logging import get_logger
from backend.services.ingestion.document_parser import ParsedDocument

logger = get_logger(__name__)

CONCEPT_EXTRACTION_PROMPT = """
You are an expert educational knowledge graph builder.

From this educational content, extract:
1. Key concepts (nouns/topics a student must understand)
2. Prerequisite relationships (concept A requires knowing concept B first)
3. Derivation relationships (concept A is derived from concept B)
4. Association relationships (concept A is related to concept B)

Subject: {subject}
Content: {content}

Respond ONLY in this exact JSON (no markdown):
{{
  "concepts": [
    {{
      "name": "concept name",
      "description": "one sentence description",
      "difficulty": "beginner|intermediate|advanced",
      "subject": "{subject}"
    }}
  ],
  "relationships": [
    {{
      "source": "concept A",
      "target": "concept B",
      "type": "PREREQUISITE|DERIVED_FROM|RELATED_TO|DEPENDS_ON",
      "description": "why this relationship exists"
    }}
  ]
}}
"""


class KnowledgeGraphBuilder:

    async def build_from_document(self, parsed_doc: ParsedDocument, document_id: str) -> dict:
        """Extract concepts and relationships, then upsert into Neo4j."""
        # Build sample content from first 10 pages
        content_sample = "\n".join(
            f"Page {p.page_number}: {p.text[:300]}"
            for p in parsed_doc.pages[:10]
            if p.text.strip()
        )
        if not content_sample.strip():
            return {"concepts": 0, "relationships": 0}

        logger.info("Extracting knowledge graph", document_id=document_id, subject=parsed_doc.subject)

        prompt = CONCEPT_EXTRACTION_PROMPT.format(
            subject=parsed_doc.subject,
            content=content_sample[:4000],
        )

        raw = await generate_text(prompt)
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("KG extraction returned non-JSON", document_id=document_id)
            return {"concepts": 0, "relationships": 0}

        concepts = data.get("concepts", [])
        relationships = data.get("relationships", [])

        await self._upsert_concepts(concepts, document_id, parsed_doc.filename)
        await self._upsert_relationships(relationships)

        logger.info("Knowledge graph updated",
                    concepts=len(concepts), relationships=len(relationships))
        return {"concepts": len(concepts), "relationships": len(relationships)}

    async def _upsert_concepts(self, concepts: list[dict], document_id: str, filename: str) -> None:
        for concept in concepts:
            cypher = """
            MERGE (c:Concept {name: $name})
            SET c.description = $description,
                c.difficulty = $difficulty,
                c.subject = $subject,
                c.source_document = $document_id,
                c.source_file = $filename
            """
            await run_query(cypher, {
                "name": concept.get("name", "").strip(),
                "description": concept.get("description", ""),
                "difficulty": concept.get("difficulty", "intermediate"),
                "subject": concept.get("subject", "General"),
                "document_id": document_id,
                "filename": filename,
            })

    async def _upsert_relationships(self, relationships: list[dict]) -> None:
        for rel in relationships:
            rel_type = rel.get("type", "RELATED_TO").upper()
            # Only allow safe relationship types
            if rel_type not in ("PREREQUISITE", "DERIVED_FROM", "RELATED_TO", "DEPENDS_ON"):
                rel_type = "RELATED_TO"

            cypher = f"""
            MATCH (a:Concept {{name: $source}})
            MATCH (b:Concept {{name: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.description = $description
            """
            await run_query(cypher, {
                "source": rel.get("source", "").strip(),
                "target": rel.get("target", "").strip(),
                "description": rel.get("description", ""),
            })

    async def get_concept_subgraph(self, concept_name: str, depth: int = 2) -> dict:
        """Get a subgraph of concepts related to the given concept."""
        cypher = """
        MATCH path = (c:Concept {name: $name})-[*1..""" + str(depth) + """]->(related:Concept)
        RETURN c, relationships(path) as rels, nodes(path) as nodes
        LIMIT 50
        """
        records = await run_query(cypher, {"name": concept_name})

        nodes_map = {}
        edges = []

        for record in records:
            for node in record.get("nodes", []):
                n = dict(node)
                nodes_map[n["name"]] = n

            for rel in record.get("rels", []):
                edges.append({
                    "source": rel.start_node["name"],
                    "target": rel.end_node["name"],
                    "relationship": type(rel).__name__,
                })

        return {"nodes": list(nodes_map.values()), "edges": edges}

    async def get_learning_path(self, target_concept: str) -> list[str]:
        """Return ordered list of concepts to learn before target."""
        cypher = """
        MATCH path = (prereq:Concept)-[:PREREQUISITE*]->(target:Concept {name: $name})
        RETURN [n IN nodes(path) | n.name] as path_names
        ORDER BY length(path)
        LIMIT 10
        """
        records = await run_query(cypher, {"name": target_concept})
        if not records:
            return [target_concept]
        # Return the longest path (most detailed)
        longest = max(records, key=lambda r: len(r.get("path_names", [])))
        return longest.get("path_names", [target_concept])

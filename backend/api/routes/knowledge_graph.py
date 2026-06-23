"""
backend/api/routes/knowledge_graph.py
Knowledge graph exploration endpoints.
"""
from fastapi import APIRouter, Query, HTTPException
from backend.core.neo4j_client import run_query
from backend.services.ingestion.knowledge_graph_builder import KnowledgeGraphBuilder
from backend.models.schemas import KnowledgeGraphResponse, KnowledgeGraphNode, KnowledgeGraphEdge

router = APIRouter(prefix="/knowledge-graph", tags=["Knowledge Graph"])
kg_builder = KnowledgeGraphBuilder()


@router.get("/concepts", response_model=list[KnowledgeGraphNode])
async def list_concepts(
    subject: str | None = Query(default=None),
    difficulty: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
):
    """List all concepts in the knowledge graph."""
    where_clauses = []
    params: dict = {"limit": limit}

    if subject:
        where_clauses.append("c.subject = $subject")
        params["subject"] = subject
    if difficulty:
        where_clauses.append("c.difficulty = $difficulty")
        params["difficulty"] = difficulty

    where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    cypher = f"MATCH (c:Concept) {where_str} RETURN c LIMIT $limit"

    records = await run_query(cypher, params)
    return [
        KnowledgeGraphNode(
            name=r["c"]["name"],
            subject=r["c"].get("subject", "General"),
            difficulty=r["c"].get("difficulty", "intermediate"),
            description=r["c"].get("description", ""),
        )
        for r in records
    ]


@router.get("/subgraph/{concept_name}", response_model=KnowledgeGraphResponse)
async def get_subgraph(
    concept_name: str,
    depth: int = Query(default=2, ge=1, le=4),
):
    """Get the knowledge subgraph centred on a concept."""
    data = await kg_builder.get_concept_subgraph(concept_name, depth)
    if not data["nodes"]:
        raise HTTPException(404, f"Concept '{concept_name}' not found in knowledge graph")

    nodes = [
        KnowledgeGraphNode(
            name=n.get("name", ""),
            subject=n.get("subject", "General"),
            difficulty=n.get("difficulty", "intermediate"),
            description=n.get("description", ""),
        )
        for n in data["nodes"]
    ]
    edges = [
        KnowledgeGraphEdge(
            source=e["source"],
            target=e["target"],
            relationship=e["relationship"],
        )
        for e in data["edges"]
    ]
    return KnowledgeGraphResponse(nodes=nodes, edges=edges)


@router.get("/learning-path/{concept_name}")
async def get_learning_path(concept_name: str):
    """Get ordered learning path to reach a concept."""
    path = await kg_builder.get_learning_path(concept_name)
    return {"concept": concept_name, "learning_path": path}


@router.get("/graph-data")
async def get_full_graph(subject: str | None = Query(default=None)):
    """Get full graph data for visualization (nodes + edges)."""
    where = "WHERE c.subject = $subject" if subject else ""
    params = {"subject": subject} if subject else {}

    node_cypher = f"MATCH (c:Concept) {where} RETURN c LIMIT 100"
    edge_cypher = f"""
    MATCH (a:Concept)-[r]->(b:Concept)
    {where.replace('c.', 'a.')}
    RETURN a.name as source, b.name as target, type(r) as relationship
    LIMIT 200
    """

    import asyncio
    nodes_raw, edges_raw = await asyncio.gather(
        run_query(node_cypher, params),
        run_query(edge_cypher, params),
    )

    return {
        "nodes": [
            {
                "id": r["c"]["name"],
                "label": r["c"]["name"],
                "subject": r["c"].get("subject", "General"),
                "difficulty": r["c"].get("difficulty", "intermediate"),
                "description": r["c"].get("description", ""),
            }
            for r in nodes_raw
        ],
        "edges": edges_raw,
    }

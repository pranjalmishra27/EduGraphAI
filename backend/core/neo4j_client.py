"""
backend/core/neo4j_client.py
Neo4j async driver singleton with context manager helpers, and a local SQLite fallback.
"""
from __future__ import annotations
import re
import asyncio
import aiosqlite
from pathlib import Path
from neo4j import AsyncGraphDatabase, AsyncDriver
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
_driver: AsyncDriver | None = None
_sqlite_conn: aiosqlite.Connection | None = None
_fallback_mode: bool | None = None  # None: uninitialized, True/False once checked


class MockRel:
    def __init__(self, source: str, target: str, rel_type: str):
        self.start_node = {"name": source}
        self.end_node = {"name": target}
        self._rel_type = rel_type

    @property
    def __class__(self):
        # Return a class named rel_type dynamically so type(rel).__name__ works
        return type(self._rel_type, (object,), {})


async def get_sqlite_conn() -> aiosqlite.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        settings = get_settings()
        db_path = Path(settings.local_graph_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_conn = await aiosqlite.connect(str(db_path))
        _sqlite_conn.row_factory = aiosqlite.Row

        # Create local graph schema
        await _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            name TEXT PRIMARY KEY,
            description TEXT,
            difficulty TEXT,
            subject TEXT,
            source_document TEXT,
            source_file TEXT
        )
        """)
        await _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            source TEXT,
            target TEXT,
            relationship TEXT,
            description TEXT,
            PRIMARY KEY (source, target, relationship)
        )
        """)
        await _sqlite_conn.commit()
        logger.info("SQLite local graph database initialised", path=str(db_path))
    return _sqlite_conn


async def get_neo4j_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver initialised", uri=settings.neo4j_uri)
    return _driver


async def check_fallback_mode() -> bool:
    global _fallback_mode
    if _fallback_mode is not None:
        return _fallback_mode

    settings = get_settings()
    if settings.use_local_graph_fallback:
        try:
            # Attempt to connect to Neo4j to see if it's running
            driver = await get_neo4j_driver()
            # Wait up to 2 seconds for verification
            await asyncio.wait_for(driver.verify_connectivity(), timeout=2.0)
            _fallback_mode = False
            logger.info("Successfully connected to Neo4j. Using Neo4j graph mode.")
        except Exception as e:
            _fallback_mode = True
            logger.warning("Could not connect to Neo4j. Falling back to SQLite local graph mode.", error=str(e))
    else:
        _fallback_mode = False

    return _fallback_mode


async def close_neo4j_driver() -> None:
    global _driver, _sqlite_conn
    if _driver:
        await _driver.close()
        _driver = None
    if _sqlite_conn:
        await _sqlite_conn.close()
        _sqlite_conn = None


async def run_query_sqlite(cypher: str, parameters: dict) -> list[dict]:
    """Intercept Cypher queries and run corresponding SQL logic on SQLite fallback."""
    conn = await get_sqlite_conn()

    # Query 1: MERGE Concept
    if "MERGE (c:Concept" in cypher:
        await conn.execute("""
            INSERT INTO concepts (name, description, difficulty, subject, source_document, source_file)
            VALUES (:name, :description, :difficulty, :subject, :document_id, :filename)
            ON CONFLICT(name) DO UPDATE SET
                description=excluded.description,
                difficulty=excluded.difficulty,
                subject=excluded.subject,
                source_document=excluded.source_document,
                source_file=excluded.source_file
        """, {
            "name": parameters.get("name", "").strip(),
            "description": parameters.get("description", ""),
            "difficulty": parameters.get("difficulty", "intermediate"),
            "subject": parameters.get("subject", "General"),
            "document_id": parameters.get("document_id", ""),
            "filename": parameters.get("filename", ""),
        })
        await conn.commit()
        return []

    # Query 2: MERGE Relationship
    elif "MERGE (a)-[r:" in cypher:
        # Extract relationship type from string e.g. MERGE (a)-[r:PREREQUISITE]->(b)
        rel_match = re.search(r"-\[r:(\w+)\]->", cypher)
        rel_type = rel_match.group(1) if rel_match else "RELATED_TO"
        
        # Verify both source and target concepts exist to match MATCH behavior
        source = parameters.get("source", "").strip()
        target = parameters.get("target", "").strip()
        
        async with conn.execute("SELECT 1 FROM concepts WHERE name = ?", (source,)) as cursor:
            has_source = await cursor.fetchone()
        async with conn.execute("SELECT 1 FROM concepts WHERE name = ?", (target,)) as cursor:
            has_target = await cursor.fetchone()
            
        if has_source and has_target:
            await conn.execute("""
                INSERT INTO relationships (source, target, relationship, description)
                VALUES (:source, :target, :relationship, :description)
                ON CONFLICT(source, target, relationship) DO UPDATE SET
                    description=excluded.description
            """, {
                "source": source,
                "target": target,
                "relationship": rel_type,
                "description": parameters.get("description", ""),
            })
            await conn.commit()
        return []

    # Query 3: List concepts
    elif "MATCH (c:Concept)" in cypher and "LIMIT" in cypher and "relationships" not in cypher and "nodes" not in cypher:
        sql = "SELECT name, description, difficulty, subject, source_document, source_file FROM concepts"
        where_clauses = []
        sql_params = {}
        if "subject" in parameters and parameters["subject"] is not None:
            where_clauses.append("subject = :subject")
            sql_params["subject"] = parameters["subject"]
        if "difficulty" in parameters and parameters["difficulty"] is not None:
            where_clauses.append("difficulty = :difficulty")
            sql_params["difficulty"] = parameters["difficulty"]
            
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            
        if "limit" in parameters:
            sql += " LIMIT :limit"
            sql_params["limit"] = parameters["limit"]
            
        rows = []
        async with conn.execute(sql, sql_params) as cursor:
            async for r in cursor:
                rows.append({
                    "c": {
                        "name": r["name"],
                        "description": r["description"],
                        "difficulty": r["difficulty"],
                        "subject": r["subject"],
                        "source_document": r["source_document"],
                        "source_file": r["source_file"]
                    }
                })
        return rows

    # Query 4: Full graph edges
    elif "MATCH (a:Concept)-[r]->(b:Concept)" in cypher:
        sql = """
            SELECT r.source, r.target, r.relationship
            FROM relationships r
            JOIN concepts c ON r.source = c.name
        """
        sql_params = {}
        if "subject" in parameters and parameters["subject"] is not None:
            sql += " WHERE c.subject = :subject"
            sql_params["subject"] = parameters["subject"]
        sql += " LIMIT 200"
        
        rows = []
        async with conn.execute(sql, sql_params) as cursor:
            async for r in cursor:
                rows.append({
                    "source": r["source"],
                    "target": r["target"],
                    "relationship": r["relationship"]
                })
        return rows

    # Query 5: Subgraph Query with depth path
    elif "MATCH path = (c:Concept" in cypher and "relationships(path)" in cypher:
        # Extract depth from Cypher e.g. [*1..2]
        depth_match = re.search(r"\[\*1\.\.(\d+)\]", cypher)
        depth = int(depth_match.group(1)) if depth_match else 2
        name = parameters.get("name", "")
        
        # Load all concepts and relationships
        concepts_by_name = {}
        async with conn.execute("SELECT * FROM concepts") as cursor:
            async for r in cursor:
                concepts_by_name[r["name"]] = dict(r)
                
        adj = {}
        async with conn.execute("SELECT source, target, relationship, description FROM relationships") as cursor:
            async for r in cursor:
                adj.setdefault(r["source"], []).append(dict(r))
                
        if name not in concepts_by_name:
            return []
            
        paths = []
        
        def find_paths(curr, current_path, current_depth):
            if current_depth > depth:
                return
            if curr not in adj:
                return
            for edge in adj[curr]:
                nxt = edge["target"]
                new_path = current_path + [edge]
                paths.append(new_path)
                find_paths(nxt, new_path, current_depth + 1)
                
        find_paths(name, [], 1)
        
        records = []
        for p in paths[:50]:
            nodes_list = [concepts_by_name[name]]
            rels_list = []
            for edge in p:
                t_name = edge["target"]
                if t_name in concepts_by_name:
                    nodes_list.append(concepts_by_name[t_name])
                rels_list.append(MockRel(edge["source"], edge["target"], edge["relationship"]))
                
            records.append({
                "nodes": nodes_list,
                "rels": rels_list
            })
        return records

    # Query 6: Prerequisite Learning Path
    elif "[:PREREQUISITE*]" in cypher:
        name = parameters.get("name", "")
        
        # Load incoming PREREQUISITE edges
        prereqs = {}
        async with conn.execute("SELECT source, target FROM relationships WHERE relationship = 'PREREQUISITE'") as cursor:
            async for r in cursor:
                prereqs.setdefault(r["target"], []).append(r["source"])
                
        paths = []
        
        def find_prereq_paths(curr, current_path):
            if curr in current_path:
                paths.append(list(reversed(current_path)))
                return
            new_path = current_path + [curr]
            parents = prereqs.get(curr, [])
            if not parents:
                paths.append(list(reversed(new_path)))
                return
            for p in parents:
                find_prereq_paths(p, new_path)
                
        find_prereq_paths(name, [])
        paths = sorted(paths, key=len)
        paths = paths[:10]
        
        return [{"path_names": p} for p in paths]

    # Query 7: Retrieval KG Context (ANY word in words)
    elif "ANY(word IN $words" in cypher:
        words = parameters.get("words", [])
        
        all_concepts = []
        async with conn.execute("SELECT name, description, difficulty FROM concepts") as cursor:
            async for r in cursor:
                name_lower = r["name"].lower()
                if any(word.lower() in name_lower for word in words):
                    all_concepts.append({
                        "concept": r["name"],
                        "description": r["description"],
                        "difficulty": r["difficulty"]
                    })
                    
        all_concepts = all_concepts[:10]
        
        results = []
        for c_data in all_concepts:
            connections = []
            async with conn.execute("SELECT target, relationship FROM relationships WHERE source = ?", (c_data["concept"],)) as cursor:
                async for r in cursor:
                    connections.append({
                        "target": r["target"],
                        "rel": r["relationship"]
                    })
            results.append({
                "concept": c_data["concept"],
                "description": c_data["description"],
                "difficulty": c_data["difficulty"],
                "connections": connections
            })
        return results

    # Fallback default: empty query return
    return []


async def run_query(cypher: str, parameters: dict | None = None) -> list[dict]:
    """Execute a Cypher query (routed to Neo4j or SQLite fallback)."""
    fallback = await check_fallback_mode()
    if fallback:
        return await run_query_sqlite(cypher, parameters or {})

    driver = await get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run(cypher, parameters or {})
        return [record.data() async for record in result]


async def init_graph_schema() -> None:
    """Create constraints and indexes for the knowledge graph."""
    fallback = await check_fallback_mode()
    if fallback:
        # Initialise SQLite connection/tables
        await get_sqlite_conn()
        logger.info("SQLite local graph schema initialised")
        return

    constraints = [
        "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
        "CREATE INDEX concept_subject IF NOT EXISTS FOR (c:Concept) ON (c.subject)",
    ]
    for cypher in constraints:
        await run_query(cypher)
    logger.info("Neo4j schema initialised")

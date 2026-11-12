"""
api/index.py
Vercel serverless entrypoint for the EduGraphAI FastAPI backend.

Vercel's filesystem is read-only except for /tmp, so on Vercel we point every
writable store (SQLite DBs, ChromaDB) at /tmp and seed it from the copies
bundled in ./data on cold start. Anything written at runtime (uploads, new
embeddings) is lost when the instance is recycled -- use a hosted Postgres
(DATABASE_URL) and Neo4j (NEO4J_URI) for durable storage.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # backend uses ./data/... relative paths

if os.environ.get("VERCEL"):
    TMP = Path("/tmp/edugraphai")
    TMP.mkdir(parents=True, exist_ok=True)

    def _seed(src: Path, dst: Path) -> None:
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    _seed(ROOT / "data" / "edugraphai.db", TMP / "edugraphai.db")
    _seed(ROOT / "data" / "knowledge_graph.db", TMP / "knowledge_graph.db")
    _seed(ROOT / "data" / "chroma_db", TMP / "chroma_db")

    # Only applied if not already set in the Vercel dashboard
    os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{TMP}/edugraphai.db")
    os.environ.setdefault("LOCAL_GRAPH_DB_PATH", str(TMP / "knowledge_graph.db"))
    os.environ.setdefault("CHROMA_PERSIST_DIR", str(TMP / "chroma_db"))
    os.environ.setdefault("APP_ENV", "production")

from backend.main import app  # noqa: E402  (Vercel looks for `app`)

"""
backend/main.py
FastAPI application entry point — registers routers, lifespan events,
CORS, exception handlers, and health check.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import documents, intelligence, knowledge_graph, analytics
from backend.core.config import get_settings
from backend.core.database import init_db
from backend.core.logging import setup_logging, get_logger
from backend.core.neo4j_client import init_graph_schema, close_neo4j_driver

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EduGraphAI backend starting up", env=settings.app_env)

    # Initialise PostgreSQL tables
    try:
        await init_db()
        logger.info("PostgreSQL tables initialised")
    except Exception as e:
        logger.error("PostgreSQL init failed", error=str(e))

    # Initialise Neo4j schema
    try:
        await init_graph_schema()
        logger.info("Neo4j schema initialised")
    except Exception as e:
        logger.warning("Neo4j init failed (is Neo4j running?)", error=str(e))

    yield

    # Shutdown
    await close_neo4j_driver()
    logger.info("EduGraphAI backend shut down")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduGraphAI API",
    description="Multimodal Educational Intelligence Platform — RAG + Knowledge Graph + ML Analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(documents.router,       prefix="/api/v1")
app.include_router(intelligence.router,    prefix="/api/v1")
app.include_router(knowledge_graph.router, prefix="/api/v1")
app.include_router(analytics.router,       prefix="/api/v1")

# ─── Exception Handlers ───────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "service": "EduGraphAI",
        "version": "1.0.0",
        "environment": settings.app_env,
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "EduGraphAI API is running.",
        "docs": "/docs",
        "redoc": "/redoc",
    }

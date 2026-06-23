"""
backend/api/routes/documents.py
Document upload, listing, and deletion endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.core.config import get_settings
from backend.models.schemas import Document, DocumentUploadResponse, DocumentOut
from backend.services.ingestion.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])
ingestion_service = IngestionService()
settings = get_settings()

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
    "image/jpeg", "image/png", "image/webp",
    "text/plain", "text/markdown",
}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    subject: str = Form(default="General"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a document (PDF, PPTX, image, or text)."""
    # Validate size
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {settings.max_upload_size_mb} MB")

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        # Be lenient — check extension
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in ("pdf", "pptx", "ppt", "jpg", "jpeg", "png", "txt", "md"):
            raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    doc = await ingestion_service.ingest_document(
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        subject=subject,
        db=db,
    )

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        message=f"Document ingested successfully. {doc.chunk_count} chunks created.",
    )


@router.get("/", response_model=list[DocumentOut])
async def list_documents(
    subject: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all ingested documents."""
    query = select(Document).order_by(Document.uploaded_at.desc())
    if subject:
        query = query.where(Document.subject == subject)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    deleted = await ingestion_service.delete_document(document_id, db)
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"message": "Document deleted successfully"}

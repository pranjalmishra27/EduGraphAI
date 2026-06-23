"""
backend/models/schemas.py
Pydantic request/response schemas + SQLAlchemy ORM models.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import String, Float, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from backend.core.database import Base


# ─── SQLAlchemy ORM ────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(512))
    doc_type: Mapped[str] = mapped_column(String(50))   # pdf | pptx | image | text
    subject: Mapped[str] = mapped_column(String(256), default="General")
    chapter: Mapped[str | None] = mapped_column(String(256), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="processing")  # processing | ready | error
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    chunks: Mapped[list[Chunk]] = relationship("Chunk", back_populates="document", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chapter: Mapped[str | None] = mapped_column(String(256), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    embedding_id: Mapped[str | None] = mapped_column(String, nullable=True)  # ChromaDB ID

    document: Mapped[Document] = relationship("Document", back_populates="chunks")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(256))
    email: Mapped[str] = mapped_column(String(256), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    quiz_attempts: Mapped[list[QuizAttempt]] = relationship("QuizAttempt", back_populates="student")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"))
    topic: Mapped[str] = mapped_column(String(256))
    score: Mapped[float] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float, default=100.0)
    difficulty: Mapped[str] = mapped_column(String(50), default="medium")
    time_spent_mins: Mapped[float] = mapped_column(Float, default=0.0)
    attempted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    student: Mapped[Student] = relationship("Student", back_populates="quiz_attempts")


# ─── Pydantic Schemas ──────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str
    subject: str
    chunk_count: int
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    subject: str | None = None
    include_diagrams: bool = True
    include_formulas: bool = True


class SourceChunk(BaseModel):
    content: str
    page_number: int | None
    chapter: str | None
    document_filename: str
    relevance_score: float


class FormulaRef(BaseModel):
    name: str
    latex: str
    explanation: str
    related_concepts: list[str]


class DiagramRef(BaseModel):
    caption: str
    chapter: str | None
    page_number: int | None
    document_filename: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    formulas: list[FormulaRef]
    diagrams: list[DiagramRef]
    knowledge_graph_context: list[dict]
    confidence_score: float


class HandwritingEvalRequest(BaseModel):
    topic: str
    expected_answer: str | None = None


class HandwritingEvalResponse(BaseModel):
    extracted_text: str
    steps_identified: list[str]
    missing_steps: list[str]
    conceptual_errors: list[str]
    calculation_errors: list[str]
    formula_errors: list[str]
    overall_score: float
    detailed_feedback: str


class KnowledgeGraphNode(BaseModel):
    name: str
    subject: str
    difficulty: str
    description: str


class KnowledgeGraphEdge(BaseModel):
    source: str
    target: str
    relationship: str


class KnowledgeGraphResponse(BaseModel):
    nodes: list[KnowledgeGraphNode]
    edges: list[KnowledgeGraphEdge]


class ExamAnalysisResponse(BaseModel):
    high_priority_topics: list[str]
    topic_frequency: dict[str, int]
    important_derivations: list[str]
    numerical_problems: list[str]
    recommendations: list[str]


class StudentCreate(BaseModel):
    name: str
    email: str


class QuizAttemptCreate(BaseModel):
    student_id: str
    topic: str
    score: float
    max_score: float = 100.0
    difficulty: str = "medium"
    time_spent_mins: float = 0.0


class LearningRecommendation(BaseModel):
    student_id: str
    weak_topics: list[str]
    recommended_order: list[str]
    revision_priorities: list[str]
    predicted_improvement: float


class PerformancePrediction(BaseModel):
    student_id: str
    predicted_score: float
    pass_probability: float
    risk_level: str          # low | medium | high
    risk_topics: list[str]
    trend: str               # improving | declining | stable
    feature_importances: dict[str, float]

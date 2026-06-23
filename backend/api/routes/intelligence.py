"""
backend/api/routes/intelligence.py
Question answering, handwriting evaluation, and exam analysis endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.models.schemas import (
    QuestionRequest, QuestionResponse,
    HandwritingEvalRequest, HandwritingEvalResponse,
    ExamAnalysisResponse,
)
from backend.services.retrieval.rag_pipeline import RAGPipeline
from backend.services.retrieval.handwriting_evaluator import HandwritingEvaluator
from backend.services.retrieval.exam_analyzer import ExamAnalyzer

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

rag_pipeline = RAGPipeline()
handwriting_evaluator = HandwritingEvaluator()
exam_analyzer = ExamAnalyzer()


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a student's question using RAG over uploaded documents.
    Returns answer + sources + formulas + diagrams + KG context.
    """
    return await rag_pipeline.answer(
        question=request.question,
        subject=request.subject,
        include_diagrams=request.include_diagrams,
        include_formulas=request.include_formulas,
    )


@router.post("/evaluate-handwriting", response_model=HandwritingEvalResponse)
async def evaluate_handwriting(
    image: UploadFile = File(...),
    topic: str = Form(...),
    expected_answer: str = Form(default=""),
):
    """
    Evaluate a handwritten student solution using Gemini Vision.
    Upload image (JPEG/PNG) of the handwritten answer.
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Only JPEG/PNG/WebP images are supported")

    image_bytes = await image.read()
    return await handwriting_evaluator.evaluate(
        image_bytes=image_bytes,
        mime_type=image.content_type,
        topic=topic,
        expected_answer=expected_answer or None,
    )


@router.post("/analyze-exam", response_model=ExamAnalysisResponse)
async def analyze_exam(file: UploadFile = File(...)):
    """
    Analyze a previous-year exam paper to identify high-priority topics.
    Upload PDF of the exam paper.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(415, "Only PDF exam papers are supported")

    file_bytes = await file.read()
    return await exam_analyzer.analyze(file_bytes, "application/pdf")

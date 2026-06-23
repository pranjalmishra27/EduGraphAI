"""
backend/api/routes/analytics.py
Student management, quiz tracking, and ML analytics endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db
from backend.models.schemas import (
    Student, QuizAttempt,
    StudentCreate, QuizAttemptCreate,
    LearningRecommendation, PerformancePrediction,
)
from backend.services.ml.analytics_service import (
    PerformancePredictor, LearningRecommender,
    build_student_features, get_topic_weaknesses,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

predictor = PerformancePredictor()
recommender = LearningRecommender()


# ── Students ──────────────────────────────────────────────────────────────────

@router.post("/students", response_model=dict)
async def create_student(data: StudentCreate, db: AsyncSession = Depends(get_db)):
    student = Student(name=data.name, email=data.email)
    db.add(student)
    await db.flush()
    return {"student_id": student.id, "name": student.name, "email": student.email}


@router.get("/students/{student_id}")
async def get_student(student_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(404, "Student not found")
    return {"id": student.id, "name": student.name, "email": student.email}


# ── Quiz Attempts ──────────────────────────────────────────────────────────────

@router.post("/quiz-attempts", response_model=dict)
async def record_quiz_attempt(data: QuizAttemptCreate, db: AsyncSession = Depends(get_db)):
    attempt = QuizAttempt(
        student_id=data.student_id,
        topic=data.topic,
        score=data.score,
        max_score=data.max_score,
        difficulty=data.difficulty,
        time_spent_mins=data.time_spent_mins,
    )
    db.add(attempt)
    await db.flush()
    return {"attempt_id": attempt.id, "message": "Quiz attempt recorded"}


@router.get("/students/{student_id}/attempts")
async def get_student_attempts(student_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(QuizAttempt)
        .where(QuizAttempt.student_id == student_id)
        .order_by(QuizAttempt.attempted_at.desc())
    )
    attempts = result.scalars().all()
    return [
        {
            "id": a.id,
            "topic": a.topic,
            "score": a.score,
            "max_score": a.max_score,
            "pct_score": round(a.score / a.max_score * 100, 1),
            "difficulty": a.difficulty,
            "time_spent_mins": a.time_spent_mins,
            "attempted_at": a.attempted_at.isoformat(),
        }
        for a in attempts
    ]


# ── ML Analytics ──────────────────────────────────────────────────────────────

@router.get("/students/{student_id}/recommendations", response_model=LearningRecommendation)
async def get_recommendations(
    student_id: str,
    subject: str = "Physics",
    db: AsyncSession = Depends(get_db),
):
    """Generate personalized learning recommendations for a student."""
    result = await db.execute(
        select(QuizAttempt).where(QuizAttempt.student_id == student_id)
    )
    attempts = result.scalars().all()

    attempts_data = [
        {
            "topic": a.topic,
            "score": a.score,
            "max_score": a.max_score,
            "difficulty": a.difficulty,
            "time_spent_mins": a.time_spent_mins,
            "attempted_at": a.attempted_at.isoformat(),
        }
        for a in attempts
    ]

    return recommender.generate_recommendation(student_id, attempts_data, subject)


@router.get("/students/{student_id}/predict", response_model=PerformancePrediction)
async def predict_performance(student_id: str, db: AsyncSession = Depends(get_db)):
    """Predict student's future exam performance."""
    result = await db.execute(
        select(QuizAttempt).where(QuizAttempt.student_id == student_id)
    )
    attempts = result.scalars().all()

    if not attempts:
        raise HTTPException(400, "No quiz attempts found. Please record some quiz attempts first.")

    attempts_data = [
        {
            "topic": a.topic,
            "score": a.score,
            "max_score": a.max_score,
            "difficulty": a.difficulty,
            "time_spent_mins": a.time_spent_mins,
            "attempted_at": a.attempted_at.isoformat(),
        }
        for a in attempts
    ]

    features = build_student_features(attempts_data)
    if features.empty:
        raise HTTPException(400, "Unable to compute features from attempt data")

    prediction = predictor.predict(features)
    weak_topics = get_topic_weaknesses(attempts_data)

    predicted_score = prediction["predicted_score"]
    pass_prob = prediction["pass_probability"]

    if pass_prob < 0.5:
        risk_level = "high"
    elif pass_prob < 0.75:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Determine trend from feature
    trend_slope = float(features.get("score_trend", [0])[0]) if not features.empty else 0
    if trend_slope > 1:
        trend = "improving"
    elif trend_slope < -1:
        trend = "declining"
    else:
        trend = "stable"

    return PerformancePrediction(
        student_id=student_id,
        predicted_score=predicted_score,
        pass_probability=pass_prob,
        risk_level=risk_level,
        risk_topics=weak_topics[:5],
        trend=trend,
        feature_importances=prediction.get("feature_importances", {}),
    )


@router.get("/students/{student_id}/dashboard")
async def get_student_dashboard(student_id: str, db: AsyncSession = Depends(get_db)):
    """Aggregated dashboard data for a student."""
    result = await db.execute(
        select(QuizAttempt).where(QuizAttempt.student_id == student_id)
    )
    attempts = result.scalars().all()

    if not attempts:
        return {"message": "No data yet", "attempts": 0}

    import pandas as pd
    df = pd.DataFrame([{
        "topic": a.topic,
        "score": a.score,
        "max_score": a.max_score,
        "pct_score": a.score / a.max_score * 100,
        "difficulty": a.difficulty,
        "attempted_at": a.attempted_at.isoformat(),
    } for a in attempts])

    topic_stats = df.groupby("topic")["pct_score"].agg(["mean", "count", "std"]).round(1)

    return {
        "total_attempts": len(attempts),
        "overall_avg": round(df["pct_score"].mean(), 1),
        "topics_studied": df["topic"].nunique(),
        "topic_breakdown": topic_stats.reset_index().to_dict(orient="records"),
        "recent_scores": df.tail(10)[["topic", "pct_score", "attempted_at"]].to_dict(orient="records"),
    }

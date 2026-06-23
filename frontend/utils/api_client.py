"""
frontend/utils/api_client.py
Thin HTTP client wrapping all EduGraphAI backend endpoints.
"""
from __future__ import annotations

import os
import httpx
from typing import Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BASE = f"{BACKEND_URL}/api/v1"
TIMEOUT = 120.0   # Gemini calls can be slow


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT)


# ─── Documents ────────────────────────────────────────────────────────────────

def upload_document(file_bytes: bytes, filename: str, content_type: str, subject: str) -> dict:
    with _client() as c:
        r = c.post(
            f"{BASE}/documents/upload",
            files={"file": (filename, file_bytes, content_type)},
            data={"subject": subject},
        )
        r.raise_for_status()
        return r.json()


def list_documents(subject: str | None = None) -> list[dict]:
    params = {}
    if subject:
        params["subject"] = subject
    with _client() as c:
        r = c.get(f"{BASE}/documents/", params=params)
        r.raise_for_status()
        return r.json()


def delete_document(document_id: str) -> dict:
    with _client() as c:
        r = c.delete(f"{BASE}/documents/{document_id}")
        r.raise_for_status()
        return r.json()


# ─── Intelligence ─────────────────────────────────────────────────────────────

def ask_question(question: str, subject: str | None, include_diagrams: bool, include_formulas: bool) -> dict:
    payload = {
        "question": question,
        "subject": subject,
        "include_diagrams": include_diagrams,
        "include_formulas": include_formulas,
    }
    with _client() as c:
        r = c.post(f"{BASE}/intelligence/ask", json=payload)
        r.raise_for_status()
        return r.json()


def evaluate_handwriting(image_bytes: bytes, filename: str, content_type: str, topic: str, expected_answer: str) -> dict:
    with _client() as c:
        r = c.post(
            f"{BASE}/intelligence/evaluate-handwriting",
            files={"image": (filename, image_bytes, content_type)},
            data={"topic": topic, "expected_answer": expected_answer},
        )
        r.raise_for_status()
        return r.json()


def analyze_exam(file_bytes: bytes, filename: str) -> dict:
    with _client() as c:
        r = c.post(
            f"{BASE}/intelligence/analyze-exam",
            files={"file": (filename, file_bytes, "application/pdf")},
        )
        r.raise_for_status()
        return r.json()


# ─── Knowledge Graph ──────────────────────────────────────────────────────────

def get_graph_data(subject: str | None = None) -> dict:
    params = {}
    if subject:
        params["subject"] = subject
    with _client() as c:
        r = c.get(f"{BASE}/knowledge-graph/graph-data", params=params)
        r.raise_for_status()
        return r.json()


def get_subgraph(concept: str, depth: int = 2) -> dict:
    with _client() as c:
        r = c.get(f"{BASE}/knowledge-graph/subgraph/{concept}", params={"depth": depth})
        r.raise_for_status()
        return r.json()


def get_learning_path(concept: str) -> dict:
    with _client() as c:
        r = c.get(f"{BASE}/knowledge-graph/learning-path/{concept}")
        r.raise_for_status()
        return r.json()


def list_concepts(subject: str | None = None) -> list[dict]:
    params = {}
    if subject:
        params["subject"] = subject
    with _client() as c:
        r = c.get(f"{BASE}/knowledge-graph/concepts", params=params)
        r.raise_for_status()
        return r.json()


# ─── Analytics ────────────────────────────────────────────────────────────────

def create_student(name: str, email: str) -> dict:
    with _client() as c:
        r = c.post(f"{BASE}/analytics/students", json={"name": name, "email": email})
        r.raise_for_status()
        return r.json()


def record_quiz_attempt(student_id: str, topic: str, score: float, max_score: float,
                        difficulty: str, time_spent: float) -> dict:
    with _client() as c:
        r = c.post(f"{BASE}/analytics/quiz-attempts", json={
            "student_id": student_id,
            "topic": topic,
            "score": score,
            "max_score": max_score,
            "difficulty": difficulty,
            "time_spent_mins": time_spent,
        })
        r.raise_for_status()
        return r.json()


def get_student_attempts(student_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"{BASE}/analytics/students/{student_id}/attempts")
        r.raise_for_status()
        return r.json()


def get_recommendations(student_id: str, subject: str) -> dict:
    with _client() as c:
        r = c.get(f"{BASE}/analytics/students/{student_id}/recommendations", params={"subject": subject})
        r.raise_for_status()
        return r.json()


def predict_performance(student_id: str) -> dict:
    with _client() as c:
        r = c.get(f"{BASE}/analytics/students/{student_id}/predict")
        r.raise_for_status()
        return r.json()


def get_dashboard(student_id: str) -> dict:
    with _client() as c:
        r = c.get(f"{BASE}/analytics/students/{student_id}/dashboard")
        r.raise_for_status()
        return r.json()

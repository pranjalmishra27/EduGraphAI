"""
backend/services/retrieval/exam_analyzer.py
Analyzes uploaded exam papers to identify high-priority topics and trends.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from backend.core.gemini_client import generate_with_document
from backend.core.logging import get_logger
from backend.models.schemas import ExamAnalysisResponse

logger = get_logger(__name__)

EXAM_ANALYSIS_PROMPT = """
You are an expert exam pattern analyser for educational institutions.

Analyse this examination paper / question set and identify:
1. All topics/concepts tested
2. Frequently appearing topics (topics tested multiple times)
3. Important derivations asked
4. Numerical problem types
5. Topic-wise question frequency count
6. Strategic recommendations for students

Respond ONLY in this JSON format:
{
  "all_topics": ["topic1", "topic2"],
  "topic_frequency": {"topic1": 3, "topic2": 2},
  "high_priority_topics": ["most frequently tested topic1", "topic2"],
  "important_derivations": ["derivation 1", "derivation 2"],
  "numerical_problems": ["numerical type 1", "numerical type 2"],
  "recommendations": [
    "Focus on topic X as it appears in 60% of questions",
    "Master derivation Y as it appears every year"
  ]
}
"""


class ExamAnalyzer:

    async def analyze(self, file_bytes: bytes, mime_type: str) -> ExamAnalysisResponse:
        logger.info("Analyzing exam paper")

        raw = await generate_with_document(EXAM_ANALYSIS_PROMPT, file_bytes, mime_type)
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Exam analysis returned non-JSON")
            data = {
                "high_priority_topics": [],
                "topic_frequency": {},
                "important_derivations": [],
                "numerical_problems": [],
                "recommendations": ["Unable to parse exam paper. Please ensure it is a valid PDF."],
            }

        return ExamAnalysisResponse(
            high_priority_topics=data.get("high_priority_topics", []),
            topic_frequency=data.get("topic_frequency", {}),
            important_derivations=data.get("important_derivations", []),
            numerical_problems=data.get("numerical_problems", []),
            recommendations=data.get("recommendations", []),
        )

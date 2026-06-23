"""
backend/services/retrieval/handwriting_evaluator.py
Evaluates handwritten student solutions using Gemini Vision.
"""
from __future__ import annotations
import json
import re
from backend.core.gemini_client import generate_with_image
from backend.core.logging import get_logger
from backend.models.schemas import HandwritingEvalResponse

logger = get_logger(__name__)

EVALUATION_PROMPT = """
You are an expert educational evaluator. Analyse this handwritten student solution.

Topic: {topic}
{expected_answer_section}

Please evaluate:
1. Extract all text from the handwritten solution (OCR)
2. Identify all logical/solution steps the student wrote
3. Identify any missing steps that should be present
4. Identify conceptual errors (misunderstood concepts)
5. Identify calculation errors (arithmetic mistakes)
6. Identify incorrect formula usage
7. Provide an overall score out of 10
8. Write detailed, constructive feedback for the student

Respond ONLY in this exact JSON format:
{{
  "extracted_text": "full OCR text from the handwriting",
  "steps_identified": ["step 1 description", "step 2 description"],
  "missing_steps": ["missing step 1", "missing step 2"],
  "conceptual_errors": ["error description"],
  "calculation_errors": ["error description"],
  "formula_errors": ["error description"],
  "overall_score": 7.5,
  "detailed_feedback": "Full paragraph feedback for the student..."
}}
"""


class HandwritingEvaluator:

    async def evaluate(
        self,
        image_bytes: bytes,
        mime_type: str,
        topic: str,
        expected_answer: str | None = None,
    ) -> HandwritingEvalResponse:
        logger.info("Evaluating handwriting", topic=topic)

        expected_section = ""
        if expected_answer:
            expected_section = f"\nExpected/Model Answer for reference:\n{expected_answer}"

        prompt = EVALUATION_PROMPT.format(
            topic=topic,
            expected_answer_section=expected_section,
        )

        raw = await generate_with_image(prompt, image_bytes, mime_type)
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Handwriting eval returned non-JSON, using fallback")
            data = {
                "extracted_text": raw,
                "steps_identified": [],
                "missing_steps": [],
                "conceptual_errors": [],
                "calculation_errors": [],
                "formula_errors": [],
                "overall_score": 0.0,
                "detailed_feedback": "Unable to parse structured evaluation. Raw extraction: " + raw[:500],
            }

        return HandwritingEvalResponse(
            extracted_text=data.get("extracted_text", ""),
            steps_identified=data.get("steps_identified", []),
            missing_steps=data.get("missing_steps", []),
            conceptual_errors=data.get("conceptual_errors", []),
            calculation_errors=data.get("calculation_errors", []),
            formula_errors=data.get("formula_errors", []),
            overall_score=float(data.get("overall_score", 0.0)),
            detailed_feedback=data.get("detailed_feedback", ""),
        )

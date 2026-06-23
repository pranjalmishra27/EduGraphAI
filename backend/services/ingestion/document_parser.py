"""
backend/services/ingestion/document_parser.py
Multimodal document parsing using Gemini + pypdf + python-pptx.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
import pypdf
from pptx import Presentation
from pptx.util import Inches

from backend.core.gemini_client import generate_with_document, generate_with_image, generate_text
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    chapter: str | None = None
    tables: list[str] = field(default_factory=list)
    figure_descriptions: list[str] = field(default_factory=list)
    raw_equations: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    filename: str
    doc_type: str
    pages: list[ParsedPage] = field(default_factory=list)
    formulas: list[dict] = field(default_factory=list)
    diagrams: list[dict] = field(default_factory=list)
    subject: str = "General"
    total_pages: int = 0


EXTRACTION_PROMPT = """
You are an expert educational content extractor. Analyse the provided document page/slide and extract:

1. MAIN TEXT: All readable text content preserving structure
2. CHAPTER/SECTION: The chapter or section heading if present
3. EQUATIONS: Any mathematical or scientific equations (provide in LaTeX notation)
4. FIGURES: Describe any diagrams, figures, or illustrations present
5. TABLES: Extract table data as text

Respond in this exact JSON format:
{
  "text": "extracted text here",
  "chapter": "chapter name or null",
  "equations": ["latex_eq_1", "latex_eq_2"],
  "figure_descriptions": ["description of figure 1", "description of figure 2"],
  "tables": ["table content as text"]
}
"""

FORMULA_ENRICHMENT_PROMPT = """
Given this LaTeX equation: {equation}
From subject: {subject}

Provide:
1. Common name of this formula
2. Plain English explanation of what it represents
3. What each symbol/variable means
4. Related concepts a student should know
5. Derivation steps (brief)

Respond in JSON:
{
  "name": "formula name",
  "latex": "latex notation",
  "explanation": "plain explanation",
  "symbol_meanings": {"symbol": "meaning"},
  "related_concepts": ["concept1", "concept2"],
  "derivation_steps": ["step1", "step2"]
}
"""

STRUCTURE_EXTRACTION_PROMPT = """
Analyse this educational document and identify:
1. Subject domain (Physics, Chemistry, Mathematics, etc.)
2. Chapter structure and sequence
3. Key concepts covered
4. Prerequisite relationships between concepts

Respond in JSON:
{
  "subject": "subject name",
  "chapters": ["ch1", "ch2"],
  "key_concepts": ["concept1", "concept2"],
  "prerequisites": [{"concept": "A", "requires": "B"}]
}
"""


class DocumentParser:

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        ext = Path(filename).suffix.lower()
        logger.info("Parsing document", filename=filename, extension=ext)

        if ext == ".pdf":
            return await self._parse_pdf(file_bytes, filename)
        elif ext in (".pptx", ".ppt"):
            return await self._parse_pptx(file_bytes, filename)
        elif ext in (".jpg", ".jpeg", ".png", ".webp"):
            return await self._parse_image(file_bytes, filename)
        elif ext in (".txt", ".md"):
            return await self._parse_text(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── PDF ──────────────────────────────────────────────────────────────────

    async def _parse_pdf(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        doc = ParsedDocument(filename=filename, doc_type="pdf")
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        doc.total_pages = len(reader.pages)

        # Use Gemini for rich extraction on the whole doc (up to 20 pages)
        # then fall back to pypdf for larger docs page by page
        if doc.total_pages <= 20:
            result = await self._gemini_extract_document(file_bytes, "application/pdf", filename)
            doc.subject = result.get("subject", "General")
            doc.pages = result.get("pages", [])
            doc.formulas = result.get("formulas", [])
            doc.diagrams = result.get("diagrams", [])
        else:
            # Chunk into 20-page batches
            await self._parse_pdf_in_batches(reader, file_bytes, doc)

        return doc

    async def _parse_pdf_in_batches(self, reader: pypdf.PdfReader, file_bytes: bytes, doc: ParsedDocument) -> None:
        """Parse large PDFs in 20-page batches using pypdf for text + Gemini for enrichment."""
        import pypdf
        writer_batch = []
        batch_size = 20

        for start in range(0, len(reader.pages), batch_size):
            end = min(start + batch_size, len(reader.pages))
            writer = pypdf.PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            buf = io.BytesIO()
            writer.write(buf)
            batch_bytes = buf.getvalue()

            result = await self._gemini_extract_document(batch_bytes, "application/pdf", doc.filename)
            # Offset page numbers
            for page in result.get("pages", []):
                if page.page_number is None:
                    page.page_number = 0
                page.page_number += start
            doc.pages.extend(result.get("pages", []))
            doc.formulas.extend(result.get("formulas", []))
            doc.diagrams.extend(result.get("diagrams", []))

        if not doc.subject or doc.subject == "General":
            # Infer subject from first batch text
            sample_text = " ".join(p.text[:200] for p in doc.pages[:5])
            doc.subject = await self._infer_subject(sample_text)

    async def _gemini_extract_document(self, file_bytes: bytes, mime_type: str, filename: str) -> dict:
        """Send document to Gemini for full structured extraction."""
        prompt = f"""
You are an expert educational content analyser. Extract ALL content from this document.

For EACH page/slide identify:
- Page number
- Chapter/section heading
- Full text content
- Mathematical equations (in LaTeX notation, e.g. F = ma, \\frac{{d^2x}}{{dt^2}})
- Diagram/figure descriptions
- Table content

Also identify:
- Subject domain
- Key formulas with explanations
- Visual diagrams with captions

Respond in this JSON format:
{{
  "subject": "Physics",
  "pages": [
    {{
      "page_number": 1,
      "chapter": "chapter name or null",
      "text": "full page text",
      "equations": ["F = ma", "E = mc^2"],
      "figure_descriptions": ["Circuit diagram showing..."],
      "tables": []
    }}
  ],
  "formulas": [
    {{
      "name": "Newton's Second Law",
      "latex": "F = ma",
      "explanation": "Force equals mass times acceleration",
      "related_concepts": ["momentum", "kinematics"],
      "symbol_meanings": {{"F": "force in Newtons", "m": "mass in kg", "a": "acceleration m/s^2"}}
    }}
  ],
  "diagrams": [
    {{
      "caption": "Free body diagram",
      "chapter": "chapter name",
      "page_number": 3,
      "description": "Shows forces acting on object"
    }}
  ]
}}
"""
        import json
        raw = await generate_with_document(prompt, file_bytes, mime_type)
        # Strip markdown code fences if present
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON, attempting partial parse", filename=filename)
            data = {"subject": "General", "pages": [], "formulas": [], "diagrams": []}

        # Convert page dicts to ParsedPage objects
        parsed_pages = []
        for p in data.get("pages", []):
            try:
                page_num = p.get("page_number")
                page_number = int(page_num) if page_num is not None else 0
            except (ValueError, TypeError):
                page_number = 0
            parsed_pages.append(ParsedPage(
                page_number=page_number,
                text=p.get("text", ""),
                chapter=p.get("chapter"),
                raw_equations=p.get("equations", []),
                figure_descriptions=p.get("figure_descriptions", []),
                tables=p.get("tables", []),
            ))
        data["pages"] = parsed_pages
        return data

    # ── PPTX ─────────────────────────────────────────────────────────────────

    async def _parse_pptx(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        doc = ParsedDocument(filename=filename, doc_type="pptx")
        prs = Presentation(io.BytesIO(file_bytes))
        doc.total_pages = len(prs.slides)

        # Extract text locally first
        local_pages = []
        for i, slide in enumerate(prs.slides):
            texts = []
            chapter = None
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            if shape.shape_type == 13 or (para.runs and para.runs[0].font.size and para.runs[0].font.size > 2000000):
                                chapter = t
                            texts.append(t)
            local_pages.append(ParsedPage(
                page_number=i + 1,
                chapter=chapter,
                text="\n".join(texts),
            ))

        # Enrich with Gemini for formulas/diagrams
        result = await self._gemini_extract_document(file_bytes,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", filename)
        doc.subject = result.get("subject", "General")
        doc.formulas = result.get("formulas", [])
        doc.diagrams = result.get("diagrams", [])

        # Merge local + gemini pages (local text is more reliable, gemini adds equations)
        gemini_pages = {p.page_number: p for p in result.get("pages", [])}
        for lp in local_pages:
            gp = gemini_pages.get(lp.page_number)
            if gp:
                lp.raw_equations = gp.raw_equations
                lp.figure_descriptions = gp.figure_descriptions
                if not lp.chapter:
                    lp.chapter = gp.chapter
        doc.pages = local_pages
        return doc

    # ── Image ─────────────────────────────────────────────────────────────────

    async def _parse_image(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        doc = ParsedDocument(filename=filename, doc_type="image")
        doc.total_pages = 1
        prompt = """Extract all text, equations (in LaTeX), and describe any diagrams from this image.
Respond in JSON: {"text": "...", "equations": [], "figure_descriptions": [], "chapter": null}"""
        import json
        raw = await generate_with_image(prompt, file_bytes)
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"text": raw, "equations": [], "figure_descriptions": []}

        doc.pages = [ParsedPage(
            page_number=1,
            text=data.get("text", ""),
            raw_equations=data.get("equations", []),
            figure_descriptions=data.get("figure_descriptions", []),
        )]
        return doc

    # ── Plain Text ────────────────────────────────────────────────────────────

    async def _parse_text(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        doc = ParsedDocument(filename=filename, doc_type="text")
        text = file_bytes.decode("utf-8", errors="ignore")
        doc.total_pages = 1
        doc.pages = [ParsedPage(page_number=1, text=text)]
        return doc

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _infer_subject(self, sample_text: str) -> str:
        prompt = f"What is the academic subject of this text? Reply with one word only (e.g. Physics, Chemistry, Mathematics, Biology, Engineering).\n\nText: {sample_text[:500]}"
        try:
            return (await generate_text(prompt)).strip()
        except Exception:
            return "General"

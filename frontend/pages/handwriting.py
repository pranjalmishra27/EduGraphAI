"""
frontend/pages/handwriting.py
Upload handwritten solution images and get step-by-step AI evaluation.
"""
import streamlit as st
from frontend.utils.api_client import evaluate_handwriting
from frontend.components.ui_components import section_header, empty_state

MIME_MAP = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}

TOPICS = [
    "Newton's Laws", "Kinematics", "Work-Energy Theorem",
    "Electromagnetic Induction", "Gauss's Law", "Thermodynamics",
    "Wave Optics", "Integration", "Differentiation",
    "Chemical Equilibrium", "Electrochemistry", "Other (specify below)",
]


def render():
    section_header(
        "Handwriting Evaluator",
        "Upload your handwritten answer — AI will analyse every step",
        "✍️",
    )

    col_upload, col_form = st.columns([1, 1])

    # ── Upload column ────────────────────────────────────────────────────────
    with col_upload:
        st.markdown("#### 📷 Upload Handwritten Solution")
        image_file = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png", "webp"],
            help="Clear photo of your handwritten answer. Better lighting = better evaluation.",
        )
        if image_file:
            st.image(image_file, caption="Your uploaded solution", use_container_width=True)

    # ── Form column ──────────────────────────────────────────────────────────
    with col_form:
        st.markdown("#### ⚙️ Evaluation Settings")
        topic_select = st.selectbox("Topic", TOPICS)
        topic = st.text_input(
            "Custom topic (if 'Other' selected)",
            placeholder="e.g. Bernoulli's Theorem derivation",
        ) if "Other" in topic_select else topic_select

        expected_answer = st.text_area(
            "Model answer / expected solution (optional)",
            placeholder="Paste the correct solution here for comparison…",
            height=140,
        )

        evaluate_btn = st.button(
            "🔍 Evaluate Solution", type="primary",
            use_container_width=True,
            disabled=(image_file is None),
        )

    if not image_file:
        st.markdown("---")
        empty_state("📷", "Upload an image to begin evaluation")
        return

    if evaluate_btn:
        ext = image_file.name.rsplit(".", 1)[-1].lower()
        mime = MIME_MAP.get(ext, "image/jpeg")
        with st.spinner("AI is analysing your handwriting… this takes ~20-30 seconds."):
            try:
                result = evaluate_handwriting(
                    image_bytes=image_file.read(),
                    filename=image_file.name,
                    content_type=mime,
                    topic=topic,
                    expected_answer=expected_answer or "",
                )
                st.session_state["hw_result"] = result
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    result = st.session_state.get("hw_result")
    if not result:
        return

    st.markdown("---")
    st.markdown("### 📋 Evaluation Results")

    # ── Score banner ─────────────────────────────────────────────────────────
    score = result.get("overall_score", 0)
    score_pct = score / 10 * 100
    color = "#22c55e" if score >= 7 else "#f59e0b" if score >= 5 else "#ef4444"
    st.markdown(
        f"""
<div style="background:#1e293b;border:2px solid {color};border-radius:12px;
  padding:20px;text-align:center;margin:14px 0;">
  <div style="font-size:3rem;font-weight:800;color:{color};">{score:.1f}<span style="font-size:1.5rem;color:#64748b;">/10</span></div>
  <div style="color:#94a3b8;font-size:0.95rem;margin-top:4px;">Overall Score</div>
  <div style="background:#0f172a;border-radius:8px;height:8px;margin:10px auto;max-width:300px;">
    <div style="background:{color};height:8px;border-radius:8px;width:{score_pct:.0f}%;"></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── OCR text ─────────────────────────────────────────────────────────────
    with st.expander("📝 Extracted Text (OCR)", expanded=False):
        st.code(result.get("extracted_text", ""), language=None)

    # ── Four analysis tabs ────────────────────────────────────────────────────
    tabs = st.tabs(["✅ Steps Found", "❌ Missing Steps", "⚠️ Errors", "💬 Feedback"])

    with tabs[0]:
        steps = result.get("steps_identified", [])
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(
                    f'<div style="background:#14532d22;border-left:3px solid #22c55e;'
                    f'padding:8px 14px;border-radius:0 6px 6px 0;margin:5px 0;color:#e2e8f0;">'
                    f'<b style="color:#4ade80;">Step {i}:</b> {step}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No identifiable steps extracted.")

    with tabs[1]:
        missing = result.get("missing_steps", [])
        if missing:
            for m in missing:
                st.markdown(
                    f'<div style="background:#7f1d1d22;border-left:3px solid #ef4444;'
                    f'padding:8px 14px;border-radius:0 6px 6px 0;margin:5px 0;color:#e2e8f0;">'
                    f'⚠️ {m}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No missing steps detected. 🎉")

    with tabs[2]:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**🔴 Conceptual Errors**")
            errs = result.get("conceptual_errors", [])
            for e in errs:
                st.markdown(f"• {e}")
            if not errs:
                st.success("None")
        with col_b:
            st.markdown("**🟡 Calculation Errors**")
            errs = result.get("calculation_errors", [])
            for e in errs:
                st.markdown(f"• {e}")
            if not errs:
                st.success("None")
        with col_c:
            st.markdown("**🔵 Formula Errors**")
            errs = result.get("formula_errors", [])
            for e in errs:
                st.markdown(f"• {e}")
            if not errs:
                st.success("None")

    with tabs[3]:
        feedback = result.get("detailed_feedback", "")
        if feedback:
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
                f'padding:18px 22px;color:#e2e8f0;line-height:1.7;">{feedback}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No feedback generated.")

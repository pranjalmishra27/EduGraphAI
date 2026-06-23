"""
frontend/pages/qa_chat.py
Context-aware question answering with source citations, formulas, diagrams, KG.
"""
import streamlit as st
from frontend.utils.api_client import ask_question, list_documents
from frontend.components.ui_components import (
    source_card, formula_card, section_header, empty_state
)

SUBJECTS = ["Auto-detect", "Physics", "Chemistry", "Mathematics",
            "Biology", "Engineering", "Computer Science", "Economics"]

EXAMPLE_QUESTIONS = [
    "What is Newton's Second Law and how is it derived?",
    "Explain Gauss's Law with an example.",
    "How does Bernoulli's equation relate to fluid dynamics?",
    "What are Maxwell's Equations?",
    "Explain the ideal gas equation and its assumptions.",
]


def render():
    section_header("Ask EduGraphAI", "Ask any question — answered from your uploaded documents", "💬")

    # ── Check documents exist ────────────────────────────────────────────────
    try:
        docs = list_documents()
    except Exception:
        docs = []

    if not docs:
        empty_state(
            "📂",
            "No documents uploaded yet",
            "Head to Document Library and upload a textbook or lecture notes first.",
        )
        return

    # Retrieve query settings from global session state (set in sidebar)
    subject_opt = st.session_state.get("subject_filter", "Auto-detect")
    subject = None if subject_opt == "Auto-detect" else subject_opt
    include_formulas = st.session_state.get("include_formulas", True)
    include_diagrams = st.session_state.get("include_diagrams", True)

    # ── Sidebar controls (Page-specific) ────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**💡 Example questions**")
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex[:55] + "…" if len(ex) > 55 else ex, key=f"ex_{ex[:20]}"):
                st.session_state["qa_question"] = ex

    # ── Chat history ────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Input ────────────────────────────────────────────────────────────────
    question = st.text_area(
        "Your question",
        value=st.session_state.pop("qa_question", ""),
        placeholder="e.g. Explain Newton's Second Law with a derivation…",
        height=100,
        key="qa_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("🔍 Ask", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ Clear history", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

    if ask and question.strip():
        with st.spinner("Retrieving from your documents and generating answer…"):
            try:
                result = ask_question(
                    question=question.strip(),
                    subject=subject,
                    include_diagrams=include_diagrams,
                    include_formulas=include_formulas,
                )
                st.session_state.chat_history.insert(0, {"question": question.strip(), "result": result})
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Render chat history ──────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        q = entry["question"]
        r = entry["result"]

        # Question bubble
        st.markdown(
            f'<div style="background:#1e3a5f;border-radius:10px 10px 2px 10px;'
            f'padding:12px 18px;margin:14px 0 6px;color:#bfdbfe;font-weight:600;">'
            f'🧑‍🎓 {q}</div>',
            unsafe_allow_html=True,
        )

        # Answer
        st.markdown(
            f'<div style="background:#1e293b;border:1px solid #334155;border-radius:2px 10px 10px 10px;'
            f'padding:16px 20px;margin-bottom:8px;">',
            unsafe_allow_html=True,
        )
        st.markdown(r.get("answer", ""), unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tabs for evidence
        tabs = st.tabs(["📚 Sources", "🔢 Formulas", "🖼️ Diagrams", "🕸️ Concepts"])

        with tabs[0]:
            sources = r.get("sources", [])
            if sources:
                for i, src in enumerate(sources, 1):
                    source_card(src, i)
            else:
                st.caption("No source chunks retrieved.")

        with tabs[1]:
            formulas = r.get("formulas", [])
            if formulas:
                for f in formulas:
                    formula_card(f)
            else:
                st.caption("No formulas retrieved for this question.")

        with tabs[2]:
            diagrams = r.get("diagrams", [])
            if diagrams:
                for d in diagrams:
                    st.markdown(
                        f'<div style="background:#1e293b;border:1px solid #334155;'
                        f'border-radius:8px;padding:12px 16px;margin:6px 0;">'
                        f'<b style="color:#93c5fd;">🖼️ {d.get("caption","")}</b><br>'
                        f'<span style="color:#64748b;font-size:0.82rem;">'
                        f'{d.get("chapter","") or ""}'
                        f'{" · Page " + str(d["page_number"]) if d.get("page_number") else ""}'
                        f' — {d.get("document_filename","")}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No diagrams retrieved.")

        with tabs[3]:
            kg = r.get("knowledge_graph_context", [])
            if kg:
                for rec in kg[:6]:
                    concept = rec.get("concept", "")
                    desc = rec.get("description", "")
                    connections = rec.get("connections", [])
                    with st.expander(f"🔵 {concept}", expanded=False):
                        if desc:
                            st.markdown(f"_{desc}_")
                        if connections:
                            conn_list = [
                                f"`{c['target']}` via **{c['rel']}**"
                                for c in connections if c.get("target")
                            ]
                            st.markdown("**Related:** " + " · ".join(conn_list[:5]))
            else:
                st.caption("No knowledge graph context found.")

        st.markdown('<hr style="border-color:#1e293b;">', unsafe_allow_html=True)

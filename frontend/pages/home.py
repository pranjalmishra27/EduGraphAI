"""
frontend/pages/home.py
Landing / dashboard overview page.
"""
import streamlit as st


def render():
    st.markdown("""
    <div style="text-align:center;padding:40px 0 10px;">
      <div style="font-size:3.5rem;">🎓</div>
      <h1 style="font-size:2.6rem;font-weight:800;
        background:linear-gradient(135deg,#60a5fa,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:8px 0;">
        EduGraphAI
      </h1>
      <p style="color:#94a3b8;font-size:1.1rem;max-width:600px;margin:0 auto;">
        Your Multimodal Educational Intelligence Platform — upload textbooks,
        ask questions, explore knowledge graphs, and get personalized learning paths.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)

    features = [
        ("📁", "Document Library", "Upload PDFs, PPTX, images. AI extracts text, formulas, and diagrams automatically."),
        ("💬", "Ask EduGraphAI", "Ask any question from your uploaded material. Get cited answers with formula references."),
        ("🕸️", "Knowledge Graph", "Visualize concept dependencies and prerequisite learning paths in an interactive graph."),
        ("✍️", "Handwriting Eval", "Upload photos of handwritten answers. Get step-by-step AI feedback and error analysis."),
        ("📊", "Exam Intelligence", "Upload past exam papers to discover high-priority topics and question trends."),
        ("📈", "Student Analytics", "Track quiz performance. Get AI-powered score predictions and study plans."),
    ]

    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
              padding:20px;margin-bottom:16px;height:150px;">
              <div style="font-size:1.8rem;">{icon}</div>
              <div style="font-weight:700;color:#f1f5f9;margin:8px 0 6px;">{title}</div>
              <div style="color:#94a3b8;font-size:0.84rem;line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 **Get started** by uploading a document in **Document Library**, then head to **Ask EduGraphAI**.")

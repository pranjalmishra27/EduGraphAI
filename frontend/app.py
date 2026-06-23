"""
frontend/app.py
EduGraphAI Streamlit application — main entry with multi-page navigation.
Run with: streamlit run frontend/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="EduGraphAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark sidebar */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Hide Streamlit native sidebar navigation */
  [data-testid="stSidebarNav"] { display: none; }

  /* Main background */
  .stApp { background: #0f172a; color: #e2e8f0; }

  /* Cards */
  .edu-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 18px 22px; margin: 10px 0;
  }

  /* Buttons */
  .stButton > button {
    background: #3b82f6; color: white; border: none;
    border-radius: 6px; font-weight: 600;
  }
  .stButton > button:hover { background: #2563eb; }

  /* Input fields */
  .stTextInput input, .stTextArea textarea, .stSelectbox div {
    background: #1e293b !important; color: #f1f5f9 !important;
    border-color: #334155 !important;
  }

  /* Metric labels */
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
  [data-testid="stMetricValue"] { color: #f1f5f9 !important; }

  /* Expanders */
  .streamlit-expanderHeader { background: #1e293b !important; }

  /* Dividers */
  hr { border-color: #334155; }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
      <div style="font-size:2.2rem;">🎓</div>
      <div style="font-size:1.3rem;font-weight:800;color:#60a5fa;letter-spacing:1px;">EduGraphAI</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:4px;">Educational Intelligence Platform</div>
    </div>
    <hr style="border-color:#1e293b;margin:10px 0 20px;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "📁 Document Library",
            "💬 Ask EduGraphAI",
            "🕸️ Knowledge Graph",
            "✍️ Handwriting Evaluator",
            "📊 Exam Intelligence",
            "📈 Student Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Session: student ID
    with st.expander("⚙️ Session Settings"):
        if "student_id" not in st.session_state:
            st.session_state.student_id = ""
        st.session_state.student_id = st.text_input(
            "Student ID", value=st.session_state.student_id, placeholder="e.g. stu_abc123"
        )

    # Global Query Settings
    SUBJECTS_FILTER = ["Auto-detect", "Physics", "Chemistry", "Mathematics", "Biology", "Engineering", "Computer Science", "Economics"]
    with st.expander("⚙️ Query Settings", expanded=True):
        if "subject_filter" not in st.session_state:
            st.session_state.subject_filter = "Auto-detect"
        try:
            idx = SUBJECTS_FILTER.index(st.session_state.subject_filter)
        except ValueError:
            idx = 0
        st.session_state.subject_filter = st.selectbox(
            "Subject filter", options=SUBJECTS_FILTER, index=idx
        )
        
        if "include_formulas" not in st.session_state:
            st.session_state.include_formulas = True
        st.session_state.include_formulas = st.toggle(
            "Include formulas", value=st.session_state.include_formulas
        )
        
        if "include_diagrams" not in st.session_state:
            st.session_state.include_diagrams = True
        st.session_state.include_diagrams = st.toggle(
            "Include diagrams", value=st.session_state.include_diagrams
        )

    st.markdown("""
    <div style="position:fixed;bottom:20px;left:0;width:240px;text-align:center;font-size:0.7rem;color:#334155;">
      EduGraphAI Platform
    </div>
    """, unsafe_allow_html=True)

# ─── Page Routing ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from frontend.pages.home import render
elif page == "📁 Document Library":
    from frontend.pages.document_library import render
elif page == "💬 Ask EduGraphAI":
    from frontend.pages.qa_chat import render
elif page == "🕸️ Knowledge Graph":
    from frontend.pages.knowledge_graph import render
elif page == "✍️ Handwriting Evaluator":
    from frontend.pages.handwriting import render
elif page == "📊 Exam Intelligence":
    from frontend.pages.exam_intelligence import render
elif page == "📈 Student Analytics":
    from frontend.pages.student_analytics import render
else:
    from frontend.pages.home import render

render()

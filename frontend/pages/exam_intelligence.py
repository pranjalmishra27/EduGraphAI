"""
frontend/pages/exam_intelligence.py
Upload previous-year exam papers and get topic frequency analysis + recommendations.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from frontend.utils.api_client import analyze_exam
from frontend.components.ui_components import section_header, empty_state


def render():
    section_header(
        "Exam Intelligence",
        "Upload past exam papers — discover high-priority topics and question trends",
        "📊",
    )

    col_upload, col_info = st.columns([1, 1])

    with col_upload:
        st.markdown("#### 📄 Upload Exam Paper")
        exam_file = st.file_uploader(
            "Choose PDF exam paper",
            type=["pdf"],
            help="Previous-year question papers, sample papers, or assignment PDFs.",
        )

    with col_info:
        st.markdown("#### ℹ️ What you'll get")
        st.markdown("""
        - 🎯 **High-priority topics** ranked by frequency
        - 📈 **Topic frequency chart** — visualised
        - 📐 **Important derivations** identified
        - 🔢 **Numerical problem types** catalogued
        - 💡 **Personalised recommendations** for exam prep
        """)

    if not exam_file:
        st.markdown("---")
        empty_state("📋", "Upload an exam paper PDF to begin analysis")
        return

    analyze_btn = st.button("🔍 Analyse Exam Paper", type="primary", use_container_width=False)

    if analyze_btn:
        with st.spinner("Analyzing the exam paper for trends and patterns… (~20-40s)"):
            try:
                result = analyze_exam(exam_file.read(), exam_file.name)
                st.session_state["exam_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    result = st.session_state.get("exam_result")
    if not result:
        return

    st.markdown("---")
    st.markdown("### 📊 Exam Analysis Results")

    # ── Summary metrics ───────────────────────────────────────────────────────
    topic_freq = result.get("topic_frequency", {})
    high_priority = result.get("high_priority_topics", [])
    derivations = result.get("important_derivations", [])
    numericals = result.get("numerical_problems", [])
    recommendations = result.get("recommendations", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Topics Found", len(topic_freq))
    c2.metric("🎯 High Priority", len(high_priority))
    c3.metric("📐 Derivations", len(derivations))
    c4.metric("🔢 Numericals", len(numericals))

    tabs = st.tabs(["🎯 High Priority", "📈 Topic Chart", "📐 Derivations", "🔢 Numericals", "💡 Recommendations"])

    # ── High priority topics ──────────────────────────────────────────────────
    with tabs[0]:
        if high_priority:
            st.markdown("Topics you **must** master before the exam:")
            for i, topic in enumerate(high_priority, 1):
                freq = topic_freq.get(topic, "—")
                pct_width = min(int(float(freq) / max(topic_freq.values()) * 100), 100) if topic_freq and isinstance(freq, (int, float)) else 40
                st.markdown(
                    f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
  padding:12px 16px;margin:6px 0;display:flex;align-items:center;gap:14px;">
  <div style="background:#3b82f6;color:white;border-radius:50%;width:28px;height:28px;
    display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;
    flex-shrink:0;">{i}</div>
  <div style="flex:1;">
    <div style="color:#f1f5f9;font-weight:600;">{topic}</div>
    <div style="background:#0f172a;height:4px;border-radius:4px;margin-top:6px;">
      <div style="background:#3b82f6;height:4px;border-radius:4px;width:{pct_width}%;"></div>
    </div>
  </div>
  <div style="color:#60a5fa;font-weight:700;font-size:0.9rem;">×{freq}</div>
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No high-priority topics identified.")

    # ── Topic frequency bar chart ─────────────────────────────────────────────
    with tabs[1]:
        if topic_freq:
            sorted_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
            labels = [t[0] for t in sorted_topics]
            values = [t[1] for t in sorted_topics]

            fig = go.Figure(go.Bar(
                x=values, y=labels,
                orientation="h",
                marker=dict(
                    color=values,
                    colorscale="Blues",
                    showscale=False,
                ),
                text=[f"  {v}" for v in values],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0"),
                xaxis=dict(showgrid=False, color="#64748b"),
                yaxis=dict(color="#e2e8f0", autorange="reversed"),
                height=max(300, len(labels) * 32),
                margin=dict(l=10, r=60, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart for relative importance
            fig2 = px.pie(
                names=labels[:8],
                values=values[:8],
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            fig2.update_layout(
                paper_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                showlegend=True,
                legend=dict(font=dict(color="#e2e8f0")),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No topic frequency data extracted.")

    # ── Derivations ───────────────────────────────────────────────────────────
    with tabs[2]:
        if derivations:
            for d in derivations:
                st.markdown(
                    f'<div style="background:#1e293b;border-left:4px solid #a78bfa;'
                    f'padding:10px 16px;border-radius:0 8px 8px 0;margin:6px 0;color:#e2e8f0;">'
                    f'📐 {d}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No specific derivations identified.")

    # ── Numericals ────────────────────────────────────────────────────────────
    with tabs[3]:
        if numericals:
            for n in numericals:
                st.markdown(
                    f'<div style="background:#1e293b;border-left:4px solid #34d399;'
                    f'padding:10px 16px;border-radius:0 8px 8px 0;margin:6px 0;color:#e2e8f0;">'
                    f'🔢 {n}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No numerical problem types identified.")

    # ── Recommendations ───────────────────────────────────────────────────────
    with tabs[4]:
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(
                    f'<div style="background:#1e3a5f;border:1px solid #2563eb;border-radius:8px;'
                    f'padding:12px 16px;margin:7px 0;">'
                    f'<span style="color:#60a5fa;font-weight:700;">💡 {i}.</span> '
                    f'<span style="color:#e2e8f0;">{rec}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recommendations generated.")

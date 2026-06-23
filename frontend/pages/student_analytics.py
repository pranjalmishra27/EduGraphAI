"""
frontend/pages/student_analytics.py
Student profile, quiz tracker, ML recommendations, and performance prediction.
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from frontend.utils.api_client import (
    create_student, record_quiz_attempt,
    get_student_attempts, get_recommendations,
    predict_performance, get_dashboard,
)
from frontend.components.ui_components import section_header, risk_badge, empty_state, metric_row

SUBJECTS = ["Physics", "Chemistry", "Mathematics", "Biology",
            "Engineering", "Computer Science", "Economics"]

TOPICS_BY_SUBJECT = {
    "Physics": ["Kinematics", "Newton's Laws", "Work-Energy Theorem",
                "Rotational Motion", "Gravitation", "Thermodynamics",
                "Electrostatics", "Current Electricity", "Electromagnetic Induction", "Optics"],
    "Chemistry": ["Atomic Structure", "Periodic Table", "Chemical Bonding",
                  "Thermodynamics", "Equilibrium", "Electrochemistry", "Organic Chemistry"],
    "Mathematics": ["Sets & Functions", "Trigonometry", "Algebra",
                    "Coordinate Geometry", "Differential Calculus", "Integral Calculus",
                    "Vectors", "Probability"],
    "Biology": ["Cell Biology", "Genetics", "Evolution", "Physiology",
                "Ecology", "Biochemistry"],
    "Engineering": ["Circuits", "Signals & Systems", "Control Systems",
                    "Thermodynamics", "Fluid Mechanics", "Material Science"],
    "Computer Science": ["Data Structures", "Algorithms", "Operating Systems",
                         "DBMS", "Networks", "Machine Learning"],
    "Economics": ["Microeconomics", "Macroeconomics", "Statistics",
                  "Game Theory", "International Trade"],
}


def render():
    section_header("Student Analytics", "Track performance · Get AI recommendations · Predict exam scores", "📈")

    tabs = st.tabs(["👤 Profile", "📝 Log Quiz", "📊 Dashboard", "🎯 Recommendations", "🔮 Predict Score"])

    # ── Tab 1: Profile / Registration ────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Create or set your student profile")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full name", placeholder="Pranjal Mishra")
            email = st.text_input("Email", placeholder="you@example.com")
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("➕ Create Student Profile", type="primary"):
                if name and email:
                    try:
                        result = create_student(name, email)
                        st.success(f"Student created! ID: `{result['student_id']}`")
                        st.session_state.student_id = result["student_id"]
                        st.info("Your Student ID has been saved to the session.")
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Please fill in both name and email.")

        st.markdown("---")
        st.markdown("**Or use an existing Student ID (set in sidebar ⚙️ settings)**")
        sid = st.session_state.get("student_id", "")
        if sid:
            st.success(f"Active Student ID: `{sid}`")
        else:
            st.warning("No Student ID set. Create a profile above or enter your ID in the sidebar settings.")

    # ── Tab 2: Log Quiz Attempt ───────────────────────────────────────────────
    with tabs[1]:
        sid = st.session_state.get("student_id", "")
        if not sid:
            st.warning("Set your Student ID in the sidebar first.")
        else:
            st.markdown(f"#### Log quiz attempt for `{sid}`")
            c1, c2 = st.columns(2)
            with c1:
                global_sub = st.session_state.get("subject_filter", "Auto-detect")
                default_sub = "Physics" if global_sub == "Auto-detect" or global_sub not in SUBJECTS else global_sub
                try:
                    idx = SUBJECTS.index(default_sub)
                except ValueError:
                    idx = 0
                subject = st.selectbox("Subject", SUBJECTS, index=idx, key="quiz_subject")
                topics = TOPICS_BY_SUBJECT.get(subject, ["General"])
                topic = st.selectbox("Topic", topics + ["Other"])
                if topic == "Other":
                    topic = st.text_input("Custom topic")
            with c2:
                score = st.number_input("Score obtained", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
                max_score = st.number_input("Maximum marks", min_value=1.0, max_value=200.0, value=100.0, step=1.0)
                difficulty = st.select_slider("Difficulty", ["easy", "medium", "hard"], value="medium")
                time_spent = st.number_input("Time spent (minutes)", min_value=0.0, max_value=300.0, value=30.0)

            pct = score / max_score * 100
            st.markdown(f"**Score: {pct:.1f}%**")
            prog_color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
            st.markdown(
                f'<div style="background:#0f172a;height:8px;border-radius:4px;">'
                f'<div style="background:{prog_color};height:8px;border-radius:4px;width:{min(pct,100):.0f}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if st.button("💾 Save Quiz Attempt", type="primary"):
                try:
                    record_quiz_attempt(sid, topic, score, max_score, difficulty, time_spent)
                    st.success(f"Quiz attempt for **{topic}** saved! ({pct:.1f}%)")
                    # Clear cached dashboard
                    st.session_state.pop("dashboard_data", None)
                except Exception as e:
                    st.error(str(e))

    # ── Tab 3: Dashboard ──────────────────────────────────────────────────────
    with tabs[2]:
        sid = st.session_state.get("student_id", "")
        if not sid:
            st.warning("Set your Student ID in the sidebar first.")
        else:
            if st.button("🔄 Refresh Dashboard") or "dashboard_data" not in st.session_state:
                try:
                    st.session_state.dashboard_data = get_dashboard(sid)
                    st.session_state.attempts_data = get_student_attempts(sid)
                except Exception as e:
                    st.error(str(e))

            dash = st.session_state.get("dashboard_data", {})
            attempts = st.session_state.get("attempts_data", [])

            if not dash or dash.get("attempts", 0) == 0:
                empty_state("📊", "No quiz data yet", "Log some quiz attempts in the previous tab.")
            else:
                metric_row([
                    ("Total Attempts", str(dash.get("total_attempts", 0)), None),
                    ("Overall Average", f"{dash.get('overall_avg', 0):.1f}%", None),
                    ("Topics Studied", str(dash.get("topics_studied", 0)), None),
                ])

                st.markdown("---")

                # Topic breakdown bar chart
                topic_data = dash.get("topic_breakdown", [])
                if topic_data:
                    st.markdown("##### 📊 Topic Performance")
                    topics_list = [t["topic"] for t in topic_data]
                    means = [t["mean"] for t in topic_data]
                    counts = [t["count"] for t in topic_data]
                    colors = ["#22c55e" if m >= 75 else "#f59e0b" if m >= 50 else "#ef4444" for m in means]

                    fig = go.Figure(go.Bar(
                        x=topics_list, y=means,
                        marker_color=colors,
                        text=[f"{m:.0f}% ({c}x)" for m, c in zip(means, counts)],
                        textposition="outside",
                        textfont=dict(color="#e2e8f0", size=11),
                    ))
                    fig.update_layout(
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#1e293b",
                        font=dict(color="#e2e8f0"),
                        yaxis=dict(range=[0, 110], color="#64748b", title="Score %"),
                        xaxis=dict(color="#e2e8f0"),
                        height=360,
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    fig.add_hline(y=75, line_dash="dot", line_color="#22c55e",
                                  annotation_text="75% target", annotation_font_color="#22c55e")
                    fig.add_hline(y=50, line_dash="dot", line_color="#ef4444",
                                  annotation_text="50% pass mark", annotation_font_color="#ef4444")
                    st.plotly_chart(fig, use_container_width=True)

                # Recent attempts trend line
                recent = dash.get("recent_scores", [])
                if len(recent) >= 3:
                    st.markdown("##### 📈 Recent Score Trend")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, len(recent) + 1)),
                        y=[r["pct_score"] for r in recent],
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=8, color="#60a5fa"),
                        text=[r["topic"] for r in recent],
                        hovertemplate="%{text}<br>%{y:.1f}%",
                    ))
                    fig2.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                        font=dict(color="#e2e8f0"),
                        yaxis=dict(range=[0, 105], color="#64748b", title="Score %"),
                        xaxis=dict(title="Attempt #", color="#64748b"),
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Recommendations ────────────────────────────────────────────────
    with tabs[3]:
        sid = st.session_state.get("student_id", "")
        if not sid:
            st.warning("Set your Student ID in the sidebar first.")
        else:
            global_sub = st.session_state.get("subject_filter", "Auto-detect")
            default_sub = "Physics" if global_sub == "Auto-detect" or global_sub not in SUBJECTS else global_sub
            try:
                idx = SUBJECTS.index(default_sub)
            except ValueError:
                idx = 0
            subject_rec = st.selectbox("Subject for recommendations", SUBJECTS, index=idx, key="rec_subject")
            if st.button("🎯 Generate Recommendations", type="primary"):
                with st.spinner("Analyzing your learning profile…"):
                    try:
                        rec = get_recommendations(sid, subject_rec)
                        st.session_state["rec_data"] = rec
                    except Exception as e:
                        st.error(str(e))

            rec = st.session_state.get("rec_data")
            if rec:
                # Predicted improvement banner
                imp = rec.get("predicted_improvement", 0)
                st.markdown(
                    f'<div style="background:#14532d22;border:1px solid #22c55e;border-radius:10px;'
                    f'padding:16px;text-align:center;margin:10px 0;">'
                    f'<span style="color:#4ade80;font-size:1.8rem;font-weight:800;">+{imp:.1f}%</span>'
                    f'<div style="color:#94a3b8;font-size:0.88rem;">Predicted score improvement if you follow the plan</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                c_weak, c_order = st.columns(2)
                with c_weak:
                    st.markdown("##### ⚠️ Weak Topics")
                    weak = rec.get("weak_topics", [])
                    if weak:
                        for t in weak:
                            st.markdown(
                                f'<div style="background:#7f1d1d22;border-left:3px solid #ef4444;'
                                f'padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;color:#fca5a5;">'
                                f'⚠️ {t}</div>', unsafe_allow_html=True
                            )
                    else:
                        st.success("No weak topics! Keep it up. 🎉")

                with c_order:
                    st.markdown("##### 📚 Recommended Study Order")
                    order = rec.get("recommended_order", [])
                    for i, t in enumerate(order, 1):
                        st.markdown(
                            f'<div style="background:#1e293b;border-left:3px solid #3b82f6;'
                            f'padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;color:#bfdbfe;">'
                            f'<b>{i}.</b> {t}</div>', unsafe_allow_html=True
                        )

                st.markdown("##### 🔁 Revision Priorities")
                for t in rec.get("revision_priorities", []):
                    st.markdown(
                        f'<div style="background:#1e3a5f;border-left:3px solid #60a5fa;'
                        f'padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;color:#93c5fd;">'
                        f'🔁 {t}</div>', unsafe_allow_html=True
                    )

    # ── Tab 5: Performance Prediction ────────────────────────────────────────
    with tabs[4]:
        sid = st.session_state.get("student_id", "")
        if not sid:
            st.warning("Set your Student ID in the sidebar first.")
        else:
            st.markdown("Uses an ensemble learning model to predict your exam score.")
            if st.button("🔮 Predict My Performance", type="primary"):
                with st.spinner("Running ML prediction models…"):
                    try:
                        pred = predict_performance(sid)
                        st.session_state["pred_data"] = pred
                    except Exception as e:
                        st.error(str(e))

            pred = st.session_state.get("pred_data")
            if pred:
                # Main score gauge
                score = pred.get("predicted_score", 0)
                pass_prob = pred.get("pass_probability", 0)
                risk = pred.get("risk_level", "medium")
                trend = pred.get("trend", "stable")

                # Score gauge using plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    title={"text": "Predicted Exam Score", "font": {"color": "#e2e8f0", "size": 18}},
                    delta={"reference": 60, "increasing": {"color": "#22c55e"}, "decreasing": {"color": "#ef4444"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": "#3b82f6"},
                        "bgcolor": "#1e293b",
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 40], "color": "#ef444422"},
                            {"range": [40, 75], "color": "#f59e0b22"},
                            {"range": [75, 100], "color": "#22c55e22"},
                        ],
                        "threshold": {
                            "line": {"color": "#f1f5f9", "width": 3},
                            "thickness": 0.85,
                            "value": score,
                        },
                    },
                    number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 36}},
                ))
                fig.update_layout(
                    paper_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats row
                trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                metric_row([
                    ("Pass Probability", f"{pass_prob:.0%}", None),
                    ("Trend", f"{trend_icon} {trend.capitalize()}", None),
                    ("Risk Level", risk.upper(), None),
                ])
                st.markdown("<br>", unsafe_allow_html=True)
                risk_badge(risk)

                # Risk topics
                risk_topics = pred.get("risk_topics", [])
                if risk_topics:
                    st.markdown("#### ⚠️ Risk Topics — Focus here")
                    for t in risk_topics:
                        st.markdown(
                            f'<div style="background:#7f1d1d22;border-left:3px solid #ef4444;'
                            f'padding:8px 14px;border-radius:0 6px 6px 0;margin:4px 0;color:#fca5a5;">'
                            f'⚠️ {t}</div>', unsafe_allow_html=True
                        )

                # Feature importances (if available)
                fi = pred.get("feature_importances", {})
                if fi:
                    st.markdown("#### 🔍 What drives your prediction")
                    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]
                    labels = [x[0].replace("_", " ").title() for x in fi_sorted]
                    values = [x[1] for x in fi_sorted]

                    fig2 = go.Figure(go.Bar(
                        x=values, y=labels, orientation="h",
                        marker_color="#3b82f6",
                        text=[f"{v:.3f}" for v in values],
                        textposition="outside",
                        textfont=dict(color="#e2e8f0"),
                    ))
                    fig2.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                        font=dict(color="#e2e8f0"),
                        yaxis=dict(autorange="reversed", color="#e2e8f0"),
                        xaxis=dict(color="#64748b", title="Feature Importance"),
                        height=300,
                        margin=dict(l=10, r=60, t=10, b=10),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

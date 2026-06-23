"""
frontend/pages/knowledge_graph.py
Interactive knowledge graph explorer with pyvis network + learning path finder.
"""
from __future__ import annotations
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from frontend.utils.api_client import get_graph_data, get_learning_path, list_concepts
from frontend.components.ui_components import section_header, empty_state

SUBJECTS = ["All", "Physics", "Chemistry", "Mathematics",
            "Biology", "Engineering", "Computer Science"]

DIFFICULTY_COLOR = {
    "beginner":     "#22c55e",
    "intermediate": "#f59e0b",
    "advanced":     "#ef4444",
}


def render():
    section_header("Knowledge Graph", "Explore concept dependencies and learning paths", "🕸️")

    tabs = st.tabs(["🌐 Full Graph", "🗺️ Learning Path", "📋 Concept List"])

    # ── Tab 1: Full Graph ────────────────────────────────────────────────────
    with tabs[0]:
        col1, col2 = st.columns([2, 5])
        with col1:
            global_sub = st.session_state.get("subject_filter", "Auto-detect")
            default_sub = "All" if global_sub == "Auto-detect" else global_sub
            if default_sub not in SUBJECTS:
                default_sub = "All"
            try:
                idx = SUBJECTS.index(default_sub)
            except ValueError:
                idx = 0
            subject_filter = st.selectbox("Filter subject", SUBJECTS, index=idx, key="kg_subject")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh = st.button("🔄 Refresh Graph")

        subject = None if subject_filter == "All" else subject_filter

        if "kg_data" not in st.session_state or refresh:
            with st.spinner("Loading knowledge graph…"):
                try:
                    st.session_state.kg_data = get_graph_data(subject)
                except Exception as e:
                    st.error(f"Failed to load graph: {e}")
                    st.session_state.kg_data = {"nodes": [], "edges": []}

        data = st.session_state.kg_data
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not nodes:
            empty_state(
                "🕸️",
                "Knowledge graph is empty",
                "Upload documents first — concepts are extracted automatically during ingestion.",
            )
        else:
            st.markdown(
                f"**{len(nodes)} concepts · {len(edges)} relationships**",
            )
            _render_pyvis_graph(nodes, edges)

            # Legend
            st.markdown("**Difficulty legend:**")
            lc = st.columns(3)
            lc[0].markdown("🟢 Beginner")
            lc[1].markdown("🟡 Intermediate")
            lc[2].markdown("🔴 Advanced")

    # ── Tab 2: Learning Path ─────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("Enter a concept to see its prerequisite learning path.")
        concept_input = st.text_input(
            "Target concept",
            placeholder="e.g. Electromagnetic Induction",
            key="lp_concept",
        )

        if st.button("📍 Get Learning Path", type="primary") and concept_input.strip():
            with st.spinner("Computing learning path…"):
                try:
                    result = get_learning_path(concept_input.strip())
                    path = result.get("learning_path", [])
                    if path:
                        st.markdown(f"### Learning path to: **{concept_input}**")
                        _render_path(path)
                    else:
                        st.info("No prerequisite path found. This may be a root concept.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 3: Concept List ──────────────────────────────────────────────────
    with tabs[2]:
        col_s, col_d = st.columns(2)
        with col_s:
            sub_list = st.selectbox("Subject", SUBJECTS, key="cl_subject")
        with col_d:
            diff_filter = st.selectbox("Difficulty", ["All", "beginner", "intermediate", "advanced"])

        try:
            concepts = list_concepts(None if sub_list == "All" else sub_list)
        except Exception as e:
            st.error(str(e))
            concepts = []

        if diff_filter != "All":
            concepts = [c for c in concepts if c.get("difficulty") == diff_filter]

        if not concepts:
            empty_state("🔵", "No concepts found", "Try a different filter.")
        else:
            st.markdown(f"**{len(concepts)} concepts**")
            for c in concepts:
                diff = c.get("difficulty", "intermediate")
                color = DIFFICULTY_COLOR.get(diff, "#94a3b8")
                st.markdown(
                    f'<div style="background:#1e293b;border-left:4px solid {color};'
                    f'border-radius:0 8px 8px 0;padding:10px 14px;margin:5px 0;">'
                    f'<b style="color:#f1f5f9;">{c["name"]}</b> '
                    f'<span style="color:#64748b;font-size:0.8rem;">[{diff}]</span><br>'
                    f'<span style="color:#94a3b8;font-size:0.82rem;">{c.get("description","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _render_pyvis_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Render an interactive pyvis network inside Streamlit."""
    try:
        from pyvis.network import Network
        import tempfile, os

        net = Network(
            height="560px",
            width="100%",
            bgcolor="#0f172a",
            font_color="#e2e8f0",
            directed=True,
        )
        net.set_options(json.dumps({
            "nodes": {
                "shape": "dot",
                "size": 16,
                "font": {"size": 13, "color": "#e2e8f0"},
                "borderWidth": 2,
            },
            "edges": {
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
                "color": {"color": "#334155", "highlight": "#60a5fa"},
                "smooth": {"type": "curvedCW", "roundness": 0.2},
                "font": {"size": 10, "color": "#64748b"},
            },
            "physics": {
                "barnesHut": {"gravitationalConstant": -8000, "springLength": 120},
                "stabilization": {"iterations": 150},
            },
            "interaction": {"hover": True, "tooltipDelay": 100},
        }))

        node_ids = {n["id"] for n in nodes}

        for node in nodes:
            diff = node.get("difficulty", "intermediate")
            color = DIFFICULTY_COLOR.get(diff, "#f59e0b")
            net.add_node(
                node["id"],
                label=node["label"],
                title=f"<b>{node['label']}</b><br>{node.get('description','')}<br><i>{diff}</i>",
                color=color,
                size=18 if diff == "advanced" else 14,
            )

        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src in node_ids and tgt in node_ids:
                net.add_edge(
                    src, tgt,
                    label=edge.get("relationship", ""),
                    title=edge.get("relationship", ""),
                )

        import uuid
        temp_name = f"graph_{uuid.uuid4().hex}.html"
        temp_path = os.path.join(tempfile.gettempdir(), temp_name)
        try:
            net.save_graph(temp_path)
            with open(temp_path, "r", encoding="utf-8") as file:
                html_content = file.read()
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

        components.html(html_content, height=570, scrolling=False)

    except ImportError:
        # Fallback: plotly scatter graph
        _render_plotly_graph(nodes, edges)


def _render_plotly_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Plotly fallback for graph visualization."""
    import math
    n = len(nodes)
    positions = {
        node["id"]: (
            math.cos(2 * math.pi * i / n),
            math.sin(2 * math.pi * i / n),
        )
        for i, node in enumerate(nodes)
    }

    edge_x, edge_y = [], []
    node_ids = set(positions.keys())
    for edge in edges:
        s, t = edge.get("source"), edge.get("target")
        if s in positions and t in positions:
            x0, y0 = positions[s]
            x1, y1 = positions[t]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    node_x = [positions[n["id"]][0] for n in nodes]
    node_y = [positions[n["id"]][1] for n in nodes]
    node_text = [n["label"] for n in nodes]
    node_colors = [DIFFICULTY_COLOR.get(n.get("difficulty","intermediate"), "#f59e0b") for n in nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#334155"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             marker=dict(size=14, color=node_colors, line=dict(width=1, color="#0f172a")),
                             text=node_text, textposition="top center",
                             textfont=dict(color="#e2e8f0", size=11)))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        showlegend=False, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_path(path: list[str]) -> None:
    """Render learning path as a visual step-by-step flow."""
    cols = st.columns(min(len(path), 5))
    for i, concept in enumerate(path):
        col_idx = i % len(cols)
        with cols[col_idx]:
            is_last = i == len(path) - 1
            bg = "#1e3a5f" if is_last else "#1e293b"
            border = "#3b82f6" if is_last else "#334155"
            label = "🎯 Goal" if is_last else f"Step {i+1}"
            st.markdown(
                f'<div style="background:{bg};border:2px solid {border};border-radius:8px;'
                f'padding:12px;text-align:center;margin:4px 0;">'
                f'<div style="color:#64748b;font-size:0.75rem;">{label}</div>'
                f'<div style="color:#f1f5f9;font-weight:700;margin-top:4px;font-size:0.9rem;">{concept}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if i < len(path) - 1 and (i + 1) % len(cols) == 0:
            st.markdown(
                '<div style="text-align:center;font-size:1.5rem;color:#334155;">↓</div>',
                unsafe_allow_html=True,
            )

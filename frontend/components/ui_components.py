"""
frontend/components/ui_components.py
Reusable Streamlit UI widgets used across pages.
"""
from __future__ import annotations
import streamlit as st


# ─── Confidence badge ─────────────────────────────────────────────────────────

def confidence_badge(score: float) -> None:
    color = "#22c55e" if score >= 0.75 else "#f59e0b" if score >= 0.5 else "#ef4444"
    label = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">'
        f'Confidence: {label} ({score:.0%})</span>',
        unsafe_allow_html=True,
    )


# ─── Source card ──────────────────────────────────────────────────────────────

def source_card(source: dict, index: int) -> None:
    with st.expander(
        f"📄 Source {index}: {source.get('document_filename', 'Unknown')} "
        f"{'| ' + source['chapter'] if source.get('chapter') else ''} "
        f"{'| Page ' + str(source['page_number']) if source.get('page_number') else ''}",
        expanded=False,
    ):
        st.caption(f"Relevance: {source.get('relevance_score', 0):.2%}")
        st.markdown(source.get("content", ""))


# ─── Formula card ─────────────────────────────────────────────────────────────

def formula_card(formula: dict) -> None:
    st.markdown(
        f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px 18px;margin:6px 0;">
  <div style="font-weight:700;color:#93c5fd;font-size:1rem;margin-bottom:6px;">
    🔢 {formula.get('name','')}
  </div>
  <div style="font-family:'Courier New',monospace;font-size:1.1rem;color:#f1f5f9;
    background:#0f172a;padding:8px 12px;border-radius:4px;margin-bottom:8px;">
    {formula.get('latex','')}
  </div>
  <div style="color:#cbd5e1;font-size:0.88rem;">{formula.get('explanation','')}</div>
  <div style="color:#64748b;font-size:0.8rem;margin-top:6px;">
    Related: {', '.join(formula.get('related_concepts', []))}
  </div>
</div>""",
        unsafe_allow_html=True,
    )


# ─── Risk badge ───────────────────────────────────────────────────────────────

def risk_badge(level: str) -> None:
    colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}
    color = colors.get(level.lower(), "#94a3b8")
    st.markdown(
        f'<span style="background:{color};color:white;padding:4px 14px;'
        f'border-radius:12px;font-size:0.85rem;font-weight:700;">Risk: {level.upper()}</span>',
        unsafe_allow_html=True,
    )


# ─── Metric row ───────────────────────────────────────────────────────────────

def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """items: list of (label, value, delta)"""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# ─── Empty state ──────────────────────────────────────────────────────────────

def empty_state(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
<div style="text-align:center;padding:60px 20px;color:#64748b;">
  <div style="font-size:3rem;">{icon}</div>
  <div style="font-size:1.1rem;font-weight:600;margin-top:12px;color:#94a3b8;">{title}</div>
  <div style="font-size:0.88rem;margin-top:6px;">{subtitle}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ─── Section header ───────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    st.markdown(
        f"""
<div style="border-left:4px solid #3b82f6;padding:4px 0 4px 14px;margin-bottom:18px;">
  <div style="font-size:1.25rem;font-weight:700;color:#f1f5f9;">{icon} {title}</div>
  {f'<div style="font-size:0.85rem;color:#94a3b8;margin-top:2px;">{subtitle}</div>' if subtitle else ''}
</div>""",
        unsafe_allow_html=True,
    )


# ─── Status pill ──────────────────────────────────────────────────────────────

def status_pill(status: str) -> str:
    colors = {"ready": "#22c55e", "processing": "#f59e0b", "error": "#ef4444"}
    color = colors.get(status, "#94a3b8")
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'padding:2px 8px;border-radius:10px;font-size:0.78rem;font-weight:600;">{status}</span>'
    )

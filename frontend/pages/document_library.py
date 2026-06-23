"""
frontend/pages/document_library.py
Upload, list, and delete ingested documents.
"""
import streamlit as st
from frontend.utils.api_client import upload_document, list_documents, delete_document
from frontend.components.ui_components import section_header, empty_state, status_pill

MIME_MAP = {
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "txt": "text/plain",
}

SUBJECTS = ["General", "Physics", "Chemistry", "Mathematics", "Biology",
            "Engineering", "Computer Science", "Economics", "History"]


def render():
    section_header("Document Library", "Upload and manage your educational materials", "📁")

    # ── Upload Panel ──────────────────────────────────────────────────────────
    with st.expander("➕ Upload New Document", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "pptx", "jpg", "jpeg", "png", "txt"],
                help="Supports PDF, PowerPoint, images, and plain text. Max 50 MB.",
            )
        with col2:
            subject = st.selectbox("Subject", SUBJECTS)

        if uploaded_file and st.button("🚀 Upload & Ingest", use_container_width=True):
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            mime = MIME_MAP.get(ext, "application/octet-stream")
            with st.spinner("Ingesting document… AI is extracting text, formulas and diagrams. This may take 30-90s."):
                try:
                    result = upload_document(
                        file_bytes=uploaded_file.read(),
                        filename=uploaded_file.name,
                        content_type=mime,
                        subject=subject,
                    )
                    st.success(f"✅ {result['message']}")
                    st.json(result)
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")

    # ── Filter + List ─────────────────────────────────────────────────────────
    col_filter, _ = st.columns([2, 5])
    with col_filter:
        filter_subject = st.selectbox("Filter by subject", ["All"] + SUBJECTS, key="filter_sub")

    try:
        docs = list_documents(None if filter_subject == "All" else filter_subject)
    except Exception as e:
        st.error(f"Could not reach backend: {e}")
        return

    if not docs:
        empty_state("📂", "No documents yet", "Upload your first document above to get started.")
        return

    st.markdown(f"**{len(docs)} document(s)**")

    for doc in docs:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([4, 1.5, 1.5, 1.5, 1])
            with c1:
                st.markdown(f"**{doc['filename']}**")
                st.caption(f"Subject: {doc.get('subject','—')} · {doc.get('chunk_count',0)} chunks")
            with c2:
                st.caption("Type")
                st.markdown(f"`{doc.get('doc_type','—')}`")
            with c3:
                st.caption("Uploaded")
                ts = doc.get("uploaded_at", "")[:10]
                st.caption(ts)
            with c4:
                st.markdown(status_pill(doc.get("status", "unknown")), unsafe_allow_html=True)
            with c5:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                    try:
                        delete_document(doc["id"])
                        st.success("Deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            st.markdown('<hr style="margin:6px 0;border-color:#1e293b;">', unsafe_allow_html=True)

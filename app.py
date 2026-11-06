"""Streamlit web UI:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from rag.config import settings
from rag.ingest import ingest_pdf
from rag.providers import get_embeddings, get_llm
from rag.qa import answer_question

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF")
st.caption(f"RAG with LangChain + ChromaDB · provider: **{settings.provider}**")

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.messages = []
    st.session_state.doc_name = None

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("PDF file", type="pdf")
    if uploaded and st.button("Process PDF", type="primary"):
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / uploaded.name
        path.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Reading and indexing the PDF..."):
                store, reused = ingest_pdf(path, get_embeddings())
        except Exception as exc:  # show API-key / quota / scanned-PDF errors nicely
            st.error(f"Could not process the PDF: {exc}")
        else:
            st.session_state.store = store
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []
            st.success("Loaded from cache." if reused else "Indexed. Ask away!")
    if st.session_state.doc_name:
        st.info(f"Active document: {st.session_state.doc_name}")
    st.header("2. Ask")
    st.write("Answers come only from the document, with page references.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}**: {s['snippet']}...")

question = st.chat_input("Ask a question about the PDF", disabled=st.session_state.store is None)
if st.session_state.store is None:
    st.info("Upload a PDF in the sidebar and click **Process PDF** to start.")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.store, get_llm(), question)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        else:
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**Page {s['page']}**: {s['snippet']}...")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )

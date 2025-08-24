import os
import streamlit as st
from dotenv import load_dotenv
from src.config import AppConfig, load_app_config
from src.ingestion import parse_files_and_chunk
from src.store import build_or_update_indices, load_indices, DocChunk, load_docstore
from src.hybrid import hybrid_retrieve
from src.rerank import maybe_rerank
from src.generate import get_generator
from src.graph import build_graph, RAGState
from src.utils import timer, fmt_citation

load_dotenv()
st.set_page_config(page_title="RAG FAQ Bot", page_icon="❓", layout="wide")

st.sidebar.title("RAG FAQ Bot — Index")
uploaded_files = st.sidebar.file_uploader(
    "Upload FAQ files (PDF / TXT / MD / CSV)", type=["pdf","txt","md","csv"], accept_multiple_files=True
)
if "uploaded_payloads" not in st.session_state:
    st.session_state["uploaded_payloads"] = []

if uploaded_files:
    # Keep in-memory copies to parse on demand
    st.session_state["uploaded_payloads"] = [(f.name, f.getvalue()) for f in uploaded_files]

if st.sidebar.button("Build / Update Index", use_container_width=True):
    if not st.session_state.get("uploaded_payloads"):
        st.sidebar.warning("No files uploaded yet.")
    else:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            status_text.text("📄 Parsing files...")
            progress_bar.progress(0.2)
            texts = parse_files_and_chunk(st.session_state["uploaded_payloads"])
            
            status_text.text("🔍 Building search indices...")
            progress_bar.progress(0.4)
            build_or_update_indices(texts)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Index updated successfully!")
            st.sidebar.success("Index updated ✅")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
        finally:
            # Clean up progress indicators after 2 seconds
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

st.sidebar.subheader("Retrieval Options")
use_hybrid = st.sidebar.toggle("Use Hybrid Search (BM25 + Vector)", value=True)
use_rerank = st.sidebar.toggle("Use Reranker", value=True)
topk_vec = st.sidebar.slider("TopK Vector", 1, 20, 5)
topk_bm25 = st.sidebar.slider("TopK BM25", 1, 20, 5)
topk_after = st.sidebar.slider("TopK After Rerank", 1, 20, 5)

st.title("❓ Customer Support FAQ Bot")
st.caption("Upload your FAQ docs, build an index, then ask questions. See retrieval details in the debug panel.")

question = st.text_input("Ask a question about your FAQs…", value="What is the refund policy?")
ask = st.button("Ask", type="primary")

cfg: AppConfig = load_app_config()
gen = get_generator()

if ask and question.strip():
    faiss, bm25, docstore = load_indices()
    graph = build_graph(use_hybrid, use_rerank)

    with timer() as t:
        result_state: RAGState = graph.invoke({
            "question": question.strip(),
            "retrieved_vector": [],
            "retrieved_bm25": [],
            "candidates": [],
            "reranked": [],
            "context": "",
            "answer": "",
            "citations": [],
            "config": {
                "topk_vec": topk_vec,
                "topk_bm25": topk_bm25,
                "topk_after": topk_after,
                "use_hybrid": use_hybrid,
                "use_rerank": use_rerank,
            }
        })
    st.markdown(f"**Answer:** {result_state['answer']}")
    if result_state["citations"]:
        st.markdown("**Citations:**")
        for c in result_state["citations"]:
            st.markdown(f"- {fmt_citation(c)}")

    with st.expander("🔎 Debug: Retrieval Details"):
        st.write("**Vector Results:**")
        for d in result_state.get("retrieved_vector", [])[:20]:
            st.code(f"[{d.score:.3f}] {d.meta['file_name']} :: {d.text[:220]}")
        if use_hybrid:
            st.write("**BM25 Results:**")
            for d in result_state.get("retrieved_bm25", [])[:20]:
                st.code(f"[{d.score:.3f}] {d.meta['file_name']} :: {d.text[:220]}")
        st.write("**Final Candidates (after merge/rerank):**")
        for d in (result_state.get("reranked") or result_state.get("candidates"))[:20]:
            st.code(f"[{d.score:.3f}] {d.meta['file_name']} :: {d.text[:220]}")
    st.caption(f"Elapsed: {t.elapsed_ms:.0f} ms")

else:
    st.info("Tip: try uploading the sample files in `data/samples/`, build the index, then ask.")

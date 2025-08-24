# RAG FAQ Bot (LangGraph + Streamlit)

A compact, portfolio-friendly **Customer Support FAQ bot** showing **LangGraph** orchestration with **Hybrid Retrieval (FAISS + BM25)** and optional **Cross-Encoder Reranking**. Upload FAQ files (PDF/MD/TXT/CSV), build a local index, ask questions, and see grounded answers with citations and a debug panel.

## Features
- LangGraph pipeline with configurable nodes (vector, BM25, rerank, generate).
- Hybrid retrieval: FAISS (semantic) + BM25 (lexical).
- Optional reranker: `BAAI/bge-reranker-base` via `sentence-transformers`.
- Runs CPU-only; works without LLM key via a local stub.
- Streamlit UI with toggles, sliders, and expandable debug panel.
- Persistent local store in `./.rag_store/`.

## Setup
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env  # optionally add your OPENAI_API_KEY / GROQ_API_KEY
```

## Run
```bash
streamlit run app.py
```

## Usage
1. Upload one or more FAQ files (PDF, MD, TXT, CSV of Q/A).
2. Click **Build / Update Index**.
3. Ask a question. Toggle **Hybrid Search** and **Reranker** and watch the debug panel update.
4. If you don't set an API key, a local stub synthesizes an answer from the top chunks.

## Notes
- Index persists in `./.rag_store/`. Delete this folder to reset.
- Models are set in `.env` and `src/config.py`.
- CSV format assumed as columns like `question,answer` (auto-detected).

## Future Work
- Multi-query rewriting
- Response caching
- Faithfulness evaluation
- FastAPI endpoint

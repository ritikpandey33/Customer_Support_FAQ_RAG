import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    api_provider: str
    embedding_provider: str
    openai_api_key: str
    groq_api_key: str
    hf_token: str
    embed_model: str
    rerank_model: str
    faiss_path: str
    docstore_path: str
    bm25_path: str

def load_app_config() -> 'AppConfig':
    return AppConfig(
        api_provider=os.getenv("API_PROVIDER", "groq"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        rerank_model=os.getenv("RERANK_MODEL", "gpt-4o-mini"),
        faiss_path=os.getenv("FAISS_PATH", ".rag_store/faiss.index"),
        docstore_path=os.getenv("DOCSTORE_PATH", ".rag_store/docstore.json"),
        bm25_path=os.getenv("BM25_PATH", ".rag_store/bm25.pkl"),
    )

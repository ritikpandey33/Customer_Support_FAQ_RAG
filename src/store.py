from typing import List, Dict, Any, Tuple
import os, json, pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.config import load_app_config
from src.embeddings import embed_texts
from src.utils import hash_text

class DocChunk:
    def __init__(self, text: str, meta: Dict[str,Any], score: float=0.0):
        self.text=text; self.meta=meta; self.score=score

def _paths():
    cfg = load_app_config()
    return cfg.faiss_path, cfg.docstore_path, cfg.bm25_path

def _ensure_dirs():
    for p in _paths():
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def build_or_update_indices(chunks: List[Dict[str,Any]]):
    cfg = load_app_config()
    _ensure_dirs()
    
    # Load existing docs to check for duplicates
    existing_docs = load_docstore() if os.path.exists(cfg.docstore_path) else []
    existing_uids = {doc.get("uid") for doc in existing_docs if doc.get("uid")}
    
    # docstore - APPEND new chunks, skip duplicates
    docs = existing_docs.copy()
    start_id = len(docs)
    new_chunks = []
    
    for c in chunks:
        # Create unique ID for this chunk
        uid = hash_text(c["text"] + c["meta"]["file_name"] + str(c["meta"]["chunk_id"]))
        
        # Skip if already processed
        if uid in existing_uids:
            continue
            
        c["id"] = start_id
        c["uid"] = uid
        docs.append(c)
        new_chunks.append(c)
        existing_uids.add(uid)
        start_id += 1
    with open(cfg.docstore_path,'w',encoding='utf-8') as f:
        json.dump(docs,f,ensure_ascii=False)

    # embeddings + FAISS - only process NEW chunks
    if new_chunks:
        new_texts = [d["text"] for d in new_chunks]
        new_vecs = embed_texts(cfg.embed_model, new_texts).astype('float32')
        
        # Load existing FAISS index or create new one
        if os.path.exists(cfg.faiss_path):
            index = faiss.read_index(cfg.faiss_path)
            index.add(new_vecs)  # Add new vectors to existing index
        else:
            index = faiss.IndexFlatIP(new_vecs.shape[1])
            index.add(new_vecs)
        faiss.write_index(index, cfg.faiss_path)

    # BM25 - rebuild with all texts (BM25 needs all texts together)
    all_texts = [d["text"] for d in docs]
    tokenized = [t.lower().split() for t in all_texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    with open(cfg.bm25_path,'wb') as f:
        pickle.dump({"bm25": bm25, "texts": all_texts}, f)

def load_docstore():
    cfg = load_app_config()
    if not os.path.exists(cfg.docstore_path):
        return []
    with open(cfg.docstore_path,'r',encoding='utf-8') as f:
        return json.load(f)

def load_indices():
    cfg = load_app_config()
    # FAISS
    faiss_idx = None
    if os.path.exists(cfg.faiss_path):
        faiss_idx = faiss.read_index(cfg.faiss_path)
    # BM25
    bm25 = None
    if os.path.exists(cfg.bm25_path):
        import pickle
        with open(cfg.bm25_path,'rb') as f:
            obj = pickle.load(f)
            bm25 = obj.get("bm25")
    return faiss_idx, bm25, load_docstore()

def faiss_search(query: str, topk: int) -> List[DocChunk]:
    cfg = load_app_config()
    idx, _, docs = load_indices()
    if idx is None or not docs:
        return []
    import numpy as np
    qv = embed_texts(cfg.embed_model, [query]).astype('float32')
    
    # Ensure qv is 2D (n_samples, n_features) as expected by FAISS
    if qv.ndim == 1:
        qv = qv.reshape(1, -1)
    elif qv.ndim == 3:
        qv = qv.reshape(qv.shape[0], -1)
    
    sims, I = idx.search(qv, topk)
    out = []
    
    for rank, (score, i) in enumerate(zip(sims[0], I[0])):
        if i < 0 or i >= len(docs): 
            continue
        d = docs[i]
        out.append(DocChunk(d["text"], d["meta"], float(score)))
    return out

def bm25_search(query: str, topk: int) -> List[DocChunk]:
    cfg = load_app_config()
    _, bm25, docs = load_indices()
    if bm25 is None or not docs:
        return []
    texts = [d["text"] for d in docs]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    order = np.argsort(scores)[::-1][:topk]
    out = []
    
    for rank, i in enumerate(order):
        d = docs[int(i)]
        score = scores[int(i)]
        out.append(DocChunk(d["text"], d["meta"], float(score)))
    return out

from typing import List
from src.store import faiss_search, bm25_search, DocChunk

def hybrid_retrieve(question: str, topk_vec: int, topk_bm25: int, use_hybrid: bool):
    vec = faiss_search(question, topk_vec) or []
    bm = bm25_search(question, topk_bm25) if use_hybrid else []
    return vec, bm

def merge_candidates(vec: List[DocChunk], bm: List[DocChunk]):
    # Simple score normalization and union by (file_name, chunk_id)
    def key(d): return (d.meta.get("file_name"), d.meta.get("chunk_id"))
    merged = {}
    # normalize vector to [0,1], bm25 to [0,1]
    if vec:
        v_scores = [d.score for d in vec]
        v_min, v_max = min(v_scores), max(v_scores)
    else:
        v_min, v_max = 0.0, 1.0
    if bm:
        b_scores = [d.score for d in bm]
        b_min, b_max = min(b_scores), max(b_scores)
    else:
        b_min, b_max = 0.0, 1.0
    def norm(x, a, b):
        return 0.5 if a==b else (x - a) / (b - a)
    for d in vec:
        merged[key(d)] = d
        d.score = 0.5 + 0.5*norm(d.score, v_min, v_max)  # bias toward vector
    for d in bm:
        if key(d) in merged:
            merged[key(d)].score = max(merged[key(d)].score, 0.5*norm(d.score, b_min, b_max))
        else:
            d.score = 0.5*norm(d.score, b_min, b_max)
            merged[key(d)] = d
    return sorted(list(merged.values()), key=lambda x: x.score, reverse=True)

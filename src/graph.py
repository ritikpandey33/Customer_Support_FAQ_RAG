from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from src.store import DocChunk
from src.hybrid import hybrid_retrieve, merge_candidates
from src.rerank import maybe_rerank
from src.generate import get_generator

class RAGState(TypedDict):
    question: str
    retrieved_vector: List[DocChunk]
    retrieved_bm25: List[DocChunk]
    candidates: List[DocChunk]
    reranked: List[DocChunk]
    context: str
    answer: str
    citations: List[Dict]
    config: Dict

def _retrieve_vector(state: RAGState):
    vec, _ = hybrid_retrieve(state["question"], state["config"]["topk_vec"], 0, False)
    state["retrieved_vector"] = vec
    return state

def _retrieve_bm25(state: RAGState):
    _, bm = hybrid_retrieve(
    state["question"],
    state["config"]["topk_vec"],      # use correct value
    state["config"]["topk_bm25"],
    True
)
    state["retrieved_bm25"] = bm
    return state

def _merge(state: RAGState):
    vec = state.get("retrieved_vector") or []
    bm = state.get("retrieved_bm25") or []
    state["candidates"] = merge_candidates(vec, bm if state["config"].get("use_hybrid") else [])
    return state

def _rerank(state: RAGState):
    cand = state.get("candidates") or []
    state["reranked"] = maybe_rerank(state["question"], cand, state["config"]["topk_after"], state["config"].get("use_rerank"))
    return state

def _make_context(state: RAGState):
    chosen = state.get("reranked") or state.get("candidates") or []
    # Keep the topK already applied; context is join
    ctx = []
    for c in chosen:
        ctx.append(f"{c.meta.get('file_name')} :: {c.text}")
    state["context"] = "\n\n".join(ctx)
    return state

def _generate(state: RAGState):
    chosen = state.get("reranked") or state.get("candidates") or []
    gen = get_generator()
    out = gen(state["question"], chosen)
    state["answer"] = out["answer"]
    state["citations"] = out["citations"]
    return state

def build_graph(use_hybrid: bool, use_rerank: bool):
    g = StateGraph(RAGState)
    g.add_node("retrieve_vector", _retrieve_vector)
    if use_hybrid:
        g.add_node("retrieve_bm25", _retrieve_bm25)
    g.add_node("merge_candidates", _merge)
    if use_rerank:
        g.add_node("rerank_candidates", _rerank)
    g.add_node("make_context", _make_context)
    g.add_node("generate_answer", _generate)

    g.add_edge(START, "retrieve_vector")
    if use_hybrid:
        g.add_edge("retrieve_vector", "retrieve_bm25")
        g.add_edge("retrieve_bm25", "merge_candidates")
    else:
        g.add_edge("retrieve_vector", "merge_candidates")
    if use_rerank:
        g.add_edge("merge_candidates", "rerank_candidates")
        g.add_edge("rerank_candidates", "make_context")
    else:
        g.add_edge("merge_candidates", "make_context")
    g.add_edge("make_context", "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile()

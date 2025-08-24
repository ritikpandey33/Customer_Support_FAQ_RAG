import os
from typing import List, Dict
from src.store import DocChunk

PROMPT = '''You are a helpful FAQ assistant. Use ONLY the provided context to answer.
If the answer is not present, say: "I don't find this in the provided FAQs."

Question: {question}

Context:
{context}

Answer (be concise and reference the citations):'''

def _format_context(chunks: List[DocChunk]):
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.meta.get('file_name')} :: {c.text}")
    return "\n\n".join(lines)

def _citations_from(chunks: List[DocChunk]):
    cites = []
    for i, c in enumerate(chunks, start=1):
        cites.append({
            "file": c.meta.get("file_name",""),
            "page": c.meta.get("page"),
            "snippet": c.text[:200].replace('\n',' ')
        })
    return cites

def _stub_generate(question: str, chunks: List[DocChunk]) -> Dict:
    if not chunks:
        return {"answer": "I don't find this in the provided FAQs.", "citations": []}
    top = chunks[0]
    answer = f"From {top.meta.get('file_name')}: {top.text[:300]}"
    return {"answer": answer, "citations": _citations_from(chunks)}

def generate_answer_openai(question: str, chunks: List[DocChunk]) -> Dict:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        context = _format_context(chunks)
        prompt = PROMPT.format(question=question, context=context)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        return {"answer": text, "citations": _citations_from(chunks)}
    except Exception:
        return _stub_generate(question, chunks)

def generate_answer_stub(question: str, chunks: List[DocChunk]) -> Dict:
    return _stub_generate(question, chunks)

def get_generator():
    if os.getenv("OPENAI_API_KEY"):
        return generate_answer_openai
    # (Add GROQ/TOGETHER similarly if desired)
    return generate_answer_stub

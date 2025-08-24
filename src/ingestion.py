from typing import List, Tuple, Dict, Any
import io, csv, re
from pypdf import PdfReader

def _read_pdf(name: str, data: bytes) -> str:
    with io.BytesIO(data) as f:
        reader = PdfReader(f)
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts)

def _read_text_like(name: str, data: bytes) -> str:
    return data.decode('utf-8', errors='ignore')

def _read_csv(name: str, data: bytes) -> str:
    f = io.StringIO(data.decode('utf-8', errors='ignore'))
    try:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            q = row.get('question') or row.get('Question') or row.get('q') or ""
            a = row.get('answer') or row.get('Answer') or row.get('a') or ""
            if q or a:
                rows.append(f"Q: {q}\nA: {a}")
        if rows:
            return "\n\n".join(rows)
    except Exception:
        pass
    # fallback raw
    f.seek(0)
    return f.read()

def _chunk(text: str, target=1200, overlap=120) -> List[str]:
    # Special handling for Q&A format - CREATE INDIVIDUAL Q&A PAIRS
    if "Q:" in text and "A:" in text:
        # Split on Q&A pairs, each pair becomes its own chunk
        qa_pairs = re.split(r'\n\n(?=Q:)', text)
        chunks = []
        
        for pair in qa_pairs:
            pair = pair.strip()
            if pair and len(pair) > 10:  # Skip very short/empty pairs
                chunks.append(pair)
        
        return chunks if chunks else [text]
    
    # Fallback to regular chunking for non-Q&A content
    text = re.sub(r'\s+',' ', text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + target)
        chunk = text[i:j]
        chunks.append(chunk)
        i = max(j - overlap, j)
    return chunks

def parse_files_and_chunk(files: List[Tuple[str, bytes]]):
    out = []
    for name, data in files:
        ext = name.lower().split('.')[-1]
        if ext == 'pdf':
            content = _read_pdf(name, data)
        elif ext in ('txt','md'):
            content = _read_text_like(name, data)
        elif ext == 'csv':
            content = _read_csv(name, data)
        else:
            content = _read_text_like(name, data)
        for idx, ch in enumerate(_chunk(content)):
            out.append({
                "text": ch,
                "meta": {
                    "file_name": name,
                    "chunk_id": idx,
                    "page": None
                }
            })
    return out

import hashlib
import time
from contextlib import contextmanager

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]

@contextmanager
def timer():
    class T: 
        start=0; end=0; elapsed_ms=0
    t = T()
    t.start = time.time()
    try:
        yield t
    finally:
        t.end = time.time()
        t.elapsed_ms = (t.end - t.start) * 1000

def fmt_citation(c):
    loc = ""
    if c.get("page"):
        loc = f" (p.{c['page']})"
    return f"**{c['file']}**{loc}: “{c['snippet'][:160]}…”"

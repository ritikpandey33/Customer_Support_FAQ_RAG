from typing import List
from openai import OpenAI
from groq import Groq
from src.config import load_app_config
from src.store import DocChunk

def _score_with_api(question: str, passages: List[str], model: str, api_provider: str, api_key: str) -> List[float]:
    prompt = f"""Rate the relevance of each passage to the question on a scale of 0.0 to 1.0.
Question: {question}

Passages:
{chr(10).join([f"{i+1}. {passage}" for i, passage in enumerate(passages)])}

Return only a comma-separated list of scores (e.g., 0.8,0.3,0.9):"""

    try:
        if api_provider == "openai":
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            scores_text = result["choices"][0]["message"]["content"].strip()
            
        elif api_provider == "groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            scores_text = response.choices[0].message.content.strip()
        
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        scores = [float(s.strip()) for s in scores_text.split(',')]
        if len(scores) != len(passages):
            scores = [0.5] * len(passages)
            
    except Exception as e:
        print(f"Reranking error: {e}")
        scores = [0.5] * len(passages)
    
    return scores

def maybe_rerank(question: str, candidates: List[DocChunk], topk_after: int, use_rerank: bool):
    if not use_rerank or not candidates:
        return candidates[:topk_after]
    
    cfg = load_app_config()
    passages = [c.text for c in candidates]
    
    if cfg.api_provider == "openai":
        scores = _score_with_api(question, passages, cfg.rerank_model, "openai", cfg.openai_api_key)
    elif cfg.api_provider == "groq":
        scores = _score_with_api(question, passages, cfg.rerank_model, "groq", cfg.groq_api_key)
    else:
        raise ValueError(f"Unsupported API provider: {cfg.api_provider}")
    
    for c, s in zip(candidates, scores):
        c.score = float(s)
    
    return sorted(candidates, key=lambda x: x.score, reverse=True)[:topk_after]

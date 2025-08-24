from typing import List
import numpy as np
import requests
import json
from src.config import load_app_config

def embed_texts(model_name: str, texts: List[str]):
    cfg = load_app_config()
    
    if cfg.embedding_provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")
        
        headers = {
            "Authorization": f"Bearer {cfg.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": texts,
            "model": model_name
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response from OpenAI API: {str(e)}")
    
    elif cfg.embedding_provider == "huggingface":
        if not cfg.hf_token:
            raise ValueError("HuggingFace token is required for embeddings")
        
        headers = {
            "Authorization": f"Bearer {cfg.hf_token}",
            "Content-Type": "application/json"
        }
        
        # HF API for embeddings - process texts in batches for speed
        embeddings = []
        batch_size = 10  # Process 10 texts at once
        
        try:
            # Process in batches to reduce API calls from 200 to 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                data = {
                    "inputs": batch,
                    "options": {"wait_for_model": True}
                }
                
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model_name}",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                
                # HF returns list of embeddings for batch input
                if isinstance(result, list):
                    if len(batch) == 1:
                        # Single text returns single embedding
                        embeddings.append(result)
                    else:
                        # Multiple texts return list of embeddings
                        embeddings.extend(result)
                else:
                    raise ValueError(f"Unexpected response format: {type(result)}")
                
            
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HuggingFace API error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f" - Details: {error_details}"
                except:
                    error_msg += f" - Response: {e.response.text}"
            raise ValueError(error_msg)
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response from HuggingFace API: {str(e)}")
        
    else:
        raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")
    
    # Convert to proper numpy array - handle nested structure from batching
    try:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # If we have extra nesting from batching, flatten appropriately
        if embeddings_array.ndim == 3 and embeddings_array.shape[0] == len(embeddings):
            # Shape is (n_batches, 1, embedding_dim) -> (n_texts, embedding_dim)
            embeddings_array = embeddings_array.squeeze(axis=1)
        elif embeddings_array.ndim == 1 and len(embeddings) == 1:
            # Single embedding returned as 1D -> reshape to 2D
            embeddings_array = embeddings_array.reshape(1, -1)
        
        # Normalize
        embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        return embeddings_array
    except Exception as e:
        raise ValueError(f"Failed to convert embeddings to array: {e}")

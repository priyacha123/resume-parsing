import os
import requests
import numpy as np
import logging

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


def compute_embedding(text: str) -> list:
    """
    Computes a 384-dimensional dense semantic embedding vector for the given text
    using Hugging Face's MiniLM model inference API.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return [0.0] * 384

    token = os.environ.get('HF_API_TOKEN')
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": cleaned[:2000]},
            timeout=25
        )
        response.raise_for_status()
        embedding = response.json()
        arr = np.array(embedding, dtype=float)
        
        # If API returns per-token embeddings (2D matrix), mean-pool to 1D vector
        if arr.ndim > 1:
            arr = arr.mean(axis=0)

        # Flatten if needed
        flat = arr.flatten()
        if len(flat) != 384:
            logger.warning(f"Embedding length was {len(flat)}, expected 384")
            # If length mismatch, pad or truncate to 384
            if len(flat) > 384:
                flat = flat[:384]
            else:
                flat = np.pad(flat, (0, 384 - len(flat)))

        return flat.tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding service unavailable: {str(e)}")


def cosine_similarity_score(vec1, vec2) -> float:
    """
    Computes the cosine similarity percentage between two vector embeddings.
    Clamps result to [0.0, 100.0] and handles zero vectors safely.
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0 or np.isnan(norm_a) or np.isnan(norm_b):
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Cosine similarity can range from -1 to 1; for natural language embeddings it is >= 0
    clamped = max(0.0, min(1.0, float(similarity)))
    return round(clamped * 100, 2)
# from sentence_transformers import SentenceTransformer
# import numpy as np

# # Load once at module level — reloading per-request would be slow
# _model = None
# def get_model():
#     global _model
#     if _model is None:
#         _model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
#     return _model

# def compute_embedding(text: str):
#     model = get_model()
#     return model.encode(text).tolist()  # Convert numpy array to list for JSON serialization

# def cosine_similarity_score(vec1, vec2) -> float:
#     a, b = np.array(vec1), np.array(vec2)
#     similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 
#     return round(float(similarity) * 100, 2)  # Return percentage similarity rounded to 2 decimal places


import os
import requests
import numpy as np

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


def compute_embedding(text: str):
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}"}
    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": text[:2000]}, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Embedding service unavailable: {str(e)}")

    embedding = response.json()
    arr = np.array(embedding)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.tolist()


def cosine_similarity_score(vec1, vec2) -> float:
    a, b = np.array(vec1), np.array(vec2)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return round(float(similarity) * 100, 2)
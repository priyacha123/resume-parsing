from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level — reloading per-request would be slow
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
    return _model

def compute_embedding(text: str):
    model = get_model()
    return model.encode(text).tolist()  # Convert numpy array to list for JSON serialization

def cosine_similarity_score(vec1, vec2) -> float:
    a, b = np.array(vec1), np.array(vec2)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 
    return round(float(similarity) * 100, 2)  # Return percentage similarity rounded to 2 decimal places
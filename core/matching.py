from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_tfidf_score(resume_text: str, jd_text: str) -> float:
    """
    Computes the TF-IDF cosine similarity score between a resume and a job description.

    Args:
        resume_text (str): The text extracted from the resume.
        jd_text (str): The text of the job description.

    Returns:
        float: A percentage similarity score between 0.0 and 100.0 rounded to 2 decimal places.
    """
    if not resume_text or not jd_text:
        return 0.0

    r_clean = resume_text.strip()
    j_clean = jd_text.strip()
    if not r_clean or not j_clean:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2),
            max_features=5000
        )
        tfidf_matrix = vectorizer.fit_transform([r_clean, j_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(similarity[0][0]) * 100
        # Clamp score between 0 and 100
        return round(max(0.0, min(100.0, score)), 2)
    except Exception:
        # Fallback to simple word overlap if TF-IDF fails (e.g. extremely short text)
        r_words = set(re.findall(r'\w+', r_clean.lower()))
        j_words = set(re.findall(r'\w+', j_clean.lower()))
        if not j_words:
            return 0.0
        overlap = len(r_words.intersection(j_words)) / len(j_words)
        return round(max(0.0, min(100.0, overlap * 100)), 2)
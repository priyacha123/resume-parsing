from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_tfidf_score(resume_text: str, jd_text: str) -> float:
    """
    Computes the TF-IDF cosine similarity score between a resume and a job description.

    Args:
        resume_text (str): The text extracted from the resume.
        jd_text (str): The text of the job description.

    Returns:
        float: A similarity score between 0 and 1, where 1 indicates identical texts.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0  # Return 0 if either text is empty

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0: 1], tfidf_matrix[1:2])
    return round(float(similarity[0][0]) * 100, 2)  # Return percentage similarity rounded to 2 decimal places
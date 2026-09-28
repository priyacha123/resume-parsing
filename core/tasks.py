from celery import shared_task
from .models import Resume, JobDescription, MatchResult
from .parsing import extract_resume_text
from .matching import compute_tfidf_score
from .embeddings import compute_embedding, cosine_similarity_score
from .tailoring import generate_tailoring_suggestions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@shared_task
def parse_resume_task(resume_id):
    resume = Resume.objects.get(pk=resume_id)
    try:
        text = extract_resume_text(resume.file, resume.original_filename)
        resume.raw_text = text
        resume.save()
        return {"status": "success", "resume_id": resume_id}
    except Exception as e:
        logger.error(f"Resume parsing failed for resume_id={resume_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}

@shared_task
def compute_match_task(resume_id, jd_id, method='hybrid'):
    resume = Resume.objects.get(pk=resume_id)
    jd = JobDescription.objects.get(pk=jd_id)

    normalized_method = (method or 'hybrid').lower()
    if normalized_method == 'tfidf':
        score = compute_tfidf_score(resume.raw_text, jd.raw_text)
    else:
        tfidf_score = compute_tfidf_score(resume.raw_text, jd.raw_text)
        embedding_score = None

        try:
            if not resume.embedding:
                resume.embedding = compute_embedding(resume.raw_text)
                resume.save(update_fields=['embedding'])
            if not jd.embedding:
                jd.embedding = compute_embedding(jd.raw_text)
                jd.save(update_fields=['embedding'])
            embedding_score = cosine_similarity_score(resume.embedding, jd.embedding)
        except Exception as exc:
            logger.warning(
                "Embedding failed for resume_id=%s, jd_id=%s: %s",
                resume_id,
                jd_id,
                exc,
            )

        if normalized_method == 'embedding':
            score = embedding_score if embedding_score is not None else tfidf_score
        elif embedding_score is not None:
            score = (embedding_score * 0.65) + (tfidf_score * 0.35)
        else:
            score = tfidf_score

    score = max(0.0, min(100.0, round(float(score), 1)))

    match = MatchResult.objects.create(
        resume=resume,
        job_description=jd,
        score=score,
        method=normalized_method,
    )

    # Chain: kick off suggestion generation right after the match is scored
    generate_suggestions_task.delay(match.id)

    return {"match_id": match.id, "score": score}

@shared_task
def generate_suggestions_task(match_id):
    match = MatchResult.objects.get(pk=match_id)
    suggestions = generate_tailoring_suggestions(
        match.resume.raw_text,
        match.job_description.raw_text,
        method=match.method,
        score=match.score
    )
    match.suggestions = suggestions
    match.save()
    return {"match_id": match_id, "status": "done"}
from celery import shared_task
from .models import Resume, JobDescription, MatchResult
from .parsing import extract_resume_text
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
def compute_match_task(resume_id, jd_id, method='embedding'):
    resume = Resume.objects.get(pk=resume_id)
    jd = JobDescription.objects.get(pk=jd_id)

    if not resume.embedding:
        resume.embedding = compute_embedding(resume.raw_text)
        resume.save()

    if not jd.embedding:
        jd.embedding = compute_embedding(jd.raw_text)
        jd.save()

    score = cosine_similarity_score(resume.embedding, jd.embedding)

    match = MatchResult.objects.create(
        resume=resume,
        job_description=jd,
        score=score,
        method=method
    )

    # Chain: kick off suggestion generation right after the match is scored
    generate_suggestions_task.delay(match.id)

    return {"match_id": match.id, "score": score}

@shared_task
def generate_suggestions_task(match_id):
    match = MatchResult.objects.get(pk=match_id)
    suggestions = generate_tailoring_suggestions(
        match.resume.raw_text,
        match.job_description.raw_text
    )
    match.suggestions = suggestions
    match.save()
    return {"match_id": match_id, "status": "done"}
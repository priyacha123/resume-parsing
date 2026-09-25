import os
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

SUGGESTION_PROMPT = """
You are an expert ATS resume reviewer and career coach. Compare the resume below against the job description and return ONLY a valid JSON object (no commentary, no markdown fences) in this exact JSON structure:

{{
  "overall_summary": "2-3 concise sentences summarizing key strengths and the biggest gaps between candidate and requirements.",
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "matched_strengths": ["strength1", "strength2"],
  "suggestions": [
    {{
      "section": "e.g. Work Experience or Skills",
      "issue": "What specifically is missing or weakly phrased",
      "fix": "Actionable, concrete rewrite suggestion including relevant metric or action verb"
    }}
  ]
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

def extract_json_from_response(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith('```'):
        # remove starting ``` or ```json
        first_newline = raw.find('\n')
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        if raw.endswith('```'):
            raw = raw[:-3].strip()

    # Find boundaries of the JSON object
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    return json.loads(raw)


def generate_tailoring_suggestions(resume_text: str, jd_text: str) -> dict:
    """
    Generates AI-powered resume tailoring suggestions based on job description.
    Falls back gracefully if the API fails or is unavailable.
    """
    if not resume_text.strip() or not jd_text.strip():
        return {
            "overall_summary": "Please provide both resume and job description to generate suggestions.",
            "missing_keywords": [],
            "matched_strengths": [],
            "suggestions": []
        }

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = SUGGESTION_PROMPT.format(
            resume_text=resume_text[:6000],
            jd_text=jd_text[:4000]
        )

        response = model.generate_content(prompt)
        if response and response.text:
            parsed = extract_json_from_response(response.text)
            # Ensure standard keys exist
            return {
                "overall_summary": parsed.get("overall_summary", "Review complete."),
                "missing_keywords": parsed.get("missing_keywords", []),
                "matched_strengths": parsed.get("matched_strengths", []),
                "suggestions": parsed.get("suggestions", [])
            }
    except Exception as e:
        logger.warning(f"Gemini tailoring suggestions call failed: {e}")

    # Fallback heuristic analysis if Gemini API is unreachable or fails
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    
    # Common stop words to exclude from keyword extraction
    stop_words = {
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be',
        'by', 'on', 'not', 'he', 'i', 'this', 'are', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
        'there', 'their', 'we', 'him', 'been', 'has', 'when', 'who', 'more', 'no', 'if', 'out', 'so',
        'said', 'what', 'up', 'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new',
        'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'like', 'now',
        'such', 'our', 'over', 'man', 'me', 'even', 'most', 'made', 'after', 'also', 'did', 'many',
        'must', 'should', 'work', 'experience', 'skills', 'role', 'team', 'years', 'looking', 'candidate'
    }
    
    candidate_keywords = [
        w.strip('.,;():/"\'-') for w in jd_words - resume_words 
        if len(w) > 3 and w.isalpha() and w not in stop_words
    ]
    missing = candidate_keywords[:8]

    return {
        "overall_summary": "Analysis completed. Review the highlighted missing keywords and align your experience descriptions with the job requirements.",
        "missing_keywords": missing,
        "matched_strengths": ["Relevant professional background matching core job requirements"],
        "suggestions": [
            {
                "section": "Skills & Keywords",
                "issue": "Specific JD skills are not explicitly stated in the resume.",
                "fix": f"Consider incorporating key terms like: {', '.join(missing[:4]) if missing else 'industry-standard terminology'}."
            },
            {
                "section": "Experience Impact",
                "issue": "Action verbs and quantifiable metrics can be emphasized further.",
                "fix": "Use the X-Y-Z formula: Accomplished [X], as measured by [Y], by doing [Z]."
            }
        ]
    }

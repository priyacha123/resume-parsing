import os
import json
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SUGGESTION_PROMPT = """
You are a resume optimization assistant. Compare the resume below against the job description and return ONLY a JSON object (no markdown, no preamble) in this exact shape:

{{
  "missing_keywords": ["list of important JD keywords/skills absent from the resume"],
  "suggestions": [
    {{"section": "e.g. Experience bullet #2", "issue": "what's weak", "fix": "concrete rewrite suggestion"}}
  ],
  "overall_summary": "2-3 sentence summary of the biggest gaps"
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text} 
"""

def generate_tailoring_suggestions(resume_text: str, jd_text: str) -> dict:
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = SUGGESTION_PROMPT.format(resume_text=resume_text[:6000], jd_text=jd_text[:4000])

    response = model.generate_content(prompt)
    raw = response.text.strip()
    # Gemini sometimes wraps JSON in markdown fences despite instructions — strip if present
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from model response", "raw": raw}



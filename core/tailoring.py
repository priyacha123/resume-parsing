import os
import json
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

METHOD_INSTRUCTIONS = {
    'tfidf': {
        'role': 'Exact Keyword & ATS Vocabulary Specialist',
        'focus': (
            "Focus STRICTLY on EXACT KEYWORDS, hard technical skills, tools, frameworks, acronyms, and vocabulary overlap. "
            "Analyze which exact terms from the Job Description are missing from the resume text. "
            "Your rewrite suggestions must focus on inserting exact missing terminology into relevant sections (Skills, Work Experience, Summary) "
            "to maximize keyword frequency and pass strict rule-based ATS filters."
        ),
        'summary_prefix': "Keyword Density & Vocabulary Analysis",
    },
    'embedding': {
        'role': 'Semantic AI & Role Alignment Coach',
        'focus': (
            "Focus on SEMANTIC DEPTH, conceptual relevance, seniority alignment, and role scope. "
            "Do not focus merely on isolated keywords; evaluate whether the candidate demonstrates the depth of engineering, "
            "problem-solving complexity, and domain experience implied by the job requirements. "
            "Your rewrite suggestions must focus on elevating impact, articulating business outcomes, demonstrating technical leadership, "
            "and narrative alignment with what hiring managers look for."
        ),
        'summary_prefix': "Semantic AI & Conceptual Fit Analysis",
    },
    'hybrid': {
        'role': 'Comprehensive ATS & Recruiter Evaluator',
        'focus': (
            "Provide a balanced, dual-layer evaluation combining both exact ATS keyword matching (35%) and semantic narrative relevance (65%). "
            "Highlight both critical missing technical keywords that risk ATS filtering AND qualitative gaps in how achievements, "
            "responsibilities, and competencies are phrased."
        ),
        'summary_prefix': "Hybrid ATS & Narrative Analysis",
    }
}

SUGGESTION_PROMPT = """
You are an expert {role}. Compare the resume below against the job description.
The candidate was scored using the {method_name} model and received a match score of {score_str}%.

EVALUATION OBJECTIVE:
{focus_instructions}

IMPORTANT RULES:
- Include ALL missing keywords or skill gaps you find — do NOT limit to a fixed number.
- Include ALL matched strengths you identify — do NOT truncate.
- Include a separate suggestion entry for EVERY distinct section or issue you find. Do not merge unrelated issues into one bullet.
  Typical resumes may need suggestions for: Professional Summary, Technical Skills, Work Experience (per role), Education, Certifications, Projects, etc.
- Be thorough and exhaustive. A near-perfect resume (90%+) may only need 2-3 suggestions, a weak match (below 50%) may need 8-15. Adjust accordingly.
- Every suggestion's "fix" must be concrete and directly actionable — not generic advice.

Return ONLY a valid JSON object (no markdown fences, no conversational preamble) matching this exact schema:

{{
  "overall_summary": "2-4 sentences providing a thorough executive critique specifically reflecting the {method_name} evaluation perspective.",
  "missing_keywords": ["every important missing keyword, tool, skill, certification, or domain term from the JD not present in the resume"],
  "matched_strengths": ["every concrete strength or skill from the resume that aligns with the JD requirements"],
  "suggestions": [
    {{
      "section": "Section name (e.g. Professional Summary, Technical Skills, Work Experience at [Company], Projects, Certifications)",
      "issue": "Specific weakness, omission, or phrasing gap identified from this evaluation model's perspective",
      "fix": "Actionable, concrete rewrite or addition — include example phrasing or metrics where possible"
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
    if raw.startswith('```'):
        first_newline = raw.find('\n')
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        if raw.endswith('```'):
            raw = raw[:-3].strip()

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    return json.loads(raw)


def generate_tailoring_suggestions(
    resume_text: str,
    jd_text: str,
    method: str = 'hybrid',
    score: float = None
) -> dict:
    """
    Generates model-specific resume tailoring suggestions based on the selected scoring method.
    Adapts prompt focus dynamically between TF-IDF (keywords), Embedding (semantic context), and Hybrid.
    """
    if not resume_text.strip() or not jd_text.strip():
        return {
            "overall_summary": "Please provide both resume and job description to generate suggestions.",
            "missing_keywords": [],
            "matched_strengths": [],
            "suggestions": []
        }

    norm_method = method.lower()
    if 'tfidf' in norm_method:
        active_key = 'tfidf'
        method_name = "Exact Keywords (TF-IDF)"
    elif 'embedding' in norm_method:
        active_key = 'embedding'
        method_name = "Semantic AI (Dense Vector)"
    else:
        active_key = 'hybrid'
        method_name = "Smart Hybrid (Semantic + Keywords)"

    method_config = METHOD_INSTRUCTIONS[active_key]
    score_str = f"{score:.1f}" if score is not None else "N/A"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = SUGGESTION_PROMPT.format(
            role=method_config['role'],
            method_name=method_name,
            score_str=score_str,
            focus_instructions=method_config['focus'],
            resume_text=resume_text[:6000],
            jd_text=jd_text[:4000]
        )

        response = model.generate_content(prompt)
        if response and response.text:
            parsed = extract_json_from_response(response.text)
            return {
                "overall_summary": parsed.get("overall_summary", "Review complete."),
                "missing_keywords": parsed.get("missing_keywords", []),
                "matched_strengths": parsed.get("matched_strengths", []),
                "suggestions": parsed.get("suggestions", [])
            }
    except Exception as e:
        logger.warning(f"Gemini tailoring suggestions call failed: {e}")

    # Fallback heuristic analysis customized by method
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())

    stop_words = {
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'be',
        'by', 'on', 'not', 'he', 'i', 'this', 'are', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
        'there', 'their', 'we', 'him', 'been', 'has', 'when', 'who', 'more', 'no', 'if', 'out', 'so',
        'said', 'what', 'up', 'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new',
        'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'like', 'now',
        'such', 'our', 'over', 'man', 'me', 'even', 'most', 'made', 'after', 'also', 'did', 'many',
        'must', 'should', 'work', 'experience', 'skills', 'role', 'team', 'years', 'looking', 'candidate'
    }

    diff_words = sorted(
        {w.strip('.,;():/"\'-') for w in jd_words - resume_words
         if len(w) > 3 and w.isalpha() and w not in stop_words},
        key=lambda x: len(x), reverse=True  # longer words tend to be more meaningful
    )
    missing = diff_words  # No artificial cap — return all gaps found

    if active_key == 'tfidf':
        # Build one keyword-insertion suggestion per 3 missing keywords (group them cleanly)
        suggestions = []
        for i in range(0, max(len(missing), 1), 3):
            chunk = missing[i:i + 3]
            if chunk:
                suggestions.append({
                    "section": "Technical Skills / Keywords",
                    "issue": f"Exact JD keyword tokens absent: {', '.join(chunk)}.",
                    "fix": f"Add these exact terms to your Skills section and weave them naturally into relevant Work Experience bullets: {', '.join(chunk)}."
                })
        suggestions.append({
            "section": "Work Experience Bullets (All Roles)",
            "issue": "High-value JD terms are under-represented across your work history, reducing TF-IDF relevance density.",
            "fix": "In each role, rephrase existing bullets to naturally include relevant JD vocabulary. Repeating key terms across multiple sections meaningfully raises your ATS keyword score."
        })
        return {
            "overall_summary": f"Keyword Analysis: Candidate scored {score_str}%. The resume lacks key vocabulary tokens explicitly emphasized in the job posting. {len(missing)} distinct keyword gaps were identified.",
            "missing_keywords": missing,
            "matched_strengths": ["Shared baseline technical vocabulary with the job description"],
            "suggestions": suggestions
        }
    elif active_key == 'embedding':
        # Semantic fallback: produce one suggestion per missing domain term plus general depth suggestions
        suggestions = []
        for kw in missing:
            suggestions.append({
                "section": "Work Experience or Projects",
                "issue": f"Conceptual domain of '{kw}' is absent or underrepresented in context.",
                "fix": f"Add concrete examples demonstrating work involving {kw}, with outcome-driven framing: 'Designed and deployed [solution using {kw}] resulting in [measurable outcome].'"
            })
        # Always include structural semantic suggestions
        suggestions.extend([
            {
                "section": "Professional Summary",
                "issue": "Lacks a clear executive statement of seniority, specialization, and scope of impact.",
                "fix": "Open with: '[X]+ years of [domain] engineering experience, delivering [scale/complexity] systems used by [audience/impact]. Expert in [key technologies from JD].' — make it role-specific."
            },
            {
                "section": "Work Experience Impact",
                "issue": "Bullet points describe responsibilities rather than measurable business outcomes.",
                "fix": "Reframe using: 'Led [initiative], achieving [quantified result] by implementing [approach].' Include metrics: percentages, user counts, latency improvements, revenue impact."
            }
        ])
        return {
            "overall_summary": f"Semantic AI Analysis: Candidate scored {score_str}%. {len(missing)} conceptual domain gaps were identified beyond basic keyword matching.",
            "missing_keywords": missing,
            "matched_strengths": ["Strong conceptual background matching core domain responsibilities"],
            "suggestions": suggestions
        }
    else:
        # Hybrid: combine keyword-insertion + impact/narrative suggestions
        suggestions = []
        for i in range(0, max(len(missing), 1), 3):
            chunk = missing[i:i + 3]
            if chunk:
                suggestions.append({
                    "section": "Technical Skills / Keywords",
                    "issue": f"These JD-critical terms are absent: {', '.join(chunk)}.",
                    "fix": f"Explicitly list {', '.join(chunk)} in your Skills section. Then weave each into a Work Experience bullet to maximize both keyword frequency and semantic context."
                })
        suggestions.extend([
            {
                "section": "Professional Summary",
                "issue": "Does not immediately signal alignment with the target role's title and core competencies.",
                "fix": "Rewrite the summary to mirror the JD's language: include seniority level, core domain, and 2-3 key technologies from the job posting in the first sentence."
            },
            {
                "section": "Work Experience Impact",
                "issue": "Bullets describe tasks but lack quantifiable impact statements that signal seniority.",
                "fix": "Apply X-Y-Z framing: 'Accomplished [X], as measured by [Y], by doing [Z].' Add metrics (e.g., 40% latency reduction, 2M+ users, 99.9% uptime)."
            }
        ])
        return {
            "overall_summary": f"Hybrid Analysis: Candidate scored {score_str}%. {len(missing)} keyword and narrative gaps identified across both ATS keyword and semantic dimensions.",
            "missing_keywords": missing,
            "matched_strengths": ["Relevant professional background matching core job requirements"],
            "suggestions": suggestions
        }

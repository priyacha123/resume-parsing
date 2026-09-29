import os
import json
import logging
import re
from google import genai

logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None

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
- Suggest exactly 2 realistic portfolio project ideas that close the most important gaps for this specific job and evaluation model.
- Base each project on the candidate's actual resume background, existing projects, and the specific job requirements. Do not reuse generic project titles when the resume or job requirements provide more specific domains or technologies.
- Project ideas must be distinct from generic advice and should include practical features, technologies, and the resume value they demonstrate.

FORMATTING RULES (apply to every regular section — Professional Summary, Technical Skills, Work Experience, Education, Certifications, and any other section that applies):
- Group all suggestions under their section name — one block per section, never mixed across sections.
- Each section gets at most 5 short bullet points. If more issues exist, keep only the 5 highest-impact ones.
- Each bullet is a single concise sentence, no more than 18 words, suitable for a one-line UI block. Never concatenate multiple issues into one long sentence or paragraph.
- Deduplicate — if two potential bullets say substantially the same thing, keep only the stronger phrasing and drop the other.
- Work Experience gets one block per role, keyed by "Work Experience at [Company]" — do not combine multiple roles into a single block.
- Only include sections that are actually relevant to this resume/JD pair. Do not output an empty section.
- Be thorough but disciplined: a near-perfect resume (90%+) may only need 1-2 sections with 1-2 bullets each; a weak match (below 50%) may use the full 5-bullet cap across most sections.
- Every bullet must be concrete and directly actionable — never generic advice like "improve your summary."

PROJECT SECTION RULES (separate from the sections above):
- For each existing resume project that's relevant to this JD, and for each of the 2 new suggested project ideas, provide 3-5 concise resume-ready achievement bullets.
- Each bullet uses an action verb, is directly adaptable into a resume, and includes a measurable outcome wherever realistically possible (e.g. "Reduced query latency by 40% via indexed lookups" rather than "Improved performance").
- Keep each project bullet to one line — no run-on sentences.

Return ONLY a valid JSON object (no markdown fences, no conversational preamble) matching this exact schema:

{{
  "overall_summary": "2-4 sentences providing a thorough executive critique specifically reflecting the {method_name} evaluation perspective.",
  "missing_keywords": ["every important missing keyword, tool, skill, certification, or domain term from the JD not present in the resume"],
  "matched_strengths": ["every concrete strength or skill from the resume that aligns with the JD requirements"],
  "project_ideas": [
    {{
      "title": "Specific portfolio project name",
      "rationale": "Why this project closes a gap for this role and this evaluation model",
      "features": ["3-5 concrete features to implement"],
      "technologies": ["Relevant technologies from the JD or adjacent tools"],
      "resume_value": "The capability or measurable outcome this project could demonstrate on the resume",
      "bullets": ["3-5 resume-ready achievement bullets for this NEW project, written as if already completed"]
    }}
  ],
  "sections": {{
    "Professional Summary": ["bullet 1", "bullet 2"],
    "Technical Skills": ["bullet 1", "bullet 2"],
    "Work Experience at [Company Name]": ["bullet 1", "bullet 2"],
    "Education": ["bullet 1"],
    "Certifications": ["bullet 1"],
    "Projects": {{
      "[Existing Project Name From Resume]": ["bullet 1", "bullet 2", "bullet 3"]
    }}
  }}
}}

Only include keys in "sections" that are genuinely relevant — omit any section with nothing meaningful to say rather than returning an empty array.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

# SUGGESTION_PROMPT = """You are a resume optimization assistant. Compare the resume below against the job description and return ONLY a JSON object (no markdown, no preamble) in this exact shape:

# {{
#   "missing_keywords": ["list of important JD keywords/skills absent from the resume"],
#   "suggestions": [
#     {{"section": "e.g. Experience bullet #2", "issue": "what's weak", "fix": "concrete rewrite suggestion"}}
#   ],
#   "overall_summary": "2-3 sentence summary of the biggest gaps"
# }}

# RESUME:
# {resume_text}

# JOB DESCRIPTION:
# {jd_text}
# """


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


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _suggestion_list(value, sections=None) -> list[dict]:
    suggestions = []
    if isinstance(sections, dict):
        for section, content in sections.items():
            if isinstance(content, dict):
                for subsection, bullets in content.items():
                    suggestions.append(_section_block(
                        f"{section}: {subsection}",
                        f"Improve the {subsection} subsection.",
                        bullets,
                    ))
            else:
                suggestions.append(_section_block(
                    section,
                    f"Improve the {section} section.",
                    content,
                ))
        return _limit_suggestions(suggestions)

    if not isinstance(value, list):
        return []
    for item in value:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(key), str) for key in ("section", "issue", "fix")):
            continue
        fixes = item.get("recommendations")
        if not isinstance(fixes, list):
            fixes = _split_recommendations(item["fix"])
        suggestions.append(_section_block(item["section"], item["issue"], fixes))
    return _limit_suggestions(suggestions)


def _section_block(section: str, issue: str, fixes) -> dict:
    bullets = []
    if isinstance(fixes, list):
        for fix in fixes:
            if isinstance(fix, str):
                bullets.extend(_split_recommendations(fix))
    elif isinstance(fixes, str):
        bullets = _split_recommendations(fixes)
    unique_bullets = []
    seen = set()
    for bullet in bullets:
        concise = _shorten_text(bullet, 180)
        key = concise.lower()
        if concise and key not in seen:
            seen.add(key)
            unique_bullets.append(concise)
    return {
        "section": section.strip() or "General Resume",
        "issue": _shorten_text(issue, 150),
        "fix": unique_bullets[0] if unique_bullets else "Add a specific, measurable improvement for this section.",
        "recommendations": unique_bullets[:5],
    }


def _split_recommendations(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+", text.strip())
    cleaned = []
    for part in parts:
        value = re.sub(r"^\s*[-•]\s*", "", part).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned or [text.strip()]


def _shorten_text(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rsplit(' ', 1)[0]}..."


def _limit_suggestions(suggestions: list[dict]) -> list[dict]:
    grouped = {}
    order = []
    for suggestion in suggestions:
        section = suggestion["section"].strip() or "General Resume"
        key = section.lower()
        block = _section_block(
            section,
            suggestion["issue"],
            suggestion.get("recommendations") or [suggestion["fix"]],
        )
        if key not in grouped:
            grouped[key] = block
            order.append(key)
            continue
        existing = grouped[key]["recommendations"]
        for bullet in block["recommendations"]:
            if bullet.lower() not in {item.lower() for item in existing} and len(existing) < 5:
                existing.append(bullet)
        grouped[key]["fix"] = existing[0] if existing else grouped[key]["fix"]
        if len(order) == 10:
            break
    return [grouped[key] for key in order[:10]]


def _project_ideas_from_model(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    ideas = []
    for item in value[:2]:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(key), str) for key in ("title", "rationale", "resume_value")):
            continue
        features = _string_list(item.get("features"))
        technologies = _string_list(item.get("technologies"))
        if features and technologies:
            ideas.append({
                "title": item["title"].strip(),
                "rationale": item["rationale"].strip(),
                "features": features,
                "technologies": technologies,
                "resume_value": item["resume_value"].strip(),
            })
    return ideas


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
            "project_ideas": [],
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
        prompt = SUGGESTION_PROMPT.format(
            role=method_config['role'],
            method_name=method_name,
            score_str=score_str,
            focus_instructions=method_config['focus'],
            resume_text=resume_text[:6000],
            jd_text=jd_text[:4000]
        )

        if gemini_client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        if response and response.text:
            parsed = extract_json_from_response(response.text)
            project_ideas = _project_ideas_from_model(parsed.get("project_ideas"))
            if len(project_ideas) < 2:
                fallback_ideas = _project_ideas([], active_key, resume_text, jd_text)
                project_ideas.extend(fallback_ideas[:2 - len(project_ideas)])
            return {
                "overall_summary": parsed.get("overall_summary", "Review complete."),
                "missing_keywords": _string_list(parsed.get("missing_keywords")),
                "matched_strengths": _string_list(parsed.get("matched_strengths")),
                "project_ideas": project_ideas[:2],
                "suggestions": _suggestion_list(
                    parsed.get("suggestions"),
                    parsed.get("sections"),
                ),
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
        keyword_bullets = [
            f"Add {keyword} to Technical Skills and one relevant experience bullet."
            for keyword in missing[:5]
        ]
        suggestions = [
            _section_block(
                "Technical Skills",
                "Important job-description keywords are missing from the resume.",
                keyword_bullets or ["List the most relevant tools from the job description in your Skills section."],
            ),
            _section_block(
                "Work Experience",
                "Experience bullets do not show enough evidence for the target requirements.",
                ["Rewrite relevant bullets using the job description's terminology and measurable outcomes."],
            ),
        ]
        return {
            "overall_summary": f"Keyword Analysis: Candidate scored {score_str}%. The resume lacks key vocabulary tokens explicitly emphasized in the job posting. {len(missing)} distinct keyword gaps were identified.",
            "missing_keywords": missing,
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _limit_suggestions(suggestions)
        }
    elif active_key == 'embedding':
        domain_bullets = [
            f"Show a concrete project or achievement using {keyword}."
            for keyword in missing[:5]
        ]
        suggestions = [
            _section_block(
                "Professional Summary",
                "The summary does not clearly communicate role fit and impact.",
                ["State your specialization, seniority, and strongest technologies in the opening sentence."],
            ),
            _section_block(
                "Work Experience",
                "Experience bullets emphasize duties more than outcomes.",
                domain_bullets or ["Add measurable outcomes, technical decisions, and business impact to key bullets."],
            ),
        ]
        return {
            "overall_summary": f"Semantic AI Analysis: Candidate scored {score_str}%. {len(missing)} conceptual domain gaps were identified beyond basic keyword matching.",
            "missing_keywords": missing,
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _limit_suggestions(suggestions)
        }
    else:
        keyword_bullets = [
            f"Add {keyword} to Technical Skills and one relevant experience bullet."
            for keyword in missing[:5]
        ]
        suggestions = [
            _section_block(
                "Technical Skills",
                "Important job-description keywords are missing from the resume.",
                keyword_bullets or ["List the most relevant tools from the job description in your Skills section."],
            ),
            _section_block(
                "Professional Summary",
                "The summary does not clearly signal alignment with the target role.",
                ["Mention the target role, seniority, domain, and two relevant technologies."],
            ),
            _section_block(
                "Work Experience",
                "Experience bullets need stronger keyword context and measurable impact.",
                ["Rewrite key bullets with relevant technologies, actions, and measurable outcomes."],
            ),
        ]
        return {
            "overall_summary": f"Hybrid Analysis: Candidate scored {score_str}%. {len(missing)} keyword and narrative gaps identified across both ATS keyword and semantic dimensions.",
            "missing_keywords": missing,
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _limit_suggestions(suggestions)
        }


def _matched_strengths(resume_text: str, jd_text: str, method: str) -> list[str]:
    """Return a useful, model-specific fallback when the generative response is unavailable."""
    resume_tokens = set(re.findall(r"[a-z0-9+#.]+", resume_text.lower()))
    jd_tokens = set(re.findall(r"[a-z0-9+#.]+", jd_text.lower()))
    shared = sorted(
        {
            word
            for word in resume_tokens.intersection(jd_tokens)
            if len(word) > 3
        }
    )
    if method == 'tfidf':
        return [f"Exact JD vocabulary overlap: {', '.join(shared[:8])}" if shared else "Some exact JD vocabulary overlap detected"]
    if method == 'embedding':
        return [
            "Relevant conceptual background aligned with the role's responsibilities",
            "Transferable problem-solving experience that can support the target domain",
        ]
    return [
        f"Shared technical vocabulary: {', '.join(shared[:8])}" if shared else "Relevant technical vocabulary overlap",
        "A foundation of experience that can be strengthened with targeted impact evidence",
    ]


def _project_ideas(
    missing: list[str],
    method: str,
    resume_text: str = "",
    jd_text: str = "",
) -> list[dict]:
    """Provide actionable project ideas that vary with the selected evaluation model."""
    relevant_terms = _project_terms(missing, resume_text, jd_text)
    gap_text = ', '.join(relevant_terms[:3]) or 'the highest-priority job requirements'
    primary = relevant_terms[0] if relevant_terms else "the target role"
    secondary = relevant_terms[1] if len(relevant_terms) > 1 else "production reliability"
    if method == 'tfidf':
        return [
            {
                "title": f"{primary.title()} Skills Validation Platform",
                "rationale": f"Build a focused application around {gap_text} so the resume gains direct, credible evidence for the highest-priority ATS gaps.",
                "features": [f"Core workflow demonstrating {primary}", f"Searchable {secondary} results", "Automated tests", "CI pipeline with coverage"],
                "technologies": relevant_terms[:4] or ["REST API", "Docker", "PostgreSQL"],
                "resume_value": f"Adds measurable implementation evidence for {primary} and related job-specific terminology.",
            },
            {
                "title": f"{secondary.title()} Automation Toolkit",
                "rationale": f"Create a practical tool that applies {secondary} to a realistic workflow and documents the exact technologies missing from the resume: {gap_text}.",
                "features": ["Import and export workflows", f"Automated {primary} processing", "Integration tests", "CI/CD status reporting"],
                "technologies": relevant_terms[4:8] or relevant_terms[:4] or ["Python", "REST API", "GitHub Actions"],
                "resume_value": f"Demonstrates hands-on use of {secondary}, testing, and delivery practices rather than unsupported keyword listing.",
            },
        ]
    if method == 'embedding':
        return [
            {
                "title": f"Scalable {primary.title()} Operations Platform",
                "rationale": f"Demonstrate end-to-end ownership and system depth by solving a realistic operational problem involving {gap_text}.",
                "features": ["Role-based workflow", f"Domain-specific {primary} processing", "Observability dashboard", "Failure recovery and audit history"],
                "technologies": relevant_terms[:4] or ["Python", "PostgreSQL", "Redis", "Docker"],
                "resume_value": f"Shows architecture decisions, reliability thinking, and measurable outcomes in {primary}.",
            },
            {
                "title": f"{secondary.title()} Decision Intelligence System",
                "rationale": f"Build a data-informed system that connects {primary} with {secondary}, proving conceptual fit beyond isolated technology keywords.",
                "features": ["Event-driven data ingestion", "Role-based dashboards", "Retry and dead-letter handling", "Performance and availability metrics"],
                "technologies": relevant_terms[4:8] or relevant_terms[:4] or ["FastAPI", "Redis", "PostgreSQL", "Docker"],
                "resume_value": f"Adds evidence of scalability, observability, fault tolerance, and engineering impact across {primary} and {secondary}.",
            },
        ]
    return [
        {
            "title": f"{primary.title()} Intelligence Workspace",
            "rationale": f"Combine {gap_text} in a user-facing workflow that demonstrates both implementation depth and product judgment for this target role.",
            "features": ["Personalized recommendations", "Search and filtering", "Async processing", "Metrics dashboard", "Secure API"],
            "technologies": relevant_terms[:4] or ["React", "TypeScript", "Python", "PostgreSQL"],
            "resume_value": f"Provides keyword coverage, semantic relevance, measurable outcomes, and clear tradeoffs centered on {primary}.",
        },
        {
            "title": f"{secondary.title()} Collaboration Hub",
            "rationale": f"Build a collaborative product that applies the target role's requirements to a workflow involving {primary} and {secondary}.",
            "features": ["Secure authentication", "Real-time collaboration", "Background notifications", "Search and filtering", "Deployment monitoring"],
            "technologies": relevant_terms[4:8] or relevant_terms[:4] or ["Next.js", "TypeScript", "Python", "Docker"],
            "resume_value": f"Shows full-stack delivery, product thinking, deployment, and the ability to connect {primary} with {secondary}.",
        },
    ]


def _project_terms(missing: list[str], resume_text: str, jd_text: str) -> list[str]:
    terms = []
    for term in missing:
        cleaned = term.strip()
        if cleaned and cleaned.lower() not in {item.lower() for item in terms}:
            terms.append(cleaned)

    jd_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", jd_text)
    resume_tokens = {token.lower() for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", resume_text)}
    stop_words = {
        "the", "and", "for", "with", "from", "this", "that", "are", "you", "your",
        "will", "have", "has", "our", "their", "into", "using", "work", "years",
        "experience", "role", "team", "skills", "strong", "must", "about",
    }
    for token in jd_tokens:
        lowered = token.lower()
        if lowered in stop_words or lowered in resume_tokens or len(token) < 4:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= 8:
            break
    return terms[:8]

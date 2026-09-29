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
- Include a separate suggestion entry for EVERY distinct section or issue you find. Do not merge unrelated issues into one bullet.
  Typical resumes may need suggestions for: Professional Summary, Technical Skills, Work Experience (per role), Education, Certifications, Projects, etc.
- Be thorough and exhaustive. A near-perfect resume (90%+) may only need 2-3 suggestions, a weak match (below 50%) may need 8-15. Adjust accordingly.
- Every suggestion's "fix" must be concrete and directly actionable — not generic advice.

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
      "resume_value": "The capability or measurable outcome this project could demonstrate on the resume"
    }}
  ],
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


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _suggestion_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    suggestions = [
        {
            "section": item.get("section", "").strip(),
            "issue": item.get("issue", "").strip(),
            "fix": item.get("fix", "").strip(),
        }
        for item in value
        if isinstance(item, dict)
        and all(isinstance(item.get(key), str) for key in ("section", "issue", "fix"))
    ]
    return _group_suggestions(suggestions)


def _suggestion_group_key(section: str) -> str:
    normalized = re.sub(r"\s+", " ", section.strip().lower())
    if "project" in normalized:
        return normalized
    if any(term in normalized for term in ("skill", "keyword", "technology", "tech stack")):
        return "technical skills"
    if "summary" in normalized or "objective" in normalized:
        return "professional summary"
    if any(term in normalized for term in ("work experience", "experience", "employment", "career")):
        return "work experience"
    if "education" in normalized:
        return "education"
    if "certif" in normalized:
        return "certifications"
    return normalized


def _group_suggestions(suggestions: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for suggestion in suggestions:
        section = suggestion["section"] or "General Resume"
        key = _suggestion_group_key(section)
        if key not in grouped:
            grouped[key] = {
                "section": section if "project" in key else key.title(),
                "issue": suggestion["issue"],
                "fix": suggestion["fix"],
            }
            order.append(key)
            continue

        grouped[key]["issue"] += f" {suggestion['issue']}"
        grouped[key]["fix"] += f" {suggestion['fix']}"

    return [grouped[key] for key in order]


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
                "suggestions": _suggestion_list(parsed.get("suggestions")),
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
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _group_suggestions(suggestions)
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
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _group_suggestions(suggestions)
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
            "matched_strengths": _matched_strengths(resume_text, jd_text, active_key),
            "project_ideas": _project_ideas(missing, active_key, resume_text, jd_text),
            "suggestions": _group_suggestions(suggestions)
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

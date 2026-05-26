import re
import spacy

nlp = spacy.load("en_core_web_sm")

# ── Known tech/soft skills to look for in job descriptions ──────────
KNOWN_SKILLS = {
    # Technical
    "python", "java", "javascript", "c++", "c#", "sql", "nosql", "r",
    "machine learning", "deep learning", "data science", "ai", "nlp",
    "cloud computing", "aws", "azure", "gcp", "docker", "kubernetes",
    "react", "node.js", "nodejs", "angular", "vue", "django", "flask",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
    "networking", "cybersecurity", "devops", "git", "linux", "restapi",
    "rest api", "graphql", "mongodb", "postgresql", "mysql", "redis",
    "spark", "hadoop", "tableau", "powerbi", "excel", "html", "css",
    # Soft
    "communication", "problem-solving", "problem solving", "leadership",
    "teamwork", "time management", "adaptability", "creativity",
    "project management", "critical thinking", "organization",
    "collaboration", "analytical", "attention to detail",
}


def extract_jd_skills(job_description: str) -> set:
    """
    Extracts only meaningful skill keywords from a job description.
    Uses two strategies:
      1. Direct match against KNOWN_SKILLS list (multi-word aware)
      2. NLP noun/proper-noun tokens that are not stopwords
    """
    jd_lower = job_description.lower()

    # Strategy 1 — match known skills directly (handles multi-word e.g. "machine learning")
    found = set()
    for skill in KNOWN_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
            found.add(skill)

    # Strategy 2 — NLP single-word nouns not in stoplist
    doc = nlp(jd_lower)
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.pos_ in ("NOUN", "PROPN")
            and len(token.text) > 2
        ):
            found.add(token.lemma_)

    return found


def calculate_match_score(job_description: str, resume_skills: list) -> int:
    """
    Calculates job match % by comparing resume skills against
    skill-specific keywords extracted from the job description.

    Returns an integer 0-100.
    """
    if not job_description or not resume_skills:
        return 0

    # Extract meaningful skill keywords from JD
    jd_skills = extract_jd_skills(job_description)

    # Normalise resume skills to lowercase + lemma variants
    resume_set = set()
    for skill in resume_skills:
        s = skill.lower().strip()
        resume_set.add(s)
        # also add spaCy lemma so "managing" matches "management" etc.
        doc = nlp(s)
        for token in doc:
            resume_set.add(token.lemma_)

    # Debugging
    print("\n=== DEBUG: Job Match ===")
    print(f"JD skill keywords ({len(jd_skills)}): {sorted(jd_skills)}")
    print(f"Resume skills ({len(resume_set)}): {sorted(resume_set)}")

    if not jd_skills:
        return 0

    matched = jd_skills.intersection(resume_set)
    print(f"Matched ({len(matched)}): {sorted(matched)}")
    print("========================\n")

    # Score = matched skills / total JD skills * 100
    # Cap denominator at len(jd_skills) so score is always meaningful
    raw = (len(matched) / len(jd_skills)) * 100

    # Boost: if resume covers >50% of JD skills, apply a readability boost
    # so a genuinely good match shows 60-80% instead of 20-30%
    if raw >= 30:
        raw = min(raw * 1.6, 98)
    elif raw >= 15:
        raw = min(raw * 1.3, 60)

    return int(round(raw))

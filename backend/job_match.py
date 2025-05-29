import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def calculate_match_score(job_description, resume_skills):
    """
    Calculates how well a resume matches a job description based on skills.
    """
    if not job_description or not resume_skills:
        return 0  # Avoid division by zero or empty values

    job_desc_doc = nlp(job_description.lower())
    job_keywords = {token.text for token in job_desc_doc if not token.is_stop and not token.is_punct}

    resume_skill_set = set(skill.lower() for skill in resume_skills)

    # ✅ Debugging Logs
    print("\n=== DEBUG: Job Match Calculation ===")
    print(f"Extracted Job Keywords: {job_keywords}")
    print(f"Resume Skills: {resume_skill_set}")
    print("====================================\n")

    # ✅ Matching Logic
    matched_skills = job_keywords.intersection(resume_skill_set)
    match_percentage = (len(matched_skills) / len(job_keywords)) * 100 if job_keywords else 0

    return round(match_percentage, 2)

import fitz  # PyMuPDF for PDF extraction
import spacy
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# ✅ Skill Sets
TECHNICAL_SKILLS = {
    "Python", "Java", "JavaScript", "C++", "SQL", "Machine Learning", "Deep Learning",
    "Data Science", "AI", "Cloud Computing", "AWS", "Azure", "Docker", "Kubernetes",
    "React", "Node.js", "TensorFlow", "Keras", "Networking", "Cybersecurity"
}

SOFT_SKILLS = {
    "Communication", "Problem-Solving", "Leadership", "Teamwork", "Time Management",
    "Adaptability", "Creativity", "Project Management", "Critical Thinking", "Organization"
}

# ✅ Extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

# ✅ Extract name from resume
def extract_name(text):
    """Attempts to extract the candidate's name."""
    doc = nlp(text)
    name_candidates = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # 🔍 DEBUG: Print extracted names
    print(f"DEBUG: Possible Names Extracted: {name_candidates}")

    return name_candidates[0] if name_candidates else "Not Found"

# ✅ Extract skills from resume
def extract_skills(text):
    """Extracts technical and soft skills from resume text."""
    text_lower = text.lower()

    technical_skills_found = {skill for skill in TECHNICAL_SKILLS if skill.lower() in text_lower}
    soft_skills_found = {skill for skill in SOFT_SKILLS if skill.lower() in text_lower}

    # 🔍 Debugging Output
    print("\n=== DEBUG: Skills Extraction ===")
    print(f"Technical Skills Found: {technical_skills_found}")
    print(f"Soft Skills Found: {soft_skills_found}")
    print("=================================\n")

    return list(technical_skills_found), list(soft_skills_found)

# ✅ Calculate ATS Score
def calculate_ats_score(text):
    """Calculates ATS score based on keyword matching, formatting, and structure."""
    score = 0
    max_score = 100

    # ✅ 1. Keyword Matching (40 Points)
    keywords = TECHNICAL_SKILLS.union(SOFT_SKILLS)  # Combine skill sets
    keyword_matches = sum(1 for word in keywords if word.lower() in text.lower())
    score += min(keyword_matches * 2, 40)  # Adjusted for better weighting

    # ✅ 2. Formatting & Structure (20 Points)
    if any(term in text for term in ["Education", "Work Experience", "Skills", "Projects"]):
        score += 20

    # ✅ 3. Contact Information (15 Points)
    email_pattern = r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+"
    phone_pattern = r"\+?\d{10,13}"
    if re.search(email_pattern, text) and re.search(phone_pattern, text):
        score += 15

    # ✅ 4. Readability & Length (15 Points)
    word_count = len(text.split())
    if 300 <= word_count <= 1000:  # Improved readability range
        score += 15
    elif word_count < 200:
        score -= 5  # Less strict penalties for short resumes
    elif word_count > 1200:
        score -= 5  # Less strict penalties for long resumes

    return min(max(score, 0), max_score)  # Ensure score stays within 0-100%

# ✅ Analyze Resume Function
def analyze_resume(pdf_path):
    """Analyzes the resume and provides an ATS score and extracted data."""
    text = extract_text_from_pdf(pdf_path)

    name = extract_name(text)
    technical_skills, soft_skills = extract_skills(text)
    ats_score = calculate_ats_score(text)

    # 🔍 Debugging: Print Extracted Data
    print("\n=== DEBUG: Resume Analysis ===")
    print(f"Name: {name}")
    print(f"Technical Skills: {technical_skills}")
    print(f"Soft Skills: {soft_skills}")
    print(f"ATS Score: {ats_score}")
    print("=================================\n")

    return {
        "name": name,
        "technical_skills": technical_skills,
        "soft_skills": soft_skills,
        "ats_score": ats_score,
        "recommendations": "Improve keyword usage, formatting, and include all essential sections."
    }

import re
import random

def fetch_linkedin_profile(url):
    # Placeholder mock function (simulate content fetch)
    return """
    John Doe is a software engineer with expertise in Python, data analysis, and cloud computing.
    Currently working at XYZ Corp. Passionate about AI and machine learning.
    """

def analyze_linkedin_content(text):
    score = 0
    keywords = []

    # Basic keyword extraction (mock logic)
    key_terms = ["Python", "AI", "Machine Learning", "Leadership", "Cloud", "Data Analysis", "Communication"]
    for kw in key_terms:
        if kw.lower() in text.lower():
            keywords.append(kw)

    score = min(100, len(keywords) * 10 + random.randint(0, 20))  # Simulated scoring

    suggestions = []
    if "Python" not in keywords:
        suggestions.append("Add Python expertise if applicable.")
    if "Leadership" not in keywords:
        suggestions.append("Highlight leadership experience.")
    if "Communication" not in keywords:
        suggestions.append("Mention communication skills or achievements.")

    suggestions.append("Update headline to include keywords.")
    suggestions.append("Add more endorsements or recommendations.")

    return score, keywords, suggestions

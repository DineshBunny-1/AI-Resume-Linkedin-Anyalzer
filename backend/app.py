import os
import time
import streamlit as st
import mysql.connector
from resume_processing import analyze_resume
from job_match import calculate_match_score
from database import save_to_db, fetch_past_resumes
from linkedin_analyzer import fetch_linkedin_profile, analyze_linkedin_content

# === Ensure Required spaCy & NLTK Models Installed ===
import spacy
import nltk

# Load or download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data (only once; safe to call multiple times)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 🔧 Set wide layout
st.set_page_config(layout="wide")

# === Sidebar Navigation ===
st.sidebar.title("🔍 Choose Analyzer")
selected_option = st.sidebar.radio("Select Option", ["Resume Analyzer", "LinkedIn Profile Analyzer"])

# === Resume Analyzer ===
if selected_option == "Resume Analyzer":
    st.title("📄 AI Resume Analyzer")

    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("📝 Paste Job Description (Optional)")

    if uploaded_file:
        with st.status("📊 Analyzing resume...", expanded=True) as status:
            time.sleep(1.5)
            st.write("✔️ Extracting Experience & Skills")
            time.sleep(1.5)
            st.write("✔️ Scanning for ATS Compatibility")
            time.sleep(1.5)
            st.write("✔️ Generating Report & Insights")
            time.sleep(1.5)
            status.update(label="✅ Analysis Complete!", state="complete")

        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Analyze Resume
        resume_data = analyze_resume(file_path)
        name = resume_data.get("name", "Unknown")
        ats_score = resume_data.get("ats_score", 0)

        # Merge both skill sets for job match
        tech_skills = resume_data.get("technical_skills", [])
        soft_skills = resume_data.get("soft_skills", [])
        all_skills = tech_skills + soft_skills

        education = resume_data.get("education", "Not Found")
        experience = resume_data.get("experience", "Not Found")

        # Calculate match score
        match_score = 0
        if job_description:
            match_score = calculate_match_score(job_description, all_skills)

        # Save to DB
        save_to_db(name, ats_score, all_skills, education, experience, match_score)

        # === Results Display ===
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 ATS Score")
            st.metric(label="Applicant Tracking Score", value=f"{ats_score}%")

            st.subheader("🔍 Job Match Score")
            st.metric(label="Resume Match with Job", value=f"{match_score}%")

        with col2:
            st.subheader("🛠 Skills Overview")
            st.write(f"**Technical Skills:** {', '.join(tech_skills) if tech_skills else 'None found'}")
            st.write(f"**Soft Skills:** {', '.join(soft_skills) if soft_skills else 'None found'}")

        st.success("✅ Resume data saved to database successfully!")

        # === Past Resume Analyses ===
        st.subheader("📜 Past Resume Analyses")
        past_resumes = fetch_past_resumes()
        for res in past_resumes:
            st.write(f"🔹 **{res[0]}** | ATS Score: {res[1]}% | Job Match: {res[3]}%")

# === LinkedIn Profile Analyzer ===
elif selected_option == "LinkedIn Profile Analyzer":
    st.title("🔗 LinkedIn Profile Analyzer")
    st.subheader("🔗 Paste Your LinkedIn Profile URL")

    linkedin_url = st.text_input("🔍 Enter LinkedIn Profile URL")

    if linkedin_url:
        with st.status("🤖 Analyzing LinkedIn profile...", expanded=True) as status:
            time.sleep(1)
            st.write("✔️ Fetching Profile Content...")
            profile_text = fetch_linkedin_profile(linkedin_url)

            st.write("✔️ Running AI Analysis...")
            score, keywords, suggestions = analyze_linkedin_content(profile_text)

            status.update(label="✅ LinkedIn Analysis Complete!", state="complete")

        st.metric(label="🔎 Profile Strength Score", value=f"{score} / 100")

        st.subheader("🔑 Detected Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords found.")

        st.subheader("📈 Suggestions to Improve Profile")
        for suggestion in suggestions:
            st.write(f"✅ {suggestion}")

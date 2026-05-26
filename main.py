import os
import sys
import time

# ── Path setup ───────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
sys.path.insert(0, BACKEND_DIR)

# ── spaCy / NLTK bootstrap ───────────────────────────────────────────
import spacy, nltk
try:
    spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download; download("en_core_web_sm")

for _r in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{_r}" if _r == "punkt" else f"corpora/{_r}")
    except LookupError:
        nltk.download(_r, quiet=True)

# ── Streamlit ────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="AI Resume & LinkedIn Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
css_path = os.path.join(FRONTEND_DIR, "styles.css")
_extra   = open(css_path).read() if os.path.exists(css_path) else ""

st.markdown(f"""
<style>
{_extra}
[data-testid="stAppViewContainer"] {{ background:#0f1117; }}
[data-testid="stSidebar"] {{ background:#161b27; border-right:1px solid #2a2f3f; }}
[data-testid="stFileUploader"] {{
    border:2px dashed #3b82f6 !important;
    border-radius:14px !important;
    padding:20px !important;
    background:#131722 !important;
}}
div.stButton > button {{
    width:100%; padding:14px 0; font-size:17px; font-weight:700;
    background:linear-gradient(135deg,#3b82f6,#6366f1);
    color:white; border:none; border-radius:12px; margin-top:10px;
    transition:opacity .2s;
}}
div.stButton > button:hover {{ opacity:.85; }}
[data-testid="metric-container"] {{
    background:#1a1f2e; border:1px solid #2a2f3f;
    border-radius:14px; padding:18px 22px;
}}
.step-item {{
    display:flex; align-items:center; gap:12px;
    padding:10px 16px; margin:5px 0;
    background:#1a1f2e; border-radius:10px;
    border-left:3px solid #3b82f6;
    font-size:15px; color:#e2e8f0;
}}
.step-done {{ border-left-color:#22c55e; }}
.step-spin {{ border-left-color:#f59e0b; }}
.skill-tag {{
    display:inline-block; background:#1e3a5f; color:#93c5fd;
    border-radius:6px; padding:3px 10px; margin:3px; font-size:13px;
}}
.skill-tag-soft {{
    background:#2d1b4e; color:#c4b5fd;
}}
.ai-box {{
    background:#0d1f12; border:1px solid #16a34a;
    border-radius:14px; padding:20px 24px; margin:10px 0;
    color:#dcfce7; line-height:1.7; font-size:15px;
}}
.ai-box h4 {{ color:#4ade80; margin-bottom:10px; }}
.groq-badge {{
    display:inline-block; background:#1a1f2e;
    border:1px solid #6366f1; border-radius:8px;
    padding:4px 12px; font-size:12px; color:#a5b4fc;
    margin-bottom:12px;
}}
.api-warning {{
    background:#1c1008; border:1px solid #f59e0b;
    border-radius:10px; padding:14px 18px;
    color:#fde68a; font-size:14px;
}}
</style>
""", unsafe_allow_html=True)

# ── Backend imports ──────────────────────────────────────────────────
from resume_processing import analyze_resume
from job_match import calculate_match_score
from database import save_to_db, fetch_past_resumes
from linkedin_analyzer import fetch_linkedin_profile, analyze_linkedin_content

# ════════════════════════════════════════════════════════════════════
# AI HELPER  (Groq — free, unlimited, Llama 3.3 70B)
# ════════════════════════════════════════════════════════════════════
def get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)

def ai_analyze_job_match(api_key, resume_text_skills, job_description, ats_score, match_score):
    """
    Uses Groq (Llama 3.3-70B) to deeply analyze the resume vs job description
    and return a detailed match report + recommendations.
    """
    try:
        client = get_groq_client(api_key)
        prompt = f"""
You are an expert ATS and career coach AI. Analyze the following resume skills against a job description.

RESUME SKILLS DETECTED:
{resume_text_skills}

JOB DESCRIPTION:
{job_description}

CURRENT SCORES:
- ATS Score: {ats_score}%
- Keyword Match Score: {match_score}%

Your task:
1. Give a DETAILED JOB MATCH ANALYSIS — what skills match, what's missing, how strong the fit is.
2. Give a MATCH PERCENTAGE VERDICT with reasoning (be honest).
3. List TOP 5 MISSING SKILLS or keywords the candidate should add to their resume for this role.
4. Give 5 SPECIFIC, ACTIONABLE RECOMMENDATIONS to improve the resume for this job.
5. Give an OVERALL VERDICT in 2-3 sentences.

Format your response clearly with these exact section headers:
### 🎯 Job Match Analysis
### 📊 Match Verdict
### ❌ Missing Skills / Keywords
### 💡 Actionable Recommendations
### ✅ Overall Verdict
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI analysis error: {e}"


def ai_general_recommendations(api_key, tech_skills, soft_skills, ats_score):
    """
    General resume recommendations when no job description is provided.
    """
    try:
        client = get_groq_client(api_key)
        prompt = f"""
You are an expert resume coach and ATS optimization specialist.

A candidate has the following resume profile:
- Technical Skills: {', '.join(tech_skills) if tech_skills else 'None detected'}
- Soft Skills: {', '.join(soft_skills) if soft_skills else 'None detected'}
- ATS Score: {ats_score}%

Give them:
1. An honest RESUME STRENGTH ASSESSMENT based on their skills and ATS score.
2. TOP 5 SKILLS they should add to become more hireable in tech/business roles.
3. 5 SPECIFIC RESUME IMPROVEMENT TIPS to boost their ATS score above 80%.
4. A CAREER PATH SUGGESTION based on their current skill set.

Format with these exact headers:
### 📋 Resume Strength Assessment
### ⬆️ Skills to Add
### 🔧 Resume Improvement Tips
### 🚀 Career Path Suggestion
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI analysis error: {e}"


def ai_linkedin_suggestions(api_key, score, keywords, profile_text):
    """AI-powered LinkedIn profile improvement suggestions."""
    try:
        client = get_groq_client(api_key)
        prompt = f"""
You are a LinkedIn optimization expert.

Profile details:
- Profile Strength Score: {score}/100
- Keywords detected: {', '.join(keywords) if keywords else 'None'}
- Profile content excerpt: {profile_text[:800] if profile_text else 'Not available'}

Give:
1. PROFILE SECTION ANALYSIS — which sections are strong/weak.
2. TOP 5 KEYWORDS to add for better recruiter visibility.
3. HEADLINE & SUMMARY suggestions.
4. 5 ACTIONABLE STEPS to reach All-Star profile status.

Format with these exact headers:
### 📋 Profile Section Analysis
### 🔑 Keywords to Add
### ✍️ Headline & Summary Tips
### ⭐ Steps to All-Star Status
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI analysis error: {e}"


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🤖 AI Analyzer")
st.sidebar.markdown("---")

selected_option = st.sidebar.radio(
    "Navigation",
    ["📄 Resume Analyzer", "🔗 LinkedIn Analyzer", "📜 History"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Groq API Key")
st.sidebar.markdown(
    '<div class="groq-badge">⚡ Free · Llama 3.3-70B · No limits</div>',
    unsafe_allow_html=True,
)
if "groq_api_key" not in st.session_state:
    # Load from Streamlit secrets if available, else empty
    st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY", "")

def _save_key():
    st.session_state.groq_api_key = st.session_state._groq_key_input

st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_xxxxxxxxxxxx",
    label_visibility="collapsed",
    key="_groq_key_input",
    on_change=_save_key,
    help="Paste once — saved for the whole session",
)

groq_api_key = st.session_state.groq_api_key
if groq_api_key:
    st.sidebar.success("✅ API key saved for this session!")
else:
    st.sidebar.markdown(
        '<div class="api-warning">⚠️ No API key — AI insights disabled.<br>'
        '<a href="https://console.groq.com" target="_blank" style="color:#fbbf24;">Get free key →</a></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("Upload a PDF resume · Add job description · Get AI-powered insights")


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — RESUME ANALYZER
# ════════════════════════════════════════════════════════════════════
if selected_option == "📄 Resume Analyzer":

    st.markdown("## 📄 AI Resume Analyzer")
    st.markdown("Get your **ATS Score**, **Skill Breakdown**, **Job Match**, and **AI-powered recommendations**.")
    st.markdown("---")

    # STEP 1 — Upload
    st.markdown("### Step 1 — Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Drop your PDF here or click to browse",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")

    st.markdown("")

    # STEP 2 — Job Description
    st.markdown("### Step 2 — Paste Job Description *(optional but recommended)*")
    job_description = st.text_area(
        "job_desc",
        height=150,
        placeholder="Paste the full job description here.\nAI will deeply analyze your fit for this specific role.\nLeave blank for a general resume review.",
        label_visibility="collapsed",
    )
    if job_description.strip():
        st.caption(f"📝 {len(job_description.split())} words · AI will analyze your match for this role.")
    else:
        st.caption("💡 Add a job description to unlock AI Job Match Analysis.")

    st.markdown("")

    # STEP 3 — Analyze button
    st.markdown("### Step 3 — Run Analysis")
    analyze_clicked = st.button(
        "🚀  Analyze Resume",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )
    if uploaded_file is None:
        st.caption("⬆️ Upload a resume first to enable the Analyze button.")

    # ── RESULTS ──────────────────────────────────────────────────────
    if analyze_clicked and uploaded_file:

        # Save file
        upload_dir = os.path.join(BACKEND_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("---")
        st.markdown("### ⚙️ Analyzing…")

        # Animation steps
        steps = [
            ("📂", "Reading PDF & extracting text…"),
            ("🧠", "Running NLP & identifying entities…"),
            ("🛠", "Detecting technical & soft skills…"),
            ("📊", "Calculating ATS compatibility score…"),
            ("💼", "Matching against job description…"),
            ("🤖", "Running AI deep analysis…" if groq_api_key else "Generating recommendations…"),
            ("💾", "Saving results to database…"),
        ]

        placeholders = []
        for icon, label in steps:
            ph = st.empty()
            ph.markdown(f'<div class="step-item">{icon} &nbsp;{label}</div>', unsafe_allow_html=True)
            placeholders.append((ph, label))

        # Run analysis while animating
        resume_data  = None
        match_score  = 0
        ai_result    = None

        for i, (ph, label) in enumerate(placeholders):
            ph.markdown(f'<div class="step-item step-spin">⏳ &nbsp;{label}</div>', unsafe_allow_html=True)

            if i == 0:
                time.sleep(0.5)
            elif i == 1:
                resume_data = analyze_resume(file_path)
                time.sleep(0.5)
            elif i == 2:
                time.sleep(0.7)
            elif i == 3:
                time.sleep(0.6)
            elif i == 4:
                tech_skills  = resume_data.get("technical_skills", [])
                soft_skills  = resume_data.get("soft_skills", [])
                all_skills   = tech_skills + soft_skills
                if job_description.strip():
                    match_score = calculate_match_score(job_description, all_skills)
                time.sleep(0.5)
            elif i == 5:
                # AI call
                if groq_api_key:
                    tech_skills = resume_data.get("technical_skills", [])
                    soft_skills = resume_data.get("soft_skills", [])
                    ats_score   = resume_data.get("ats_score", 0)
                    skills_str  = f"Technical: {', '.join(tech_skills)}\nSoft: {', '.join(soft_skills)}"
                    if job_description.strip():
                        ai_result = ai_analyze_job_match(
                            groq_api_key, skills_str, job_description, ats_score, match_score
                        )
                    else:
                        ai_result = ai_general_recommendations(
                            groq_api_key, tech_skills, soft_skills, ats_score
                        )
                time.sleep(0.4)
            elif i == 6:
                try:
                    name      = resume_data.get("name", "Unknown")
                    ats_score = resume_data.get("ats_score", 0)
                    education = resume_data.get("education", "Not Found")
                    experience= resume_data.get("experience", "Not Found")
                    save_to_db(name, ats_score, all_skills, education, experience, match_score)
                except Exception:
                    pass
                time.sleep(0.3)

            ph.markdown(f'<div class="step-item step-done">✅ &nbsp;{label}</div>', unsafe_allow_html=True)

        # ── Extract final values ──────────────────────────────────────
        name            = resume_data.get("name", "Unknown")
        ats_score       = resume_data.get("ats_score", 0)
        tech_skills     = resume_data.get("technical_skills", [])
        soft_skills     = resume_data.get("soft_skills", [])
        all_skills      = tech_skills + soft_skills
        recommendations = resume_data.get(
            "recommendations",
            "Improve keyword usage, formatting, and include all essential sections."
        )

        # ── RESULTS SECTION ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("## ✅ Results")

        # Score cards
        c1, c2, c3 = st.columns(3)
        c1.metric("👤 Candidate", name)
        c2.metric("🎯 ATS Score", f"{ats_score}%",
                  delta="Good ✅" if ats_score >= 60 else "Needs Work ⚠️")
        c3.metric("💼 Job Match", f"{match_score}%",
                  delta="Strong 🔥" if match_score >= 60 else (
                      "—" if not job_description.strip() else "Low 📉"))

        st.markdown("")

        # Progress bars
        ats_color = "#22c55e" if ats_score >= 70 else ("#f59e0b" if ats_score >= 40 else "#ef4444")
        st.markdown(f'<div style="margin-bottom:6px;font-weight:600;color:#e2e8f0;">ATS Compatibility — <span style="color:{ats_color}">{ats_score}%</span></div>', unsafe_allow_html=True)
        st.progress(ats_score / 100)

        if job_description.strip():
            mc = "#22c55e" if match_score >= 60 else ("#f59e0b" if match_score >= 30 else "#ef4444")
            st.markdown(f'<div style="margin:14px 0 6px;font-weight:600;color:#e2e8f0;">Job Match Score — <span style="color:{mc}">{match_score}%</span></div>', unsafe_allow_html=True)
            st.progress(match_score / 100)

        st.markdown("---")

        # Skills
        col_t, col_s = st.columns(2)
        with col_t:
            st.markdown("#### 🛠 Technical Skills")
            if tech_skills:
                st.markdown("".join(f'<span class="skill-tag">{s}</span>' for s in sorted(tech_skills)), unsafe_allow_html=True)
            else:
                st.info("No technical skills detected.")
        with col_s:
            st.markdown("#### 🤝 Soft Skills")
            if soft_skills:
                st.markdown("".join(f'<span class="skill-tag skill-tag-soft">{s}</span>' for s in sorted(soft_skills)), unsafe_allow_html=True)
            else:
                st.info("No soft skills detected.")

        st.markdown("---")

        # ── AI INSIGHTS ───────────────────────────────────────────────
        if ai_result:
            st.markdown("#### 🤖 AI-Powered Insights")
            st.markdown('<span class="groq-badge">⚡ Powered by Groq · Llama 3.3-70B</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-box">{ai_result}</div>', unsafe_allow_html=True)
        else:
            st.markdown("#### 💡 Recommendations")
            st.info(recommendations)
            if not groq_api_key:
                st.markdown(
                    '<div class="api-warning">🤖 <strong>Want AI-powered insights?</strong> Add your free Groq API key in the sidebar to get deep job match analysis and personalized recommendations.</div>',
                    unsafe_allow_html=True,
                )

        st.success("🎉 Analysis complete!")


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — LINKEDIN ANALYZER
# ════════════════════════════════════════════════════════════════════
elif selected_option == "🔗 LinkedIn Analyzer":
    st.markdown("## 🔗 LinkedIn Profile Analyzer")
    st.markdown("Get a **Profile Strength Score**, keyword insights, and **AI-powered improvement tips**.")
    st.markdown("---")

    st.markdown("### Step 1 — Enter LinkedIn URL")
    linkedin_url = st.text_input(
        "url",
        placeholder="https://www.linkedin.com/in/your-profile/",
        label_visibility="collapsed",
    )

    st.markdown("")
    st.markdown("### Step 2 — Run Analysis")
    linkedin_clicked = st.button(
        "🚀  Analyze LinkedIn Profile",
        disabled=(not linkedin_url.strip()),
        use_container_width=True,
    )

    if linkedin_clicked:
        if not linkedin_url.startswith("https://www.linkedin.com"):
            st.error("❌ Please enter a valid LinkedIn URL.")
        else:
            st.markdown("---")
            st.markdown("### ⚙️ Analyzing…")

            steps = [
                ("🌐", "Fetching LinkedIn profile…"),
                ("🔍", "Scanning keywords & skills…"),
                ("📈", "Scoring profile strength…"),
                ("🤖", "Running AI analysis…" if groq_api_key else "Generating suggestions…"),
            ]

            placeholders = []
            for icon, label in steps:
                ph = st.empty()
                ph.markdown(f'<div class="step-item">{icon} &nbsp;{label}</div>', unsafe_allow_html=True)
                placeholders.append((ph, label))

            profile_text = None
            score = keywords = suggestions = None
            ai_li_result = None

            for i, (ph, label) in enumerate(placeholders):
                ph.markdown(f'<div class="step-item step-spin">⏳ &nbsp;{label}</div>', unsafe_allow_html=True)
                if i == 0:
                    profile_text = fetch_linkedin_profile(linkedin_url)
                    time.sleep(0.8)
                elif i == 1:
                    time.sleep(0.7)
                elif i == 2:
                    score, keywords, suggestions = analyze_linkedin_content(profile_text)
                    time.sleep(0.6)
                elif i == 3:
                    if groq_api_key:
                        ai_li_result = ai_linkedin_suggestions(groq_api_key, score, keywords, profile_text)
                    time.sleep(0.5)
                ph.markdown(f'<div class="step-item step-done">✅ &nbsp;{label}</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("## ✅ Results")

            sc1, _ = st.columns([1, 2])
            sc1.metric("🔎 Profile Strength", f"{score} / 100",
                       delta="Strong 🔥" if score >= 70 else "Needs Work ⚠️")
            st.progress(score / 100)

            st.markdown("---")
            st.markdown("#### 🔑 Detected Keywords")
            if keywords:
                st.markdown("".join(f'<span class="skill-tag">{kw}</span>' for kw in keywords), unsafe_allow_html=True)
            else:
                st.info("No strong keywords found. Enrich your profile summary.")

            st.markdown("---")

            if ai_li_result:
                st.markdown("#### 🤖 AI-Powered Profile Suggestions")
                st.markdown('<span class="groq-badge">⚡ Powered by Groq · Llama 3.3-70B</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="ai-box">{ai_li_result}</div>', unsafe_allow_html=True)
            else:
                st.markdown("#### 📈 Suggestions")
                if suggestions:
                    for i, s in enumerate(suggestions, 1):
                        st.markdown(f"**{i}.** ✅ {s}")
                else:
                    st.success("Your profile looks great!")
                if not groq_api_key:
                    st.markdown(
                        '<div class="api-warning">🤖 Add your free Groq API key in the sidebar for AI-powered profile suggestions.</div>',
                        unsafe_allow_html=True,
                    )


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ════════════════════════════════════════════════════════════════════
elif selected_option == "📜 History":
    st.markdown("## 📜 Past Resume Analyses")
    st.markdown("---")
    try:
        past_resumes = fetch_past_resumes()
        if past_resumes:
            st.markdown(f"**{len(past_resumes)} record(s) found.**")
            for res in past_resumes:
                with st.expander(f"🔹 {res[0]}  —  ATS: {res[1]}%  |  Match: {res[3]}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ATS Score", f"{res[1]}%")
                    c2.metric("Job Match", f"{res[3]}%")
                    c3.metric("Candidate", res[0])
                    if len(res) > 2: st.markdown(f"**Skills:** {res[2]}")
                    if len(res) > 4: st.markdown(f"**Education:** {res[4]}")
        else:
            st.info("No past analyses found. Upload a resume to get started!")
    except Exception as db_err:
        st.error(f"❌ Could not load history: {db_err}")
        st.caption("Make sure MySQL is running and database.py is configured.")


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:13px;'>"
    "🤖 AI Resume & LinkedIn Analyzer &nbsp;|&nbsp; Powered by Groq · Llama 3.3-70B &nbsp;|&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    pass

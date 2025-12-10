import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Core logic ----------

def preprocess_text(text: str) -> str:
    """
    Lowercase, remove non-alphanumeric characters, and strip extra spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = text.strip()
    return text


def compute_similarity(job_description: str, resume_text: str) -> float:
    """
    Takes raw job description and resume text, cleans them,
    and returns cosine similarity score between 0 and 1.
    """
    clean_jd = preprocess_text(job_description)
    clean_resume = preprocess_text(resume_text)

    documents = [clean_jd, clean_resume]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity_matrix[0][0]
    return score


def analyze_skills(
    job_description: str,
    resume_text: str,
    skills_to_check: list[str],
) -> tuple[list[str], list[str]]:
    """
    Given a list of skills, returns (matched_skills, missing_skills).

    Uses the job description to filter skills:
    - Only skills that appear in the JD (by words) are considered.
    """
    clean_jd = preprocess_text(job_description)
    clean_resume = preprocess_text(resume_text)

    jd_tokens = set(clean_jd.split())
    resume_tokens = set(clean_resume.split())

    matched: list[str] = []
    missing: list[str] = []

    for skill in skills_to_check:
        # Work in lowercase, but keep original skill string for display
        words = skill.lower().split()

        # Only consider skills that JD actually mentions
        if not all(word in jd_tokens for word in words):
            continue

        in_resume = all(word in resume_tokens for word in words)

        if in_resume:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


# ---------- Streamlit UI ----------

st.title("AI-Powered ATS Resume Matcher")
st.write(
    "Compare a job description and a resume, get a match score and a basic skill gap analysis."
)

st.subheader("Step 1: Paste Job Description")
job_description_input = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here...",
)

st.subheader("Step 2: Paste Resume")
resume_input = st.text_area(
    "Resume Text",
    height=200,
    placeholder="Paste your resume (or a summary) here...",
)

st.subheader("Step 3: Configure Expected Skills (optional)")

default_skills = [
    "python",
    "machine learning",
    "deep learning",
    "data preprocessing",
    "model evaluation",
    "pandas",
    "numpy",
    "scikit learn",
    "fastapi",
    "flask",
    "deployment",
]

skills_text = st.text_area(
    "Expected skills (comma separated)",
    value=", ".join(default_skills),
    height=100,
)

# Convert comma-separated string to list of skills
expected_skills = [s.strip() for s in skills_text.split(",") if s.strip()]

if st.button("Analyze Match"):
    if not job_description_input.strip() or not resume_input.strip():
        st.error("Please provide both Job Description and Resume text.")
    else:
        # 1) Similarity score
        similarity = compute_similarity(job_description_input, resume_input)
        match_percentage = round(similarity * 100, 2)

        st.markdown("### Results")
        st.write(f"**Match Percentage:** {match_percentage}%")

        # 2) Skill analysis
        matched_skills, missing_skills = analyze_skills(
            job_description_input,
            resume_input,
            expected_skills,
        )

        st.markdown("### Matched Skills")
        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.write("No expected skills found in resume.")

        st.markdown("### Missing Skills")
        if missing_skills:
            st.warning(", ".join(missing_skills))
        else:
            st.write("No skills missing from the expected list (based on JD).")

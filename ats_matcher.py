import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



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
    similarity_score = similarity_matrix[0][0]
    return similarity_score



def analyze_skills(
    
    resume_text: str,
    expected_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Given a list of expected skills, returns (matched_skills, missing_skills).
    """
    clean_resume = preprocess_text(resume_text)
    resume_tokens = set(clean_resume.split())

    matched_list = []
    missing_list = []

    for skill in expected_skills:
        skill_words = skill.split()
        in_resume = all(word in resume_tokens for word in skill_words)

        if in_resume:
            matched_list.append(skill)
        else:
            missing_list.append(skill)

    return matched_list, missing_list



def main():
    print("=== ATS Resume Keyword Matcher (File Version) ===\n")

    
    with open("job_description.txt", "r", encoding="utf-8") as f:
        job_description = f.read()

    
    with open("resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    
    expected_skills = [
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

  
    score = compute_similarity(job_description, resume_text)
    match_percentage = round(score * 100, 2)

    print("Job Description (from job_description.txt):")
    print(job_description)

    print("\nResume (from resume.txt):")
    print(resume_text)

    print("\n=== RESULTS ===")
    print(f"Match Percentage: {match_percentage}%")

    
    matched_skills, missing_skills = analyze_skills(
         resume_text, expected_skills
    )

    print("\nMatched Skills:")
    if matched_skills:
        print(", ".join(matched_skills))
    else:
        print("None")

    print("\nMissing Skills:")
    if missing_skills:
        print(", ".join(missing_skills))
    else:
        print("None")



if __name__ == "__main__":
    main()

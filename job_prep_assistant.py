# app/main.py

import os
import re
from typing import Tuple, List
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import google.generativeai as genai

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# 1) Configure Gemini API (set your API key as env var: GEMINI_API_KEY)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# 2) Embedding model
EMBEDDING_MODEL = "models/embedding-001"  # Gemini text embedding

# 3) Weights for final score
WEIGHTS = {
    "skills": 0.50,
    "title":  0.20,
    "education": 0.10,
    "experience": 0.20
}

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def simple_keyword_overlap(source: str, target: str, stopwords: List[str]) -> float:
    tokens = {w.lower().strip(".,!?\"'()") for w in re.findall(r"\b\w+\b", source)}
    tokens -= set(stopwords)
    words_in_target = {w.lower() for w in re.findall(r"\b\w+\b", target)}
    return (len(tokens & words_in_target) / len(tokens)) if tokens else 0.0

def detect_junior_senior_level(text: str) -> int:
    m = re.search(r"(\d{1,2})(?:\+)?\s+years?", text.lower())
    return min(int(m.group(1)), 10) if m else 0

def simple_education_score(text: str) -> float:
    degrees = ["phd", "doctorate", "master", "bachelor", "bs", "ba"]
    text_l = text.lower()
    for deg in degrees:
        if deg in text_l:
            return {"phd":1.0, "doctorate":1.0, "master":0.8}.get(deg, 0.6)
    return 0.3

def extract_title_match(resume: str, jd: str) -> float:
    titles = ["data scientist", "data analyst", "ml engineer",
              "machine learning engineer", "ai engineer"]
    resume_l, jd_l = resume.lower(), jd.lower()
    matches = sum(1 for t in titles if t in resume_l and t in jd_l)
    return min(matches / len(titles), 1.0)

# ─── GEMINI EMBEDDINGS ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    resp = genai.embeddings.create(
        model=EMBEDDING_MODEL,
        text=text
    )
    return resp["embeddings"][0]

def embed_and_compare(a: str, b: str) -> float:
    emb_a = np.array(get_embedding(a)).reshape(1, -1)
    emb_b = np.array(get_embedding(b)).reshape(1, -1)
    return float(cosine_similarity(emb_a, emb_b)[0,0])

# ─── MAIN SCORING FUNCTION ─────────────────────────────────────────────────────

def compute_ats_score(resume_text: str, jd_text: str,
                      stopwords: List[str]) -> Tuple[float, dict]:
    skill_score = simple_keyword_overlap(jd_text, resume_text, stopwords)
    title_score = extract_title_match(resume_text, jd_text)
    edu_score   = simple_education_score(resume_text)

    exp_years  = detect_junior_senior_level(resume_text)
    jd_years   = detect_junior_senior_level(jd_text)
    years_score = min(exp_years / max(jd_years, 1), 1.0)

    embed_score = embed_and_compare(resume_text, jd_text)
    experience_score = 0.5 * years_score + 0.5 * embed_score

    final = (
        WEIGHTS["skills"]     * skill_score +
        WEIGHTS["title"]      * title_score +
        WEIGHTS["education"]  * edu_score +
        WEIGHTS["experience"] * experience_score
    ) * 100

    breakdown = {
        "Skill Match (%)":      round(skill_score * 100, 2),
        "Title Fit (%)":        round(title_score * 100, 2),
        "Education Fit (%)":    round(edu_score * 100, 2),
        "Experience Fit (%)":   round(experience_score * 100, 2),
        "Semantic Similarity (%)": round(embed_score * 100, 2),
        "Overall ATS Score (%)":   round(final, 2)
    }
    return breakdown["Overall ATS Score (%)"], breakdown

# ─── STREAMLIT UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="ATS Resume Matcher", layout="wide")
st.title("ATS System")

with st.sidebar:
    st.header("Settings")
    stopword_input = st.text_area(
        "Stopwords (comma-separated)",
        value="and,the,of,to,for,with,experience,years"
    )

resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
jd_text      = st.text_area("Paste the Job Description (plain text)", height=200)

if st.button("Compute Match Score"):
    if not resume_file or not jd_text.strip():
        st.error("Please upload a resume PDF and paste the JD text.")
    else:
        with st.spinner("Analyzing…"):
            resume_text = extract_text_from_pdf(resume_file)
            stopwords   = [w.strip().lower() for w in stopword_input.split(",") if w.strip()]
            score, breakdown = compute_ats_score(resume_text, jd_text, stopwords)

        st.subheader(f"📝 Overall ATS Match Score: {score}%")
        st.table(breakdown)
        st.markdown(
            "----\n"
            "**How it works:**  \n"
            "- **Skills**: keyword overlap filtered by stopwords  \n"
            "- **Title**: role-title exact matches  \n"
            "- **Education**: degree-level scoring  \n"
            "- **Experience**: mix of years match + semantic embedding similarity  \n"
            "- **Embeddings**: Gemini `embed-text-bison-001` cosine similarity"
        )

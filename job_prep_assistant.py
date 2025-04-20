import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
import nltk
import re

# Download only if missing
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
embed_model = "models/embedding-001"
text_model = genai.GenerativeModel("models/gemini-2.5-pro-exp-03-25")

# --- Utilities ---
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    return set([word for word in words if word.isalpha() and word not in stop_words])

def get_keyword_match_score(jd_text, resume_text):
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    matched = jd_keywords.intersection(resume_keywords)
    return len(matched) / len(jd_keywords) if jd_keywords else 0

def get_experience_title_match_score(resume_text, jd_text):
    titles = ["Data Scientist", "Data Analyst", "Machine Learning Engineer", "AI Researcher"]
    score = 0
    for title in titles:
        if title.lower() in resume_text.lower() and title.lower() in jd_text.lower():
            score += 1
    return min(score / len(titles), 1)

def get_embedding(text):
    response = genai.embed_content(model=embed_model, content=text, task_type="RETRIEVAL_DOCUMENT")
    return response['embedding']

def extract_score_from_gemini(text):
    match = re.search(r"\b(\d{1,3})\b", text)
    if match:
        score = int(match.group(1))
        return min(score, 100)
    return 0

def get_gemini_smart_score(resume, jd):
    prompt = f"""
Review the following resume and job description. Assign an ATS compatibility score based on:

1. Matching skills
2. Tools and technologies used
3. Education level
4. Experience relevance

Output only a numeric match score (0 to 100), give a descriptive feedback based on the differences and what matches and suggestion for improvement.

Resume:
{resume}

Job Description:
{jd}
"""
    response = text_model.generate_content(prompt).text
    score = extract_score_from_gemini(response)
    return score, response

# --- UI ---
st.title(" ATS Match Score & Feedback")

uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type="pdf")
jd = st.text_area("Paste the Job Description")

resume_text = ""

if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.success("Resume text extracted!")
    except Exception as e:
        st.error(f"❌ PDF error: {e}")

if st.button(" Get ATS Match & Feedback"):
    if resume_text and jd:
        keyword_score = get_keyword_match_score(jd, resume_text)
        exp_score = get_experience_title_match_score(resume_text, jd)
        gemini_score, gemini_feedback = get_gemini_smart_score(resume_text, jd)
        final_score = round(0.4 * keyword_score * 100 + 0.4 * gemini_score + 0.2 * exp_score * 100, 2)

        st.markdown("### Final ATS Compatibility Score")
        st.metric("Score (0-100)", f"{final_score}%")
        
        st.markdown("### Gemini Feedback")
        st.write(gemini_feedback)

        st.markdown("### Component Scores")
        st.write(f"**Keyword Match Score:** {keyword_score*100:.2f}%")
        st.write(f"**Gemini Smart Score:** {gemini_score}%")
        st.write(f"**Title Match Score:** {exp_score*100:.2f}%")
    else:
        st.warning("Please upload a resume and paste the job description.")

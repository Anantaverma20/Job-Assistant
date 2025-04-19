import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configure Gemini securely
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
embed_model = "models/embedding-001"
text_model = genai.GenerativeModel("models/gemini-2.5-pro-exp-03-25")

# --- NLP & Scoring Utilities ---
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word.isalpha() and word not in stop_words]
    return set(keywords)

def get_keyword_match_score(jd_text, resume_text):
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    matched = jd_keywords.intersection(resume_keywords)
    return len(matched) / len(jd_keywords) if jd_keywords else 0

def get_weighted_score(cosine_score, jd_text, resume_text):
    keyword_score = get_keyword_match_score(jd_text, resume_text)
    return 0.6 * cosine_score + 0.4 * keyword_score

def get_embedding(text):
    response = genai.embed_content(model=embed_model, content=text, task_type="RETRIEVAL_DOCUMENT")
    return response['embedding']

def get_similarity_score(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

def get_feedback(resume, jd):
    prompt = f"""
Compare the following resume and job description:

Resume:
{resume}

Job Description:
{jd}

Provide:
- A match score (0–100)
- Missing skills
- Format improvement tips
- Summary of strengths and weaknesses
"""
    return text_model.generate_content(prompt).text

def find_jobs(resume):
    prompt = f"""
You are a job search assistant for data science roles. Based on the following resume:

{resume}

Suggest:
1. 3–5 job titles the candidate should apply for
2. What companies are likely hiring for these roles
3. 2 job search platforms to look at
4. Keywords the candidate should use in their search
"""
    return text_model.generate_content(prompt).text

# --- Streamlit UI ---
st.title("🧠 AI Job Prep Assistant")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF format)", type="pdf")
jd = st.text_area("📝 Paste Job Description")

resume_text = ""

if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.success("✅ Resume text extracted successfully.")
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")

if st.button("Analyze & Score"):
    if resume_text and jd:
        st.session_state.r_emb = get_embedding(resume_text)
        st.session_state.jd_emb = get_embedding(jd)
        score = get_similarity_score(st.session_state.r_emb, st.session_state.jd_emb)
        feedback = get_feedback(resume_text, jd)
        st.success(f"🔍 Match Score: {score*100:.2f}%")
        st.markdown("### 💬 Resume Feedback")
        st.write(feedback)
        missing_keywords = extract_keywords(jd) - extract_keywords(resume_text)
        st.markdown("### ❌ Missing Keywords:")
        st.write(", ".join(sorted(missing_keywords)))
    else:
        st.warning("⚠️ Please upload a resume and paste a job description.")

if st.button("📊 Show Score Breakdown"):
    if "r_emb" in st.session_state and "jd_emb" in st.session_state:
        cosine_score = get_similarity_score(st.session_state.r_emb, st.session_state.jd_emb)
        keyword_score = get_keyword_match_score(jd, resume_text)
        weighted_score = get_weighted_score(cosine_score, jd, resume_text)
        st.markdown("### 📈 Accuracy Dashboard")
        st.metric("🔗 Cosine Similarity", f"{cosine_score * 100:.2f}%")
        st.metric("🧩 Keyword Match", f"{keyword_score * 100:.2f}%")
        st.metric("🏆 Weighted ATS Score", f"{weighted_score * 100:.2f}%")

if st.button("🎯 Job Finder Agent"):
    if resume_text:
        jobs = find_jobs(resume_text)
        st.markdown("### 🔍 Job Suggestions")
        st.write(jobs)
    else:
        st.warning("⚠️ Upload your resume first.")

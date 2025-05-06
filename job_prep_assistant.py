import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import re
import numpy as np

# Configuration and Setup
st.set_page_config(page_title="ATS Resume Matcher", page_icon="📄")

# API Configuration
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Gemini API key not found. Please set up your API key in Streamlit secrets.")

# Models
embed_model = "models/embedding-001"
text_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Advanced Keyword Extraction and Matching
class ATSScanner:
    @staticmethod
    def extract_keywords(text, min_word_length=3):
        """
        Enhanced keyword extraction with more robust filtering
        """
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Advanced filtering
        keywords = [
            word.strip(".,!?()[]{}\"':;") 
            for word in words 
            if (word.isalpha() and 
                len(word) >= min_word_length and 
                word not in ENGLISH_STOP_WORDS)
        ]
        
        return list(set(keywords))

    @staticmethod
    def calculate_keyword_match(jd_text, resume_text):
        """
        Calculate keyword match with TF-IDF for more sophisticated matching
        """
        # Extract keywords
        jd_keywords = ATSScanner.extract_keywords(jd_text)
        resume_keywords = ATSScanner.extract_keywords(resume_text)
        
        # Create vectorizer
        vectorizer = TfidfVectorizer().fit([' '.join(jd_keywords), ' '.join(resume_keywords)])
        
        # Transform keywords
        jd_vector = vectorizer.transform([' '.join(jd_keywords)])
        resume_vector = vectorizer.transform([' '.join(resume_keywords)])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(jd_vector, resume_vector)[0][0]
        
        return similarity

    @staticmethod
    def calculate_semantic_similarity(jd_text, resume_text):
        """
        Use Gemini embeddings for semantic similarity
        """
        try:
            jd_embedding = genai.embed_content(
                model=embed_model, 
                content=jd_text, 
                task_type="RETRIEVAL_DOCUMENT"
            )['embedding']
            
            resume_embedding = genai.embed_content(
                model=embed_model, 
                content=resume_text, 
                task_type="RETRIEVAL_DOCUMENT"
            )['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(jd_embedding, resume_embedding) / (
                np.linalg.norm(jd_embedding) * np.linalg.norm(resume_embedding)
            )
            
            return similarity
        except Exception as e:
            st.error(f"Semantic similarity calculation error: {e}")
            return 0

    @staticmethod
    def generate_comprehensive_feedback(resume_text, jd_text):
        """
        Generate detailed feedback using Gemini
        """
        try:
            prompt = f"""Provide a comprehensive ATS compatibility analysis:

1. Skill Matching: Identify matching and missing skills
2. Experience Relevance: Compare resume experience with job requirements
3. Keyword Optimization: Suggest keywords to improve
4. Overall Improvement Recommendations

Job Description:
{jd_text}

Resume:
{resume_text}

Output Format:
- Matching Skills: [List]
- Missing Skills: [List]
- Experience Match: [Percentage and Brief Analysis]
- Keyword Recommendations: [List]
- Overall Improvement Suggestions: [Concise Advice]
"""
            response = text_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Feedback generation error: {e}")
            return "Unable to generate detailed feedback."

# Streamlit App
def main():
    st.title("🔍 Advanced ATS Resume Matcher")
    
    # Sidebar for additional controls
    st.sidebar.header("ATS Matching Settings")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)", 
        type="pdf", 
        help="Upload your resume in PDF format"
    )
    
    # Job Description Input
    jd = st.text_area(
        "Paste Job Description", 
        height=200, 
        help="Copy and paste the full job description here"
    )
    
    # Matching Button
    if st.button(" Analyze Resume Match"):
        if uploaded_file and jd:
            # Extract Resume Text
            try:
                pdf_reader = PdfReader(uploaded_file)
                resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
                
                # Calculate Scores
                keyword_match = ATSScanner.calculate_keyword_match(jd, resume_text)
                semantic_match = ATSScanner.calculate_semantic_similarity(jd, resume_text)
                
                # Generate Comprehensive Feedback
                detailed_feedback = ATSScanner.generate_comprehensive_feedback(resume_text, jd)
                
                # Combine Scores (you can adjust weights)
                final_score = round((keyword_match + semantic_match) * 50, 2)
                
                # Display Results
                st.markdown("### ATS Compatibility Analysis")
                
                # Score Visualization
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Keyword Match", f"{keyword_match*100:.2f}%")
                with col2:
                    st.metric("Semantic Match", f"{semantic_match*100:.2f}%")
                with col3:
                    st.metric("Overall ATS Score", f"{final_score}%")
                
                # Detailed Feedback
                st.markdown("### Detailed Feedback")
                st.write(detailed_feedback)
                
                # Recommendations Visualization
                if final_score < 60:
                    st.warning("Low ATS compatibility. Consider major resume revisions.")
                elif final_score < 80:
                    st.info("Moderate ATS compatibility. Some improvements needed.")
                else:
                    st.success("High ATS compatibility. Great job!")
            
            except Exception as e:
                st.error(f"Error processing resume: {e}")
        else:
            st.warning("Please upload a resume and paste the job description.")

# Run the app
if __name__ == "__main__":
    main()

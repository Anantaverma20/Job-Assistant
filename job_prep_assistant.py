import streamlit as st
import google.generativeai as genai
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import spacy

# Load spaCy model for advanced NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install spaCy and en_core_web_sm model. Run:\n"
             "pip install spacy\n"
             "python -m spacy download en_core_web_sm")
    st.stop()

class AdvancedATSMatcher:
    def __init__(self, gemini_api_key=None):
        """
        Initialize ATS Matcher with advanced NLP and AI capabilities
        """
        # Configure Gemini API if key is provided
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.text_model = genai.GenerativeModel("gemini-pro")
                self.embed_model = "models/embedding-001"
            except Exception as e:
                st.warning(f"Gemini API configuration failed: {e}")
                self.text_model = None
        else:
            self.text_model = None

    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_entities(self, text):
        """
        Extract named entities and key information
        """
        doc = nlp(text)
        
        entities = {
            'skills': [],
            'organizations': [],
            'job_titles': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'WORK_OF_ART':
                entities['job_titles'].append(ent.text)
        
        return entities

    def advanced_skill_extraction(self, text):
        """
        Advanced skill extraction using NLP and custom keywords
        """
        # Predefined skill categories
        skill_categories = {
            'Technical Skills': [
                'python', 'r', 'sql', 'java', 'c++', 'javascript',
                'machine learning', 'deep learning', 'data science',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 
                'numpy', 'apache spark', 'hadoop', 'cloud computing'
            ],
            'Soft Skills': [
                'communication', 'leadership', 'teamwork', 
                'problem solving', 'critical thinking', 
                'adaptability', 'creativity'
            ],
            'Methodological Skills': [
                'agile', 'scrum', 'kanban', 'waterfall',
                'data analysis', 'statistical modeling',
                'experimental design', 'hypothesis testing'
            ]
        }
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract skills
        extracted_skills = {
            'Technical Skills': [],
            'Soft Skills': [],
            'Methodological Skills': []
        }
        
        # Check for predefined skills
        text_lower = text.lower()
        for category, skills in skill_categories.items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills[category].append(skill)
        
        # Extract potential skills from noun chunks
        additional_skills = [
            chunk.text.lower() 
            for chunk in doc.noun_chunks 
            if len(chunk.text.split()) <= 3
        ]
        
        # Remove duplicates and add to technical skills
        additional_skills = list(set(additional_skills))
        extracted_skills['Technical Skills'].extend(additional_skills)
        
        return extracted_skills

    def calculate_skill_match(self, resume_skills, jd_skills):
        """
        Calculate sophisticated skill match
        """
        # Flatten skill dictionaries
        resume_all_skills = sum(resume_skills.values(), [])
        jd_all_skills = sum(jd_skills.values(), [])
        
        # Calculate matches
        technical_match = len(
            set(resume_skills['Technical Skills']) & 
            set(jd_skills['Technical Skills'])
        ) / max(len(jd_skills['Technical Skills']), 1)
        
        soft_skills_match = len(
            set(resume_skills['Soft Skills']) & 
            set(jd_skills['Soft Skills'])
        ) / max(len(jd_skills['Soft Skills']), 1)
        
        methodological_match = len(
            set(resume_skills['Methodological Skills']) & 
            set(jd_skills['Methodological Skills'])
        ) / max(len(jd_skills['Methodological Skills']), 1)
        
        # Weighted skill match
        skill_match_score = (
            technical_match * 0.5 + 
            soft_skills_match * 0.3 + 
            methodological_match * 0.2
        )
        
        return {
            'overall_skill_match': skill_match_score * 100,
            'technical_match': technical_match * 100,
            'soft_skills_match': soft_skills_match * 100,
            'methodological_match': methodological_match * 100
        }

    def semantic_similarity(self, resume_text, jd_text):
        """
        Advanced semantic similarity using TF-IDF and cosine similarity
        """
        # Preprocess texts
        resume_processed = self.preprocess_text(resume_text)
        jd_processed = self.preprocess_text(jd_text)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
        
        # Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return cosine_sim * 100

    def ai_enhanced_evaluation(self, resume_text, jd_text):
        """
        AI-powered comprehensive evaluation
        """
        if not self.text_model:
            return "AI evaluation not available"
        
        try:
            prompt = f"""Perform a comprehensive evaluation of the resume against the job description:

Job Description:
{jd_text}

Resume:
{resume_text}

Provide a detailed assessment covering:
1. Skill Alignment
2. Experience Relevance
3. Potential Cultural Fit
4. Areas of Improvement
5. Overall Recommendation

Format the response as a structured professional assessment."""
            
            response = self.text_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI evaluation error: {e}"

    def comprehensive_ats_score(self, resume_text, jd_text):
        """
        Comprehensive ATS scoring mechanism
        """
        # Extract skills
        resume_skills = self.advanced_skill_extraction(resume_text)
        jd_skills = self.advanced_skill_extraction(jd_text)
        
        # Skill Match
        skill_match = self.calculate_skill_match(resume_skills, jd_skills)
        
        # Semantic Similarity
        semantic_sim = self.semantic_similarity(resume_text, jd_text)
        
        # Experience Evaluation (basic implementation)
        experience_keywords = [
            'internship', 'project', 'developed', 'implemented', 
            'managed', 'led', 'created', 'designed'
        ]
        experience_score = sum(
            resume_text.lower().count(keyword) for keyword in experience_keywords
        ) / len(experience_keywords)
        
        # Weighted Scoring
        overall_score = (
            skill_match['overall_skill_match'] * 0.4 +
            semantic_sim * 0.3 +
            experience_score * 30
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'skill_match': skill_match,
            'semantic_similarity': round(semantic_sim, 2),
            'experience_score': round(experience_score * 100, 2)
        }

def main():
    st.set_page_config(page_title="Advanced ATS Resume Matcher", page_icon="🔍")
    
    # Title and Description
    st.title("🔍 Professional ATS Resume Matcher")
    st.markdown("""
    Advanced Applicant Tracking System (ATS) Analysis Tool
    - Comprehensive Resume Evaluation
    - AI-Powered Insights
    - Detailed Skill Matching
    """)
    
    # Gemini API Key Input (Optional)
    gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
    
    # Initialize ATS Matcher
    ats_matcher = AdvancedATSMatcher(gemini_api_key)
    
    # Resume Upload
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
    # Job Description Input
    jd_text = st.text_area("Paste Detailed Job Description", height=200)
    
    # Analyze Button
    if st.button("Analyze Resume"):
        # Input Validation
        if not uploaded_resume:
            st.warning("Please upload a resume PDF.")
            return
        
        if not jd_text:
            st.warning("Please paste the job description.")
            return
        
        try:
            # Extract Resume Text
            pdf_reader = PdfReader(uploaded_resume)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            
            # Perform Comprehensive Analysis
            ats_score = ats_matcher.comprehensive_ats_score(resume_text, jd_text)
            
            # Display Scoring
            st.markdown("### 📊 Comprehensive ATS Evaluation")
            
            # Score Columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Match", f"{ats_score['overall_score']}%")
            with col2:
                st.metric("Skill Alignment", f"{ats_score['skill_match']['overall_skill_match']:.2f}%")
            with col3:
                st.metric("Semantic Similarity", f"{ats_score['semantic_similarity']}%")
            
            # Detailed Skill Breakdown
            st.markdown("### 🔬 Skill Matching Breakdown")
            skill_col1, skill_col2, skill_col3 = st.columns(3)
            with skill_col1:
                st.metric("Technical Skills", f"{ats_score['skill_match']['technical_match']:.2f}%")
            with skill_col2:
                st.metric("Soft Skills", f"{ats_score['skill_match']['soft_skills_match']:.2f}%")
            with skill_col3:
                st.metric("Methodological Skills", f"{ats_score['skill_match']['methodological_match']:.2f}%")
            
            # Advanced AI Evaluation
            if ats_matcher.text_model:
                st.markdown("### 🤖 AI-Powered Insights")
                ai_evaluation = ats_matcher.ai_enhanced_evaluation(resume_text, jd_text)
                st.write(ai_evaluation)
            
            # Visualization of Match
            if ats_score['overall_score'] < 50:
                st.error("🚨 Low Compatibility: Significant Resume Revision Needed")
            elif ats_score['overall_score'] < 70:
                st.warning("⚠️ Moderate Compatibility: Some Improvements Required")
            else:
                st.success("✅ High Compatibility: Strong Candidate Profile")
        
        except Exception as e:
            st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()

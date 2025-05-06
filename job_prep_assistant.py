import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedATSMatcher:
    @staticmethod
    def preprocess_text(text):
        """
        Preprocess text for matching
        - Convert to lowercase
        - Remove special characters
        - Split into words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text

    @staticmethod
    def extract_skill_keywords(text):
        """
        Extract specific skill-related keywords
        """
        # List of key skills and technical terms
        skill_keywords = [
            # Technical Skills
            'python', 'machine learning', 'deep learning', 'data science', 
            'statistical modeling', 'nlp', 'data mining', 'sql', 
            'data visualization', 'pandas', 'numpy', 'scikit-learn', 
            'tensorflow', 'keras', 'pytorch', 'llms', 'ai',
            
            # Methodological Skills
            'causal inference', 'metrics design', 'a/b testing', 
            'statistical reasoning', 'predictive modeling',
            
            # Business Skills
            'business acumen', 'stakeholder management', 
            'project management', 'financial performance',
            
            # Engineering Skills
            'code testing', 'infrastructure design', 
            'software engineering', 'agile', 'ci/cd',
            
            # Soft Skills
            'leadership', 'collaboration', 'communication', 
            'problem solving', 'critical thinking'
        ]
        
        # Find matches in the text
        found_skills = []
        for skill in skill_keywords:
            if skill in text.lower():
                found_skills.append(skill)
        
        return found_skills

    @staticmethod
    def calculate_skill_match_score(resume_text, jd_text):
        """
        Calculate skill match score
        """
        # Preprocess texts
        resume_processed = AdvancedATSMatcher.preprocess_text(resume_text)
        jd_processed = AdvancedATSMatcher.preprocess_text(jd_text)
        
        # Extract skills
        resume_skills = AdvancedATSMatcher.extract_skill_keywords(resume_processed)
        jd_skills = AdvancedATSMatcher.extract_skill_keywords(jd_processed)
        
        # Calculate skill overlap
        matching_skills = set(resume_skills) & set(jd_skills)
        
        # Scoring logic
        skill_match_score = len(matching_skills) / len(set(jd_skills)) if jd_skills else 0
        
        return {
            'match_percentage': skill_match_score * 100,
            'matching_skills': list(matching_skills),
            'total_jd_skills': len(jd_skills),
            'matched_skills': len(matching_skills)
        }

    @staticmethod
    def calculate_experience_relevance(resume_text, jd_text):
        """
        Calculate experience relevance score
        """
        # Key experience indicators
        experience_keywords = [
            'internship', 'project', 'developed', 'implemented', 
            'optimized', 'improved', 'created', 'analyzed'
        ]
        
        # Count relevant experience indicators
        resume_experience_indicators = sum(
            resume_text.lower().count(keyword) for keyword in experience_keywords
        )
        jd_experience_indicators = sum(
            jd_text.lower().count(keyword) for keyword in experience_keywords
        )
        
        # Normalize and score
        try:
            experience_relevance = min(
                resume_experience_indicators / jd_experience_indicators, 
                1.0
            )
        except ZeroDivisionError:
            experience_relevance = 0.5
        
        return experience_relevance * 100

    @staticmethod
    def calculate_comprehensive_ats_score(resume_text, jd_text):
        """
        Calculate comprehensive ATS score
        """
        # Skill Match Score
        skill_match = AdvancedATSMatcher.calculate_skill_match_score(resume_text, jd_text)
        
        # Experience Relevance
        experience_score = AdvancedATSMatcher.calculate_experience_relevance(resume_text, jd_text)
        
        # TF-IDF Cosine Similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Weighted Scoring
        # Adjust weights based on importance
        skill_weight = 0.4
        experience_weight = 0.3
        cosine_weight = 0.3
        
        comprehensive_score = (
            skill_match['match_percentage'] * skill_weight +
            experience_score * experience_weight +
            cosine_sim * 100 * cosine_weight
        )
        
        return {
            'overall_score': round(comprehensive_score, 2),
            'skill_match': skill_match,
            'experience_score': round(experience_score, 2),
            'cosine_similarity': round(cosine_sim * 100, 2)
        }

# Example usage and testing
def analyze_resume_match(resume_text, jd_text):
    """
    Comprehensive ATS matching analysis
    """
    match_result = AdvancedATSMatcher.calculate_comprehensive_ats_score(resume_text, jd_text)
    
    # Generate detailed feedback
    feedback = f"""
ATS Compatibility Analysis:
--------------------------
Overall Match Score: {match_result['overall_score']}%

Detailed Breakdown:
1. Skill Match: {match_result['skill_match']['match_percentage']}%
   - Total Skills in Job Description: {match_result['skill_match']['total_jd_skills']}
   - Matched Skills: {match_result['skill_match']['matched_skills']}
   - Matching Skills: {', '.join(match_result['skill_match']['matching_skills'])}

2. Experience Relevance: {match_result['experience_score']}%

3. Semantic Similarity: {match_result['cosine_similarity']}%

Recommendations:
- Focus on highlighting skills that match the job description
- Quantify achievements in your resume
- Align your experience with the specific requirements
"""
    
    return match_result, feedback


# Streamlit App
def main():
    st.title("Advanced ATS Resume Matcher")
    
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

import os
import json
import re
import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

class JobAssistant:
    def __init__(self, config_path: str = 'job_config.json'):
        """
        Initialize the Job Assistant with configuration settings.
        
        :param config_path: Path to the configuration JSON file
        """
        self.config = self.load_config(config_path)
        self.job_applications = []
        self.skills_inventory = self.config.get('skills', [])
        self.application_log_path = self.config.get('application_log_path', 'job_applications.json')
        
        # Load existing applications
        self.load_job_applications()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        :param config_path: Path to the configuration file
        :return: Dictionary of configuration settings
        """
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Creating default configuration.")
            return {
                'skills': [],
                'application_log_path': 'job_applications.json',
                'resume_template_path': 'resume_template.txt'
            }
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {config_path}. Using default configuration.")
            return {}

    def add_skill(self, skill: str) -> None:
        """
        Add a new skill to the skills inventory.
        
        :param skill: Skill to be added
        """
        if skill.lower() not in [s.lower() for s in self.skills_inventory]:
            self.skills_inventory.append(skill)
            self._save_skills_inventory()
        
    def remove_skill(self, skill: str) -> None:
        """
        Remove a skill from the skills inventory.
        
        :param skill: Skill to be removed
        """
        original_length = len(self.skills_inventory)
        self.skills_inventory = [s for s in self.skills_inventory if s.lower() != skill.lower()]
        
        if len(self.skills_inventory) < original_length:
            self._save_skills_inventory()

    def _save_skills_inventory(self) -> None:
        """
        Save skills inventory to the config file.
        """
        try:
            config = self.load_config('job_config.json')
            config['skills'] = self.skills_inventory
            with open('job_config.json', 'w') as config_file:
                json.dump(config, config_file, indent=2)
        except IOError:
            st.error(f"Error saving skills inventory to job_config.json")

    def record_job_application(self, job_details: Dict[str, Any]) -> None:
        """
        Record details of a job application.
        
        :param job_details: Dictionary containing job application details
        """
        # Validate required fields
        required_fields = ['company', 'position', 'application_date', 'status']
        for field in required_fields:
            if field not in job_details:
                raise ValueError(f"Missing required field: {field}")
        
        # Add unique identifier
        job_details['id'] = len(self.job_applications) + 1
        
        self.job_applications.append(job_details)
        self._save_job_applications()

    def _save_job_applications(self) -> None:
        """
        Save job applications to a JSON file.
        """
        try:
            with open(self.application_log_path, 'w') as log_file:
                json.dump(self.job_applications, log_file, indent=2)
        except IOError:
            st.error(f"Error saving job applications to {self.application_log_path}")

    def load_job_applications(self) -> List[Dict[str, Any]]:
        """
        Load job applications from the log file.
        
        :return: List of job applications
        """
        try:
            with open(self.application_log_path, 'r') as log_file:
                self.job_applications = json.load(log_file)
            return self.job_applications
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            st.error(f"Error reading job applications from {self.application_log_path}")
            return []

    def generate_resume_customization_notes(self, job_description: str) -> Dict[str, List[str]]:
        """
        Generate resume customization suggestions based on job description.
        
        :param job_description: Text of the job description
        :return: Dictionary of customization suggestions
        """
        # Basic NLP-like processing (simplified)
        job_description = job_description.lower()
        
        customization_notes = {
            'highlighted_skills': [],
            'potential_keywords': [],
            'recommended_adjustments': []
        }
        
        # Extract potential skills and keywords
        for skill in self.skills_inventory:
            if skill.lower() in job_description:
                customization_notes['highlighted_skills'].append(skill)
        
        # Extract potential keywords (simplified)
        potential_keywords = re.findall(r'\b[a-z]{3,}\b', job_description)
        customization_notes['potential_keywords'] = list(set(potential_keywords))[:10]
        
        # Basic recommendations
        if len(customization_notes['highlighted_skills']) < 3:
            customization_notes['recommended_adjustments'].append(
                "Consider adding more relevant skills to your resume"
            )
        
        return customization_notes

    def update_application_status(self, application_id: int, new_status: str) -> None:
        """
        Update the status of a specific job application.
        
        :param application_id: Unique identifier of the job application
        :param new_status: New status of the application
        """
        for application in self.job_applications:
            if application['id'] == application_id:
                application['status'] = new_status
                self._save_job_applications()
                return
        
        st.error(f"No application found with ID {application_id}")

def main():
    # Initialize job assistant
    job_assistant = JobAssistant()

    # Streamlit app title
    st.title('Job Application Tracker')

    # Sidebar for navigation
    menu = st.sidebar.selectbox('Menu', 
        ['Job Applications', 'Skills Inventory', 'Resume Customization'])

    if menu == 'Job Applications':
        st.header('Job Applications')
        
        # Form to add new job application
        with st.form('job_application_form'):
            company = st.text_input('Company Name')
            position = st.text_input('Position')
            status_options = ['Applied', 'Interview', 'Offer', 'Rejected']
            status = st.selectbox('Application Status', status_options)
            submit = st.form_submit_button('Record Application')

            if submit:
                if company and position:
                    job_application = {
                        'company': company,
                        'position': position,
                        'application_date': datetime.datetime.now().isoformat(),
                        'status': status
                    }
                    job_assistant.record_job_application(job_application)
                    st.success('Job application recorded successfully!')
                else:
                    st.error('Please fill in all required fields.')

        # Display existing job applications
        if job_assistant.job_applications:
            df = pd.DataFrame(job_assistant.job_applications)
            st.dataframe(df)

            # Allow status update
            app_to_update = st.selectbox(
                'Select Application to Update Status', 
                [app['id'] for app in job_assistant.job_applications]
            )
            new_status = st.selectbox('New Status', 
                ['Applied', 'Interview', 'Offer', 'Rejected'])
            if st.button('Update Status'):
                job_assistant.update_application_status(app_to_update, new_status)
                st.success('Application status updated!')
                st.experimental_rerun()

    elif menu == 'Skills Inventory':
        st.header('Skills Inventory')
        
        # Add new skill
        new_skill = st.text_input('Add a New Skill')
        if st.button('Add Skill'):
            if new_skill:
                job_assistant.add_skill(new_skill)
                st.success(f'Skill {new_skill} added!')

        # Display and remove existing skills
        if job_assistant.skills_inventory:
            st.subheader('Your Skills')
            skill_to_remove = st.selectbox(
                'Select Skill to Remove', 
                job_assistant.skills_inventory
            )
            if st.button('Remove Skill'):
                job_assistant.remove_skill(skill_to_remove)
                st.success(f'Skill {skill_to_remove} removed!')
                st.experimental_rerun()

            # Display skills list
            st.write('Current Skills:')
            for skill in job_assistant.skills_inventory:
                st.write(f"- {skill}")

    elif menu == 'Resume Customization':
        st.header('Resume Customization')
        
        # Job description input
        job_description = st.text_area('Paste Job Description')
        
        if st.button('Analyze Job Description'):
            if job_description:
                customization_notes = job_assistant.generate_resume_customization_notes(job_description)
                
                st.subheader('Resume Customization Suggestions')
                
                # Highlighted Skills
                st.write('**Highlighted Skills:**')
                if customization_notes['highlighted_skills']:
                    for skill in customization_notes['highlighted_skills']:
                        st.write(f"- {skill}")
                else:
                    st.write("No direct skill matches found.")
                
                # Potential Keywords
                st.write('**Potential Keywords:**')
                if customization_notes['potential_keywords']:
                    st.write(', '.join(customization_notes['potential_keywords']))
                else:
                    st.write("No potential keywords identified.")
                
                # Recommended Adjustments
                st.write('**Recommended Adjustments:**')
                if customization_notes['recommended_adjustments']:
                    for adjustment in customization_notes['recommended_adjustments']:
                        st.write(f"- {adjustment}")
                else:
                    st.write("No specific adjustments recommended.")
            else:
                st.error('Please enter a job description to analyze.')

if __name__ == '__main__':
    # Ensure config and log files exist
    if not os.path.exists('job_config.json'):
        with open('job_config.json', 'w') as f:
            json.dump({'skills': [], 'application_log_path': 'job_applications.json'}, f)
    
    if not os.path.exists('job_applications.json'):
        with open('job_applications.json', 'w') as f:
            json.dump([], f)
    
    main()

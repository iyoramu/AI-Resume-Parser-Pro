import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import pickle
from pathlib import Path
import json
from typing import Dict, List, Optional, Any, Union
import re

class MLModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skill_normalizer = self._load_skill_normalizer()
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.job_vectorizer = TfidfVectorizer(stop_words='english')
        
    def _load_skill_normalizer(self) -> Dict[str, str]:
        """Load skill normalization mappings"""
        normalizer_file = Path(__file__).parent.parent / "data" / "skill_normalizer.pkl"
        try:
            with open(normalizer_file, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.PickleError):
            # Create default normalizer if file doesn't exist
            default_normalizer = {
                "python programming": "Python",
                "machine learning": "Machine Learning",
                "natural language processing": "Natural Language Processing",
                "deep learning": "Deep Learning",
                "data analysis": "Data Analysis"
            }
            return default_normalizer
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names to standard taxonomy"""
        normalized = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in self.skill_normalizer:
                normalized.append(self.skill_normalizer[skill_lower])
            else:
                # Use most similar existing skill
                similarities = [
                    (norm_skill, self._skill_similarity(skill_lower, norm_skill.lower()))
                    for norm_skill in set(self.skill_normalizer.values())
                ]
                if similarities:
                    best_match, best_score = max(similarities, key=lambda x: x[1])
                    if best_score > 0.7:  # Similarity threshold
                        normalized.append(best_match)
                    else:
                        normalized.append(skill.title())
                else:
                    normalized.append(skill.title())
        return sorted(list(set(normalized)))  # Remove duplicates
    
    def _skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills"""
        emb1 = self.sbert_model.encode(skill1, convert_to_tensor=True)
        emb2 = self.sbert_model.encode(skill2, convert_to_tensor=True)
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    
    def calculate_compatibility(
        self, 
        resume_data: Dict[str, Any], 
        job_description: Union[Dict[str, Any], str]
    ) -> Dict[str, float]:
        """Calculate compatibility score between resume and job description"""
        # Prepare text for comparison
        resume_text = self._prepare_resume_text(resume_data)
        job_text = self._prepare_job_text(job_description)
        
        # Calculate TF-IDF similarity
        try:
            tfidf_matrix = self.job_vectorizer.fit_transform([resume_text, job_text])
            tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            tfidf_sim = 0.0
        
        # Calculate semantic similarity with SBERT
        sbert_sim = self._calculate_sbert_similarity(resume_text, job_text)
        
        # Calculate skill match
        skill_match = self._skill_match_score(
            resume_data.get('skills', []), 
            job_description
        )
        
        # Combined score (weighted average)
        combined_score = 0.5 * sbert_sim + 0.3 * tfidf_sim + 0.2 * skill_match
        
        return {
            "overall_score": float(np.clip(combined_score, 0, 1)),
            "tfidf_similarity": float(np.clip(tfidf_sim, 0, 1)),
            "semantic_similarity": float(np.clip(sbert_sim, 0, 1)),
            "skill_match": float(np.clip(skill_match, 0, 1))
        }
    
    def _prepare_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume text for comparison"""
        sections = []
        
        if resume_data.get('education'):
            sections.append(" ".join(
                f"{edu.get('degree', '')} {edu.get('institution', '')}" 
                for edu in resume_data['education']
            ))
        
        if resume_data.get('experience'):
            sections.append(" ".join(
                f"{exp.get('position', '')} {exp.get('company', ')}" 
                for exp in resume_data['experience']
            ))
        
        if resume_data.get('skills'):
            sections.append(" ".join(resume_data['skills']))
            
        if resume_data.get('projects'):
            sections.append(" ".join(resume_data['projects']))
            
        return " ".join(sections)
    
    def _prepare_job_text(self, job_description: Union[Dict[str, Any], str]) -> str:
        """Prepare job description text for comparison"""
        if isinstance(job_description, dict):
            return " ".join([
                job_description.get('title', ''), 
                job_description.get('description', ''),
                " ".join(job_description.get('requirements', [])),
                " ".join(job_description.get('preferred_qualifications', []))
            ])
        return str(job_description)
    
    def _calculate_sbert_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using SBERT"""
        embedding1 = self.sbert_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sbert_model.encode(text2, convert_to_tensor=True)
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    
    def _skill_match_score(
        self, 
        resume_skills: List[str], 
        job_description: Union[Dict[str, Any], str]
    ) -> float:
        """Calculate skill match score"""
        if not resume_skills:
            return 0.0
            
        if isinstance(job_description, dict) and 'requirements' in job_description:
            job_skills = job_description['requirements']
        else:
            # Extract skills from raw job description text
            job_skills = self._extract_skills_from_text(str(job_description))
            
        if not job_skills:
            return 0.0
            
        # Normalize skills
        norm_resume_skills = set(self.normalize_skills(resume_skills))
        norm_job_skills = set(self.normalize_skills(job_skills))
        
        if not norm_job_skills:
            return 0.0
            
        # Calculate matching score
        intersection = norm_resume_skills.intersection(norm_job_skills)
        return len(intersection) / len(norm_job_skills)
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from job description text"""
        # First try to find explicit skill lists
        list_patterns = [
            r"requirements:\s*(.*?)(?=\n\n)",
            r"skills required:\s*(.*?)(?=\n\n)",
            r"technical skills:\s*(.*?)(?=\n\n)"
        ]
        
        for pattern in list_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                skills_section = match.group(1)
                # Split by commas, bullets, or newlines
                skills = re.split(r'[,•\n]', skills_section)
                return [skill.strip() for skill in skills if skill.strip()]
        
        # Fallback to keyword-based extraction
        skills = set()
        skill_keywords = [
            "knowledge of", "experience with", "proficient in", 
            "familiar with", "expertise in", "working knowledge of"
        ]
        
        for sent in re.split(r'[.!?]', text):
            sent_lower = sent.lower()
            for keyword in skill_keywords:
                if keyword in sent_lower:
                    # Extract the phrase after the keyword
                    parts = sent_lower.split(keyword)
                    if len(parts) > 1:
                        skill_phrase = parts[1].strip()
                        # Take the first few words as skill
                        skill = ' '.join(skill_phrase.split()[:3]).title()
                        skills.add(skill)
                        
        return list(skills)
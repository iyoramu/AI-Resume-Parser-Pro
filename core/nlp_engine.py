import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

class NlpEngine:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._initialize_matchers()
        self._load_patterns()
        
        # Add custom pipeline components
        Span.set_extension("score", default=1.0, force=True)
        
    def _initialize_matchers(self):
        """Initialize various matchers for entity extraction"""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Education patterns
        education_patterns = [
            [{"LOWER": {"IN": ["bsc", "b.sc", "bs", "b.s"]}}, {"LOWER": "."}, {"IS_ALPHA": True}],
            [{"LOWER": {"IN": ["btech", "b.tech"]}}, {"LOWER": "."}, {"IS_ALPHA": True}],
            [{"LOWER": {"IN": ["mtech", "m.tech"]}}, {"LOWER": "."}, {"IS_ALPHA": True}],
            [{"LOWER": "ms"}, {"LOWER": "in"}, {"IS_ALPHA": True}],
            [{"LOWER": "master"}, {"LOWER": "of"}, {"LOWER": "science"}],
            [{"LOWER": "bachelor"}, {"LOWER": "of"}, {"LOWER": "science"}],
            [{"LOWER": "ph"}, {"LOWER": "d"}, {"LOWER": "."}],
            [{"LOWER": "doctor"}, {"LOWER": "of"}, {"LOWER": "philosophy"}]
        ]
        self.matcher.add("EDUCATION", education_patterns)
        
        # Experience patterns
        experience_patterns = [
            [{"LOWER": {"IN": ["worked", "work", "experience"]}}, {"LOWER": "as"}, {"ENT_TYPE": "PERSON"}],
            [{"LOWER": {"IN": ["worked", "work", "experience"]}}, {"LOWER": "at"}, {"ENT_TYPE": "ORG"}],
            [{"LOWER": {"IN": ["worked", "work", "experience"]}}, {"LOWER": "in"}, {"ENT_TYPE": "ORG"}],
            [{"LOWER": "join"}, {"LOWER": "as"}, {"ENT_TYPE": "PERSON"}],
            [{"LOWER": "join"}, {"LOWER": "at"}, {"ENT_TYPE": "ORG"}]
        ]
        self.matcher.add("EXPERIENCE", experience_patterns)
        
    def _load_patterns(self):
        """Load patterns from external files"""
        data_dir = Path(__file__).parent.parent / "data"
        
        # Load skills
        skills_file = data_dir / "skills.json"
        with open(skills_file, 'r') as f:
            skills_data = json.load(f)
        self.skill_patterns = list(self.nlp.pipe(skills_data["skills"]))
        self.phrase_matcher.add("SKILLS", self.skill_patterns)
        
        # Load companies
        companies_file = data_dir / "companies.json"
        with open(companies_file, 'r') as f:
            companies_data = json.load(f)
        self.company_patterns = list(self.nlp.pipe(companies_data["companies"]))
        self.phrase_matcher.add("COMPANIES", self.company_patterns)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from resume text"""
        doc = self.nlp(text)
        
        # Run matchers
        matches = self.matcher(doc)
        phrase_matches = self.phrase_matcher(doc)
        
        entities = {
            "name": self._extract_name(doc),
            "contact": self._extract_contact_info(doc),
            "education": self._extract_education(doc, matches),
            "experience": self._extract_experience(doc, matches, phrase_matches),
            "skills": self._extract_skills(doc, phrase_matches),
            "certifications": self._extract_certifications(doc),
            "projects": self._extract_projects(doc)
        }
        
        return entities
    
    def _extract_name(self, doc) -> Optional[str]:
        """Extract candidate name from document"""
        # First look for PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if it's likely a name (has at least two words)
                if len(ent.text.split()) >= 2:
                    return ent.text
        
        # Fallback: look for title case patterns at the beginning of the document
        for sent in doc.sents:
            if sent.text == doc.text[:len(sent.text)]:  # First sentence
                for token in sent:
                    if token.is_title and token.text.isalpha():
                        name_parts = [token.text]
                        for next_token in token.head.children:
                            if next_token.is_title and next_token.text.isalpha():
                                name_parts.append(next_token.text)
                        if len(name_parts) >= 2:
                            return ' '.join(name_parts)
        return None
    
    def _extract_contact_info(self, doc) -> Dict[str, Optional[str]]:
        """Extract contact information (email, phone)"""
        contact = {"email": None, "phone": None}
        
        # Extract email
        email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_regex, doc.text)
        if emails:
            contact["email"] = emails[0]
            
        # Extract phone numbers
        phone_regex = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        phones = re.findall(phone_regex, doc.text)
        if phones:
            # Take the longest phone number found
            contact["phone"] = max(phones, key=len)
            
        return contact
    
    def _extract_education(self, doc, matches) -> List[Dict[str, str]]:
        """Extract education information"""
        education = []
        
        # Extract using NER
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(token.text.lower() in ["university", "college", "institute"] for token in ent):
                education.append({"institution": ent.text})
                
        # Extract using patterns
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "EDUCATION":
                span = doc[start:end]
                education.append({"degree": span.text})
                
        # Deduplicate
        unique_education = []
        seen = set()
        for edu in education:
            identifier = (edu.get('institution', ''), edu.get('degree', ''))
            if identifier not in seen:
                seen.add(identifier)
                unique_education.append(edu)
                
        return unique_education
    
    def _extract_experience(self, doc, matches, phrase_matches) -> List[Dict[str, Optional[str]]]:
        """Extract work experience"""
        experience = []
        current_company = None
        current_position = None
        current_duration = None
        
        # Extract companies using phrase matcher
        companies = []
        for match_id, start, end in phrase_matches:
            if self.nlp.vocab.strings[match_id] == "COMPANIES":
                companies.append(doc[start:end].text)
        
        # Extract positions and durations
        for sent in doc.sents:
            text = sent.text.lower()
            
            # Look for position indicators
            position_keywords = ["worked as", "position of", "role of", "as a", "position:"]
            for keyword in position_keywords:
                if keyword in text:
                    parts = text.split(keyword)
                    if len(parts) > 1:
                        current_position = parts[1].split('.')[0].strip().title()
            
            # Look for duration indicators
            duration_pattern = r"\((.*?)\)"  # Dates often in parentheses
            durations = re.findall(duration_pattern, sent.text)
            if durations:
                current_duration = durations[0]
                
            # When we find a company name, create a new experience entry
            for company in companies:
                if company.lower() in text.lower():
                    if current_company and current_company != company:
                        # Add previous experience before starting new one
                        if current_position or current_duration:
                            experience.append({
                                "company": current_company,
                                "position": current_position,
                                "duration": current_duration
                            })
                    
                    current_company = company
                    current_position = None
                    current_duration = None
                    break
        
        # Add the last experience if exists
        if current_company:
            experience.append({
                "company": current_company,
                "position": current_position,
                "duration": current_duration
            })
                    
        return experience
    
    def _extract_skills(self, doc, phrase_matches) -> List[str]:
        """Extract skills from document"""
        skills = set()
        
        # Extract using phrase matcher
        for match_id, start, end in phrase_matches:
            if self.nlp.vocab.strings[match_id] == "SKILLS":
                skills.add(doc[start:end].text.lower())
                
        # Extract from noun chunks that contain technical terms
        tech_terms = {"programming", "development", "engineering", "framework", 
                     "language", "technology", "tool", "software", "system"}
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(term in chunk_text for term in tech_terms):
                # Clean and add the chunk
                clean_chunk = re.sub(r'[^a-zA-Z0-9\s]', '', chunk_text).strip()
                if 1 <= len(clean_chunk.split()) <= 3:
                    skills.add(clean_chunk)
                    
        return sorted(list(skills))
    
    def _extract_certifications(self, doc) -> List[str]:
        """Extract certifications"""
        certs = set()
        cert_keywords = ["certified", "certification", "license", "licensed", "certificate"]
        
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(keyword in sent_lower for keyword in cert_keywords):
                # Clean and extract the certification name
                clean_sent = re.sub(r'[^a-zA-Z0-9\s]', ' ', sent.text).strip()
                certs.add(clean_sent)
                
        return sorted(list(certs))
    
    def _extract_projects(self, doc) -> List[str]:
        """Extract projects"""
        projects = set()
        project_keywords = ["project", "developed", "created", "built", "designed", "implemented"]
        
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(keyword in sent_lower for keyword in project_keywords):
                # Clean and extract the project description
                clean_sent = re.sub(r'[^a-zA-Z0-9\s]', ' ', sent.text).strip()
                projects.add(clean_sent)
                
        return sorted(list(projects))
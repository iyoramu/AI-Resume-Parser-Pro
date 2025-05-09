import pytest
from core.file_processor import FileProcessor
from core.nlp_engine import NlpEngine
from core.ml_models import MLModels
import os
from pathlib import Path

@pytest.fixture
def file_processor():
    return FileProcessor()

@pytest.fixture
def nlp_engine():
    return NlpEngine()

@pytest.fixture
def ml_models():
    return MLModels()

def test_pdf_processing(file_processor):
    test_file = Path(__file__).parent / "sample_resumes" / "sample.pdf"
    if not test_file.exists():
        pytest.skip("Sample PDF file not found")
    text = file_processor.extract_text(str(test_file))
    assert len(text) > 0
    assert "experience" in text.lower() or "education" in text.lower()

def test_docx_processing(file_processor):
    test_file = Path(__file__).parent / "sample_resumes" / "sample.docx"
    if not test_file.exists():
        pytest.skip("Sample DOCX file not found")
    text = file_processor.extract_text(str(test_file))
    assert len(text) > 0
    assert "experience" in text.lower() or "education" in text.lower()

def test_name_extraction(nlp_engine):
    test_text = "John Doe\nSenior Software Engineer\njohn.doe@example.com"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["name"] == "John Doe"

def test_contact_extraction(nlp_engine):
    test_text = "Contact: john.doe@example.com, (123) 456-7890"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["contact"]["email"] == "john.doe@example.com"
    assert "123" in entities["contact"]["phone"]

def test_skill_normalization(ml_models):
    skills = ["python programming", "ml", "data analysis"]
    normalized = ml_models.normalize_skills(skills)
    assert "Python" in normalized
    assert "Machine Learning" in normalized
    assert "Data Analysis" in normalized

def test_compatibility_scoring(ml_models):
    resume_data = {
        "skills": ["Python", "Machine Learning"],
        "experience": [{"company": "Google", "position": "Data Scientist"}],
        "education": [{"degree": "PhD", "institution": "Stanford University"}]
    }
    job_desc = {
        "title": "Data Scientist",
        "description": "Looking for a Python expert with ML experience",
        "requirements": ["Python", "Machine Learning", "Data Analysis"]
    }
    score = ml_models.calculate_compatibility(resume_data, job_desc)
    assert score["overall_score"] > 0.5
    assert score["skill_match"] > 0.5
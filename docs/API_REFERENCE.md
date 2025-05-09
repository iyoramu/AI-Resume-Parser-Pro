# API Reference

## Base URL
`https://your-api-url.com/v1`

## Authentication
All endpoints require an API key sent in the `X-API-Key` header.

## Endpoints

### POST /parse-resume
Parse a resume file and extract structured information.

**Request:**
- `file`: Resume file (PDF, DOCX, TXT, or image)
- `job_description` (optional): JSON string of job description

**Response:**
```json
{
    "success": true,
    "data": {
        "name": "IRUTABYOSE Yoramu",
        "contact": {
            "email": "yirutabyose@gmail.com",
            "phone": "(+250) 781-101-4782"
        },
        "education": [
            {"degree": "PhD Computer Science", "institution": "Stanford University"}
        ],
        "experience": [
            {"company": "Google", "position": "Senior Software Engineer", "duration": "2015-2020"}
        ],
        "skills": ["Python", "Machine Learning"],
        "certifications": ["AWS Certified Developer"],
        "projects": ["Built recommendation system with 95% accuracy"]
    },
    "compatibility": {
        "overall_score": 0.85,
        "tfidf_similarity": 0.82,
        "semantic_similarity": 0.87,
        "skill_match": 0.90
    }
}
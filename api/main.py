from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import uuid
from datetime import datetime
import json

from core.file_processor import FileProcessor
from core.nlp_engine import NlpEngine
from core.ml_models import MLModels

app = FastAPI(
    title="AI-Powered Resume Parser API",
    description="World-class resume parsing system with NLP and ML capabilities",
    version="1.0.0",
    openapi_tags=[{
        "name": "resume",
        "description": "Operations with resume parsing and analysis"
    }]
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
file_processor = FileProcessor()
nlp_engine = NlpEngine()
ml_models = MLModels()

class JobDescription(BaseModel):
    title: str
    description: str
    requirements: List[str]
    preferred_qualifications: Optional[List[str]] = None

@app.post("/parse-resume", tags=["resume"])
async def parse_resume(
    file: UploadFile = File(...), 
    job_description: Optional[str] = None
):
    """Parse resume and optionally match against job description"""
    try:
        # Save uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1]
        temp_file = f"temp_{uuid.uuid4()}{file_ext}"
        
        with open(temp_file, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Process file
        text = file_processor.extract_text(temp_file)
        entities = nlp_engine.extract_entities(text)
        
        # Normalize skills
        if 'skills' in entities:
            entities['skills'] = ml_models.normalize_skills(entities['skills'])
            
        # Calculate compatibility if job description provided
        compatibility = None
        if job_description:
            try:
                job_data = json.loads(job_description)
            except json.JSONDecodeError:
                job_data = job_description
                
            compatibility = ml_models.calculate_compatibility(entities, job_data)
        
        # Clean up
        os.remove(temp_file)
        
        return {
            "success": True,
            "data": entities,
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-resume", tags=["resume"])
async def match_resume(
    resume_data: Dict[str, Any], 
    job_description: JobDescription
):
    """Match existing resume data against job description"""
    try:
        # Normalize skills if not already done
        if 'skills' in resume_data:
            resume_data['skills'] = ml_models.normalize_skills(resume_data['skills'])
            
        compatibility = ml_models.calculate_compatibility(resume_data, job_description)
        
        return {
            "success": True,
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ResumeEntity(BaseModel):
    name: Optional[str]
    contact: Dict[str, Optional[str]]
    education: List[Dict[str, str]]
    experience: List[Dict[str, Optional[str]]]
    skills: List[str]
    certifications: Optional[List[str]]
    projects: Optional[List[str]]

class CompatibilityScore(BaseModel):
    overall_score: float
    tfidf_similarity: float
    semantic_similarity: float
    skill_match: float

class ParseResponse(BaseModel):
    success: bool
    data: ResumeEntity
    compatibility: Optional[CompatibilityScore]
    timestamp: str

class MatchResponse(BaseModel):
    success: bool
    compatibility: CompatibilityScore
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
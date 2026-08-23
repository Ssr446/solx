from pydantic import BaseModel
from typing import Dict, Optional

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    original_text: str
    normalized_text: str
    scores: Dict[str, float]
    primary_label: str
    explanation: str
    cultural_reference: Optional[str] = None
    severity: str

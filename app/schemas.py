from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionResponse(BaseModel):
    predicted_caption: str
    confidence: float
    predicted_class: Optional[str] = None
    timestamp: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

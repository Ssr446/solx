from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from schemas import AnalyzeRequest, AnalyzeResponse
from normalizer import TanglishNormalizer
from classifier import SOLXClassifier
from context_engine import CulturalContextEngine

app = FastAPI(
    title="SOLX API",
    description="Sarcasm & Online harm Locator with eXplainability API",
    version="1.0.0"
)

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components
normalizer = TanglishNormalizer()
classifier = SOLXClassifier(use_mock=False)
context_engine = CulturalContextEngine()

@app.get("/")
def read_root():
    return {"message": "Welcome to the SOLX API. Use /analyze to analyze Tanglish text."}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    original_text = request.text
    
    # Stage 1: Normalize
    normalized_text = normalizer.normalize(original_text)
    
    # Stage 2: Classify
    scores = classifier.predict(normalized_text)
    error_msg = scores.pop("error_msg", None)
    
    # Stage 3: Explain (Cultural Context)
    context_result = context_engine.lookup(normalized_text)
    
    # Determine primary label and overall severity based on scores
    primary_label = max(scores, key=scores.get) if scores else "Safe"
    if scores[primary_label] < 30.0:
        primary_label = "Safe"
        
    # Build explanation
    if primary_label == "Safe":
        context = {
            "explanation": "This comment does not appear to contain harmful or sarcastic intent.",
            "severity": "Safe",
            "primary_label": "Safe",
            "cultural_reference": context_result.source
        }
    else:
        context = {
            "explanation": f"Flagged for {primary_label} due to the phrase '{context_result.phrase}'. {context_result.explanation}",
            "severity": context_result.severity,
            "primary_label": primary_label,
            "cultural_reference": context_result.source
        }
        
    # Compile response
    explanation = context["explanation"]
    if error_msg:
        explanation = "BACKEND INFERENCE ERROR: " + error_msg
        
    return AnalyzeResponse(
        original_text=original_text,
        normalized_text=normalized_text,
        scores=scores,
        primary_label=context["primary_label"],
        explanation=explanation,
        cultural_reference=context["cultural_reference"],
        severity=context["severity"]
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

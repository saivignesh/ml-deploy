from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Sentiment prediction")

model = joblib.load("logistic.joblib")


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to analyze")


class ModelResponse(BaseModel):
    sentiment: Literal["Positive", "Negative"]
    confidence: float | None = None


@app.get("/")
def welcome():
    return {"message": "Welcome to review sentiment prediction."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ModelResponse)
def predict(request: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pred = model.predict([request.text])[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    confidence = None

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([request.text]).max())

    return ModelResponse(sentiment=sentiment, confidence=confidence)

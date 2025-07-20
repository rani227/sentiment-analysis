from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(title="Sentiment Analysis API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or "*" for dev
    allow_credentials=True,
    allow_methods=["*"],                      # ← this allows OPTIONS
    allow_headers=["*"],                      # ← this allows Content-Type etc.
)



MODEL_PATH = "model"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

class SentimentRequest(BaseModel):
    text: str

logging.basicConfig(level=logging.INFO)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    logging.info(f"Received: {request.text}")
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map.get(predicted_class, "Unknown")

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "raw_scores": probs.squeeze().tolist()
    }

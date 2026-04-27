import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from text_utils import preprocess_text


class CivicTriageEngine:
    def __init__(self):
        print("Loading the RoBERTa & Scikit-Learn Assets")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )

        # 🔹 Sentiment labels
        self.labels = ["Negative", "Neutral", "Positive"]


    # -------------------------------
    # Existing function (unchanged)
    # -------------------------------
    def get_sentiment_probabilities(self, clean_text):
        inputs = self.tokenizer(clean_text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        return probs


    # -------------------------------
    # NEW: Sentiment prediction
    # -------------------------------
    def predict_sentiment(self, text):
        clean_text = preprocess_text(text)
        probs = self.get_sentiment_probabilities(clean_text)

        idx = np.argmax(probs)
        sentiment = self.labels[idx]
        confidence = float(probs[idx])

        return sentiment, confidence


    # -------------------------------
    # NEW: Urgency scoring
    # -------------------------------
    def compute_urgency(self, sentiment, text):
     base_score = {
        "Positive": 1,
        "Neutral": 2,
        "Negative": 3
   }

     urgent_keywords = [
        "urgent", "asap", "immediately",
        "not working", "error", "failed", "complaint", "issue"
    ]

     strong_critical = [
        "not working", "failed", "broken", "danger", "emergency"
    ]

     text_lower = text.lower()

     boost = 0

    # 🔥 Strong override (makes it high priority)
     for word in strong_critical:
        if word in text_lower:
            return 4

    # Normal keyword boost
     for word in urgent_keywords:
        if word in text_lower:
            boost = 1
            break

     score = base_score[sentiment] + boost
     return min(score, 5)


    # -------------------------------
    # NEW: Priority mapping
    # -------------------------------
    def get_priority(self, score):
        if score >= 4:
            return "High"
        elif score == 3:
            return "Medium"
        else:
            return "Low"


    # -------------------------------
    # NEW: Full pipeline
    # -------------------------------
    def analyze(self, text):
        sentiment, confidence = self.predict_sentiment(text)
        urgency = self.compute_urgency(sentiment, text)
        priority = self.get_priority(urgency)

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "urgency_score": urgency,
            "priority": priority
        }


# -------------------------------
# MAIN TEST
# -------------------------------
if __name__ == "__main__":
    triage = CivicTriageEngine()

    test_text = "street light not working, fix this ASAP"

    result = triage.analyze(test_text)

    print(result)
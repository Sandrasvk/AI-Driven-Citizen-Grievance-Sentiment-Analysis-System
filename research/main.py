import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentiment_triage import CivicTriageEngine


app = FastAPI(title = "AI-Driven Citizen Grievance & Sentiment Analysis System", 
              description = """
              ### Automated 311 Triage Pipeline
              ***ML Engine:** Logistic Regression (Dept Routing)
              ***AI Engine:** RoBERTa (Sentiment, Priority & Urgency)
              """,
              version = "1.0.0"
              )

try:
    lr_model = joblib.load("models/lr_model.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    triage_engine = CivicTriageEngine()
    print("SUCCESS: All model loaded successfully.")

except Exception as e:
    print(F"ERROR: Could not load assets. Check your file paths. Error:{e}")

@app.get("/")
def home():
    return{"status": "Online",
           "project": "AI-Driven Citizen Grievance System",
           "version": "1.0.0",
           "models_loaded": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)


